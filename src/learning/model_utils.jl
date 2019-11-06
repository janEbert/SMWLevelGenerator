module ModelUtils

import Base.clamp!
using SparseArrays

using BSON
using CUDAnative
using CuArrays
using CuArrays.CUSPARSE
import Flux
using Flux.Tracker: gradient
#using Zygote

export LearningModel, AbstractDiscriminator, AbstractGenerator
export makeloss, dataiteratorparams, calculate_loss, step!
export makesoftloss, soft_criterion
export toggle_gpu, should_use_gpu, togpu
export mae, bce, leakyrelu
export ConvNoBias, ConvTransposeNoBias

"""
    LearningModel

An abstract type for trainable models holding the model, parameters and other
important information.

First off, to make the underlying model trainable, make the `LearningModel`
a `Flux.@treelike`.
A `LearningModel` has to implement one field `hyperparams::Dict{Symbol, Any}` which holds any
parameters that differentiate the model from another model of the same type (or not).
The following keys are required in the `hyperparams` dictionary.
   - dimensionality::Symbol: What kind of level data will this be used with, which input
                             will it accept? Can be any `Symbol` of "1d", "2d", "3dtiles"
                             or "3d".

An implementation `YourModel <: LearningModel` should also implement the
following functions:
   - `(model::YourModel)(input)`: Apply the underlying model to the input.
   - `makeloss(model::YourModel, criterion)`: See [`makeloss`](@ref).
   - `dataiteratorparams(model::YourModel)`: See [`dataiteratorparams`](@ref).
   - `calculate_loss(model::YourModel, loss, input, target)`: See [`calculate_loss`](@ref).
   - `step!(model::YourModel, parameters, optimizer, loss, input, target)`:
     See [`step!`](@ref).
   - Optionally: `soft_criterion(model::YourModel, y_hat, y, criterion)`:
                 See [`soft_criterion`](@ref).
It may be possible to use the standard definitions of some of the above functions. See their
respective implementation to be sure. They are chosen to be sensible defaults but may not
be correct for `YourModel`.
"""
abstract type LearningModel end

abstract type AbstractDiscriminator <: LearningModel end
abstract type AbstractGenerator <: LearningModel end

# TODO allow setting the precision of the models (may not work due to Flux)

# The defaults here were chosen to be the most intuitive; they modify the data the least.
# Reasoning about a sequence of data in that same format seems easier.
# They also don't correspond to a single model.

# TODO Uncomment and remove class overrides when Julia 1.3 is stable
# (model::LearningModel)(input) = model.model(input)

"""
    makeloss(model, criterion)

Return a 2-argument loss function applying the model to the input sequence `x` and return
the loss of the prediction in relation to the target sequence `y`.
"""
function makeloss(model, criterion)
    function loss(x, y)
        y_hat = model.(x)
        l = sum(criterion.(y_hat, y))
        return l
    end
end

"""
    dataiteratorparams(model)

Return a `NamedTuple` of keyword arguments to pass to the `DataIterator.dataiterator`
function so the model gets fed correctly formatted data.
"""
dataiteratorparams(::Any) = (join_pad=false, as_matrix=false)

"""
    calculate_loss(model, loss, input, target)

Return the loss obtained by applying the loss function to the given input and target.
The model is passed in case the model needs to be prepared beforehand (for example
via [`Flux.reset!`](@ref)).
"""
function calculate_loss(model, loss, input, target)
    # Reset hidden state (can be called on any `Flux.@treelike`)
    Flux.reset!(model)
    loss(input, target)
end

"""
    step!(model, parameters, optimizer, loss, input, target)

Update the given model with a single training step on the given input with the given target.
Return the loss.
"""
function step!(model, parameters, optimizer, loss, input, target)
    l = calculate_loss(model, loss, input, target)

    # Take gradients
    grads = gradient(() -> l, parameters)

    # Update weights
    Flux.Optimise.update!(optimizer, parameters, grads)
    return l
end


"""
    makesoftloss(model, criterion)

Return a 2-argument loss function applying the model to the input sequence `x` and return
the loss of the prediction in relation to the target sequence `y`.
"""
function makesoftloss(model, criterion)
    function loss(x, y)
        y_hat = model.(x)
        l = soft_criterion(model, y_hat, y, criterion)
        return l
    end
end

# A possible criterion to reduce over-generalizing. Loss function must be changed for this
# as this criterion shouldn't be broadcasted.
"""
Return a reduced loss from predicting a sequence element differently when the two previous
sequence elements were the same.
"""
function soft_criterion(#=model=#::Any, y_hat, y, criterion)
    total = 0.0f0
    last_elem = nothing
    second_to_last = nothing
    for (e_hat, e) in zip(y_hat, y)
        if last_elem != second_to_last
            total += criterion(e_hat, e)
        else
            total += 0.1f0 * criterion(e_hat, e)
        end
        second_to_last = last_elem
        last_elem = e
    end
    return total
end


"Toggle GPU usage."
toggle_gpu() = (ENV["SMWLG_IGNORE_GPU"] = !should_use_gpu())

"""
    should_use_gpu()

Whether to use the GPU. The behaviour can be controlled by setting `ENV["SMWLG_IGNORE_GPU"]`
to `true` to actively ignore the GPU even if it is available.
"""
function should_use_gpu()
    return !haskey(ENV, "SMWLG_IGNORE_GPU") || !parse(Bool, ENV["SMWLG_IGNORE_GPU"])
end

"""
    togpu(x)

Map `x` to the GPU.
Set `ENV["SMWLG_IGNORE_GPU"]` to `true` to actively ignore the GPU even if it is available.
"""
togpu(x) = should_use_gpu() ? Flux.gpu(x) : identity(x)

# When CUSPARSE supports indexing, this can be uncommented for a nice speedup in
# larger dimensions.
# togpu(x::AbstractSparseMatrix) = should_use_gpu() ? CuSparseMatrixCSR(x) : identity(x)

# togpu(x::AbstractSparseVector) = should_use_gpu() ? CuSparseVector(x) : identity(x)


"Mean absolute error."
mae(y_hat, y) = sum(abs.(y_hat .- y)) * 1 // length(y)

CuArrays.@cufunc function Flux.binarycrossentropy(ŷ, y; ϵ=eps(ŷ))
    return -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)
end

"Approximate PyTorch binary cross entropy loss."
bce(y_hat, y) = Flux.binarycrossentropy(y_hat, y; ϵ=1f-12)

leakyrelu(x) = Flux.leakyrelu(x, 0.2f0)


"A convolutional layer without learnable bias."
function ConvNoBias(k::NTuple{N, Integer}, ch::Pair{<:Integer, <:Integer},
                    activation=identity; init=Flux.glorot_uniform, stride=1, pad=0,
                    dilation=1) where N
    Flux.Conv(Flux.param(init(k..., ch...)), zeros(Float32, ch[2]), activation,
              stride=stride, pad=pad, dilation=dilation)
end

"A convolutional transpose layer without learnable bias."
function ConvTransposeNoBias(k::NTuple{N, Integer}, ch::Pair{<:Integer, <:Integer},
                             activation=identity; init=Flux.glorot_uniform, stride=1, pad=0,
                             dilation=1) where N
    Flux.ConvTranspose(Flux.param(init(k..., reverse(ch)...)), zeros(Float32, ch[2]),
                       activation, stride=stride, pad=pad, dilation=dilation)
end

function Base.clamp!(a::CuVecOrMat, low, high)
    function kernel(a, low, high)
        col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        if col <= size(a, 1) && row <= size(a, 2)
            a[col, row] = clamp(a[col, row], low, high)
        end
        return
    end

    max_threads = 256
    threads_x = min(max_threads, size(a, 1))
    threads_y = min(div(max_threads, threads_x), size(a, 2))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (size(a, 1), size(a, 2)) ./ threads)

    @cuda threads=threads blocks=blocks kernel(a, low, high)
    return a
end




"""
Convert all checkpoints in the given directory according to the given 1-argument
conversion function taking the previously loaded checkpoint and returning a `NamedTuple`
of the new checkpoint.
"""
function convertcps(dir, conversion_function)
    for cppath in map(f -> joinpath(dir, f),
                      filter(f -> endswith(f, ".bson"), readdir(dir)))
        cp = BSON.load(cppath)
        new_cp = conversion_function(cp)
        bson(cppath; new_cp...)
    end
end

end # module

