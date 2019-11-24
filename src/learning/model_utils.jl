module ModelUtils

import Base.clamp!
using SparseArrays

using Adapt: adapt
using BSON
using CUDAnative
using CuArrays
using CuArrays.CUSPARSE
import Flux
using Flux.Tracker: gradient
#using Zygote

# This import is only for documentation reference purposes.
import ..LevelFormatter

export LearningModel, AbstractDiscriminator, AbstractGenerator
export makeloss, dataiteratorparams, calculate_loss, step!
export makesoftloss, soft_criterion, makenoisebatch
export toggle_gpu, should_use_gpu, togpu, tocpu
export mae, bce, leakyrelu
export BatchToMatrix, MatrixTo3DBatch, MatrixTo2DBatch, ConvNoBias, ConvTransposeNoBias

"""
    LearningModel

An abstract type for trainable models holding the model, parameters and other
important information.

First off, to make the underlying model trainable, make the `LearningModel`
a `Flux.@treelike`.
A `LearningModel` has to implement one field `hyperparams::Dict{Symbol, Any}` which holds
any parameters that differentiate the model from another model of the same type (or not).
The following keys are required in the `hyperparams` dictionary.
   - dimensionality::Symbol: What kind of level data will this be used with, which input
                             will it accept? Can be any `Symbol` in
                             [`LevelFormatter.dimensionality_defaultflags`](@ref).

An implementation `YourModel <: LearningModel` should also implement the
following functions:
   - `(model::YourModel)(input)`: Apply the underlying model to the input.
   - `makeloss(model::YourModel, criterion)`: See [`makeloss`](@ref).
   - `dataiteratorparams(model::YourModel)`: See [`dataiteratorparams`](@ref).
   - `calculate_loss(model::YourModel, loss, input, target)`: See [`calculate_loss`](@ref).
   - `step!(model::YourModel, parameters, optimizer, loss, input, target)`:
     See [`step!`](@ref).
   - Optionally:
      - `makesoftloss(model::YourModel, criterion)`: See [`makesoftloss`](@ref).
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
    # Loss for a batch of `AbstractVector{<:AbstractMatrix}` data.
    function loss(x, y)
        y_hat = model.(x)
        rows = size(first(y), 1)
        l = sum(sum(criterion.(eachcol(e_hat), eachcol(e))
                    for (e_hat, e) in zip(y_hat, y)))
        # l = @inbounds @views sum(criterion(y_hat[i][:, col], y[i][:, col])
        #                          for i in eachindex(y) for col in axes(y[i], 2))
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
    # Loss for a batch of `AbstractVector{<:AbstractMatrix}` data.
    for i in eachindex(y)
        for col in axes(y[i], 2)
            @inbounds @views e_hat, e = y_hat[i][:, col], y[i][:, col]
            if last_elem != second_to_last
                total += criterion(e_hat, e)
            else
                total += 0.1f0 * criterion(e_hat, e)
            end
            second_to_last = last_elem
            last_elem = e
        end
    end
    return total
end


# Standard GAN discriminator loss.
"""
Return a 2-argument loss function applying the discriminator to the batch of input images
`x` and return the loss of the predictions in relation to whether the images were real
(`y[i] == 1`).
"""
function makeloss(model::AbstractDiscriminator, criterion)
    function loss(x, y)
        # Discriminator loss
        y_hat = vec(model(x))
        l = sum(criterion.(y_hat, y))
        return l
    end
end

function calculate_loss(::AbstractDiscriminator, loss, input, target)
    loss(input, target)
end

# Standard GAN discriminator training step.
function step!(d_model::AbstractDiscriminator, d_params, d_optim, d_loss,
               real_batch, real_target, g_model, fake_target, curr_batch_size)
    # Train on real batch
    d_l_real = calculate_loss(d_model, d_loss, real_batch, real_target)

    # Train on fake batch
    noise_batch = makenoisebatch(g_model, curr_batch_size)
    fake_batch = g_model(noise_batch)

    d_l_fake = calculate_loss(d_model, d_loss, fake_batch, fake_target)
    d_l = d_l_real + d_l_fake

    grads = gradient(() -> d_l, d_params)

    # Update
    Flux.Optimise.update!(d_optim, d_params, grads)
    return d_l, d_l_real, d_l_fake
end


# Standard GAN generator loss.
"""
Return a 2-argument loss function applying the generator to the random input `x` and return
the loss in relation to the discriminator's loss.
"""
function makeloss(g_model::AbstractGenerator, d_model, criterion)
    function loss(x, y)
        # Generator loss
        fakes = g_model(x)
        y_hat = d_model(fakes)
        l = sum(criterion.(y_hat, y))
        return l
    end
end

function calculate_loss(::AbstractGenerator, loss, input, target)
    loss(input, target)
end

# Standard GAN generator training step.
function step!(g_model::AbstractGenerator, g_params, g_optim, g_loss,
               real_target, curr_batch_size)
    noise_batch = makenoisebatch(g_model, curr_batch_size)

    # Use real labels for modified loss function
    g_l = calculate_loss(g_model, g_loss, noise_batch, real_target)

    grads = gradient(() -> g_l, g_params)

    # Update
    Flux.Optimise.update!(g_optim, g_params, grads)
    return g_l
end

function makenoisebatch(g_model, curr_batch_size)
    generator_inputsize = g_model.hyperparams[:inputsize]
    if g_model.hyperparams[:dimensionality] === Symbol("1d")
        noise_batch = togpu(randn(1, generator_inputsize, curr_batch_size))
    else
        noise_batch = togpu(randn(1, 1, generator_inputsize, curr_batch_size))
    end
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

function togpu(optim::Flux.ADAM)
    gpu_states = IdDict()
    for (k, (m, v, beta)) in optim.state
        gpu_states[togpu(k)] = (togpu(m), togpu(v), beta)
    end
    return Flux.ADAM(optim.eta, optim.beta, gpu_states)
end

function togpu(optim::Flux.RMSProp)
    gpu_acc = IdDict()
    for (k, accval) in optim.acc
        gpu_acc[togpu(k)] = togpu(accval)
    end
    return Flux.RMSProp(optim.eta, optim.rho, gpu_acc)
end

"""
    tocpu(x)

Map `x` to the CPU.
"""
tocpu(x::CuArray) = adapt(Array, x)
tocpu(x) = Flux.cpu(x)

function tocpu(optim::Flux.ADAM)
    cpu_states = IdDict()
    for (k, state_tuple) in optim.state
        cpu_states[tocpu(k)] = tocpu.(state_tuple)
    end
    return Flux.ADAM(optim.eta, optim.beta, cpu_states)
end

function tocpu(optim::Flux.RMSProp)
    cpu_acc = IdDict()
    for (k, accval) in optim.acc
        cpu_acc[tocpu(k)] = tocpu(accval)
    end
    return Flux.RMSProp(optim.eta, optim.rho, cpu_acc)
end


"Mean absolute error."
mae(y_hat, y) = sum(abs.(y_hat .- y)) * 1 // length(y)

CuArrays.@cufunc function Flux.binarycrossentropy(ŷ, y; ϵ=eps(ŷ))
    return -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)
end

"Approximate PyTorch binary cross entropy loss."
bce(y_hat, y) = Flux.binarycrossentropy(y_hat, y; ϵ=1f-12)

leakyrelu(x) = Flux.leakyrelu(x, 0.2f0)


"""
Reshape `x` of a size like `(a, b, c, B)` or `(a, b, B)`, where `B` is the batch size to a
matrix of size `(num_features, B)` (`num_features` should be  `a * b * c`).
"""
struct BatchToMatrix
# TODO had to remove the below as reshape does not support all `Integer` types.
# struct BatchToMatrix{I<:Integer}
    num_features::Int
end

(b::BatchToMatrix)(x) = reshape(x, b.num_features, :)

"""
Reshape the matrix `x` of size (`N, B`) to `(a, b, c, B)`, where `B` is the batch size.
`a`, `b` and `c` are given by `outputsize`.
"""
struct MatrixTo3DBatch{I<:Integer}
    outputsize1::I
    outputsize2::I
    outputsize3::I
end

# TODO have to convert to `Int` as reshape does not support all `Integer` types.
MatrixTo3DBatch(outputsize) = MatrixTo3DBatch(Int.(outputsize)...)

(b::MatrixTo3DBatch)(x) = reshape(x, b.outputsize1, b.outputsize2, b.outputsize3, :)

"""
Reshape the matrix `x` of size (`N, B`) to `(a, b, B)`, where `B` is the batch size.
`a` and `b` are given by `outputsize`.
"""
struct MatrixTo2DBatch{I<:Integer}
    outputsize1::I
    outputsize2::I
end

# TODO have to convert to `Int` as reshape does not support all `Integer` types.
MatrixTo2DBatch(outputsize) = MatrixTo2DBatch(Int.(outputsize)...)

(b::MatrixTo2DBatch)(x) = reshape(x, b.outputsize1, b.outputsize2, :)

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

function Base.clamp!(a::CuArray, low, high)
    function kernel(a, low, high)
        I = CuArrays.@cuindex a
        a[I...] = clamp(a[I...], low, high)
        return
    end
    blocks, threads = CuArrays.cudims(a)
    @cuda blocks=blocks threads=threads kernel(a, low, high)
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

