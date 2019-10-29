module ModelUtils

using SparseArrays

using BSON
using CuArrays.CUSPARSE
import Flux
using Flux.Tracker: gradient
#using Zygote

export LearningModel, makeloss, dataiteratorparams, calculate_loss, step!
export should_use_gpu, togpu

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
   - (model::YourModel)(input): Apply the underlying model to the input.
   - makeloss(model::YourModel, criterion): See [`makeloss`](@ref).
   - dataiteratorparams(model::YourModel): See [`dataiteratorparams`](@ref).
   - calculate_loss(model::YourModel, loss, input, target): See [`calculate_loss`](@ref).
   - step!(model::YourModel, parameters, optimizer,
           loss, input, target): See [`step!`](@ref).
It may be possible to use the standard definitions of some of the above functions. See their
respective implementation to be sure. They are chosen to be sensible defaults but may not
be correct for `YourModel`.
"""
abstract type LearningModel end

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
dataiteratorparams(::Any) = (join_pad=false, pad_each=false)

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

Update the given model with a single training step on the next data point in the
given `trainiter`. Return the loss.
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

