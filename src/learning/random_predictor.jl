module RandomPredictor

using ..InputStatistics
using ..ModelUtils
import ..ModelUtils: calculate_loss, step!

export RandomModel, random1d, random2d, random3dtiles, random3d

# TODO make random chance "learnable"? (online mean)
struct RandomModel <: LearningModel
    activationchance::Float32
    inputsize::Int
    outputsize::Int
    hyperparams::Dict{Symbol, Any}

    function RandomModel(activationchance::Real,
                         inputsize::Integer, outputsize::Integer, dimensionality::Symbol)
        @assert 0 <= activationchance < 1 "invalid activation chance. Must be 0 <= p < 1."
        new(activationchance, inputsize, outputsize, Dict{Symbol, Any}(
            :dimensionality => dimensionality,

            :activationchance => activationchance,
            :inputsize        => inputsize,
            :outputsize       => outputsize,
        ))
    end
end

function (model::RandomModel)(input)
    inputlength = length(input)
    @assert inputlength == model.inputsize "input size does not match"
    result = Vector{eltype(input)}(undef, model.outputsize)
    fillrandom!(result, model.activationchance)
end

calculate_loss(::RandomModel, loss, input, target) = loss(input, target)

function step!(model::RandomModel, #=parameters=#::Any, #=optimizer=#::Any,
               loss, input, target)
    return calculate_loss(model, loss, input, target)
end


function fillrandom!(array::AbstractArray, activationchance::Real)
    for i in eachindex(array)
        array[i] = rand() < activationchance ? 1 : 0
    end
    return array
end

function random1d(activationchance::Real=0.5f0, inputsize::Integer=inputsize1d,
                  outputsize::Integer=outputsizeof(inputsize))
    RandomModel(activationchance, inputsize, outputsize, Symbol("1d"))
end

function random2d(activationchance::Real=0.5f0, inputsize::Integer=inputsize2d,
                  outputsize::Integer=outputsizeof(inputsize))
    RandomModel(activationchance, inputsize, outputsize, Symbol("2d"))
end

function random3dtiles(activationchance::Real=0.5f0, inputsize::Integer=inputsize3dtiles,
                       outputsize::Integer=outputsizeof(inputsize))
    RandomModel(activationchance, inputsize, outputsize, Symbol("3dtiles"))
end

function random3d(activationchance::Real=0.5f0, inputsize::Integer=inputsize3d,
                  outputsize::Integer=outputsizeof(inputsize))
    RandomModel(activationchance, inputsize, outputsize, Symbol("3d"))
end

end # module

