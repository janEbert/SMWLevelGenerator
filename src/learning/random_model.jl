module RandomModel

using ..InputStatistics

export random1d, random2d, random3dtiles, random3d

function fillrandom!(array::AbstractArray, activationchance::Real)
    for i in eachindex(array)
        array[i] = rand() < activationchance ? 1 : 0
    end
    return array
end

function makerandommodel(activationchance::Real, inputsize::Integer)
    function randommodel(input)
        inputlength = length(input)
        @assert inputlength == inputsize
        result = Vector{eltype(input)}(undef, inputlength)
        fillrandom!(result, activationchance)
    end
end

function random1d(activationchance::Real=0.5)
    makerandommodel(activationchance, inputsize1d)
end

function random2d(activationchance::Real=0.5, inputsize::Integer=inputsize2d)
    makerandommodel(activationchance, inputsize)
end

function random3dtiles(activationchance::Real=0.5, inputsize::Integer=inputsize3dtiles)
    makerandommodel(activationchance, inputsize)
end

function random3d(activationchance::Real=0.5, inputsize::Integer=inputsize3d)
    makerandommodel(activationchance, inputsize)
end

end # module

