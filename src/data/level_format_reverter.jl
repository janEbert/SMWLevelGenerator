module LevelFormatReverter

using SparseArrays

using ..LevelBuilder
using ..LevelFormatter
using ..InputStatistics

# TODO from2d

# TODO (higher priority, but may be able to use from2d) from3d

# Do not use this to try and construct a level.
function from1d(data::AbstractVector,
                flags::Union{AbstractString, AbstractChar}=to1d_defaultflags)
    round.(UInt16, data)
end

function from2d(data::AbstractMatrix, keep::UInt16=0x100, empty::UInt16=0x025,
                flags::Union{AbstractString, AbstractChar}=to2d_defaultflags)
    level = round.(UInt16, data)
    keepindices = level .== 1
    level[keepindices]   .= keep
    level[.~keepindices] .= empty
    return level
end

function from3d(data::AbstractArray{T, 3}, empty::UInt16=0x025,
                flags::Union{AbstractString, AbstractChar}=to3d_defaultflags) where T
    from3d(data, empty, 't' in flags)
end

function from3d(data::AbstractArray{T, 3}, empty::UInt16=0x025, tiles::Bool=true) where T
    level = round.(UInt16, data)
    from3d(level, empty, Val(tiles))
end

function from3d(data::AbstractArray{UInt16, 3}, empty::UInt16=0x025,
                #=tiles=#::Val{true}) where T
    if 't' in flags # yes
    end

    # TODO
    currindex = 1
    for layer in eachslice(level, dims=3)
    end
end

function from3d(data::AbstractArray{UInt16, 3}, empty::UInt16=0x025,
                #=tiles=#::Val{false}) where T
    if 't' in flags # nope
    end
    # TODO

    for layer in eachslice(level, dims=3)
    end
end

function from3d(data::AbstractVector{SparseMatrixCSC{T}}, args...) where T
    from3d(unsparse(data), args...)
end

end # module

