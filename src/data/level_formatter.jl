"""
Format raw level data to a representation suitable for learning.

There are three types of output formats:
  - 1D: Take the row (or column) with the most ground tiles (0x100) and output it as a
        1-dimensional sequence containing ground and empty tiles (0x25).
  - 2D: Output only ground and empty tiles as the sequence in 2D.
  - 3D: Use the whole level as a cuboid where each layer in the cuboid represents a
        different tile (binary mapping).

If the level has exits to sub-levels ("screens"), they are concatenated to the parent level.
Screen exits are ignored for the 1D output format.
"""
module LevelFormatter

using SparseArrays
import SparseArrays: sparse

using ..LevelStatistics
using ..LevelBuilder

export to1d, to2d, to3d, to1d_defaultflags, to2d_defaultflags, to3d_defaultflags


const to1d_defaultflags = "t"
const to2d_defaultflags = "t"
const to3d_defaultflags = "tesx"


"""
    to1d(level::Level; keep::UInt16=0x100, empty::UInt16=0x025, binaryout=true,
         rettype=Int8, singleline=true)
    to1d(file, flags="t"; kwargs...)

Return the contents of the given `level` or `file` as a `Vector{rettype}` where `keep`
indicates which tiles are kept in the output; other tiles are substituted with `empty`.
Only the first layer of the given level ([:, :, 1]) is used for construction.

If `singleline` is `true`, use only the first line containing the maximum amount of `keep`
tiles. Otherwise, squash or flatten over all rows (or columns in vertical levels) so that
you obtain one line of all `keep` tiles in the level.
If `binaryout` is `true`, `keep` tiles are converted to ones and `empty` tiles to zeros.

`rettype` must be wider than `UInt8` if `binaryout` is `false` (`sizeof(rettype) > 1`).
To convert the output to a `String` (instead of a `Vector{String}`), use [`tostring`](@ref).
`file` can be any type that [`buildlevel`](@ref) accepts. See there for the description of
`flags` as well.

## `singleline` behaviour
Take the following horizontal level ('#' are `keep` tiles, '0' are `empty` tiles,
'*' are any other tiles):
```
    #0*#0#
    ##0#*#
    #0##0#
```
With `singleline=true`, this will be converted to:
```
    ##0#0#
```
With `singleline=false`, we get the following:
```
    ####0#
```

# Examples
```julia-repl or jldoctest
```
"""
function to1d(level::Level; kwargs...)
    isvertical(level.stats) && @warn ("this method is not useful with "
                                      * "vertical levels")
    tiles = view(level.data, :, :, firstindex(level.data, 3))
    to1d(tiles; kwargs...)
end

function to1d(level::AbstractMatrix{UInt16}; keep::UInt16=0x100, empty::UInt16=0x025,
              binaryout::Bool=true, rettype::Type=Int8, singleline::Bool=true)
    binaryout || @assert sizeof(rettype) > 1 "too small `rettype`. Try `widen`ing it."

    if singleline
        # Tested – works!
        counts = map(row -> count(isequal(keep), row), eachrow(level))
        maxline = argmax(counts)
        result = similar(view(level, maxline, :), rettype)

        # Indices we want to contain `empty` values (not contained there yet).
        emptyindices = view(level, maxline, :) .!= keep
        if binaryout
            # Faster (and more readable) this way than when converting before.
            result[.~emptyindices] .= 1
            result[emptyindices]   .= 0
        else
            result[.~emptyindices] .= keep
            result[emptyindices]   .= empty
        end
    else
        # Indices we want to contain `keep` values (not contained there yet).
        keepindices = map(col -> any(isequal(keep), Iterators.reverse(col)), eachcol(level))
        result = Vector{rettype}(undef, size(level, 2))

        if binaryout
            result[keepindices]   .= 1
            result[.~keepindices] .= 0
        else
            result[keepindices]   .= keep
            result[.~keepindices] .= empty
        end
    end
    return result
end

function to1d(file, flags::Union{AbstractString, AbstractChar}=to1d_defaultflags; kwargs...)
    level = buildlevel(file, flags)
    to1d(level; kwargs...)
end


"""
    to2d(level::Level; keep::UInt16=0x100, empty::UInt16=0x025, binaryout=true,
         rettype=Int8)
    to2d(file, flags="t"; kwargs...)

Return the contents of the given `level` or `file` as a `Matrix{rettype}` where `keep`
indicates which tiles are kept in the output; other tiles are substituted with `empty`.
Only the first layer of the given level (`[:, :, 1]`) is used for construction.
If `binaryout` is `true`, `keep` tiles are converted to ones and `empty` tiles to zeros.

`rettype` must be wider than `UInt8` if `binaryout` is `false` (`sizeof(rettype) > 1`).
To convert the output to a `String` (instead of a `Matrix{String}`), use [`tostring`](@ref).
`file` can be any type that [`buildlevel`](@ref) accepts. See there for the description of
`flags` as well.

# Example
```julia-repl or jldoctest
```
"""
function to2d(level::Level; kwargs...)
    tiles = view(level.data, :, :, firstindex(level.data, 3))
    to2d(tiles; kwargs...)
end

function to2d(level::AbstractMatrix{UInt16}; keep::UInt16=0x100, empty::UInt16=0x025,
              binaryout::Bool=true, rettype::Type=Int8)
    binaryout || @assert sizeof(rettype) > 1 "too small `rettype`. Try `widen`ing it."

    # Indices we want to contain `keep` values (not contained there yet).
    keepindices = level .== keep
    result = similar(level, rettype)

    if binaryout
        result[keepindices]   .= 1
        result[.~keepindices] .= 0
    else
        result[keepindices]   .= keep
        result[.~keepindices] .= empty
    end
    return result
end

function to2d(file, flags::Union{AbstractString, AbstractChar}=to2d_defaultflags; kwargs...)
    level = buildlevel(file, flags)
    to2d(level; kwargs...)
end

# TODO Could refactor this into `to2d` using `1 - Int(keepempty)` or
# `0 + Int(keepempty)`.
# TODO bad name
""""
    to2d_keepempty(level::AbstractMatrix{UInt16}; empty::UInt16=0x000, binaryout=true,
                   rettype=Int8)

Like [`to2d`](@ref) but keep all values that are not `empty` unless `binaryout` is `true` in
which case non-`empty` values are converted to ones.
"""
function to2d_keepempty(level::AbstractMatrix{UInt16}; empty::UInt16=0x000,
                        binaryout::Bool=true, rettype::Type=Int8)
    binaryout || @assert sizeof(rettype) > 1 "too small `rettype`. Try `widen`ing it."

    if binaryout
        # Indices we want to contain `one(rettype)` (not contained there yet).
        keepindices = level .!= empty
        result = similar(level, rettype)
        result[keepindices] .= 1
        result[.~keepindices] .= 0
    else
        result = convert(Matrix{rettype}, level)
        if empty != 0x000
            emptyindices = level .== 0x000
            result[emptyindices] .= empty
        end
    end
    return result
end


"""
    to3d(level::Level, tiles=true; empty::UInt16=0x025, binaryout=true, rettype=Int8,
         tiles=true)
    to3d(file, flags="tesx"; kwargs...)

Return the contents of the given `level` or `file` as an `Array{rettype, 3}` where each
layer represents all positions of one object type. Tiles on which the object does not occur
are set to `empty`.

If `tiles` is `true`, the first layer in the level ([:, :, 1]) is assumed to be a layer of
different tiles that will be deconstructed into individual layers per object type
(automatically set if `flags` contains `'t'`).
If `binaryout` is `true`, all non-`empty` tiles are converted to ones and `empty` tiles
to zeros.

# Example
```jldoctest
julia> level = to3d(0x105);
[...]

julia> size(level)[1:2]
(27, 320)

julia> size(level, 3) > 0  # This size can vary.
true
```
"""
function to3d(level::Level, args...; kwargs...)
    # `level.data::Array{UInt16, 3}` so call corresponding method.
    to3d(level.data, args...; kwargs...)
end

function to3d(level::AbstractArray{UInt16, 3}, tiles::Bool=true; binaryout::Bool=true,
              rettype::Type=Int8, kwargs...)
    binaryout || @assert sizeof(rettype) > 1 "too small `rettype`. Try `widen`ing it."

    to3d(level, Val(tiles); binaryout=binaryout, rettype=rettype, kwargs...)
end

function to3d(level::AbstractArray{UInt16, 3}, #=tiles=#::Val{true}; empty::UInt16=0x025,
              binaryout::Bool=true, rettype::Type=Int8)
    # Set up tiles...
    tilelayer = view(level, :, :, firstindex(level, 3))
    nontilelayers = view(level, :, :, firstindex(level, 3) + 1:lastindex(level, 3))

    tileresult = to3d(tilelayer, empty=empty, binaryout=binaryout, rettype=rettype)
    result = to3d(nontilelayers, Val(false), empty=0x000, binaryout=binaryout,
                  rettype=rettype)
    return cat(tileresult, result, dims=3)
end

function to3d(level::AbstractArray{UInt16, 3}, #=tiles=#::Val{false}; empty::UInt16=0x000,
              binaryout::Bool=true, rettype::Type=Int8)
    # Fill result with non-tile layers.
    if rettype === UInt16 && !binaryout && empty == 0x000
        return level
    end

    result = similar(level, rettype)
    # Indexing method should be the same for `level` and `result` due to `similar`.
    for i in axes(level, 3)
        result[:, :, i] = to2d_keepempty(view(level, :, :, i), empty=empty,
                                         binaryout=binaryout, rettype=rettype)
    end
    return result
end

function to3d(level::AbstractMatrix{UInt16}; empty::UInt16=0x025, binaryout::Bool=true,
              rettype::Type=Int8)
    result = Array{rettype, 3}(undef, size(level)..., uniquevanillatiles)
    # We iterate from 0 to `uniquevanillatiles - 1`, but as we would add 1 due to 1-based
    # indexing, we instead iterate from 1 to `uniquevanillatiles` and do not add 1
    # when indexing.
    for tileindex in vcat(0x1:empty - 0x1, empty + 0x1:uniquevanillatiles)
        result[:, :, tileindex] = to2d(level, keep=tileindex, empty=empty,
                                       binaryout=binaryout, rettype=rettype)
    end
    return result
end

function to3d(file, flags::Union{AbstractString, AbstractChar}=to3d_defaultflags; kwargs...)
    level = buildlevel(file, flags)
    to3d(level, 't' in flags; kwargs...)
end


"Return the given `AbstractArray{T, 3}` as a `Vector{SparseMatrixCSC{T}}`."
function sparse(array::AbstractArray{T, 3}) where T
    map(sparse, eachslice(array, dims=3))
end

"Return the given `AbstractVector{SparseMatrixCSC{T}}` as an `Array{T, 3}`."
function unsparse(sparsearray::AbstractVector{SparseMatrixCSC{T}}) where T
    reduce((a, b) -> cat(a, b, dims=3), map(Array, sparsearray))
end

"""
    tostring(level::AbstractVector)
    tostring(level::AbstractMatrix, flatten=Val(false), separator='\n')

Return the given level as a `String`. If `flatten` is `false`, return an `Array{String}`;
otherwise, a `String` where each column is separated by `separator`.
Use a `Val` argument to get a type stable version.
"""
tostring(level::AbstractVector) = join(level)

tostring(level::AbstractMatrix, args...) = tostring(level, Val(false), args...)

function tostring(level::AbstractMatrix, flatten::Bool, args...)
    tostring(level, Val(flatten), args...)
end

function tostring(level::AbstractMatrix, flatten::Val{false}, args...)
    map(tostring, eachcol(level))
end

function tostring(level::AbstractMatrix, flatten::Val{true}, separator::AbstractChar)
    join(tostring(level, Val(false)), '\n')
end

function tostring(level::AbstractArray{T, 3}, args...) where T
    # TODO
    error("not implemented")
end

function transpose(string::String)
    # TODO String transpose function for better level viewing.
    # Split on newlines, map(collect, *), flatten to array, ...
    error("not implemented")
end

end # module

