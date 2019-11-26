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

using ..LevelStatistics
using ..LevelBuilder

export to1d, to2d, to3d, dimensionality_defaultflags


"""
Default flags for each type (or dimensionality) of training data.
All of these must start with either "1d", "2d" or "3d".
"""
const dimensionality_defaultflags = Dict{Symbol, String}(
    Symbol("1d") => "t",
    Symbol("2d") => "t",
    Symbol("3dtiles") => "t",
    Symbol("3d") => "tesx",
)


"""
    to1d(level::Level; flags="t", keep::UInt16=0x100, empty::UInt16=0x025, binaryout=true,
         rettype=Int8, squash=false)
    to1d(file, flags="t"; kwargs...)

Return the contents of the given `level` or `file` as a `Vector{rettype}` where `keep`
indicates which tiles are kept in the output; other tiles are substituted with `empty`.
Only the first layer of the given level ([:, :, 1]) is used for construction.

If `squash` is `false`, use only the first line containing the maximum amount of `keep`
tiles. Otherwise, squash or flatten over all rows (or columns in vertical levels) so that
you obtain one line of all `keep` tiles in the level.
If `binaryout` is `true`, `keep` tiles are converted to ones and `empty` tiles to zeros.

`rettype` must be wider than `UInt8` if `binaryout` is `false` (`sizeof(rettype) > 1`).
To convert the output to a `String` (instead of a `Vector{String}`), use [`tostring`](@ref).
`file` can be any type that [`buildlevel`](@ref) accepts. See there for the description of
`flags` as well.

## `squash` behaviour
Take the following horizontal level ('#' are `keep` tiles, '0' are `empty` tiles,
'*' are any other tiles):
```
    #0*#0#
    ##0#*#
    #0##0#
```
With `squash=false`, this will be converted to:
```
    ##0#0#
```
With `squash=true`, we get the following:
```
    ####0#
```
"""
function to1d(level::Level; flags::Union{AbstractString,
                                 AbstractChar}=dimensionality_defaultflags[Symbol("1d")],
              kwargs...)
    isvertical(level.stats) && @warn ("this method is not useful with "
                                      * "vertical levels")
    if 't' in flags
        level, groundtile = find_ground_tile(level, flags)
        tiles = view(level.data, :, :, firstindex(level.data, 3))
        to1d(tiles; keep=groundtile, kwargs...)
    else
        tiles = view(level.data, :, :, firstindex(level.data, 3))
        to1d(tiles; kwargs...)
    end
end

function to1d(level::AbstractMatrix{UInt16}; keep::UInt16=0x100, empty::UInt16=0x025,
              binaryout::Bool=true, rettype::Type=Int8, squash::Bool=false)
    binaryout || @assert sizeof(rettype) > 1 "too small `rettype`. Try `widen`ing it."

    if !squash
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

function to1d(file, flags::Union{AbstractString,
                                 AbstractChar}=dimensionality_defaultflags[Symbol("1d")];
              kwargs...)
    level = buildlevel(file, flags * 'e')
    to1d(level; flags=flags, kwargs...)
end


"""
    to2d(level::Level; flags="t", keep::UInt16=0x100, empty::UInt16=0x025, binaryout=true,
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
function to2d(level::Level; flags::Union{AbstractString,
                                 AbstractChar}=dimensionality_defaultflags[Symbol("2d")],
              kwargs...)
    if 't' in flags
        level, groundtile = find_ground_tile(level, flags)
        tiles = view(level.data, :, :, firstindex(level.data, 3))
        to2d(tiles; keep=groundtile, kwargs...)
    else
        tiles = view(level.data, :, :, firstindex(level.data, 3))
        to2d(tiles; kwargs...)
    end
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

function to2d(file, flags::Union{AbstractString,
                                 AbstractChar}=dimensionality_defaultflags[Symbol("2d")];
              kwargs...)
    level = buildlevel(file, flags)
    to2d(level; flags=flags, kwargs...)
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
    to3d(level::Level, tiles=true; flags="tesx", empty::UInt16=0x025, binaryout=true,
         rettype=Int8, tiles=true)
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

julia> ndims(level)  # Size of dimension 3 can vary.
3
```
"""
function to3d(level::Level, args...;
              flags::Union{AbstractString,
                           AbstractChar}=dimensionality_defaultflags[Symbol("3d")],
              kwargs...)
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
    for (i, j) in zip(axes(level, 3), axes(result, 3))
        result[:, :, j] = to2d_keepempty(view(level, :, :, i), empty=empty,
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
    # Assign the empty layer.
    result[:, :, empty] = to2d(level, keep=empty, empty=0x000, binaryout=binaryout,
                               rettype=rettype)
    return result
end

function to3d(file, flags::Union{AbstractString,
                                 AbstractChar}=dimensionality_defaultflags[Symbol("3d")];
              kwargs...)
    level = buildlevel(file, flags)
    to3d(level, 't' in flags; kwargs...)
end


"""
    find_ground_tile(file, flags)

Return the level in the given file loaded with the given flags (see [`buildlevel`](@ref))
and the first tile in the level below Mario's starting position that is non-empty (in
vertical levels the first non-empty tile to the right due to the transposition).
"""
function find_ground_tile(file, flags)
    @assert 't' in flags "no tiles in level; cannot find ground tile"
    if 'e' in flags
        level = buildlevel(file, flags)
    else
        level = buildlevel(file, flags * 'e')
    end
    find_ground_tile(level, flags)
end

function find_ground_tile(level::Level, flags)
    remove_entrances = !('e' in flags)

    entrance_y, entrance_x = mainentrance(level.stats)
    # Start at the first tile below Mario's actual starting position.
    entrance_y += 2
    # Copy so vertical caching is faster. Also, the horizontal copy is very small.
    tilelayer = LevelBuilder.tilelayer(level)
    if isvertical(level.stats)
        if entrance_x > size(tilelayer, 1)
            @warn "entrance at x=$(entrance_x) out of bounds; using default ground tile."
            searchregion = nothing
        else
            searchregion = tilelayer[entrance_x, entrance_y:end]
        end
    else
        if entrance_x > size(tilelayer, 2)
            @warn "entrance at x=$(entrance_x) out of bounds; using default ground tile."
            searchregion = nothing
        else
            searchregion = tilelayer[entrance_y:end, entrance_x]
        end
    end

    if isnothing(searchregion)
        tileindex = nothing
    else
        tileindex = findfirst(!isequal(0x25), searchregion)
    end

    if isnothing(tileindex)
        groundtile = 0x100
    else
        groundtile = searchregion[tileindex]
    end

    remove_entrances && (level = remove_entrance_layers(level))
    return level, groundtile
end

"Remove the entrance layers from the given level and return a new one."
function remove_entrance_layers(level::Level)
    offsets = copy(level.offsets)
    # We could just do `haskey` but this is more resistant to changes to `buildlevel`.
    get(offsets, :entrances, 0) == 0 && return copy(level)

    data = level.data
    data = @views cat(
        data[:, :, firstindex(data, 3):level.offsets[:entrances] - 1],
        data[:, :, level.offsets[:entrances] + 2:end],
        dims=3)

    if get(offsets, :sprites, 0) > 2
        offsets[:sprites] -= 2
    end
    if get(offsets, :goalsprites, 0) > 2
        offsets[:goalsprites] -= 2
    end
    if get(offsets, :secondaryentrances, 0) > 2
        offsets[:secondaryentrances] -= 2
    end
    return Level(data, offsets, level.stats, level.spriteheader)
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

