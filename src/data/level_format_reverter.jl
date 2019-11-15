module LevelFormatReverter

using SparseArrays: SparseMatrixCSC

using ..DataCompressor: unsparse
using ..LevelBuilder: flagdict
using ..LevelFormatter
using ..LevelStatistics: uniquevanillatiles, screenrowshori
using ..SecondaryLevelStats
using ..Sprites

export from1d, from2d, from3d

function from1d(data::AbstractVector,
                ::Union{AbstractString,
                        AbstractChar}=dimensionality_defaultflags[Symbol("1d")],
                keep::UInt16=0x100, empty::UInt16=0x025)
    level = Matrix{UInt16}(undef, screenrowshori, length(data))
    row = round.(UInt16, data)
    keep_rowindices = row .== 1
    level[firstindex(level, 1):end - 3, :] .= empty
    # Ground row of level 0x105.
    level[end - 2, keep_rowindices]   .= keep
    level[end - 2, .~keep_rowindices] .= empty
    level[end - 1:end, :] .= empty
    return level
end

function from2d(data::AbstractMatrix,
                ::Union{AbstractString,
                        AbstractChar}=dimensionality_defaultflags[Symbol("2d")],
                keep::UInt16=0x100, empty::UInt16=0x025)
    level = round.(UInt16, data)
    keepindices = level .== 1
    level[keepindices]   .= keep
    level[.~keepindices] .= empty
    return level
end

"""
Return a `Dict{Symbol, <:AbstractArray{UInt16, 3}}` of data in the given level that
contains the given layers where `empty` controls how unassigned tiles are filled
(only relevant if `tiles=true`).

The dictionary always contains the following keys (layers that are not given have
default values):
   - :tiles (a single layer of tiles)
   - :entrances (two layers, one each for the main and midway entrance)
   - :sprites (one layer each for each sprite)
   - :exits (one layer each for each screen exit)
   - :secondaryexits (one layer each for each secondary exit)
   - :secondaryentrances (one layer each for each secondary entrance)
"""
function from3d(data::AbstractArray{T, 3}, empty::UInt16; tiles=true, entrances=true,
                sprites=true, goalsprites=true, secondaryentrances=true) where T
    level = round.(UInt16, data)
    layeroffset = firstindex(level, 3)
    firstlayer = view(level, :, :, layeroffset)
    # A dictionary mapping flags to the corresponding layers.
    leveldict = Dict{Symbol, typeof(level)}()

    # TODO use offsets dict from `buildlevel` instead of flags. much better but takes time
    if tiles
        tilelayer = similar(firstlayer)
        assigned_indices = falses(size(tilelayer))
        keepindices = similar(assigned_indices)
        # `tileid` starting from one is correct here as we don't need to discern it from
        # the "empty" zero anymore.
        for (tileid, layerindex) in enumerate(layeroffset:uniquevanillatiles)
            layer = view(level, :, :, layerindex)
            keepindices .= layer .== 1
            assigned_indices .|= keepindices
            tilelayer[keepindices] .= tileid
        end
        # Add one for the same reason as above.
        tilelayer[.~assigned_indices] .= empty + 1
        leveldict[:tiles] = reshape(tilelayer, size(tilelayer)..., 1)
        layeroffset += uniquevanillatiles
    else
        # Add one for the same reason as above.
        leveldict[:tiles] = fill(empty + 1, size(firstlayer)..., 1)
    end

    if entrances
        leveldict[:entrances] = level[:, :, layeroffset:layeroffset + 1]
        layeroffset += 2
    else
        entrancelayers = zeros(size(firstlayer)..., 2)
        entrancelayers[23, 2, :] .= 1
        leveldict[:entrances] = entrancelayers
    end

    if sprites
        spritelayers = layers_from_mapping(level, layeroffset, uniquesprites,
                                                   defaultspritemapping)
        leveldict[:sprites] = spritelayers
        layeroffset += uniquesprites
    elseif !goalsprites
        leveldict[:sprites] = zeros(size(firstlayer)..., uniquesprites)
    end

    if goalsprites && !sprites
        spritelayers = zeros(size(firstlayer)..., uniquesprites)
        for spriteid in Sprites.goalsprites
            keepindices = level[:, :,
                                layeroffset + defaultspritemapping[spriteid] - 1] .== 1
            # Add one so `spriteid` is never zero.
            spritelayers[:, :, defaultspritemapping[spriteid]][keepindices] .= spriteid + 1
        end
        leveldict[:sprites] = spritelayers
        layeroffset += uniquesprites
        # else-case already handled above
    end

    if secondaryentrances
        exitlayers = layers_from_mapping(level, layeroffset, maximumexits, exitmapping)
        layeroffset += maximumexits

        secondaryexitlayers = layers_from_mapping(level, layeroffset, maximumsecondaryexits,
                                                  secondaryexitmapping)
        layeroffset += maximumsecondaryexits

        entrancelayers = layers_from_mapping(level, layeroffset, maximumentrances,
                                             entranceemapping)
        layeroffset += maximumentrances

        leveldict[:exits] = exitlayers
        leveldict[:secondaryexits] = secondaryexitlayers
        leveldict[:secondaryentrances] = entrancelayers
        # `layeroffset` already updated
    else
        leveldict[:exits] = zeros(size(firstlayer)..., maximumexits)
        leveldict[:secondaryexits] = zeros(size(firstlayer)..., maximumsecondaryexits)
        leveldict[:secondaryentrances] = zeros(size(firstlayer)..., maximumentrances)
    end

    @assert !isempty(leveldict) "no layers specified. Need at least one."
    @assert layeroffset - 1 == size(level, 3) "layer count does not match actual size."
    return leveldict
end

function from3d(data::AbstractArray{T, 3},
                flags::Union{AbstractString,
                             AbstractChar}=dimensionality_defaultflags[Symbol("3d")],
                empty::UInt16=0x025) where T
    argdict = Dict{Symbol, Bool}(zip(values(flagdict), Iterators.repeated(false)))
    for flag in flags
        if !(flag in keys(flagdict))
            @warn "skipping unknown flag '$flag'"
            continue
        end
        argdict[flagdict[flag]] = true
    end
    from3d(data, empty; argdict...)
end

function from3d(data::AbstractVector{SparseMatrixCSC{T}}, args...) where T
    from3d(unsparse(data), args...)
end

function layers_from_mapping(level::AbstractArray{UInt16, 3}, startlayer::Integer,
                                     offset::Integer, forwardmapping::AbstractDict)
    reversemapping = Dict{valtype(forwardmapping),
                                keytype(forwardmapping)}(
                                    # Add one so the value is never zero.
                                    v => k + 1 for (k, v) in forwardmapping)
    layers = level[:, :, startlayer:startlayer + offset - 1]
    keepindices = falses(size(level[:, :, startlayer]))
    for (id, layer) in enumerate(eachslice(layers, dims=3))
        keepindices .= layer .== 1
        layer[keepindices] .= reversemapping[id]
    end
    return layers
end

end # module

