"Build an arbitrarily complete level."
module LevelBuilder

using ..LevelStatistics
using ..Tiles
using ..SecondaryLevelStats
using ..Sprites

export Level, buildlevel

"Mapping from characters flags to argument names."
const flagdict = Dict{Char, Symbol}(
    't' => :tiles,
    'e' => :entrances,
    's' => :sprites,
    'g' => :goalsprites,
    'x' => :secondaryentrances,
)

"A level consisting of a cuboid of non-meta level data."
struct Level{T}
    "Unformatted level data."
    data::Array{UInt16, 3}
    "First indices into different layers."
    offsets::Dict{Symbol, UInt16}
    "Level statistics."
    stats::LevelStats
    "Sprite header. May be `nothing`."
    spriteheader::Union{NamedTuple{T}, Nothing}
end

function Level(data::Array{UInt16, 3}, offsets::Dict{Symbol, UInt16}, stats::LevelStats,
               ::Nothing)
    Level{Nothing}(data, offsets, stats, nothing)
end

# TODO generate all `methodswith(LevelStats)` to be callable on a `Level`
# TODO large performance increase: change levels to be `(layers, height, width)` instead
# of `(height, width, layers)`. Will need some refactoring and regenerated DBs.

"""
    buildlevel(level; tiles=true, entrances=true, sprites=true, goalsprites=true,
               secondaryentrances=true)
    buildlevel(level, flags::AbstractString)

Return an `Tuple` containing an `Array{UInt16, 3}` describing a level containing the
specified values and a `Dict{Symbol, UInt16}` of offsets for where the values of each
layer begin.
`level` can be any `AbstractString` of a file belonging to a certain level. To collect the
necessary information, its extension (if any) will be replaced with the necessary ones.

For convenience, `level` can also be a number that will choose the file corresponding to the
number from the current directory, or – if the level does not exist in the current
directory – from [`originalleveldir`](@ref).

With `goalsprites=true`, only sprites necessary to finish the level are added (unless
`sprites` is also `true` in which case they are already contained there).
`secondaryentrances` actually includes layers for secondary exits and screen exits as well.

The `kwargs` can also be specified as an `AbstractString`, where each character in the
string maps to an argument. If a flag is given, the argument is `true`, otherwise it
is `false`.
The corresponding characters for each flag are the following:
   - tiles: t
   - entrances: e
   - sprites: s
   - goalsprites: g
   - secondaryentrances: x

If `offsets` contains a value of zero, that means there are no layers for that value.
"""
function buildlevel(file::AbstractString; tiles=true, entrances=true, sprites=true,
                    goalsprites=true, secondaryentrances=true, verbose=true)
    filebasename = splitext(file)[1]
    layers = AbstractArray[]
    layercount = 0
    offsets = Dict{Symbol, UInt16}()

    stats = getstats(filebasename * ".ent")

    # TODO refactor each of these into their own method
    if tiles
        verbose && println("Building tile layer...")
        tilelayer = Tiles.getmap(filebasename * ".map", stats)
        map16file = joinpath(dirname(filebasename), "map16fgG.bin")
        map16mappings = parsemap16(map16file)
        applymap16!(tilelayer, map16mappings)
        push!(layers, tilelayer)
        offsets[:tiles] = layercount + 1
        layercount += 1
    end

    if entrances
        verbose && println("Building entrance layers...")
        mainentrancelayer   = makemainentrancelayer(stats)
        midwayentrancelayer = makemidwayentrancelayer(stats)
        push!(layers, mainentrancelayer, midwayentrancelayer)
        offsets[:entrances] = layercount + 1
        layercount += 2
    end

    if sprites
        verbose && println("Building sprite layers...")
        spritematrix, spriteheader = getsprites(filebasename * ".sp")
        spritelayers = makespritelayers(spritematrix, stats, true)

        push!(layers, spritelayers)
        totalsize = size(spritelayers, 3)
        if totalsize > 0
            offsets[:sprites] = layercount + 1
        else
            offsets[:sprites] = 0
        end
        layercount += totalsize
    end

    if goalsprites && !sprites
        verbose && println("Building goal sprite layers...")
        spritematrix, spriteheader = getsprites(filebasename * ".sp")
        # Copying makes more sense here due to the low amount of values.
        # Not benchmarked though.
        spriteidvector = spritematrix[4, :]
        mask = map(x -> x in Sprites.goalsprites, spriteidvector)
        unpaddedspritelayers = makespritelayers(spritematrix[:, mask], stats, false)

        # We now manually pad so the size of this layer stays the same.
        # TODO This padding should probably be a functionality in the sprites module...
        spritelayers = Array{UInt8, 3}(undef, size(unpaddedspritelayers)[1:2]...,
                                       length(goalsprites))

        presentsprites = sort(unique(spriteidvector[4, mask]))
        for (i, goalsprite) in enumerate(sort(collect(Sprites.goalsprites)))
            unpaddedindex = findfirst(goalsprite, presentsprites)
            if isnothing(unpaddedindex)
                spritelayers[:, :, i] .= 0
            else
                spritelayers[:, :, i] = view(unpaddedspritelayers, :, :, unpaddedindex)
            end
        end

        push!(layers, spritelayers)
        totalsize = size(spritelayers, 3)
        if totalsize > 0
            offsets[:goalsprites] = layercount + 1
        else
            offsets[:goalsprites] = 0
        end
        layercount += totalsize
    end

    # TODO test this (v)
    if secondaryentrances
        verbose && println("Building secondary entrance layers...")
        exitvector = getexits(filebasename * ".exits")
        exitlayers = makeexitlayers(exitvector, stats, true)

        secondaryentrancevector = getentrances(filebasename * ".ent2")
        secondaryentrancelayers = makeentrancelayers(secondaryentrancevector, stats, true)

        push!(layers, exitlayers, secondaryentrancelayers)
        totalsize = size(exitlayers, 3) + size(secondaryentrancelayers, 3)
        if totalsize > 0
            offsets[:secondaryentrances] = layercount + 1
        else
            offsets[:secondaryentrances] = 0
        end
        layercount += totalsize
    end

    @assert !isempty(layers) "no layers specified. Need at least one."

    result = reduce((a, b) -> cat(a, b, dims=3), layers)
    # If the following happens, the offsets are also wrong.
    @assert size(result, 3) == layercount "layer count does not match actual size."
    # `layercount` is 1, catting produced a 2D result. Reshape to get a 3D array.
    ndims(result) < 3 && (result = reshape(result, size(result)..., 1))
    verbose && println("Done.")
    if sprites || goalsprites
        return Level(result, offsets, stats, spriteheader)
    else
        return Level(result, offsets, stats, nothing)
    end
end

function buildlevel(file::AbstractString, flags::AbstractString; kwargs...)
    argdict = Dict{Symbol, Bool}(zip(values(flagdict), Iterators.repeated(false)))
    for flag in flags
        if !(flag in keys(flagdict))
            @warn "skipping unknown flag '$flag'"
            continue
        end
        argdict[flagdict[flag]] = true
    end
    buildlevel(file; argdict..., kwargs...)
end

function buildlevel(file::AbstractString, flags::AbstractChar; kwargs...)
    buildlevel(file, string(flags); kwargs...)
end

function buildlevel(number::UInt16; kwargs...)
    filename = numbertofilename(number)
    buildlevel(filename; kwargs...)
end

function buildlevel(number::UInt16, args...)
    filename = numbertofilename(number)
    buildlevel(filename, args...)
end

"""
Build a filename from the given number and check whether the file exists. If it does not,
search in the [`originalleveldir`](@ref); if the file is still not found, throw an error.
"""
function numbertofilename(number::UInt16)
    filename = "Level" * uppercase(string(number, base=16, pad=3)) * ".map"
    isfile(filename) && return filename

    newfilename = joinpath(originalleveldir, filename)
    @assert isfile(newfilename) "a level with the given number does not exist"
    println(filename, " not found in the current directory. ",
            "Using original level instead (", newfilename, ").")
    return newfilename
end

"Return the layer containing all tiles."
function tilelayer(level::Level)
    return view(level.data, :, :, level.offsets[:tiles])
end

"Return the layer containing the main entrance."
function mainentrancelayer(level::Level)
    return view(level.data, :, :, level.offsets[:entrances])
end

"Return the layer containing the midway entrance."
function midwayentrancelayer(level::Level)
    return view(level.data, :, :, level.offsets[:entrances] + 1)
end

"Return the layer containing all sprites with the given sprite ID."
function spritelayer(level::Level, spriteid::Unsigned)
    # We subtract one due to 1-based indexing.
    return view(level.data, :, :,
                level.offsets[:sprites] + Sprites.defaultspritemapping[spriteid] - 1)
end


"Return the layer containing all secondary entrances with the given number."
function secondaryentrancelayer(level::Level, number::Unsigned)
    # We subtract one due to 1-based indexing.
    return view(level.data, :, :,
                level.offsets[:secondaryentrances]
                    + SecondaryLevelStats.entrancemapping[number]
                    + maximumexits + maximumsecondaryexits - 1)
end

"Return the layer containing all screen exits with the given destination."
function screenexitlayer(level::Level, destination::Unsigned)
    # We subtract one due to 1-based indexing.
    return view(level.data, :, :,
                level.offsets[:secondaryentrances]
                    + SecondaryLevelStats.exitmapping[destination] - 1)
end

"Return the layer containing all secondary exits with the given number."
function secondaryexitlayer(level::Level, number::Unsigned)
    # We subtract one due to 1-based indexing.
    return view(level.data, :, :,
                level.offsets[:secondaryentrances]
                    + SecondaryLevelStats.secondaryexitmapping[number]
                    + maximumexits - 1)
end

end # module

