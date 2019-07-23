"Handle sprite (enemies, mushrooms, shells, ...) data and formatting."
module Sprites

using DelimitedFiles

import ..LevelStatistics

export makespritelayers, getsprites
export uniquevanillasprites, uniqueextendedsprites, uniquesprites
export vanillaspritemapping, extendedspritemapping, defaultspritemapping

"Whether to print the message that precalculation of values is possible."
const printprecalculatereminder = true

const statsdir = abspath(joinpath(@__DIR__, "..", "..", "stats"))
const statsfile = joinpath(statsdir, "sprite_stats.csv")
const vanillamappingfile = joinpath(statsdir, "sprite_vanilla_mapping.csv")
const extendedmappingfile = joinpath(statsdir, "sprite_extended_mapping.csv")

"""
Set of sprites that are used as goals.

Used to be able to filter out the most important sprites. Blue and silver P switches are
included as they are most often required to _reach_ the goal.
"""
const goalsprites = Set{UInt8}((
    0x4a, # orb
    0x80, # key
    0x0e, # keyhole
    0x3e # P switches (both use the same sprite)
))

"""
    makespritelayers(sprites::AbstractMatrix{UInt8}, stats::LevelStats, padded::Bool=true)

Return an `Array{UInt8, 3}` of layers of sprites for the given sprites and level statistics.

If `padded` is `false`, the cuboid only contains layers for the sprites in `sprites`.
Otherwise, for each possible type of sprite, an empty layer is constructed.

See [`uniquesprites`](@ref) for the amount of layers.
"""
function makespritelayers(sprites::AbstractMatrix{UInt8}, stats::LevelStatistics.LevelStats,
                          padded::Bool=true)::Array{UInt8, 3}
    if padded
        spritecount = uniquesprites
        spritemapping = defaultspritemapping
    else
        # Copy for cache efficiency (not benchmarked)
        uniquespritearray = unique(sprites[4, :])
        spritecount = length(uniquespritearray)
        spritemapping = Dict(k => i for (i, k) in enumerate(sort(uniquespritearray)))
    end
    vertical = LevelStatistics.isvertical(stats)
    if vertical
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(LevelStatistics.screenrowsvert)
        out = zeros(UInt8, LevelStatistics.screencolumns(stats),
                    columnsperscreen * stats.screens, spritecount)
    else
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(LevelStatistics.screencols)
        out = zeros(UInt8, LevelStatistics.screenrows(stats),
                    columnsperscreen * stats.screens, spritecount)
    end
    # We can ignore that y and x are swapped for vertical levels as vertical levels are in
    # a transposed matrix anyway.
    for (y, x, screen, spriteid) in eachcol(sprites)
        # Add 1 for 1-based indexing.
        y += 0x1
        absolute_x = x + columnsperscreen * screen + 0x1
        if !haskey(spritemapping, spriteid)
            @error "sprite id $spriteid not found in $(stats.filename)."
        elseif y > size(out, 1)
            @warn ("makespritelayers: in $(stats.filename): y=$y out of bounds. "
                   * "Skipping...")
            continue
        elseif absolute_x > size(out, 2)
            @warn ("makespritelayers: in $(stats.filename): x=$absolute_x out of bounds. "
                   * "Skipping...")
            continue
        end
        # We save the `spriteid + 1` to make sure it's never zero. If it was zero, we would
        # not be able to  distinguish it from the other zeros in the matrix.
        out[y, absolute_x, spritemapping[spriteid]] = spriteid + 0x1
    end
    return out
end

"""
    getsprites(io::IO, filename::AbstractString="<Unknown>")
    getsprites(file::Abstractstring; force=false)

Return all sprite values as a `Matrix{UInt8}` and the sprite header as a `NamedTuple`,
both contained in the given IO or file.

In the matrix, each column corresponds to data for one sprite.
The ordering of values per column corresponds to [`parsespritebits`](@ref):
`y, x, screen, sprite`, where `y` and `x` are swapped for vertical levels.
`filename` is used for error reporting.
Files associated with this method end in ".sp".
"""
function getsprites(io::IO, filename::AbstractString="<Unknown>")
    spritebytes = read(io)::Vector{UInt8}
    if isempty(spritebytes)
        @warn "empty sprite file $filename."
        return Matrix{UInt8}(undef, 0, 0), nothing
    end
    # Make sure the data ends the way Lunar Magic would expect it to.
    if spritebytes[end] == 0xff
        endindex = lastindex(spritebytes) - 1
    elseif length(spritebytes) > 1 && spritebytes[end-1:end] == [0xff, 0xfe]
        endindex = lastindex(spritebytes) - 2
    else
        error("sprite file \"$filename\" ends unexpectedly.")
    end
    spriteheader = parsespriteheaderbits(bitstring(spritebytes[1]))

    spriteview = view(spritebytes, 2:endindex)
    if length(spriteview) % 3 != 0
        @warn ("unexpected amount of sprites ($(length(spriteview))) in $filename. "
               * "Skipping...")
        return Matrix{UInt8}(undef, 0, 0), spriteheader
    end

    isempty(spriteview) && return Matrix{UInt8}(undef, 0, 0), spriteheader

    spritebits = LevelStatistics.tobitstring(spriteview)
    sprites = reduce(hcat, map(x -> collect(parsespritebits(String(x))),
                     Iterators.partition(spritebits, 24)))
    return sprites, spriteheader
end

function getsprites(file::AbstractString; force::Bool=false)
    force || @assert endswith(file, ".sp") ("please use files with the .sp extension or "
                                             * "set `force=true`.")
    open(f -> getsprites(f, file), file)
end

"Return a `NamedTuple` sprite header information parsed from the given bitstring."
function parsespriteheaderbits(headerbits::AbstractString)
    @assert length(headerbits) == 8 "unexpected bitstring size."
    buoyancy = parse(Bool, headerbits[1])
    # Disable sprite buoyancy on layer 2
    disablelayer2buoyancy = parse(Bool, headerbits[2])
    newsystem = parse(Bool, headerbits[3])
    mode = parse(UInt8, headerbits[4:8], base=2)
    return (buoyancy=buoyancy, disablelayer2buoyancy=disablelayer2buoyancy,
            newsystem=newsystem, mode=mode)
end

"""
Return the values parsed from the given 3 byte bitstring of sprite data as a `NamedTuple`.

Both `y` and `x` correspond to the position in the screen given by `screen`, not the
absolute position in the level!
In vertical levels, the `y` and `x` values are swapped.
"""
function parsespritebits(spritebits::AbstractString, quiet::Bool=false)
    @assert length(spritebits) == 24 "unexpected bitstring size."
    # Both y and x are referring to the sprite's relative position in the screen,
    # not in the level.
    ybits = spritebits[8] * spritebits[1:4]
    y = parse(UInt8, ybits, base=2)
    # extrabits = spritebits[5:6]
    xbits = spritebits[9:12]
    x = parse(UInt8, xbits, base=2)
    screenbits = spritebits[7] * spritebits[13:16]
    screen = parse(UInt8, screenbits, base=2)
    spritebits = spritebits[17:24]
    sprite = parse(UInt8, spritebits, base=2)
    return (y=y, x=x, screen=screen, sprite=sprite)
end

function countsprites(dir::AbstractString, quiet::Bool=false)
    spritedict = Dict{UInt8, Int32}()
    countsprites!(spritedict, dir, quiet)
end

function countsprites!(spritedict::AbstractDict{UInt8, T},
                       dir::AbstractString, quiet::Bool=false) where {T <: Number}
    for l in map(x -> joinpath(dir, x), filter(f -> endswith(f, ".sp"), readdir(dir)))
        local sprites
        try
            sprites = getsprites(l)[1]
        catch e
            if e isa AssertionError
                quiet || @error e
                continue
            end
            rethrow(e)
        end
        isempty(sprites) && continue
        for id in sprites[4, :]
            if haskey(spritedict, id)
                spritedict[id] += one(T)
            else
                spritedict[id] = one(T)
            end
        end
    end
    return spritedict
end

function countsprites!(spritedict::AbstractDict{String, Dict{UInt8, T}},
                       dir::AbstractString, quiet::Bool=false) where {T <: Number}
    for l in map(x -> joinpath(dir, x), filter(f -> endswith(f, ".sp"), readdir(dir)))
        spritedict[l] = Dict{UInt8, T}()
        local sprites
        try
            sprites = getsprites(l)[1]
        catch e
            if e isa AssertionError
                quiet || @error e
                continue
            end
            rethrow(e)
        end
        isempty(sprites) && continue
        subdict = spritedict[l]
        for id in sprites[4, :]
            if haskey(subdict, id)
                subdict[id] += one(T)
            else
                subdict[id] = one(T)
            end
        end
    end
    return spritedict
end

function countallsprites(dir::AbstractString=LevelStatistics.leveldir, quiet::Bool=false)
    spritedict = Dict{String, Dict{String, Dict{UInt8, Int32}}}()
    for (root, dirs, _) in walkdir(dir)
        for dir in dirs
            spritedict[dir] = Dict{String, Dict{UInt8, Int32}}()
            countsprites!(spritedict[dir], joinpath(root, dir), quiet)
        end
    end
    return spritedict
end

function findnonvanillasprites(
        spritedict::Dict{String, Dict{String, Dict{UInt8, Int32}}}=countallsprites(
                LevelStatistics.leveldir),
        origspritedict::AbstractDict{UInt8, <:Number}=countsprites(
                LevelStatistics.originalleveldir))
    nonvanilla = Dict{UInt8, Dict{String, Vector{String}}}()
    for (dir, dict) in spritedict
        for (level, subdict) in dict
            for (sprite, count) in subdict
                if !haskey(origspritedict, sprite)
                    if haskey(nonvanilla, sprite)
                        if haskey(nonvanilla[sprite], dir)
                            push!(nonvanilla[sprite][dir], level)
                        else
                            nonvanilla[sprite][dir] = [level]
                        end
                    else
                        nonvanilla[sprite] = Dict{String, Vector{String}}(dir => [level])
                    end
                end
            end
        end
    end
    return nonvanilla
end

function countuniquevanillasprites(spritedict::AbstractDict{UInt8, T}=countsprites(
        LevelStatistics.originalleveldir)) where {T <: Number}
    length(spritedict)
end

function countuniqueextendedsprites(
        spritedict::Dict{String, Dict{String, Dict{UInt8, Int32}}}=countallsprites(
                LevelStatistics.leveldir, true))
    length(unique(k for d in values(spritedict)
                  for sd in values(d) for k in keys(sd)))
end

function makevanillaspritemapping(spritedict::AbstractDict{UInt8, T}=countsprites(
        LevelStatistics.originalleveldir)) where {T <: Number}
    Dict(k => i for (i, k) in enumerate(
           sort(collect(keys(spritedict)))))
end

function makeextendedspritemapping(
        spritedict::Dict{String, Dict{String, Dict{UInt8, Int32}}}=countallsprites(
                LevelStatistics.leveldir, true))
    Dict(k => i for (i, k) in enumerate(
            sort(collect(unique(k for d in values(spritedict)
                                for sd in values(d) for k in keys(sd))))))
end

function calculatespritestats()
    uniquevanillasprites = countuniquevanillasprites()
    uniqueextendedsprites = countuniqueextendedsprites()
    return uniquevanillasprites, uniqueextendedsprites
end

function generatestatsfile(path::AbstractString)
    println("Generating sprite stats file $path...")
    uniquevanillasprites, uniqueextendedsprites = calculatespritestats()
    stats = ["uniquevanillasprites" "uniqueextendedsprites";
              uniquevanillasprites  uniqueextendedsprites]
    writedlm(path, stats, ';')
    println("Done.")
end

function generatevanillamappingfile(path::AbstractString)
    println("Generating vanilla mapping file $path...")
    mapping = makevanillaspritemapping()
    writedlm(path, ("key" => "value", mapping...), ';')
    println("Done.")
end

function generateextendedmappingfile(path::AbstractString)
    println("Generating extended mapping file $path...")
    mapping = makeextendedspritemapping()
    writedlm(path, ("key" => "value", mapping...), ';')
    println("Done.")
end

function readstatsfile(path::AbstractString)
    uniquevanillasprites, uniqueextendedsprites = readdlm(path, ';', Int, skipstart=1)
end

function readmappingfile(path::AbstractString)
    mapping = readdlm(path, ';', Int, skipstart=1)
    Dict{UInt8, Int}(eachrow(mapping))
end

"Print a message on how to precalculate values to speed up initialization."
function printprecalculatemsg()
    @info """
          You can speed up the initialization process:
          Precalculate by running `$(parentmodule(generateall)).generateall()`.
          To disable this message, edit `printprecalculatereminder` to `false`.
          """
end

function initspritestats(statsfile::AbstractString)
    if isfile(statsfile)
        uniquevanillasprites, uniqueextendedsprites = readstatsfile(statsfile)
    else
        println("Calculating sprite stats...")
        uniquevanillasprites, uniqueextendedsprites = calculatespritestats()
        if (printprecalculatereminder && isfile(vanillamappingfile)
                && isfile(extendedmappingfile))
            printprecalculatemsg()
        end
    end
    return uniquevanillasprites, uniqueextendedsprites
end

function initvanillaspritemapping(vanillamappingfile::AbstractString)
    if isfile(vanillamappingfile)
        vanillaspritemapping = readmappingfile(vanillamappingfile)
    else
        println("Calculating vanilla sprite mapping...")
        vanillaspritemapping = makevanillaspritemapping()
        if printprecalculatereminder && isfile(extendedmappingfile)
            printprecalculatemsg()
        end
    end
    return vanillaspritemapping
end

function initextendedspritemapping(extendedmappingfile::AbstractString)
    if isfile(extendedmappingfile)
        extendedspritemapping = readmappingfile(extendedmappingfile)
    else
        println("Calculating extended sprite mapping...")
        extendedspritemapping = makeextendedspritemapping()
        if printprecalculatereminder
            printprecalculatemsg()
        end
    end
    return extendedspritemapping
end

function generateall()
    generatestatsfile(statsfile)
    generatevanillamappingfile(vanillamappingfile)
    generateextendedmappingfile(extendedmappingfile)
end

function checkdefaults(uniquesprites, defaultspritemapping)
    if uniquesprites === uniquevanillasprites
        @assert defaultspritemapping === vanillaspritemapping (
            "if you change `uniquesprites`, also change `defaultspritemapping` and the "
            * "other way around."
        )
    elseif uniquesprites === uniqueextendedsprites
        @assert defaultspritemapping === extendedspritemapping (
            "if you change `uniquesprites`, also change `defaultspritemapping` and the "
            * "other way around."
        )
    else
        error("unknown `uniquesprites` reference")
    end
end

const uniquevanillasprites, uniqueextendedsprites = initspritestats(statsfile)

"Amount of unique sprite ID values in the original game."
uniquevanillasprites  # = 196

"""
Amount of unique sprite ID values in all hacks found.
Does not count custom (longer than 3 byte) sprites (or possibly counts them wrong).
"""
uniqueextendedsprites  # = 241 (can change)

"Mapping of vanilla sprite IDs to layer indices."
const vanillaspritemapping  = initvanillaspritemapping(vanillamappingfile)
"Mapping of extended sprite IDs to layer indices."
const extendedspritemapping = initextendedspritemapping(extendedmappingfile)


"""
An alias to either [`uniquevanillasprites`](@ref) or [`uniqueextendedsprites`](@ref)
depending on the support of different hacks desired.

So please edit this if you wish! But do not forget to edit [`defaultspritemapping`](@ref)
as well.
"""
const uniquesprites = uniqueextendedsprites

"""
An alias to either [`vanillaspritemapping`](@ref) or [`extendedspritemapping`](@ref)
depending on the support of different hacks desired.

So please edit this if you wish! But do not forget to edit [`uniquesprites`](@ref)
as well.
"""
const defaultspritemapping = extendedspritemapping


checkdefaults(uniquesprites, defaultspritemapping)

end # module

