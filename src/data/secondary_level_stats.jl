"""
Collect secondary level statistics not present in the main file such as secondary entrances
and screen exits.

Screen exits are exits that have a level number as a destination.
Secondary exits have the number of a secondary entrance as their destination.
Both cover a whole screen.
"""
module SecondaryLevelStats

using DelimitedFiles

using ..XYTables
import ..LevelStatistics
import ..DefaultDictionary

export exitenabledtiles
export makeexitlayers, getexits, maximumexits, maximumsecondaryexits
export makescreenexitlayers, makesecondaryexitlayers
export makeentrancelayers, getentrances, maximumentrances
export entrancemapping, exitmapping, secondaryexitmapping
export parsemap16, applymap16!, defaultmap16tile


"Whether to print the message that precalculation of values is possible."
const printprecalculatereminder = true

const statsdir = abspath(joinpath(@__DIR__, "..", "..", "stats"))
const secondarystatsfile       = joinpath(statsdir, "secondary_level_stats.csv")
const entrancemappingfile      = joinpath(statsdir, "entrance_mapping.csv")
const exitmappingfile          = joinpath(statsdir, "exit_mapping.csv")
const secondaryexitmappingfile = joinpath(statsdir, "secondary_exit_mapping.csv")

"Set of tiles that are used to exit a level via screen or secondary exits."
const exitenabledtiles = Set{UInt16}((
        0x01f, # Door top, also used for small doors.
        0x020, # Door bottom.
        0x027, # Invisible blue P door top, also used for small P doors.
        0x028, # Invisible blue P door bottom.
        # TODO Maybe all boss door tiles except the bottom parts can be removed.
        0x098, # Boss door top left.
        0x099, # Boss door top right.
        0x09A, # Boss door middle left.
        0x09B, # Boss door middle right.
        0x09C, # Boss door bottom; both parts. Only in FG/BG GFX setting 1.
        0x137, # Vertical pipe top or bottom, left part.
        0x138, # Vertical pipe top or bottom, right part.
        0x13F, # Horizontal pipe left or right, bottom part. top part is never exit-enabled.
))

"The default value used for extended Map16 tiles."
const defaultmap16tile = 0x130  # stone block tile
"`defaultmap16tile` as a 2-byte array in little endian order."
const defaultmap16tile_le = begin
    second = div(defaultmap16tile, 0x100)
    first = defaultmap16tile - second * 0x100
    UInt8[first, second]
end

"""
    makeexitlayers(exits::AbstractVector{NamedTuple}, padded::Bool=true)

Return an `Array{UInt16, 3}` containing the given screen and secondary exits.

If `padded` is `false`, the cuboid only contains layers for the exits in `exits`.
Otherwise, for each possible exit destination, an empty layer is constructed.
"""
function makeexitlayers(exits::AbstractVector{NamedTuple},
                        stats::LevelStatistics.LevelStats, padded::Bool=true)
    secondaryexitmask = map(x -> x.secondaryexit, exits)
    screenexitlayers = makescreenexitlayers(view(exits, .~secondaryexitmask), stats, padded)
    secondaryexitlayers = makesecondaryexitlayers(view(exits, secondaryexitmask),
                                                  stats, padded)
    cat(screenexitlayers, secondaryexitlayers, dims=3)
end

function makescreenexitlayers(exits::AbstractVector{NamedTuple},
                              stats::LevelStatistics.LevelStats, padded::Bool=true)
    createexitlayers(exits, stats, padded, maximumexits, exitmapping)
end

function makesecondaryexitlayers(exits::AbstractVector{NamedTuple},
                                 stats::LevelStatistics.LevelStats, padded::Bool=true)
    createexitlayers(exits, stats, padded, maximumsecondaryexits, secondaryexitmapping)
end

function createexitlayers(exits::AbstractVector{NamedTuple},
                          stats::LevelStatistics.LevelStats,
                          padded::Bool=true, maximumexits::Int=0,
                          exitmapping::Dict{UInt16, Int}=Dict{UInt16, Int}())
    if padded
        @assert maximumexits > 0 ("`maximumexits` must be greater than 0 when `padded` "
                                  * "is `true`.")
        @assert !isempty(exitmapping) ("`exitmapping` must not be empty when `padded` "
                                       * "is `true`.")
        exitcount = maximumexits
    else
        uniqueexits = unique(exits)
        exitcount = length(uniqueexits)
        exitmapping = Dict(k => i
                for (i, k) in enumerate(sort(map(e -> e.number, uniqueexits))))
    end
    vertical = LevelStatistics.isvertical(stats)
    if vertical
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(LevelStatistics.screenrowsvert)
        out = zeros(UInt16, LevelStatistics.screencolumns(stats),
                    columnsperscreen * stats.screens, exitcount)
    else
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(LevelStatistics.screencols)
        out = zeros(UInt16, LevelStatistics.screenrows(stats),
                    columnsperscreen * stats.screens, exitcount)
    end
    # We can ignore that y and x are swapped for vertical levels as vertical levels are in
    # a transposed matrix anyway.
    for exit in exits
        # This marked a whole screenful which is inefficient in terms of storage.
        # xend = columnsperscreen * exit.screen
        # xstart = xend - columnsperscreen + 0x1
        # destination = exit.destination
        # if xend > size(out, 2)
        #     @warn ("createexitlayers: in $(stats.filename): x=$xend out of bounds. "
        #            * "Skipping...")
        # else
        #     out[:, xstart:xend, exitmapping[destination]] .= destination
        # end
        xstart = columnsperscreen * (exit.screen - 0x1) + 0x1
        destination = exit.destination
        if xstart > size(out, 2)
            @warn ("createexitlayers: in $(stats.filename): x=$xstart out of bounds. "
                   * "Skipping...")
        else
            # Add one so the destination is never zero.
            out[1, xstart, exitmapping[destination]] = destination + 1
        end
    end
    return out
end

#=
"""
    shrinkexits!(exits::AbstractArray{UInt16, 3}, level::AbstractArray{UInt16, 3})

Mutate the given array of exits so that exits are only marked at possible exit locations.

In other words, after application, each exit does not cover up a whole screen but instead is
only located at possible exit locations such as exit-enabled pipes or doors.

See also [`exitenabledtiles`](@ref) to easily extend this method with other exit locations.
"""
function shrinkexits!(exits::AbstractArray{UInt16, 3}, level::AbstractArray{UInt16, 3})
    # TODO
    error("not implemented.")
end
=#

"""
    getexits(io::IO, filename::AbstractString="<Unknown>")
    getexits(file::AbstractString; force::Bool=false)

Return a `Vector` of all screen and secondary exits in the given IO or file.

`filename` is used for error reporting.
Files associated with this method end in ".exits".
"""
function getexits(io::IO, filename::AbstractString="<Unknown>")
    exitbytes = read(io, 128)
    @assert length(exitbytes) == 128 "unexpected amount of exit bytes in $filename."
    convert(Vector{NamedTuple},
            # Somehow we cannot deconstruct this tuple.
            # x[1] = screen, x[2] = bytes
            filter(!isnothing, map(x -> parseexit(x[2], convert(UInt8, x[1])),
                                   enumerate(Iterators.partition(exitbytes, 4)))))
end

function getexits(file::AbstractString; force::Bool=false)
    force || @assert endswith(file, ".exits") ("please use files with the .exits "
                                                * "extension or set `force=true`.")
    open(f -> getexits(f, file), file)
end

"""
Return a `NamedTuple` of data parsed from the given 4 byte exit object or `nothing` if
there is no exit on the given `screen`.
"""
function parseexit(exitbytes::AbstractVector{UInt8}, screen::UInt8)
    @assert length(exitbytes) == 4 "unexpected amount of bytes for exit."
    # Test the `isexit` byte.
    Bool(exitbytes[3]) || return nothing
    # If `secondaryexit` is `true` (see below), this always references a secondary entrance
    # number. Otherwise, this is a level number.
    destinationbits = bitstring(exitbytes[1])

    extras = bitstring(exitbytes[2])
    # If this bit is set, the exit leads to a reference (of a secondary entrance),
    # not a level.
    secondaryexit = extras[7] == '1'
    destinationbits = extras[8] * destinationbits
    secondaryexit && (destinationbits = extras[1:4] * destinationbits)
    destination = parse(UInt16, destinationbits, base=2)
    # midwaywaterbits = extras[5]
    lmmodified = extras[6] == '1'

    # exitbytes[4] is unused
    return (destination=destination, screen=screen, secondaryexit=secondaryexit,
            lmmodified=lmmodified, )
end

"""
    makeentrancelayers(entrances::AbstractVector{NamedTuple},
                       stats::LevelStatistics.LevelStats, padded::Bool=true)

Return an `Array{UInt16, 3}` containing the given `entrances`.

If `padded` is `false`, the cuboid only contains layers for the entrances in `entrances`.
Otherwise, for each possible entrance number, an empty layer is constructed.
"""
function makeentrancelayers(entrances::AbstractVector{T},
                            stats::LevelStatistics.LevelStats, padded::Bool=true
                            ) where {T <: NamedTuple}
    if padded
        entrancecount = maximumentrances
        entrancemappingdict = entrancemapping
    else
        uniqueentrances = unique(entrances)
        entrancecount = length(uniqueentrances)
        entrancemappingdict = Dict(k => i
                for (i, k) in enumerate(sort(map(e -> e.number, uniqueentrances))))
    end
    vertical = LevelStatistics.isvertical(stats)
    if vertical
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(LevelStatistics.screenrowsvert)
        out = zeros(UInt16, LevelStatistics.screencolumns(stats),
                    columnsperscreen * stats.screens, entrancecount)
    else
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(LevelStatistics.screencols)
        out = zeros(UInt16, LevelStatistics.screenrows(stats),
                    columnsperscreen * stats.screens, entrancecount)
    end

    skiperrors = false
    # We can ignore that y and x are swapped for vertical levels as vertical levels are in
    # a transposed matrix anyway.
    for entrance in entrances
        y, x = entranceposition(entrance, stats)
        number = entrance.number
        if !haskey(entrancemappingdict, number)
            if endswith(stats.filename, "000.ent") || endswith(stats.filename, "100.ent")
                skiperrors = true
            end
            skiperrors && continue

            @error ("no mapping for entrance 0x$(string(number, base=16)) in "
                    * "$(stats.filename). Skipping...")
        else
            if x > size(out, 2)
                @warn ("makeentrancelayers: in $(stats.filename): x=$x out of "
                       * "bounds. Skipping...")
            elseif y > size(out, 1)
                @warn ("makeentrancelayers: in $(stats.filename): y=$y out of "
                       * "bounds. Skipping...")
            else
                # Add one so the number is never zero.
                out[y, x, entrancemappingdict[number]] = number + 1
            end
        end
    end
    return out
end

"""
    getentrances(io::IO, filename::AbstractString="<Unknown>")
    getentrances(file::AbstractString; force::Bool=false)

Return a `Vector` of all secondary entrances in the given IO or file.

Secondary entrances are all entrances into a level aside from the main entrance and
midway point entrance.
For which objects are found in the vector, see [`parseentrance`](@ref).
`filename` is used for error reporting.
Files associated with this method end in ".ent2".
"""
function getentrances(io::IO, filename::AbstractString="<Unknown>")
    entrancebytes = read(io)::Vector{UInt8}
    @assert length(entrancebytes) % 8 == 0 ("unexpected amount of entrance bytes "
                                            * "in $filename.")
    map(f -> parseentrance(f, filename), Iterators.partition(entrancebytes, 8))
end

function getentrances(file::AbstractString; force::Bool=false)
    force || @assert endswith(file, ".ent2") ("please use files with the .ent2 extension "
                                              * "or set `force=true`.")
    open(f -> getentrances(f, file), file)
end

"Return a `NamedTuple` of data parsed from the given 8 byte secondary entrance object."
function parseentrance(entrancebytes::AbstractVector{UInt8},
                       filename::AbstractString="<Unknown>")
    @assert length(entrancebytes) == 8 "unexpected amount of bytes for secondary entrance."
    number = entrancebytes[2] * 0x100 + entrancebytes[1]

    extras = mapreduce(bitstring, *, entrancebytes[3:7])
    # bgpositionbits = extras[1:2]
    # fgpositionbits = extras[3:4]
    # We have these bits twice because the behaviour depends on `relativefgbg`.
    # fgbgoffsetbits = extras[26] * bgpositionbits * fgpositionbits
    xymethod2 = extras[18] == '1'
    ybits = extras[5:8]
    xbits = extras[9:11]
    if xymethod2
        xbits = extras[19:20] * xbits
        ybits = extras[27:32] * ybits
        y = parse(UInt16, ybits, base=2)
        x = parse(UInt8, xbits, base=2)
        horizontalsubscreenbit = 0x0
        verticalsubscreenbit   = 0x0
    else
        yref = parse(UInt8, ybits, base=2) + 1
        xref = parse(UInt8, xbits, base=2) + 1
        y = UInt16(ys[yref])  # Convert for type stability
        x = xs[xref]
        # If the level is not expanded (vanilla size) and the bit is set, the value is
        # increased by the amount of rows (or columns for vertical levels) per
        # first subscreen.
        horizontalsubscreenbit = horizontalsubscreenbits[yref]
        verticalsubscreenbit   = verticalsubscreenbits[xref]
    end
    screenbits = extras[12:16]
    screen = parse(UInt8, screenbits, base=2)
    # slipperybits = extras[17]
    # This bit could be used to check whether the high bit of the level this is read from
    # matches this. Otherwise it is useless.
    # destinationbits = extras[21]
    # entranceactionbits = extras[22:24]
    exittooverworldbits = extras[25]
    if !(endswith(filename, "Level000.ent2") || endswith(filename, "Level100.ent2"))
        @assert exittooverworldbits == '0' ("cannot handle behaviour specified by "
                                            * "secondary screen exit.")
    end
    # relativefgbgbits = extras[33]
    # faceleftbits = extras[34]
    # waterbits = extras[35]
    # extras[36:40] is unused

    # entrancebytes[8] is unused
    return (y=y, x=x, horizontalsubscreenbit=horizontalsubscreenbit,
            verticalsubscreenbit=verticalsubscreenbit, screen=screen,
            number=number, xymethod2=xymethod2)
end

"Return the absolute y and x position of the given entrance for the given level."
function entranceposition(entrance::NamedTuple, stats::LevelStatistics.LevelStats)
    y = entrance.y
    x = widen(entrance.x)
    if LevelStatistics.isvertical(stats)
        y += entrance.screen * LevelStatistics.screenrows(stats)
        entrance.verticalsubscreenbit == 0x1 && (x += LevelStatistics.screencols)
    else
        x += entrance.screen * LevelStatistics.screencolumns(stats)
        entrance.horizontalsubscreenbit == 0x1 && (y += LevelStatistics.screenrowshoritop)
    end
    # Add 1 for 1-based indexing.
    return (y + 0x1, x + 0x1)
end

"""
    parsemap16(io::IO, filename::AbstractString="<Unknown>")
    parsemap16(file::AbstractString; force::Bool=false)

Return a [`DefaultDict{UInt16, UInt16}`](@ref) mapping the Map16 values to vanilla tiles,
where the Map16 data is in the given IO or file.

Tiles without a mapping resolve to [`defaultmap16tile`](@ref). After tile 0x1FF has been
parsed, mappings to this value will not be added to the result (the default value of the
returned `DefaultDict` is `defaultmap16tile`).
`filename` is used for error reporting.
Files associated with this method are called "map16fgG.bin".
"""
function parsemap16(io::IO, filename::AbstractString="<Unknown>")
    mappings = DefaultDictionary.DefaultDict{UInt16, UInt16}(defaultmap16tile)
    vanillatiles = read(io, 2 * LevelStatistics.uniquevanillatiles)
    @assert length(vanillatiles) == 2 * LevelStatistics.uniquevanillatiles ("unexpected "
            * "amount of vanilla Map16 bytes.")
    for (number, littleendiantile) in enumerate(Iterators.partition(vanillatiles, 2))
        tile = littleendiantile[2] * 0x100 + littleendiantile[1]
        # Subtract one as tiles start at zero.
        mappings[convert(UInt16, number - 1)] = tile
    end
    extendedtiles = read(io, 0xfc00)
    @assert length(extendedtiles) == 0xfc00 "unexpected amount of extended Map16 bytes."
    for (number, littleendiantile) in enumerate(
            Iterators.filter(!isequal(defaultmap16tile_le),
                             Iterators.partition(extendedtiles, 2)))
        tile = littleendiantile[2] * 0x100 + littleendiantile[1]
        # Subtract one as tiles start at zero.
        mappings[convert(UInt16, number + LevelStatistics.uniquevanillatiles - 1)] = tile
    end
    resolvemap16!(mappings)
    return mappings
end

function parsemap16(file::AbstractString, force::Bool=false)
    force || @assert splitdir(file)[end] == "map16fgG.bin" ("please use map16fgG.bin files "
                                                            * "or set `force=true`")
    open(f -> parsemap16(f, file), file)
end

"""
Modify the given Map16 mappings so each key points directly to the vanilla value it resolves
to, making lookup faster in the future.
"""
function resolvemap16!(mappings::DefaultDictionary.DefaultDict{UInt16, UInt16})
    remainingkeys = Set{keytype(mappings)}(keys(mappings))
    while !isempty(remainingkeys)
        key = pop!(remainingkeys)
        value = mappings[key]
        key == value && continue
        resolvekeys = Set{keytype(mappings)}()
        while key != value
            if key in resolvekeys
                key < value && (value = key)
                @warn ("infinite recursion during Map16 resolution. "
                       * "Resolved with lower value.")
                break
            else
                push!(resolvekeys, key)
            end
            key = value
            value = mappings[key]
        end
        foreach(k -> (mappings[k] = value), resolvekeys)
        setdiff!(remainingkeys, resolvekeys)
    end
end

"""
    applymap16!(tiles::AbstractMatrix{UInt16},
                mappings::DefaultDictionary.DefaultDict{UInt16, UInt16})
    applymap16!(tiles::AbstractMatrix{UInt16}, file::AbstractString)

Modify the given tile matrix in place, replacing each value with the resolved Map16 value.
The mappings can be given directly or via a file name to parse them from.
"""
function applymap16!(tiles::AbstractMatrix{UInt16},
                     mappings::DefaultDictionary.DefaultDict{UInt16, UInt16})
    for i in eachindex(tiles)
        tiles[i] = mappings[tiles[i]]
    end
end

function applymap16!(tiles::AbstractMatrix{UInt16}, file::AbstractString)
    applymap16!(tiles, parsemap16(file))
end

"""
Return a `Tuple` of `Dict`s counting all existing (found in files) entrances, screen exits
and secondary exits in the given directory.
A set of existing level names is also in the returned tuple.
"""
function countexisting(dir::AbstractString)
    entrancedict      = Dict{UInt16, Int}()
    exitdict          = Dict{UInt16, Int}()
    secondaryexitdict = Dict{UInt16, Int}()
    levelset = Set{UInt16}()
    levelnumber_re = r".*Level([0-9A-F]{3})\."

    for file in map(f -> joinpath(dir, f), filter(
            f -> endswith(f, ".ent2") || endswith(f, ".exits"), readdir(dir)))
        levelnumber = parse(UInt16, match(levelnumber_re, file).captures[1], base=16)
        levelnumber in levelset || push!(levelset, levelnumber)
        if endswith(file, ".ent2")
            for entrance in getentrances(file)
                number = entrance.number
                haskey(entrancedict, number) || (entrancedict[number] = 0)
                entrancedict[number] += 1
            end
        elseif endswith(file, ".exits")
            exits = getexits(file)
            for exit in exits
                destination = exit.destination
                if exit.secondaryexit
                    if !haskey(secondaryexitdict, destination)
                        secondaryexitdict[destination] = 0
                    end
                    secondaryexitdict[destination] += 1
                else
                    haskey(exitdict, destination) || (exitdict[destination] = 0)
                    exitdict[destination] += 1
                end
            end
        else
            error("wrong file extension")
        end
    end
    return entrancedict, exitdict, secondaryexitdict, levelset
end

# TODO function to fix missing levels? using symlink only?
# TODO look over missing levels, find out why non-bosses are prevalent
# answer: switch palace, goal post screens, short or bonus levels
# TODO better checksum stuff, check other files as well – not only .map
# maybe just symlink _every_ duplicate file
# OR (better solution) symlink only those files for which not _every_ checksum is the same (otherwise they are deleted/later fixed with the fix-missing-levels function)

# global TODO change counters like below to UInt

# TODO do we need this?
"""
Return a `NamedTuple` of `NamedTuple`s containing the maximum amount of secondary entrances,
screen exits and secondary exits over all hacks in `parentdir`, a set of those maximal
secondary entrances, screen exits and secondary exits as well as the names of the hacks each
maximum belongs to.
"""
function getsecondarystatsperrom(parentdir::AbstractString)
    maximumentrances = 0
    maximumexits = 0
    maximumsecondaryexits = 0
    local maximumentranceshack, maximumexitshack, maximumsecondaryexitshack
    local entranceset, exitset, secondaryexitset
    for (root, dirs, files) in walkdir(parentdir)
        for dir in dirs
            entrancedict, exitdict, secondaryexitdict, _ = countexisting(
                    joinpath(root, dir))

            # Only count used (connected) secondary exits.
            entranceset  = intersect(keys(entrancedict), keys(secondaryexitdict))
            numentrances = length(entranceset)
            if numentrances > maximumentrances
                maximumentrances    = numentrances
                maximumentranceshack = dir
            end

            exitset  = keys(exitdict)
            numexits = length(exitset)
            if numexits > maximumexits
                maximumexits    = numexits
                maximumexitshack = dir
            end

            secondaryexitset  = keys(secondaryexitdict)
            numsecondaryexits = length(secondaryexitset)
            if numsecondaryexits > maximumsecondaryexits
                maximumsecondaryexits    = numsecondaryexits
                maximumsecondaryexitshack = dir
            end
        end
    end
    return (maxima=(entrances=maximumentrances, exits=maximumexits,
                    secondaryexits=maximumsecondaryexits),
            sets=(entrances=entranceset, exits=exitset,
                  secondaryexits=secondaryexitset),
            hacks=(entrances=maximumentranceshack, exits=maximumexitshack,
                   secondaryexits=maximumsecondaryexitshack))
end

"""
Return a `NamedTuple` of `NamedTuple`s containing the total amount of different secondary
entrances, screen exits and secondary exits over all hacks in `parentdir` and sets of all
those secondary entrances, screen exits and secondary exits.
"""
function getsecondarystats(parentdir::AbstractString)
    entranceset = Set{UInt16}()
    exitset = Set{UInt16}()
    secondaryexitset = Set{UInt16}()
    for (root, dirs, files) in walkdir(parentdir)
        for dir in dirs
            entrancedict, exitdict, secondaryexitdict, _ = countexisting(
                    joinpath(root, dir))

            # Only count used (connected) secondary exits.
            union!(entranceset, intersect(keys(entrancedict), keys(secondaryexitdict)))
            union!(exitset, keys(exitdict))
            union!(secondaryexitset, keys(secondaryexitdict))
        end
    end

    maximumentrances = length(entranceset)
    maximumexits = length(exitset)
    maximumsecondaryexits = length(secondaryexitset)
    return (maxima=(entrances=maximumentrances, exits=maximumexits,
                    secondaryexits=maximumsecondaryexits),
            sets=(entrances=entranceset, exits=exitset,
                  secondaryexits=secondaryexitset))
end

function makemapping(set::AbstractSet{<:Number})
    Dict(k => i for (i, k) in enumerate(sort(collect(set))))
end

# TODO most likely delete
function generatesecondarystatsfileperrom(path::AbstractString)
    println("Generating secondary level stats file $path...")
    maxima, _, hacks = getsecondarystats(LevelStatistics.leveldir)
    stats = ["entrances"      "exits"      "secondary exits";
             hacks.entrances  hacks.exits  hacks.secondaryexits;
             maxima.entrances maxima.exits maxima.secondaryexits]
    writedlm(path, stats, ';')
    println("Done.")
end

function generatesecondarystatsfile(path::AbstractString)
    println("Generating secondary level stats file $path...")
    maxima = getsecondarystats(LevelStatistics.leveldir)[1]
    stats = ["entrances"      "exits"      "secondary exits";
             maxima.entrances maxima.exits maxima.secondaryexits]
    writedlm(path, stats, ';')
    println("Done.")
end

function generatemappingfile(path::AbstractString, set::AbstractSet{<:Number})
    println("Generating mapping file $path...")
    mapping = makemapping(set)
    writedlm(path, ("key" => "value", mapping...), ';')
    println("Done.")
end

function generatemappings(entrancemappingfile, exitmappingfile, secondaryexitmappingfile)
    println("Fetching mappings...")
    sets = getsecondarystats(LevelStatistics.leveldir).sets
    generatemappingfile(entrancemappingfile, sets.entrances)
    generatemappingfile(exitmappingfile, sets.exits)
    generatemappingfile(secondaryexitmappingfile, sets.secondaryexits)
end

function generateall()
    generatesecondarystatsfile(secondarystatsfile)
    generatemappings(entrancemappingfile, exitmappingfile, secondaryexitmappingfile)
end

"Print a message on how to precalculate values to speed up initialization."
function printprecalculatemsg()
    @info """
          You can speed up the initialization process:
          Precalculate by running `$(parentmodule(generateall)).generateall()`.
          To disable this message, edit `printprecalculatereminder` to `false`.
          """
end

# TODO delete most likely
function readsecondarystatsfileperrom(path::AbstractString)
    maximumentrances, maximumexits, maximumsecondaryexits = readdlm(path, ';', Int,
                                                                    skipstart=1)
end

function readsecondarystatsfile(path::AbstractString)
    maximumentrances, maximumexits, maximumsecondaryexits = readdlm(path, ';', Int,
                                                                    skipstart=1)
end

function readmappingfile(path::AbstractString)
    mapping = readdlm(path, ';', Int, skipstart=1)
    Dict{UInt16, Int}(eachrow(mapping))
end

function initsecondarystats(statsfile::AbstractString)
    if isfile(statsfile)
        maximumentrances, maximumexits, maximumsecondaryexits = readsecondarystatsfile(
                statsfile)
    else
        println("Calculating secondary level stats...")
        maximumentrances, maximumexits, maximumsecondaryexits = getsecondarystats(
                LevelStatistics.leveldir).maxima
        if (printprecalculatereminder && isfile(entrancemappingfile)
                && isfile(exitmappingfile) && isfile(secondaryexitmappingfile))
            printprecalculatemsg()
        end
    end
    return maximumentrances, maximumexits, maximumsecondaryexits
end

function initmapping(mappingfile::AbstractString, set::AbstractSet{<:Number},
                     mapping::AbstractString="")
    if isfile(mappingfile)
        mapping = readmappingfile(mappingfile)
    else
        if isempty(mapping)
            println("Calculating mapping...")
        else
            println("Calculating ", mapping, " mapping...")
        end
        mapping = makemapping(set)
    end
    return mapping
end

function initmappings(entrancemappingfile, exitmappingfile, secondaryexitmappingfile)
    notallfilesexist = !(isfile(entrancemappingfile) && isfile(exitmappingfile)
                         && isfile(secondaryexitmappingfile))
    if notallfilesexist
        sets = getsecondarystats(LevelStatistics.leveldir).sets
        entrancemapping = initmapping(entrancemappingfile, sets.entrances, "entrance")
        exitmapping = initmapping(exitmappingfile, sets.exits, "exit")
        secondaryexitmapping = initmapping(
                secondaryexitmappingfile, sets.secondaryexits, "secondary exit")
    else
        entrancemapping = readmappingfile(entrancemappingfile)
        exitmapping = readmappingfile(exitmappingfile)
        secondaryexitmapping = readmappingfile(secondaryexitmappingfile)
    end
    if printprecalculatereminder && notallfilesexist
        printprecalculatemsg()
    end
    return entrancemapping, exitmapping, secondaryexitmapping
end


const maximumentrances, maximumexits, maximumsecondaryexits = initsecondarystats(
        secondarystatsfile)

"Maximal amount of secondary entrances over all ROMs."
maximumentrances

"Maximal amount of screen exits over all ROMs."
maximumexits

"Maximal amount of secondary exits over all ROMs."
maximumsecondaryexits

const entrancemapping, exitmapping, secondaryexitmapping = initmappings(entrancemappingfile,
        exitmappingfile, secondaryexitmappingfile)

"Mapping of secondary entrance numbers to layer indices."
entrancemapping

"Mapping of screen exit numbers to layer indices."
exitmapping

"Mapping of secondary exit numbers to layer indices."
secondaryexitmapping

"""
Return the amount of connected secondary entrances/exits and screen exits/levels.
If there are missing entrances for secondary exits or missing levels for screen exits,
return those as well (otherwise they are `nothing`).

If `quiet` is `false`, display messages regarding possible errors.
"""
function countconnected(dir::AbstractString, quiet::Bool=false)
    entrancedict, exitdict, secondaryexitdict, levelset = countexisting(dir)
    if !quiet && any(values(entrancedict) .> 1)
        @warn "In $dir: duplicate secondary entrance"
    end
    connectedsecondary = length(intersect(keys(entrancedict), keys(secondaryexitdict)))
    connected = length(intersect(levelset, keys(exitdict)))
    if connectedsecondary != length(secondaryexitdict)
        quiet || @warn ("In $dir: "
                * "$connectedsecondary connected secondary entrances/exits, "
                * "$(length(entrancedict)) entrances, "
                * "$(length(secondaryexitdict)) secondary exits")
        missingentrances = setdiff(keys(secondaryexitdict), keys(entrancedict))
        if !isempty(missingentrances) && !quiet
            @warn "Secondary exit(s) without connected entrance(s) in $(dir)!"
        end
    else
        missingentrances = nothing
    end
    if connected != length(exitdict)
        quiet || @warn ("In $dir: $connected connected levels/exits, "
                * "$(length(levelset)) levels, $(length(exitdict)) exits")
        missinglevels = setdiff(keys(exitdict), levelset)
    else
        missinglevels = nothing
    end
    # Even though `missingentrances` is sometimes not `nothing` or empty, all occurrences
    # have been fixed and documented in "stats/missing_entrances.txt".
    return connectedsecondary, connected, missingentrances, missinglevels
end

"""
Print all missing entrances with the corresponding hack's name in the given parent
directory.

Even though there are occurrences, everything has been looked over and fixed or documented
in "stats/missing_entrances.txt".
"""
function printallmissingentrances(parentdir::AbstractString=LevelStatistics.leveldir)
    for dir in filter(d -> isdir(joinpath(parentdir, d)), readdir(parentdir))
        missingentrances = countconnected(joinpath(parentdir, dir), true)[3]
        if !(isnothing(missingentrances) || isempty(missingentrances))
            println(rpad(dir * ": ", 30), missingentrances)
        end
    end
end

"""
Print all missing levels and return a `Dict{UInt16, Int}` counting all unique missing
level numbers.
"""
function printallmissinglevels(parentdir::AbstractString=LevelStatistics.leveldir,
                               quiet::Bool=false)
    missingdict = Dict{UInt16, Int}()
    # TODO implement this instead (maybe in new function)
    # missingdict = Dict{String, Dict{UInt16, Int}}()
    for dir in filter(d -> isdir(joinpath(parentdir, d)), readdir(parentdir))
        missinglevels = countconnected(joinpath(parentdir, dir), true)[4]
        if !isnothing(missinglevels)
            for level in missinglevels
                haskey(missingdict, level) || (missingdict[level] = 0)
                missingdict[level] += 1
            end
        end
        if !(quiet || isnothing(missinglevels) || isempty(missinglevels))
            println(rpad(dir * ": ", 30), missinglevels)
        end
    end
    return missingdict
end

"Check whether all ROMs in the given parent directory have the same xy lookup table."
function checkxytables(parentdir::AbstractString=LevelStatistics.leveldir)
    function changeendian(byte::UInt8)
        second = div(byte, 0x10)
        first = byte - second * 0x10
        return first * 0x10 + second
    end

    comparison = vcat(map(changeendian, ys), horizontalsubscreenbits,
                      map(changeendian, xs), verticalsubscreenbits)

    for (root, dirs, files) in walkdir(parentdir)
        for file in map(f -> joinpath(root, f), filter(f -> endswith(f, ".smc"), files))
            open(file) do f
                skip(f, 0x2d930)  # Skip up to where the x-y-table should be
                tablebytes = read(f, 48)
                @assert tablebytes == comparison "different x-y-table in $file."
            end
        end
    end
end

end # module

