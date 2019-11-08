"Collect level statistics in a structure capturing essential information."
module LevelStatistics

# TODO separate header stuff (new module) and level statistics like screen rows,
# bytes per tile, ...

import Base.show, Base.size

using ..XYTables

export LevelStats, getstats, isvertical, isboss, islayer2, screencolumns, screenrows
export tilesperscreen, hack, name
export mainentrance, midwayentrance, mainentranceaction, midwayentranceaction
export makemainentrancelayer, makemidwayentrancelayer
export screencols, screenrowshoritop, screenrowshoribottom, screenrowshori, screenrowsvert
export leveldir, originalleveldir, uniquevanillatiles

# Please customize the following two variables if necessary!
"Directory containing all levels."
const leveldir = abspath(joinpath(@__DIR__, "..", "..", "levels"))
"Directory containing all levels of the original game."
const originalleveldir = joinpath(leveldir, "Super Mario World (U) [!]")


"""
The amount of different tiles in the original game. There are more behaviours than tiles due
to tiles behaving differently with different FG/BG GFX settings!
"""
const uniquevanillatiles = 0x200


# All the following sizes assume vanilla level sizes! Remove levels with an expanded format.
# In the dataset, only a single level had to be removed due to using the expanded format.

"How many bytes each tile consists of."
const bytespertile = 0x2

"The maximum amount of screens in a level."
const maxscreens = 0x20

"Number of columns of tiles in each screen (or subscreen for vertical levels)."
const screencols = 0x10

"Maximum amount of columns in a horizontal level."
const maxcolshori = widen(maxscreens) * screencols

"Number of rows of tiles in each top subscreen in horizontal levels."
const screenrowshoritop = 0x10
"Number of rows of tiles in each bottom subscreen in horizontal levels."
const screenrowshoribottom = 0x0b

"Number of rows of tiles in each screen in horizontal levels."
const screenrowshori = screenrowshoritop + screenrowshoribottom
"Number of rows of tiles in each screen in vertical levels."
const screenrowsvert = 0x10

"""
Number of parts of different sizes per screen in horizontal levels.
The other subscreen is to the bottom of the first one.
"""
const subscreenshori = 0x2

"""
Number of parts of `screenrowsvert * screencols` per screen in vertical levels.
The other subscreen is to the right of the first one.
"""
const subscreensvert = 0x2

"How many tiles per subscreen (only relevant for vertical levels)."
const tilespersubscreenvert = widen(screenrowsvert) * screencols

"How many tiles per screen in horizontal levels."
const tilesperscreenhori = widen(screenrowshori) * screencols
"How many tiles per screen (_not_ subscreen) in vertical levels."
const tilesperscreenvert = tilespersubscreenvert * subscreensvert

"Maximum amount of tiles in a horizontal level."
const maxtileshori = maxcolshori * screenrowshori


# Level modes

"Only these modes appear in the vanilla game."
const vanillamodes = (0x0, 0x1, 0x2, 0x8, 0x9, 0xa, 0xb, 0xc, 0xe, 0x10, 0x11)

# Order of these due to efficiency (earlier modes appear more often in the vanilla game).
# Non-vanilla modes are not considered in the order.
"Modes which indicate a vertical level."
const vertmodes = (0xa, 0x8, 0x7, 0xd)
"Modes which indicate a boss level."
const bossmodes = (0x9, 0xb, 0x10)  # All horizontal
"Modes which indicate a layer 2 interactive level."
const layer2modes = (0x2, 0x8, 0x1f)

# These modes are less important.
"Standard mode for most levels without any extras."
const standardmode = 0x0
"Modes which indicate a layer 2 non-interactive level."
const layer2noninteractivemodes = (0x1, 0x7, 0xf)
"Modes which indicate a dark background."
const darkbgmodes = (0xc, 0x11, 0xd)  # All horizontal
"""
Mode which indicates the underwater ghost ship level. Pretty much a normal
horizontal level.
"""
const ghostshipmode = 0xe

# These modes settings do not appear in the vanilla game.
"Modes which indicate a transparent level."
const translucentmodes = (0x1e, 0x1f)
"Modes that Lunar Magic recommends not to use."
const unusablemodes = (0x3,  0x4, 0x5, 0x6, 0x12, 0x13, 0x14, 0x15,
                       0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d)


abstract type Entrance end

"Information about a level's main entrance."
struct MainEntrance <: Entrance
    "Relative y position of the main entrance."
    y::UInt16
    "Relative x position of the main entrance."
    x::UInt8
    "Whether the entrance is in the lower part of the screen (in horizontal levels)."
    horizontalsubscreenbit::UInt8
    "Whether the entrance is in the right part of the screen (in vertical levels)."
    verticalsubscreenbit::UInt8
    "Screen the main entrance resides in."
    screen::UInt8

    "Entrance action. Also determines whether the level is a water level."
    entranceaction::UInt8
end

"Information about a level's midway entrance."
struct MidwayEntrance <: Entrance
    "Relative y position of the midway entrance."
    y::UInt16
    "Relative x position of the midway entrance."
    x::UInt8
    "Screen the midway entrance resides in."
    screen::UInt8

    "Whether information is copied from the main entrance."
    separateentrance::Bool
    "Whether the midway entrance is redirected to another level."
    redirect::Bool
    "Entrance action. Also determines whether the level is a water level."
    entranceaction::UInt8
end

# TODO create new structs for headers, insert those directly, write methods to reference
# stats from there. would be much better for extensibility and more compact and readable.
# also refactor main- and midwayentrance then to calculate from the headers directly.
"Contains various statistics about a level. Create using [`getstats`](@ref)."
struct LevelStats
    "The complete filename the level was read from."
    filename::String
    "The level number."
    number::UInt16
    "Amount of screens."
    screens::UInt8
    "Mode information. Relevant to determine verticality or boss levels."
    mode::UInt8
    "FG/BG GFX setting. Determines the behaviour and availability of certain tiles."
    fgbggfx::UInt8
    "What Lunar Magic expanded format the level uses."
    lmexpandedformat::UInt8

    "The level's main entrance."
    mainentrance::MainEntrance
    "The level's midway entrance."
    midwayentrance::MidwayEntrance
end

function Base.show(io::IO, x::LevelStats)
    print(io, """
            File:     $(x.filename)
            Number:   $(x.number) (0x$(string(x.number, base=16)))
            Screens:  $(x.screens) (0x$(string(x.screens, base=16)))
            Mode:     $(x.mode) (0x$(string(x.mode, base=16)))
            Vertical: $(isvertical(x) ? "Yes" : "No")""")
end

"Return whether the given level's orientation is vertical."
isvertical(stats::LevelStats) = stats.mode in vertmodes
"Return whether the given level is a boss level."
isboss(stats::LevelStats) = stats.mode in bossmodes
"Return whether the given level is a layer 2 interactive level (moving parts)."
islayer2(stats::LevelStats) = stats.mode in layer2modes

"Number of columns of tiles in each screen in the given level."
function screencolumns(stats::LevelStats)
    isvertical(stats) ? screencols * subscreensvert : screencols
end
"Number of rows of tiles in each screen in the given level."
screenrows(stats::LevelStats) = isvertical(stats) ? screenrowsvert : screenrowshori

"""
How many tiles per screen in the given level.
Based on the vanilla game – may be different for hacks.
"""
function tilesperscreen(stats::LevelStats)
    isvertical(stats) ? tilesperscreenvert : tilesperscreenhori
end

"""
Return the level's size in bytes.
As this method depends on [`tilesperscreen`](@ref), the result may be wrong for hacks.
"""
size(stats::LevelStats) = stats.screens * widen(tilesperscreen(stats)) * bytespertile

"""
    hack(stats::LevelStats)

Return the name of the hack the given level belongs to.
This is the last directory of `stats.filename`.
"""
hack(stats::LevelStats) = splitpath(stats.filename)[end - 1]

"""
Return the name of the level.
It consists of the directory containing the level and its filename without an extension,
separated by a forward slash (`'/'`).
"""
function name(stats::LevelStats)
    dir, base = splitpath(stats.filename)[end - 1:end]
    join((dir, splitext(base)[1]), '/')
end

"Return the absolute y and x coordinates of the given level's main entrance."
function mainentrance(stats::LevelStats)
    entrance = stats.mainentrance
    y = entrance.y
    x = widen(entrance.x)
    screen = entrance.screen
    if isvertical(stats)
        y += screen * screenrows(stats)
        entrance.verticalsubscreenbit == 0x1 && (x += screencols)
    else
        x += screen * screencolumns(stats)
        entrance.horizontalsubscreenbit == 0x1 && (y += screenrowshoritop)
    end
    # Add 1 for 1-based indexing.
    return (y + 0x1, x + 0x1)
end

"Return the absolute y and x coordinates of the given level's midway entrance."
function midwayentrance(stats::LevelStats)
    entrance = stats.midwayentrance
    if entrance.separateentrance
        # We do not want to consider the screen; these are absolute.
        return (entrance.y + 0x1, entrance.x + 0x1)
    else
        mainentrance = stats.mainentrance
        y = mainentrance.y
        x = widen(mainentrance.x)
        screen = entrance.screen
        if isvertical(stats)
            y += screen * screenrows(stats)
            mainentrance.verticalsubscreenbit == 0x1 && (x += screencols)
        else
            x += screen * screencolumns(stats)
            mainentrance.horizontalsubscreenbit == 0x1 && (y += screenrowshoritop)
        end
        # Add 1 for 1-based indexing.
        return (y + 0x1, x + 0x1)
    end
end

function mainentranceaction(stats::LevelStats)
    return stats.mainentrance.entranceaction
end

function midwayentranceaction(stats::LevelStats)
    entrance = stats.midwayentrance
    if entrance.separateentrance
        return entrance.entranceaction
    else
        return stats.mainentrance.entranceaction
    end
end

"Return a `Matrix{UInt8}` of zeros with a one at the main entrance."
function makemainentrancelayer(stats::LevelStats)
    makeentrancelayer(mainentrance(stats)..., stats)
end

"Return a `Matrix{UInt8}` of zeros with a one at the midway entrance."
function makemidwayentrancelayer(stats::LevelStats)
    makeentrancelayer(midwayentrance(stats)..., stats)
end

"Return a `Matrix{UInt8}` of zeros with a one at the given coordinates."
function makeentrancelayer(y, x, stats::LevelStats)
    if isvertical(stats)
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(screenrowsvert)
        out = zeros(UInt8, screencolumns(stats), columnsperscreen * stats.screens)
        # Seems like x and y are _not_ swapped for entrances.
        # The vertical level is transposed, so we would assume they were swapped if we had
        # the same assignment as below.
        if y > size(out, 2)
            @warn "in $(stats.filename): y=$x out of bounds"
        else
            out[x, y] = 0x1
        end
    else
        # Columns per screen in the level matrix (not the actual level columns).
        columnsperscreen = widen(screencols)
        out = zeros(UInt8, screenrows(stats), columnsperscreen * stats.screens)
        if x > size(out, 2)
            @warn "makeentrancelayer: in $(stats.filename): x=$x out of bounds. Skipping..."
        else
            out[y, x] = 0x1
        end
    end
    return out
end


"""
    getstats(level::IO, filename="<Unknown>")
    getstats(level::AbstractString, filename=level; force=false)

Calculate statistics like amount of screens or orientation on the given level.
`filename` can be used to customize the field of the same name in the
returned [`LevelStats`](@ref).
Files associated with this method end in ".ent".

# Examples
```jldoctest
julia> function printstats(header)
           stream = IOBuffer(hex2bytes(header))
           stats = getstats(stream)
           println(join((stats.screens, stats.mode, stats.vertical), ' '))
       end

julia> l105 = "05015b009a000000000000000000001a000000000000000000000000000000003340088027";

julia> printstats(l105)  # Stats for level 105
20 0 false

julia> l109 = "09010d0102660000000000000000001a00000000000000000000000000000000c66a129213";

julia> printstats(l109)  # Stats for level 105
7 10 true
```
"""
function getstats(level::IO, filename::AbstractString="<Unknown>")
    info = getlevelinfo(level)
    headerbits = tobitstring(info.header)
    headerstats = parseheaderbits(headerbits)
    secondaryheaderbits = tobitstring(info.secondaryheader)
    secondaryheaderstats = parsesecondaryheaderbits(secondaryheaderbits)
    midwayentrancebits = tobitstring(info.midwayentrance)
    midwayentrancestats = parsemidwayentrancebits(midwayentrancebits)

    mainentrance = MainEntrance(secondaryheaderstats.y, secondaryheaderstats.x,
                                secondaryheaderstats.horizontalsubscreenbit,
                                secondaryheaderstats.verticalsubscreenbit,
                                secondaryheaderstats.screen,
                                secondaryheaderstats.entranceaction)
    midwayentrance = createmidwayentrance(midwayentrancestats, secondaryheaderstats)

    return LevelStats(filename, info.number, headerstats.screens, headerstats.mode,
                      headerstats.fgbggfx, info.lmexpandedformat,
                      mainentrance, midwayentrance)
end

function getstats(level::AbstractString, filename::AbstractString=level; force::Bool=false)
    force || @assert endswith(level, ".ent") ("please use files with the .ent extension or "
                                              * "set `force=true`.")
    open(f -> getstats(f, filename), level)
end

"""
Return a `NamedTuple` of the given level's information.
The tuple contains the following (description: key (type)):
  * primary header: `header` (Vector{UInt8})
  * secondary header: `secondaryheader` (Vector{UInt8})
  * level number: `number` (UInt16)
  * midway point entrance: `midwayentrance` (Vector{UInt8})
  * Lunar Magic expanded format: `lmexpandedformat` (UInt8)
"""
function getlevelinfo(level::IO)
    numberarr = read(level, 2)
    number = numberarr[2] * 0x100 + numberarr[1]  # Convert little endian to value
    secondaryheader = read(level, 5)
    skip(level, 2)
    midwayentrance = read(level, 4)
    skip(level, 1)
    append!(secondaryheader, read(level, 2))
    # Mostly values of 0x00 (most) and 0x80 (some).
    # 0x00 and 0x80 behave the same; maybe the higher bits are a bug in Lunar Magic
    # as it only supports values between 0x0 and 0x1f.
    #
    # One occasion of 0x1b (level 025 in SMW^2 DEMO).
    # 0x1b levels have a screen height of 0x1c0 tiles.
    # However, due to simplicity we do not accept these levels.
    lmexpandedformat = read(level, 1)[1]
    @assert lmexpandedformat in (0x0, 0x80) "expanded level format is not supported."
    # 17 bytes parsed;
    # need to parse 15 more to be at ofset 0x20 (position of the primary header).
    skip(level, 0x0f)
    header = read(level, 5)
    (header=header, secondaryheader=secondaryheader, number=number,
            midwayentrance=midwayentrance, lmexpandedformat=lmexpandedformat)
end

"""
Return a `NamedTuple` of all the statistics found in the given bitstring of a
primary header.
"""
function parseheaderbits(headerbits::AbstractString)
    @assert length(headerbits) == 40 "unexpected bitstring size."
    # bgbits = headerbits[1:3]
    screensbits = headerbits[4:8]
    screens = parse(UInt8, screensbits, base=2) + 1
    # bgcolorbits = headerbits[9:11]
    modebits = headerbits[12:16]
    mode = parse(UInt8, modebits, base=2)
    # layer3prioritybits = headerbits[17]
    # musicbits = headerbits[18:20]
    # spritegfxbits = headerbits[21:24]
    # timerbits = headerbits[25:26]
    # spritebits = headerbits[27:29]
    # fgbits = headerbits[30:32]
    # itemmemorybits = headerbits[33:34]
    # verticalscrollbits = headerbits[35:36]
    fgbggfxbits = headerbits[37:40]
    fgbggfx = parse(UInt8, fgbggfxbits, base=2)
    (screens=screens, mode=mode, fgbggfx=fgbggfx)
end

"""
Return a `NamedTuple` of all the statistics found in the given bitstring of a
secondary header.
"""
function parsesecondaryheaderbits(headerbits::AbstractString)
    @assert length(headerbits) == 56 "unexpected bitstring size."
    # layer2scrollbits = headerbits[1:4]
    # Both y and x are referring to the main entrance's relative position in the screen,
    # not in the level.
    xymethod2 = headerbits[35] == '1'
    ybits = headerbits[5:8]
    xbits = headerbits[14:16]
    if xymethod2
        xbits = headerbits[36:37] * xbits
        ybits = headerbits[43:48] * ybits
        y = parse(UInt16, ybits, base=2)
        x = parse(UInt8, xbits, base=2)
        horizontalsubscreenbit = 0x0
        verticalsubscreenbit   = 0x0
    else
        yref = parse(UInt8, ybits, base=2) + 1
        xref = parse(UInt8, xbits, base=2) + 1
        y = UInt16(ys[yref])  # Convert for type stability
        x = xs[xref]
        horizontalsubscreenbit = horizontalsubscreenbits[yref]
        verticalsubscreenbit   = verticalsubscreenbits[xref]
    end
    # layer3settingbits = headerbits[9:10]
    entranceactionbits = headerbits[11:13]
    entranceaction = parse(UInt8, entranceactionbits, base=2)
    # Do **not** parse the following bits! Another bit needs to be prepended from the
    # midway entrance information.
    midwayentrancescreentailbits = headerbits[17:20]
    # fgpositionbits = headerbits[21:22]
    # bgpositionbits = headerbits[23:24]
    # We have these bits twice because the behaviour depends on `relativefgbg`.
    # fgbgoffsetbits = headerbits[42] * fgpositionbits * bgpositionbits
    # enableyoshiintrobits = headerbits[25]
    # unusedverticalpositionbits = headerbits[26]
    # verticalpositionbits = headerbits[27]
    screenbits = headerbits[28:32]
    screen = parse(UInt8, screenbits, base=2)
    # slipperybits = headerbits[33]
    # waterbits = headerbits[34]
    # smartspawnbits = headerbits[38]
    # spawnrangebits = headerbits[39:40]
    # relativebgbits = headerbits[41]
    # relativefgbgbits = headerbits[49]
    # faceleftbits = headerbits[50]
    # headerbits[51] is unused
    # bgpositionbits = headerbits[52:56]
    (y=y, x=x, horizontalsubscreenbit=horizontalsubscreenbit,
        verticalsubscreenbit=verticalsubscreenbit, screen=screen,
        midwayentrancescreentailbits=midwayentrancescreentailbits, xymethod2=xymethod2,
        entranceaction=entranceaction, )
end

"""
Return a `NamedTuple` of all the statistics found in the given bitstring of
midway entrance information.
"""
function parsemidwayentrancebits(midwaybits::AbstractString)
    @assert length(midwaybits) == 32 "unexpected bitstring size."
    separateentrancebits = midwaybits[3]
    separateentrance = separateentrancebits == '1'
    # Do **not** parse the following bit! Other bits need to be appended from the
    # secondary header.
    screenheadbits = midwaybits[4]
    if separateentrance
        # slipperybits = midwaybits[1]
        # waterbits = midwaybits[2]
        xbits = midwaybits[5] * midwaybits[13:16]
        x = parse(UInt8, xbits, base=2)
        entranceactionbits = midwaybits[6:8]
        entranceaction = parse(UInt8, entranceactionbits, base=2)
        ybits = midwaybits[27:32] * midwaybits[9:12]
        y = parse(UInt16, ybits, base=2)
        # relativefgbgbits = midwaybits[17]
        # faceleftbits = midwaybits[18]
        redirect = midwaybits[19] == '1'
        # midwaybits[20] is unused
        # fgpositionbits = headerbits[21:22]
        # bgpositionbits = headerbits[23:24]
        # midwaybits[25] is unused
        # We have these bits twice because the behaviour depends on `relativefgbg`.
        # fgbgoffsetbits = headerbits[26] * fgpositionbits * bgpositionbits
        return (y=y, x=x, screenheadbits=screenheadbits,
                separateentrance=separateentrance, redirect=redirect,
                entranceaction=entranceaction)
    else
        return (y=0x000, x=0x0, screenheadbits=screenheadbits,
                separateentrance=separateentrance, redirect=false, entranceaction=0x0)
    end
end

"Return a `MidwayEntrance` containing the given midway entrance information."
function createmidwayentrance(midwayentrancestats::NamedTuple,
                              secondaryheaderstats::NamedTuple)
    midwayentrancescreenbits = (midwayentrancestats.screenheadbits
                                * secondaryheaderstats.midwayentrancescreentailbits)
    midwayentrancescreen = parse(UInt8, midwayentrancescreenbits, base=2)

    # The `separateentrance` bit will be evaluated later.
    MidwayEntrance(midwayentrancestats.y, midwayentrancestats.x,
                   midwayentrancescreen, midwayentrancestats.separateentrance,
                   midwayentrancestats.redirect, midwayentrancestats.entranceaction)
end

"""
    tobitstring(header::AbstractVector, separator="")

Convert the given `AbstractVector` to one continuous bitstring.
Optionally, each byte's bitstring can be delimited by the given separator.
"""
function tobitstring(header::AbstractVector, separator::AbstractString="")
    mapreduce(bitstring, (x, y) -> x * separator * y, header)
end

function tobitstring(header::AbstractVector, separator::AbstractChar)
    mapreduce(bitstring, (x, y) -> x * string(separator) * y, header)
end

"""
Test whether all levels in the original game are correctly classified as vertical or
boss level.
"""
function test_verticalorboss(leveldir::AbstractString=originalleveldir)
    vertlevels = String[]
    bosslevels = String[]
    for level in map(x -> joinpath(leveldir, x),
                     filter(l -> endswith(l, ".ent"), readdir(leveldir)))
        stats = getstats(level)
        if isvertical(stats)
            push!(vertlevels, level)
        end
        if isboss(stats)
            push!(bosslevels, level)
        end
    end

    testvert = map(x -> joinpath(leveldir, "Level$x.ent"), String[
            # Many of these may be removed.
            "095"
            "096"
            "097"
            "098"
            "099"
            "09A"
            "09B"
            "0CC"
            "0D5"
            "0D9"
            "0DF"
            "0E2"
            "0E5"
            "195"
            "196"
            "197"
            "198"
            "199"
            "19A"
            "19B"
            "1C7"
            "1DE"
            "1EB"
            "1F6"
    ])

    testvert = map(x -> joinpath(leveldir, "Level$x.ent"), String[
            "0C2"
            "0DB"
            "0E7"
            "0EA"
            "0F7"
            "108"  # May be removed
            "109"
            "12A"
            "134"
            "1CE"
            "1ED"
    ])
    @assert length(vertlevels) == length(testvert) ("length is not the same. Maybe a level "
                                                    * "was removed.")
    @assert all(map(t -> t in vertlevels, testvert))

    return vertlevels
end

end # module

