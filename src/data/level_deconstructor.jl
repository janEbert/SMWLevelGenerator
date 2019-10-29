module LevelDeconstructor

import ..LevelStatistics
using ..Tiles: torawdata
using ..XYTables

export deconstructconstantinput, deconstructlevel

function deconstructconstantinput(constantinput::AbstractVector{Float32})
    deconstructconstantinput(convert.(UInt16, constantinput))
end

"Return the statistics stored in the constant input."
function deconstructconstantinput(constantinput::AbstractVector{UInt16})
    names = (
        :number,
        :screens,
        :mode,
        :fgbggfx,
        :sprites_mode,
    )
    return NamedTuple{names}(constantinput)
end

"""
    deconstructlevel(level::AbstractArray{UInt16, 3})

Return a `NamedTuple` containing
   - `tiles`,
   - `mainentrance`,
   - `midwayentrance`,
   - `sprites`,
   - `exits`,
   - `secondaryexits`,
   - `secondaryentrances`, and
   - `stats`
of the given level in the same format as when read and parsed from a corresponding
Lunar Magic dump file. This does not go for the `stats` field, though, which collects any
extra statistics or flags found.
"""
function deconstructlevel(level::AbstractDict{Symbol, <:AbstractArray{UInt16, 3}})
    tiles = deconstructtiles(level[:tiles])
    mainentrance, midwayentrance, stats = deconstructentrances(level[:entrances])
    sprites = deconstructsprites(level[:sprites])
    exits = deconstructexits(level[:exits], false)
    secondaryexits = deconstructexits(level[:secondaryexits], true)
    secondaryentrances = deconstructsecondaryentrances(level[:secondaryentrances],
                                                       stats.xymethod2)
    return (tiles=tiles, mainentrance=mainentrance, midwayentrance=midwayentrance,
            sprites=sprites, exits=exits, secondaryexits=secondaryexits,
            secondaryentrances=secondaryentrances, stats=stats)
end

"""
    deconstructlevel(level::AbstractMatrix{UInt16})

Return the raw byte data for the given tile matrix.
"""
deconstructlevel(level::AbstractMatrix{UInt16}) = torawdata(level)

"Return the relative x offset and screen of the given absolute x coordinate."
function makerelative(absx)
    # Remove one due to 1-based indexing.
    absx -= 1
    screen = UInt8(div(absx, LevelStatistics.screencols))
    x = UInt8(absx - screen * LevelStatistics.screencols)
    return x, screen
end

deconstructtiles(tiles) = torawdata(tiles[:, :, 1])

function deconstructentrances(entrances)
    mainabsy, mainx, mainscreen, mainabsx = deconstructentrance(entrances[:, :, 1])
    midabsy,  midx,  midscreen,  midabsx  = deconstructentrance(entrances[:, :, 2])

    # This does not influence the midway entrance directly.
    xymethod2 = !(checkbounds(Bool, ys, mainabsy) && checkbounds(Bool, xs, mainx))
    # This is unrelated to xymethod2 which only changes behaviour for the main entrance.
    separateentrance = mainx != midx || mainabsy != midabsy

    if xymethod2
        mainy = UInt16(mainabsy)
        mainx = UInt8(mainabsx)
    else
        reverse_ys = Dict{valtype(ys), keytype(ys)}(v => k - 1 for (k, v) in enumerate(ys))
        reverse_xs = Dict{valtype(xs), keytype(xs)}(v => k - 1 for (k, v) in enumerate(xs))

        mainy = UInt8(mainabsy)
        # These are not saved in the file so do not return them.
        horizontalsubscreenbit = mainy >= LevelStatistics.screenrowshoritop
        horizontalsubscreenbit && (mainy -= LevelStatistics.screenrowshoritop)
        yref = reverse_ys[mainy]
        xref = reverse_xs[mainx]
    end

    if separateentrance
        midy = UInt16(midabsy)
        midx = UInt8(midabsx)
    else
        @assert midx == mainx ("got different relative x offset for main and midway"
                               * "entrance even though they are not separate.")
        midy = 0x000
        midx = 0x0
    end

    if xymethod2
        mainentrance = (y=mainy, x=mainx, screen=mainscreen)
    else
        mainentrance = (y=yref,  x=xref,  screen=mainscreen)
    end
    midwayentrance = (y=midy, x=midx, screen=midscreen, separateentrance=separateentrance)
    stats = (xymethod2=xymethod2,)
    return mainentrance, midwayentrance, stats
end

function deconstructentrance(entrance)
    absy, absx = Tuple(findfirst(isequal(1), entrance))
    x, screen = makerelative(absx)
    # Remove the one for 1-based indexing
    return absy - 1, x, screen, absx - 1
end

function deconstructsprites(sprites)
    spritelist = NamedTuple[]
    for spritelayer in eachslice(sprites, dims=3)
        sprites = Tuple.(findall(!isequal(0), spritelayer))
        # TODO can pull id using defaultspritemapping
        # TODO could also forego this step in the LevelFormatReverter module
        for (absy, absx) in sprites
            y = UInt8(absy - 1)
            x, screen = makerelative(absx)
            id = UInt8(spritelayer[absy, absx])
            push!(spritelist, (y=y, x=x, screen=screen, id=id))
        end
    end
    return spritelist
end

function deconstructexits(exits, secondaryexits::Bool)
    exitlist = NamedTuple[]
    for exitlayer in eachslice(exits, dims=3)
        exits = Tuple.(findall(!isequal(0), exitlayer))
        for (absy, absx) in exits
            destination = UInt16(exitlayer[absy, absx])
            screen = makerelative(absx)[2]
            push!(exitlist, (destination=destination, screen=screen,
                             secondaryexit=secondaryexits))
        end
    end
    return exitlist
end

function deconstructsecondaryentrances(entrances, xymethod2::Bool)
    entrancelist = NamedTuple[]
    if xymethod2
        reverse_ys = Dict{valtype(ys), keytype(ys)}(v => k - 1 for (k, v) in enumerate(ys))
        reverse_xs = Dict{valtype(xs), keytype(xs)}(v => k - 1 for (k, v) in enumerate(xs))
    end

    for entrancelayer in eachslice(entrances, dims=3)
        entrances = Tuple.(findall(!isequal(0), entrancelayer))
        for (absy, absx) in entrances
            x, screen = makerelative(absx)
            number = UInt16(entrancelayer[absy, absx])
            if xymethod2
                y = UInt16(absy - 1)
                x = UInt8(absx - 1)
                push!(entrancelist, (y=y, x=x, screen=screen, number=number,
                                     xymethod2=xymethod2))
            else
                y = UInt8(absy - 1)
                # These are not saved in the file so do not return them.
                horizontalsubscreenbit = y >= LevelStatistics.screenrowshoritop
                horizontalsubscreenbit && (y -= LevelStatistics.screenrowshoritop)
                yref = reverse_ys[y]
                xref = reverse_xs[x]
                push!(entrancelist, (y=yref, x=xref, screen=screen, number=number,
                                     xymethod2=xymethod2))
            end
        end
    end
    return entrancelist
end

end # module

