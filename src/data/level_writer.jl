module LevelWriter

export writeback, writemap

function find_free_name(name, prefix="", suffix=".map")
    !ispath(prefix * name * suffix) && return name
    i = 1
    new_name = name * '_' * string(i)
    while ispath(prefix * new_name * suffix)
        i += 1
        new_name = name * '_' * string(i)
    end
    return new_name
end

function writeback(leveldata::NamedTuple, constantinputdata::NamedTuple,
                   name::AbstractString="Level" * lpad(string(
                           constantinputdata.number, base=16), 3, '0'))
    name = find_free_name(name)
    totalbytes = writemap(leveldata.tiles, name)
    totalbytes += writeent(leveldata.mainentrance, leveldata.midwayentrance,
                           leveldata.stats, constantinputdata, name)
    totalbytes += writesp(leveldata.sprites, constantinputdata, name)
    totalbytes += writeexits(leveldata.exits, leveldata.secondaryexits, name)
    totalbytes += writeent2(leveldata.secondaryentrances, constantinputdata.number, name)
    println("Generated ", name, ".{map, ent, sp, exits, ent2}")
    return totalbytes
end

function writeback(leveldata::AbstractVector{UInt8}, constantinputdata::NamedTuple,
                   remove_sprites=true, name::AbstractString="Level" * lpad(string(
                           constantinputdata.number, base=16), 3, '0'))
    writeback(leveldata, remove_sprites, name)
end

function writeback(leveldata::AbstractVector{UInt8}, remove_sprites=true,
                   name::AbstractString="Level105")
    name = find_free_name(name)
    totalbytes = writemap(leveldata, name)
    print("Generated ", name, ".map")
    if remove_sprites
        write(name * ".sp", UInt8[0x00, 0xff])
        print(" and created empty sprite file ", name, ".sp")
    end
    println()
    return totalbytes
end

function writemap(tiledata::AbstractVector{UInt8}, name::AbstractString)
    write(name * ".map", tiledata)
    # TODO write padding data as well
end

function writeent(mainentrancedata::NamedTuple, midwayentrancedata::NamedTuple,
                  stats::NamedTuple, constantinputdata::NamedTuple, name::AbstractString)
    data = createlevelinfo(mainentrancedata, midwayentrancedata, stats, constantinputdata)
    write(name * ".ent", data)
end

function createlevelinfo(mainentrancedata::NamedTuple, midwayentrancedata::NamedTuple,
                          stats::NamedTuple, constantinputdata::NamedTuple)
    data = UInt8[]
    numberfirst, numbersecond = tolittleendian(constantinputdata.number)
    secondaryheader = createsecondaryheader(mainentrancedata, midwayentrancedata, stats,
                                            constantinputdata)
    @assert length(secondaryheader) == 7 "unexpected secondary header size"
    midwayentrance = createmidwayentrancebytes(midwayentrancedata, constantinputdata)
    header = createheader(constantinputdata)
    @assert length(header) == 5 "unexpected header size"

    push!(data, numberfirst)
    push!(data, numbersecond)
    append!(data, secondaryheader[1:5])
    push!(data, 0, 0)
    append!(data, midwayentrance)
    push!(data, 0)
    append!(data, secondaryheader[6:7])
    push!(data, 0x0)  # lmexpandedformat
    append!(data, zeros(UInt8, 0xf))  # Zeros until 0x20
    append!(data, header)
    append!(data, zeros(UInt8, 0x1b))  # Zeros until 0x40
end

function createsecondaryheader(mainentrancedata::NamedTuple, midwayentrancedata::NamedTuple,
                               stats::NamedTuple, constantinputdata::NamedTuple)
    # Default values taken from level 105 in the original game
    header = BitVector()
    ybits = tobits(mainentrancedata.y)
    xbits = tobits(mainentrancedata.x)
    midwayentrancescreenbits = tobits(midwayentrancedata.screen)
    screenbits = tobits(mainentrancedata.screen)
    entranceactionbits = tobits(constantinputdata.mainentranceaction)

    append!(header, (0, 1, 0, 1))  # layer2scrollbits
    append!(header, ybits[end - 3:end])
    append!(header, (0, 0))  # layer3settingbits
    append!(header, entranceactionbits[end - 2:end])
    append!(header, xbits[end - 2:end])
    append!(header, midwayentrancescreenbits[end - 3:end])
    append!(header, (1, 0))  # fgpositionbits
    append!(header, (1, 0))  # bgpositionbits
    push!(header, 0)  # enableyoshiintrobits
    push!(header, 0)  # unusedverticalpositionbits
    push!(header, 0)  # verticalpositionbits
    append!(header, screenbits[end - 4:end])
    push!(header, 0)  # slipperybits
    push!(header, 0)  # waterbits
    push!(header, stats.xymethod2)
    append!(header, xbits[end - 4:end - 3])
    push!(header, 0)  # smartspawnbits
    append!(header, (0, 0))  # spawnrangebits
    push!(header, 0)  # relativebgbits
    push!(header, 0)  # fgbgoffsetheadbits
    append!(header, ybits[end - 9:end - 4])
    push!(header, 0)  # relativefgbgbits
    push!(header, 0)  # faceleftbits
    push!(header, 0)  # unused
    append!(header, (0, 0, 0, 0, 0))  # bgpositionbits

    @assert length(header) == 56 "unexpected secondary header bitstring size"
    return tobytes(header)
end

function createmidwayentrancebytes(midwayentrancedata::NamedTuple,
                                   constantinputdata::NamedTuple)
    midway = BitVector()
    screenbits = tobits(midwayentrancedata.screen)
    xbits = tobits(midwayentrancedata.x)
    ybits = tobits(midwayentrancedata.y)
    entranceactionbits = tobits(constantinputdata.midwayentranceaction)

    push!(midway, 0)  # slipperybits
    push!(midway, 0)  # waterbits
    push!(midway, midwayentrancedata.separateentrance)
    push!(midway, screenbits[end - 4])
    push!(midway, xbits[end - 4])
    append!(midway, entranceactionbits[end - 2:end])
    append!(midway, ybits[end - 3:end])
    append!(midway, xbits[end - 3:end])
    push!(midway, 0)  # relativefgbgbits
    push!(midway, 0)  # faceleftbits
    push!(midway, 0)  # redirectbits
    push!(midway, 1)  # unused
    append!(midway, (1, 0))  # fgpositionbits
    append!(midway, (1, 0))  # bgpositionbits
    push!(midway, 0)  # unused
    push!(midway, 0)  # fgbgoffsetheadbits
    append!(midway, ybits[end - 9:end - 4])

    @assert length(midway) == 32 "unexpected midway bitstring size"
    return tobytes(midway)
end

function createheader(constantinputdata::NamedTuple)
    header = BitVector()
    screensbits = tobits(convert(UInt8, constantinputdata.screens - 1))
    modebits    = tobits(convert(UInt8, constantinputdata.mode))
    fgbggfxbits = tobits(convert(UInt8, constantinputdata.fgbggfx))

    append!(header, (0, 0, 1))  # bgbits
    append!(header, screensbits[end - 4:end])
    append!(header, (0, 1, 0))  # bgcolorbits
    append!(header, modebits[end - 4:end])
    push!(header, 0)  # layer3prioritybits
    append!(header, (0, 0, 0))  # musicbits
    append!(header, (1, 0, 0, 0))  # spritegfxbits
    append!(header, (1, 0))  # timerbits
    append!(header, (0, 0, 0))  # spritebits
    append!(header, (0, 0, 0))  # fgbits
    append!(header, (0, 0))  # itemmemorybits
    append!(header, (1, 0))  # verticalscrollbits
    append!(header, fgbggfxbits[end - 3:end])

    @assert length(header) == 40 "unexpected header bitstring size"
    return tobytes(header)
end

function writesp(spritedata::AbstractVector{<:NamedTuple}, constantinputdata::NamedTuple,
                 name::AbstractString)
    data = UInt8[]
    header = createspriteheader(constantinputdata)
    push!(data, header)

    for sprite in spritedata
        spritebytes = createspitebytes(sprite)
        append!(data, spritebytes)
    end
    push!(data, 0xff)
    # Below was not used yet when dumping but is more "modern".
    # Also change the below assertion's subtraction if this is used.
    # append!(data, (0xff, 0xfe))
    @assert (length(data) - 2) % 3 == 0 "unexpected amount of sprite bytes"
    write(name * ".sp", data)
end

function createspriteheader(constantinputdata::NamedTuple)
    header = BitVector()
    modebits = tobits(constantinputdata.sprites_mode)

    push!(header, 0)  # buoyancybits
    push!(header, 0)  # disablelayer2buoyancybits
    push!(header, 0)  # newsystembits
    append!(header, modebits[end - 4:end])

    @assert length(header) == 8 "unexpected sprite header bitstring size"
    return parse(UInt8, filter(!isequal(' '), bitstring(header)), base=2)
end

function createspritebytes(sprite::NamedTuple)
    bytes = BitVector()
    ybits = tobits(sprite.y)
    xbits = tobits(sprite.x)
    screenbits = tobits(sprite.screen)
    idbits = tobits(convert(UInt8, sprite.id))

    append!(bytes, ybits[end - 3:end])
    append!(bytes, (0, 0))  # extrabits
    push!(bytes, screenbits[end - 4])
    push!(bytes, ybits[end - 4])
    append!(bytes, xbits[end - 3:end])
    push!(bytes, screenbits[end - 3:end])
    append!(bytes, idbits)

    @assert length(bytes) == 24 "unexpected sprite bitstring size"
    return tobytes(bytes)
end

function writeexits(exitdata::AbstractVector{<:NamedTuple},
                    secondaryexitdata::AbstractVector{<:NamedTuple}, name::AbstractString)
    data = UInt8[]
    sorted = sort(append!(exitdata, secondaryexitdata), by=exit -> exit.screen)
    i = 1

    for exit in sorted
        append!(data, zeros(UInt8, (screen - i) * 4))
        append!(data, createexit(exit))
    end

    append!(data, zeros(UInt8, 128 + 1 - i))

    @assert length(data) == 128 "unexpected amount of exit bytes"
    write(name * ".exits", data)
end

function createexit(exitdata::NamedTuple)
    exit = BitVector()
    destinationbits = tobits(exitdata.destination)

    append!(exit, destinationbits[end - 7:end])
    append!(exit, destinationbits[end - 12:end - 9])
    push!(exit, 0)  # midwaywaterbits
    push!(exit, 1)  # lmmodifiedbits
    push!(exit, exitdata.secondaryexit)
    push!(exit, destinationbits[end - 8])

    # Set `isexit` byte
    append!(exit, zeros(7))
    push!(exit, 1)
    append!(exit, zeros(8))  # unused byte

    @assert length(exit) == 32 "unexpected exit bitstring size"
    return tobytes(exit)
end

function writeent2(entrancedata::AbstractVector{<:NamedTuple}, number::UInt16,
                   name::AbstractString)
    isempty(entrancedata) && return write(name * ".ent2", UInt8[])
    data = reduce(vcat, createsecondaryentrance.(entrancedata, number))
    @assert length(data) % 8 == 0 "unexpected amount of entrance bytes"
    write(name * ".ent2", data)
end

function createsecondaryentrance(entrancedata::NamedTuple, number::UInt16)
    entrance = BitVector()
    numberfirst, numbersecond = map(tobits, tolittleendian(entrancedata.number))
    ybits = tobits(entrancedata.y)
    xbits = tobits(entrancedata.x)
    screenbits = tobits(entrancedata.screen)

    append!(entrance, numberfirst)
    append!(entrance, numbersecond)
    append!(entrance, (1, 0))  # bgpositionbits
    append!(entrance, (1, 0))  # fgpositionbits
    append!(entrance, ybits[end - 3:end])
    append!(entrance, xbits[end - 2:end])
    append!(entrance, screenbits[end - 4:end])
    push!(entrance, 0)  # slipperybits
    push!(entrance, entrancedata.xymethod2)
    append!(entrance, xbits[end - 4:end - 3])
    push!(entrance, number >= 0x100)
    append!(entrance, (1, 1, 0))  # entranceactionbits
    push!(entrance, 0)  # exittooverworldbits
    push!(entrance, 0)  # fgbgoffsetheadbits
    append!(entrance, ybits[end - 9:end - 4])
    push!(entrance, 0)  # relativefgbgbits
    push!(entrance, 0)  # faceleftbits
    push!(entrance, 0)  # waterbits
    append!(entrance, (0, 0, 0, 0, 0))  # unused
    append!(entrance, zeros(8))  # unused byte

    @assert length(entrance) == 64 "unexpected secondary entrance bitstring size"
    return tobytes(entrance)
end

function tobytes(values::BitVector)
    bitstr = filter(!isequal(' '), bitstring(values))
    map(x -> parse(UInt8, join(x), base=2), Iterators.partition(bitstr, 8))
end

tobits(value) = parse.(Bool, collect(bitstring(value)))

tolittleendian(value::UInt8) = value

function tolittleendian(value::UInt16)
    second = convert(UInt8, div(value, 0x100))
    first = convert(UInt8, value - second * 0x100)
    return first, second
end

end # module

