"Process a level's map of tiles to matrix representation and back."
module Tiles

using ..LevelStatistics

"""
    getmap(file, stats::LevelStats)
    getmap(file[, entfile::AbstractString]; force=false)

Return the contents of the given IO or file as a `Matrix{UInt16}`.
If `entfile` is not given, the extension of file is replaced with "ent" to get `entfile`.
"""
function getmap(io::IO, stats::LevelStats)::Matrix{UInt16}
    rawdata = read(io, size(stats))
    tomatrix(rawdata, stats)
end

function getmap(io::IO, entfile::AbstractString; force::Bool=false)::Matrix{UInt16}
    stats = getstats(entfile, force=force)
    getmap(io, stats)
end

function getmap(file::AbstractString, stats::LevelStats)::Matrix{UInt16}
    open(x -> getmap(x, stats), file)
end

function getmap(file::AbstractString,
                 entfile::AbstractString=file[1:findlast(isequal('.'), file)] * "ent";
                 force::Bool=false)::Matrix{UInt16}
    stats = getstats(entfile, force=force)
    getmap(file, stats)
end

# TODO begins at 0x3fe0 for level 1CE; 0x3800 for 0E7.
# 01A at 0x4380
function getlayer2map()
    
end

# TODO layer 2 levels (e.g. 0x1CE do not get parsed correctly – layer 2 is missing)
# in LM: layer 2: interact in header
"Convert the given raw byte data to a matrix."
function tomatrix(rawdata::AbstractVector{UInt8}, stats::LevelStats)::Matrix{UInt16}
    data = fromlittleendian(rawdata)
    if LevelStatistics.isvertical(stats)
        # Cuboid of subscreens (`cuboid[:, end, 2]` is the last (bottom) row of the second
        # subscreen of the first screen)
        cuboid = reshape(data, (screencols, screenrows(stats), stats.screens * 0x2))
        # Inner loop stitches the subscreens together into one screen.
        # Outer loop puts them "on top" of each other (horizontally).
        matrix = reduce(hcat, (reduce(vcat, (view(cuboid, :, :, i) for i in j:j+1))
                               for j in axes(cuboid, 3)[1:2:end]))
        # The level is transposed now. We want it that way for cache efficiency.
    else
        # Cuboid of screens (`cuboid[:, end, 1]` is the last (bottom) row of the
        # first screen).
        cuboid = reshape(data, (screencols, screenrows(stats), stats.screens))
        # Level as a 2D matrix, but transposed.
        matrix = reduce(vcat, eachslice(cuboid, dims=3))
        # Level the way a human would look at it.
        transposed = permutedims(matrix)
    end
end

"Convert the given matrix to raw byte data."
function torawdata(matrix::AbstractMatrix{UInt16})::Vector{UInt8}
    if isvertical(matrix)
        # Level is vertical

        # Put each subscreen of each screen into a new layer of the cube.
        # Inner loop splits `matrix` into individual screens.
        # Outer loop splits each screen into its two subscreens.
        # The result is concatenated in the third dimension.
        cuboid = reduce((a, b) -> cat(a, b, dims=3),
                        (view(screen, i:i + screencols - 1, :)
                         for screen in (view(matrix, :, j:j + screenrowsvert - 1)
                                        for j in axes(matrix, 2)[1:screenrowsvert:end])
                         for i in (1, screencols + 1)))
    elseif size(matrix, 1) == screenrowshori
        # Level is horizontal

        transposed = permutedims(matrix)
        # Put each screen into a new layer of the cube
        cuboid = reduce((a, b) -> cat(a, b, dims=3),
                        (view(transposed, i:i + screencols - 1, :)
                         for i in axes(transposed, 1)[1:screencols:end]))
    else
        throw("Level has non-vanilla dimensions $(size(matrix))! "
              * "Cannot determine level orientation.")
    end
    data = reshape(cuboid, :)
    tolittleendian(data)
end

"Return wether the given matrix contains a vertical level (determined heuristically)."
function isvertical(matrix::AbstractMatrix)
    size(matrix, 1) == screencols * 2 && size(matrix, 2) != screenrowshori
end

"""
    fromlittleendian(bytes::AbstractArray{UInt8})

Convert the given `AbstractArray{UInt8}` containing 16-bit values in little-endian order to
the appropriate values in a `Vector{UInt16}`.

# Examples
```jldoctest
julia> fromlittleendian([0x25 0x05; 0x00 0x01])
2-element Array{UInt16,1}:
 0x0025
 0x0105
```
"""
function fromlittleendian(bytes::AbstractArray{UInt8})::Vector{UInt16}
    map(((x, y),) -> y * 0x100 + x, Iterators.partition(bytes, 2))
end

"""
    tolittleendian(values::AbstractArray{UInt16})
Convert the given `AbstractArray{UInt16}` to a `Vector{UInt8}` containing the same values in
little-endian order.

# Examples
```jldoctest
julia> tolittleendian([0x0025 0x0105])
4-element Array{UInt8,1}:
 0x25
 0x00
 0x05
 0x01

```
"""
function tolittleendian(values::AbstractArray{UInt16})::Vector{UInt8}
    out = Vector{UInt8}(undef, length(values) * 2)
    for (i, value) in zip(1:2:length(out), values)
        second = div(value, 0x100)
        first = value - second * 0x100
        @inbounds out[i] = first
        @inbounds out[i + 1] = second
    end
    return out
end


# using ..SecondaryLevelStats: parsemap16, applymap16!

"Test the formatting functions on the given level."
function testformatting_level(file::AbstractString
        = joinpath(originalleveldir, "Level105.map"), number::Number=0, applymap16=false)
    # `file` without extension
    level = file[1:findlast(isequal('.'), file)]
    stats = getstats(level * "ent")
    rawdata = read(level * "map", size(stats))
    matrix = tomatrix(rawdata, stats)
    # applymap16 && applymap16!(matrix, parsemap16(joinpath(dirname(level), "map16fgG.bin")))
    local testrawdata
    try
        testrawdata = torawdata(matrix)
        @assert testrawdata == rawdata ("conversion failed for level $number: $(level)map.\n"
                                        * "$stats\n")
        if !all(testrawdata .=== rawdata)
            @warn """test data does not point to the same values as the original data.
                     In level $number: $(level)map."""
        end
    catch ErrorException
        @error "for $level"
    end
end

"Test the formatting functions on all levels in the given directory."
function testformatting_dir(dir::AbstractString=originalleveldir)
    applymap16 = dir == originalleveldir
    levels = filter(x -> endswith(x, ".map"), readdir(dir))
    testlevels = map(x -> joinpath(dir, x), levels)
    for (i, level) in enumerate(testlevels)
        testformatting_level(level, i, applymap16)
    end
end

"Test the formatting data on all levels in the given directory tree."
function testformatting_tree(dir::AbstractString=leveldir)
    for (root, dirs, _) in walkdir(dir)
        for subdir in dirs
            path = joinpath(root, subdir, "")
            println("Testing $path")
            testformatting_dir(path)
        end
    end
end

end # module

