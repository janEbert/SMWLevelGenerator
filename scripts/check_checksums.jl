#!/usr/bin/env julia

"""
Compute level checksums and compare them to checksums of the original game to remove
duplicate levels.
"""
module CheckChecksums

using CRC32c

using JLD


const leveldir = abspath(joinpath(@__DIR__, "..", "levels"))
const originalleveldir = joinpath(leveldir, "Super Mario World (U) [!]")
const outfile = abspath(joinpath(@__DIR__, "..", "stats", "level_checksums.jld"))


function makecsdict(origleveldir::AbstractString=originalleveldir,
                    extension::AbstractString="map")
    dict = Dict{String,UInt32}()
    levels = readdir(origleveldir)
    extension[1] == '.' || (extension = '.' * extension)
    filteredlevels = filter(l -> endswith(l, extension), levels)
    map(filteredlevels) do f
        dict[f] = open(crc32c, joinpath(origleveldir, f))
    end
    return dict
end

function checkchecksums(checksumdict::AbstractDict{String, UInt32},
                        leveldir::AbstractString=leveldir,
                        extension::AbstractString="map")
    notequal = Dict{String, UInt32}()
    equal = Dict{String, UInt32}()
    nonexistent = Dict{String, UInt32}()
    extension[1] == '.' || (extension = '.' * extension)
    for (root, dirs, files) in walkdir(leveldir)
        for file in filter(l -> endswith(l, extension), files)
            fullpath = joinpath(root, file)
            checksum = open(crc32c, fullpath)
            if !haskey(checksumdict, file)
                nonexistent[fullpath] = checksum
                @warn "$fullpath not in dict"
            end
            if checksum != checksumdict[file]
                notequal[fullpath] = checksum
            else
                equal[fullpath] = checksum
            end
        end
    end
    return notequal, equal, nonexistent
end

function removelevel(path)
    extind = findlast(isequal('.'), path)
    basepart = path[1:extind]
    foreach(ext -> rm(basepart * ext), ("bg", "ent", "ent2", "exits", "map", "sp"))
end

function removeduplicates(equalchecksums::AbstractDict{String, UInt32},
                          extension::AbstractString="map")
    # The name of the last directory for filtering.
    originallevels = splitpath(originalleveldir)[end]
    extension[1] == '.' || (extension = '.' * extension)
    # We do not remove levels 000 and 100 as they are standard levels for exits.
    foreach(removelevel, filter(l -> !occursin(originallevels, l)
                                && !endswith(l, "Level000" * extension)
                                && !endswith(l, "Level100" * extension),
                                keys(equalchecksums)))
end

function removetests(csdict::AbstractDict{String, UInt32},
                     origleveldir::AbstractString=originalleveldir)
    testcs = csdict["Level019.map"]  # First occurrence of a test level without warnings.
    for (level, cs) in csdict
        if cs == testcs
            removelevel(joinpath(origleveldir, level))
        end
    end
end

function main(leveldir::AbstractString=leveldir,
              originalleveldir::AbstractString=originalleveldir,
              outfile::AbstractString=outfile)
    checksumdict = makecsdict(originalleveldir)
    notequal, equal, nonexistent = checkchecksums(checksumdict, leveldir)
    if isempty(nonexistent)
        save(outfile, "original", checksumdict, "notequal", notequal, "equal", equal)
    else
        save(outfile, "original", checksumdict, "notequal", notequal, "equal", equal,
             "nonexistent", nonexistent)
    end

    local answer
    while true
        println("Delete duplicate levels? (y/n)")
        answer = lowercase(readline())
        answer ∉ ("y", "n") && break
        println("Please answer only with 'y' or 'n'.")
    end
    answer == "y" && (removeduplicates(equal); println("Done."))

    while true
        println("Delete test levels in original level directory? (y/n)")
        answer = lowercase(readline())
        answer ∉ ("y", "n") && break
        println("Please answer only with 'y' or 'n'.")
    end
    answer == "y" && (removetests(checksumdict, originalleveldir); println("Done."))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 3 || length(ARGS) == 1 && ARGS[1] == "-h"
        print("""
              Usage: $PROGRAM_FILE [<leveldir>] [<originalleveldir>] [<outfile>]

                 leveldir:         Directory in which the levels reside
                                   (default: ../levels (relative to this file's directory)).
                 originalleveldir: Directory in which the original levels reside
                                   (default: <leveldir>/Super Mario World (U) [!]).
                 outfile:          File to save checksums to
                                   (default: ../stats/level_checksums.jld
                                   (relative to this file's directory)).
              """)
        exit(length(ARGS) > 3 ? 1 : 0)
    end
    main(ARGS...)
end

end # module

