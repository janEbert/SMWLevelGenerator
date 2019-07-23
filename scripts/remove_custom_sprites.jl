#!/usr/bin/env julia

"""
Remove levels that contain non-standard sprite values.

It is possible that false positives remain.
"""
module RemoveCustomSprites

include(joinpath("..", "src", "sprites.jl"))


function removenonvanillaspritelevels(nonvanilla::Dict{UInt8, Dict{String, Vector{String}}}
                                      = Sprites.findnonvanillasprites())
    for dirdict in values(nonvanilla)
        for (dir, levels) in dirdict
            for level in levels
                baselevel = level[1:findlast('.', level)]
                basepath = joinpath(dir, baselevel)

                rm(joinpath(dir, level))
                for ext in ("bg", "ent", "ent2", "exits", "map")
                    rm(basepath * ext)
                end
            end
        end
    end
end

function main()
    local answer
    while true
        println("Really delete all levels with custom sprites? (y/n)")
        answer = lowercase(readline())
        answer ∉ ("y", "n") && break
        println("Please answer only with 'y' or 'n'.")
    end
    answer == "y" && (removenonvanillaspritelevels(); println("Done."))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module

