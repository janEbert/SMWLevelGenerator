module InputStatistics

import ..LevelStatistics
using ..Sprites: uniquesprites
using ..SecondaryLevelStats: maximumexits, maximumsecondaryexits, maximumentrances

export screenrows, screencols, layers3d, outputsizeof
export constantinputsize, inputsize1d, inputsize2d, inputsize3dtiles, inputsize3d
export imgsize1d, imgsize2d, imgsize3dtiles, imgsize3d

# We only process horizontal levels.
const screenrows = LevelStatistics.screenrowshori
const screencols = LevelStatistics.screencols

const layers3d = begin
    LevelStatistics.uniquevanillatiles  # tiles
    + 2                                 # main entrance and midway entrance
    + uniquesprites                     # sprites
    + maximumexits                      # screen exits
    + maximumsecondaryexits             # secondary exits
    + maximumentrances                  # secondary entrances
end

const constantinputsize = 6

# Sequence prediction
const inputsize1d = 1 + constantinputsize
const inputsize2d = screenrows + constantinputsize
const inputsize3dtiles = screenrows * LevelStatistics.uniquevanillatiles + constantinputsize
"""
This number is hardcoded. Change first summand to `size(LevelFormatter.to3d(0x105), 3)` if
this is wrong.
"""
const inputsize3d = screenrows * layers3d + constantinputsize

outputsizeof(inputsize) = inputsize - constantinputsize + 1

# GAN (first screen generation)
imgsize1d = (screencols, 1)
imgsize2d = (screencols, screenrows, 1)
imgsize3dtiles = (screencols, screenrows, LevelStatistics.uniquevanillatiles)
"""
This number is hardcoded. Change third element to `size(LevelFormatter.to3d(0x105), 3)` if
this is wrong.
"""
imgsize3d = (screencols, screenrows, layers3d)

end # module

