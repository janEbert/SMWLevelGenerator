module InputStatistics

import ..LevelStatistics

export screenrows, usegpu, togpu
export constantinputsize, inputsize1d, inputsize2d, inputsize3dtiles, inputsize3d

# We only process horizontal levels.
const screenrows = LevelStatistics.screenrowshoritop + LevelStatistics.screenrowshoribottom

const constantinputsize = 6

const inputsize1d = 1 + constantinputsize
const inputsize2d = screenrows + constantinputsize
const inputsize3dtiles = screenrows * LevelStatistics.uniquevanillatiles + constantinputsize
"""
This number is hardcoded. Change to `size(LevelFormatter.to3d(0x105), 3)` if this is wrong.
"""
const inputsize3d = screenrows * LevelStatistics.uniquevanillatiles + constantinputsize

"Set to `false` to actively ignore the GPU even if it is available."
usegpu = false

if usegpu
    togpu = Flux.gpu
else
    togpu = identity
end

"Map `x` to the GPU if `usegpu` is false, otherwise return `x`."
togpu

end # module

