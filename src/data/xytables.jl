"""
Global x and y tables used if an entrance does not use `xymethod2`.
The values are instead referenced (from these tables).
"""
module XYTables

export ys, horizontalsubscreenbits, xs, verticalsubscreenbits

const statsdir = abspath(joinpath(@__DIR__, "..", "..", "stats"))
const xytablefile = joinpath(statsdir, "xytable.txt")

"""
Return four arrays containing the y, horizontal subscreen bit, x and vertical subscreen bit
values found in the table from the given file.

These values are the same in **all** ROMs in the dataset.
"""
function readxytable(path::AbstractString)
    # Horizontal subscreen bits are taken from the table following the Y value table.
    # Vertical subscreen bits are taken from the table following the X value table.
    # The order in the file is: y, horizontal subscreen bits, x, vertical subscreen bits.
    ys, horizontalsubscreenbits, xs, verticalsubscreenbits = open(path) do file
        ys = read(file, 16)
        horizontalsubscreenbits = read(file, 16)
        xs = read(file, 8)
        verticalsubscreenbits = read(file, 8)
        return ys, horizontalsubscreenbits, xs, verticalsubscreenbits
    end
    return (ys=ys, horizontalsubscreenbits=horizontalsubscreenbits, xs=xs,
            verticalsubscreenbits=verticalsubscreenbits)
end

const ys, horizontalsubscreenbits, xs, verticalsubscreenbits = readxytable(xytablefile)

"A lookup table containing y values for secondary entrances if `xymethod2` is not set."
ys

"A lookup table containing x values for secondary entrances if `xymethod2` is not set."
xs

"""
A lookup table discerning whether the secondary entrance lies in the second
horizontal subscreen.
Only used if `xymethod2` is not set.
"""
horizontalsubscreenbits

"""
A lookup table discerning whether the secondary entrance lies in the second
vertical subscreen.
Only used if `xymethod2` is not set.
"""
verticalsubscreenbits

end # module

