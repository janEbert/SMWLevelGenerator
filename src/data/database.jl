module Database

using SparseArrays

using CSV
using DataFrames
using JuliaDB

using ..LevelStatistics
using ..LevelBuilder
using ..LevelFormatter
using ..DataCompressor

export generatedb, generatecsv

abstract type AbstractCompression end
struct NoCompression <: AbstractCompression end
struct SparseCompression <: AbstractCompression end
struct SparseIndexCompression <: AbstractCompression end
struct HardCompression <: AbstractCompression end

makecompressionfunction(::NoCompression) = identity
makecompressionfunction(::SparseCompression) = sparse
makecompressionfunction(::SparseIndexCompression) = x -> sparse(x) |> compressindices
makecompressionfunction(::HardCompression) = x -> sparse(x, dims=3) |> compressdata

compressionmap = Dict{Int, AbstractCompression}(
    0 => NoCompression(),
    1 => SparseCompression(),
    2 => SparseIndexCompression(),
    3 => HardCompression(),
)

# TODO Save _all_ level metadata (cheap but useful). Need to also change all parsers and
#      structs.
# TODO logging to file as well by default

"""
    generatedb(dest::AbstractString, rootdir::AbstractString=leveldir;
               formatfunction::Function=to3d, flags::Union{AbstractString, Nothing}=nothing,
               compression=1, verbose=true)

Generate a database at the given destination path containing data for all levels found under
the given directory.
Can also pass the function used to build up levels. It is passed one argument (the `Level`).
By default, use [`to3d`](@ref).
The flags to use in the [`buildlevel`](@ref) function can  be given in the `flags` argument.
If `flags` is `nothing`, the default flags for the given `formatfunction` will be used.
`compression` controls how far the data will be compressed, trading performance for space
efficiency. This is mostly relevant for highly-dimensional data.

As the method of [`to3d`](@ref) that takes in `Level`s does not have a `flags` argument,
flags have to be supplied independently.
"""
function generatedb(dest::AbstractString, args...; kwargs...)
    # TODO remove this/create flag to activate
    # Base.CoreLogging.disable_logging(Base.CoreLogging.LogLevel(1000))
    db = generatedb_inline(args...; kwargs...)
    println("Saving...")
    save(db, dest)
    println("Done.")
end

# TODO accept flags or Symbol for dimensionality; then get default flags for symbol
"""
    generatedb_inline(dest::AbstractString, rootdir::AbstractString=leveldir;
                      formatfunction::Function=to3d,
                      flags::Union{AbstractString, Nothing}=nothing, compression=1)

Return a database containing data for all levels found under the given directory.
Can also pass the function used to build up levels. It is passed one argument (the `Level`).
By default, use [`to3d`](@ref).
The flags to use in the [`buildlevel`](@ref) function can  be given in the `flags` argument.
If `flags` is `nothing`, the default flags for the given `formatfunction` will be used.

Like [`generatedb`](@ref) but without saving to a file.
"""
function generatedb_inline(args...; kwargs...)
    lists, datalist = generatedata(args...; kwargs...)
    println("Building database...")
    db = table(lists..., datalist; names=vcat(collect(propertynames(lists)), :data),
               pkey=:id, presorted=true, copy=false)
    println("Done.")
    return db
end


# TODO only compress if desired
"""
    generatedata(rootdir::AbstractString=leveldir; formatfunction::Function=to3d,
                 flags::Union{AbstractString, Nothing}=nothing, compression=1, verbose=true)

Return a `NamedTuple` and `Vector` containing the metadata and object data respectively
for all levels found under the given directory.
Can also pass the function used to build up levels. It is passed one argument (the `Level`).
By default, use [`to3d`](@ref).
The flags to use in the [`buildlevel`](@ref) function can  be given in the `flags` argument.
If `flags` is `nothing`, the default flags for the given `formatfunction` will be used.

`compression` controls the amount of compression applied to the data.
It can range from 0 to 3. The different values affect the behaviour in the following way:
   0: no compression
   1: convert data to sparse arrays
   2: compress sparse arrays further by adjusting their index type
   3: (only for 3D data) compress 3D sparse arrays by removing empty ones.
      This will greatly improve space efficiency but performance during data iteration
      will suffer; may not matter if using enough data iterator threads.

`verbose` controls whether progress messages are printed.
"""
function generatedata(rootdir::AbstractString=leveldir; formatfunction::Function=to3d,
                      flags::Union{AbstractString, Symbol, Nothing}=nothing,
                      compression::Union{Integer, AbstractCompression}=1,
                      verbose::Bool=true)
    i = one(UInt)
    totalsize = zero(UInt)
    funcsymbol = nameof(formatfunction)
    if isnothing(flags)
        funcdim = Symbol(String(funcsymbol)[end - 1:end])
        @assert haskey(dimensionality_defaultflags, funcdim) ("must supply flags with "
                                                              * "custom `formatfunction`")
        flags = dimensionality_defaultflags[funcdim]
    elseif flags isa Symbol
        flags = dimensionality_defaultflags[flags]
    end
    if funcsymbol === :to1d && compression > 0
        @warn ("if using the 1-dimensional `formatfunction`, it makes sense to set "
               * "`compression` to `0` as it is more space efficient.")
    end

    # Initialize compression function
    if compression isa Integer
        compression = get(compressionmap, Int(compression)) do
            throw(ArgumentError("`compression` must be between (including) 0 and 3"))
        end
    end
    compressionfunction = makecompressionfunction(compression)

    # Level data saved separately due to (yet) unknown type
    lists = (
        # DB primary key
        id = UInt[],

        # LevelStats
        hack             = String[],
        number           = UInt16[],
        screens          = UInt8[],
        mode             = UInt8[],
        fgbggfx          = UInt8[],
        lmexpandedformat = UInt8[],

        # Sprite header
        # If this first entry is `false`, the rest should be ignored.
        # We do not use `missing` because this is _assumed_ to be faster.
        hasspriteheader               = Bool[],
        sprites_buoyancy              = Bool[],
        sprites_disablelayer2buoyancy = Bool[],
        sprites_newsystem             = Bool[],
        sprites_mode                  = UInt8[],
    )

    defined_datalist = false
    local datalist

    starttime = time()
    println("Generating data...")
    # TODO parallelize
    for dir in filter(isdir, map(d -> joinpath(rootdir, d), readdir(rootdir)))
        # We choose an arbitrary extension to filter by just so we don't read in the same
        # level multiple times.
        for file in map(f -> joinpath(dir, f),
                        filter(f -> endswith(f, ".map"), readdir(dir)))
            if isnothing(flags)
                level = buildlevel(file, verbose=false)
            else
                level = buildlevel(file, flags, verbose=false)
            end
            data = formatfunction(level) |> compressionfunction
            totalsize += Base.summarysize(data)

            if !defined_datalist
                datalist = typeof(data)[]
                defined_datalist = true
            end

            addentry!(lists, datalist, i, level, data)
            printprogress(i, totalsize, verbose)
            i += 1
        end
    end
    defined_datalist || @warn "the root directory contained no levels."
    println("Done after $(round(Int, time() - starttime)) seconds.")
    @assert length(lists.id) == i - 1 ("some entries are missing. Amount of entries: "
                                       * "$(length(lists.id))/$(i - 1).")
    return lists, datalist
end

"Add an entry to the given lists, using the supplied data."
function addentry!(lists::NamedTuple, datalist::AbstractVector,
                   i::UInt, level::Level, data::AbstractArray)
    push!(lists.id, i)

    stats = level.stats
    push!(lists.hack, hack(stats))
    push!(lists.number, stats.number)
    push!(lists.screens, stats.screens)
    push!(lists.mode, stats.mode)
    push!(lists.fgbggfx, stats.fgbggfx)
    push!(lists.lmexpandedformat, stats.lmexpandedformat)

    spriteheader = level.spriteheader
    if isnothing(spriteheader)
        push!(lists.hasspriteheader, false)
        push!(lists.sprites_buoyancy, false)
        push!(lists.sprites_disablelayer2buoyancy, false)
        push!(lists.sprites_newsystem, false)
        push!(lists.sprites_mode, 0x0)
    else
        push!(lists.hasspriteheader, true)
        push!(lists.sprites_buoyancy, spriteheader.buoyancy)
        push!(lists.sprites_disablelayer2buoyancy, spriteheader.disablelayer2buoyancy)
        push!(lists.sprites_newsystem, spriteheader.newsystem)
        push!(lists.sprites_mode, spriteheader.mode)
    end

    push!(datalist, data)
end

# """
#     generatecsv(dest::AbstractString, rootdir::AbstractString=leveldir;
#                 formatfunction::Function=to3d,
#                 flags::Union{AbstractString, Nothing}=nothing,
#                 makesparse=true, writeevery=0, verbose=true)

# Generate a CSV file at `dest` containing the data for all levels found under the
# given directory.
# Can also pass the function used to build up levels. It is passed one argument (the `Level`).
# By default, use [`to3d`](@ref).
# The flags to use in the [`buildlevel`](@ref) function can  be given in the `flags` argument.
# If `flags` is `nothing`, the default flags for the given `formatfunction` will be used.
# `makesparse` controls whether a sparse representation of the data will be saved. This saves
# space for high dimensional data.
# A write instruction is issued every `writeevery` levels.
# `verbose` controls whether progress messages are printed.

# As the method of [`to3d`](@ref) that takes in `Level`s does not have a `flags` argument,
# flags have to be supplied independently.
# """
# function generatecsv(dest::AbstractString, rootdir::AbstractString=leveldir;
#                      formatfunction::Function=to3d,
#                      flags::Union{AbstractString, Nothing}=nothing, makesparse::Bool=true,
#                      writeevery::Unsigned=convert(Unsigned, 50), verbose::Bool=true)
#     i = one(UInt)
#     totalsize = zero(UInt)
#     if isnothing(flags)
#         funcsymbol = nameof(formatfunction)
#         flags = dimensionality_defaultflags[Symbol(String(funcsymbol)[end - 1:end])]
#     end

#     # Level data added later due to (yet) unknown type
#     dataframe = DataFrame(
#         # DB primary key
#         id = UInt[],

#         # LevelStats
#         hack             = String[],
#         number           = UInt16[],
#         screens          = UInt8[],
#         mode             = UInt8[],
#         fgbggfx          = UInt8[],
#         lmexpandedformat = UInt8[],

#         # Sprite header
#         # If this first entry is `false`, the rest should be ignored.
#         # We do not use `missing` because this is _assumed_ to be faster.
#         hasspriteheader               = Bool[],
#         sprites_buoyancy              = Bool[],
#         sprites_disablelayer2buoyancy = Bool[],
#         sprites_newsystem             = Bool[],
#         sprites_mode                  = UInt8[],
#     )

#     datacol_added = false

#     starttime = time()
#     println("Generating data...")
#     for dir in filter(isdir, map(d -> joinpath(rootdir, d), readdir(rootdir)))
#         # We choose an arbitrary extension to filter by just so we don't read in the same
#         # level multiple times.
#         for file in map(f -> joinpath(dir, f),
#                         filter(f -> endswith(f, ".map"), readdir(dir)))
#             if isnothing(flags)
#                 level = buildlevel(file, verbose=false)
#             else
#                 level = buildlevel(file, flags, verbose=false)
#             end
#             if makesparse
#                 # TODO only apply compressdata if 3d
#                 data = compressdata(sparse(formatfunction(level)))
#             else
#                 data = formatfunction(level)
#             end
#             totalsize += Base.summarysize(data)

#             if !datacol_added
#                 dataframe.data = typeof(data)[]
#                 CSV.write(dest, dataframe)
#                 datacol_added = true
#             end

#             addentry!(dataframe, i, level, data)

#             if i % writeevery == 0
#                 CSV.write(dest, dataframe, delim=';', append=true)
#                 deleterows!(dataframe, 1:size(dataframe, 1))
#             end
#             printprogress(i, totalsize, verbose)
#             i += 1
#         end
#     end
#     datacol_added || @warn "the root directory contained no levels."
#     println("Done after $(round(Int, time() - starttime)) seconds.")
# end

# "Add an entry to the given `AbstractDataFrame`, using the supplied data."
# function addentry!(dataframe::AbstractDataFrame, i::UInt, level::Level, data::AbstractArray)
#     stats = level.stats

#     rowhead = [
#         i,

#         hack(stats),
#         stats.number,
#         stats.screens,
#         stats.mode,
#         stats.fgbggfx,
#         stats.lmexpandedformat,
#     ]

#     spriteheader = level.spriteheader
#     if isnothing(spriteheader)
#         rowmiddle = [
#             false,
#             false,
#             false,
#             false,
#             0x0,
#         ]
#     else
#         rowmiddle = [
#             true,
#             spriteheader.buoyancy,
#             spriteheader.disablelayer2buoyancy,
#             spriteheader.newsystem,
#             spriteheader.mode,
#         ]
#     end

#     push!(dataframe, vcat(rowhead, rowmiddle, [data]))
# end

# TODO try JLD/HDF5 backend


# TODO write method for 'abstract' database containing raw `Levels` (with compressed sparse
# data) on which a formatting method has to be called. use case is saving space but
# increasing computation time (not one db for each type of data but have to format data
# when loading)

function printprogress(i, totalsize, verbose)
    if i % 50 == 0
        verbose && @info ("$i levels, $totalsize bytes of object data processed (mean: "
                          * "$(round(Int, totalsize / i)) bytes per level).")
        # These were used due to a suspected memory leak.
        # May be necessary again when using threads (unless it is fixed in Julia versions
        # greater than 1.1.1).
        # GC.gc()
        # ccall(:malloc_trim, Cvoid, (Cint,), 0)
    end
end

end # module

