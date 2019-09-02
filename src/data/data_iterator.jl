module DataIterator

using Random
using SparseArrays

using JuliaDB

using ..DataCompressor: compressedindextype, decompressdata
import ..LevelStatistics

export loaddb, traintestsplit, dataiterator


"""
    simplefilter(db::IndexedTable)

Return `true` for the given `row` if it matches the following criteria:
   - non-vertical, non-boss, non-layer-2 level
   - no "unusable" mode (as recommended by Lunar Magic)
   - with `hasspriteheader` (only if `hassprites` is `true`)
   - no `lmexpandedformat`
"""
function simplefilter(row::NamedTuple, hassprites::Bool)
    # First criterion
    isvert   = row.mode in LevelStatistics.vertmodes
    isboss   = row.mode in LevelStatistics.bossmodes
    islayer2 = row.mode in LevelStatistics.layer2modes

    # Second criterion
    isunusable = row.mode in LevelStatistics.unusablemodes

    # Third criterion (optional)
    withoutspriteheader = hassprites ? !row.hasspriteheader : false

    # Fourth criterion
    isexpanded = !(row.lmexpandedformat in (0x0, 0x80))

    # Combine
    return !(isvert || isboss || islayer2
             || isunusable
             || withoutspriteheader
             || isexpanded)
end

"""
Return the database at the given path and filter it.

If the database contains at least one entry with `hasspriteheader=true`, filter by
`hasspriteheader` as well.
"""
function loaddb(path::AbstractString)
    # Temporarily disable GC to fix https://github.com/JuliaComputing/MemPool.jl/issues/26
    Base.GC.enable(false)
    db = load(path)
    Base.GC.enable(true)
    hassprites = any(select(db, :hasspriteheader))
    return filter(row -> simplefilter(row, hassprites), db)
end

"Return a `Tuple` of indices belonging to the training and test set, respectively."
function traintestsplit(db::IndexedTable, testratio::Real=0.1, seed=0)
    @assert 0 <= testratio < 1 "Invalid test split ratio $testratio"
    indices = collect(one(UInt):convert(UInt, length(db)))
    shuffle!(MersenneTwister(seed), indices)
    lasttrainindex = round(UInt, length(indices) * (1 - testratio))
    trainindices = view(indices, firstindex(indices):lasttrainindex)
    testindices = view(indices, lasttrainindex + 1:lastindex(indices))
    return trainindices, testindices
end

function dataiteratortask(channel::AbstractChannel, db::IndexedTable,
                          splitindices::AbstractVector)
    for index in splitindices
        row = db[index]
        put!(channel, preprocess(row))
    end
end

function dataiterator(db::IndexedTable, splitindices::AbstractVector)
    channel = Channel(0)
    task = @async dataiteratortask(channel, db, splitindices)
    bind(channel, task)
    return channel
end

function prepend_isend(col::Union{Number, AbstractVector}, i::Integer, seqlength::Integer)
    i < seqlength ? vcat(zero(eltype(col)), col) : vcat(one(eltype(col)), col)
end

function prepend_isend(cols::AbstractMatrix, i::Integer, seqlength::Integer)
    if i < seqlength
        vcat(zero(eltype(cols)), reduce(vcat, cols))
    else
        vcat(one(eltype(cols)),  reduce(vcat, cols))
    end
end

function slicedata(data::AbstractVector{
                       Pair{UInt16, SparseMatrixCSC{T, compressedindextype}}})
    slicedata(decompressdata(data, false))
end

function slicedata(data::AbstractVector)
    (prepend_isend(col, i, length(data)) for (i, col) in enumerate(data))
end

function slicedata(data::AbstractMatrix)
    (pepend_isend(col, i, size(data, 2)) for (i, col) in enumerate(eachcol(data)))
end

function slicedata(data::AbstractArray{T, 3}) where T
    (prepend_isend(allcols, i, size(data, 2))
            for (i, allcols) in enumerate(eachslice(data, dims=2)))
end

"""
Return a `Generator` containing `Tuple`s of the constant part of an input and each column of
input data.
"""
function preprocess(row::NamedTuple)
    # Change `constantinputsize` in `src/learning/input_statistics.jl` if this is modified!
    constantinput = [
        row.number,
        row.screens,
        row.mode,
        row.fgbggfx,
        row.sprites_mode,
    ]
    return (vcat(constantinput, col) for col in slicedata(row.data))
end

function makebatch(dataiterator::AbstractChannel, batchsize::Integer)
    firstelem = take!(dataiterator)
    seqtype = typeof(firstelem)
    sequences = Vector{}(undef, ) # TODO hier stehen geblieben
    batch = cat((reduce(hcat, gen)
                 for gen in Iterators.take(dataiterator, batchsize))..., dims=3)
end

end # module

