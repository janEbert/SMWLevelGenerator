module DataIterator

import Base: rpad
using Random
using SparseArrays

using JuliaDB

using ..DataCompressor: compressedindextype, defaultindextype, decompressdata
import ..LevelStatistics

export loaddb, traintestsplit, dataiteratorchannel, dataiterator!, dataiterator
export gan_dataiteratorchannel, gan_dataiterator!, gan_dataiterator

# TODO normalize data
# TODO batch data using start/end symbol

function Base.rpad(v::SparseVector, n::Integer, p)
    if p == 0
        vcat(v, spzeros(max(n - length(v), 0)))
    else
        vcat(v, fill(p, max(n - length(v), 0)))
    end
end

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

function dataiteratorchannel(db::IndexedTable, buffersize, join_pad::Val, pad_each::Val)
    channeltype = dataiterator_channeltype(db[1].data, join_pad, pad_each)
    return Channel{channeltype}(buffersize)
end

function dataiterator_channeltype(data, #=join_pad=#::Val{true}, #=pad_each=#::Val)
    dataiterator_channeltype(data, Val(false))
end

function dataiterator_channeltype(data, #=join_pad=#::Val{false}, pad_each::Val{true})
    dataiterator_channeltype(data, pad_each)
end

function dataiterator_channeltype(data, #=join_pad=#::Val{false}, pad_each::Val{false})
    Vector{dataiterator_channeltype(data, pad_each)}
end

function dataiterator_channeltype(data::Union{AbstractVector{<:Pair},
                                              AbstractVector{<:SparseMatrixCSC}},
                                  pad_each::Val{true})
    SparseMatrixCSC{Float32, defaultindextype}
end

function dataiterator_channeltype(data::Union{AbstractVector{<:Pair},
                                              AbstractVector{<:SparseMatrixCSC}},
                                  pad_each::Val{false})
    SparseVector{Float32, defaultindextype}
end

dataiterator_channeltype(data, pad_each::Val{true})  = Matrix{Float32}
dataiterator_channeltype(data, pad_each::Val{false}) = Vector{Float32}

function dataiteratortask(channel::AbstractChannel, db::IndexedTable,
                          splitindices::AbstractVector, per_tile::Bool, reverse_rows::Bool,
                          join_pad::Val, pad_each::Val)
    try
        for index in splitindices
            @inbounds row = db[index]
            put!(channel, preprocess(row, per_tile, reverse_rows, join_pad, pad_each))
        end
    catch e
        print("Error in data iterator task: ")
        showerror(stdout, e)
    end
end

function dataiterator!(channel::AbstractChannel, db::IndexedTable,
                       splitindices::AbstractVector, num_threads::Integer=0,
                       per_tile=false, reverse_rows=false; join_pad=false, pad_each=false)
    @static if VERSION >= v"1.3-"
        if num_threads > 0
            per_thread_split_length = cld(length(splitindices), num_threads)
            for indices in Iterators.partition(splitindices, per_thread_split_length)
                task = Threads.@spawn dataiteratortask(channel, db, indices, per_tile,
                        reverse_rows, Val(join_pad), Val(pad_each))
            end
        else
            task = @async dataiteratortask(channel, db, splitindices, per_tile,
                                           reverse_rows, Val(join_pad), Val(pad_each))
        end
    else
        task = @async dataiteratortask(channel, db, splitindices,
                                       per_tile, reverse_rows, Val(join_pad), Val(pad_each))
    end
    return channel
end

# TODO speed test for data iterator num_threads in trainingutils
function dataiterator(db::IndexedTable, splitindices::AbstractVector,
                      num_threads::Integer=0, per_tile=false, reverse_rows=false;
                      join_pad=false, pad_each=false)
    channel = dataiteratorchannel(db, 4, Val(join_pad), Val(pad_each))
    return dataiterator!(channel, db, splitindices, num_threads, per_tile, reverse_rows;
                         join_pad=join_pad, pad_each=pad_each)
end


function prepend_hasnotended_bit(col::Union{Number, AbstractVector}, i::Integer,
                                 seqlength::Integer)
    i < seqlength ? vcat(1.0f0, col) : vcat(0.0f0, col)
end

function prepend_hasnotended_bit(col::AbstractSparseVector, i::Integer,
                                 seqlength::Integer)
    i < seqlength ? vcat(sparsevec([1.0f0]), col) : vcat(spzeros(Float32, 1), col)
end

function getconstantinput(row::NamedTuple)
    # Change `constantinputsize` in `src/learning/input_statistics.jl` if this is modified!
    # Also change `deconstructconstantinput` in `src/data/level_deconstructor.jl`.
    return [
        row.number,
        row.screens,
        row.mode,
        row.fgbggfx,
        row.sprites_mode,
    ]
end

decompress(data) = data

function decompress(
                data::AbstractVector{Pair{UInt16, SparseMatrixCSC{T, compressedindextype}}}
            ) where T
    decompressdata(data, true)
end

function slicedata(data::AbstractVector)
    (prepend_hasnotended_bit(col, i, length(data)) for (i, col) in enumerate(data))
end

function slicedata(data::AbstractMatrix)
    (prepend_hasnotended_bit(col, i, size(data, 2))
            for (i, col) in enumerate(eachcol(data)))
end

function slicedata(data::AbstractArray{T, 3}) where T
    (prepend_hasnotended_bit(vec(allcols), i, size(data, 2))
            for (i, allcols) in enumerate(eachslice(data, dims=2)))
end

function slicedata(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}},
                   #=reverse_rows=#::Val{true}) where T
    (prepend_hasnotended_bit(vec(reverse(allcols, dims=1)), i, length(data))
            for (i, allcols) in enumerate(data))
end

function slicedata(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}},
                   #=reverse_rows=#::Val{false}) where T
    (prepend_hasnotended_bit(vec(allcols), i, length(data))
            for (i, allcols) in enumerate(data))
end

"""
Return a `Vector` where each element is the constant part of an input and a part of
input data.
Whether the part of input data is a whole column or just one tile is controlled
by `per_tile`.
Data will be read from bottom to top instead of top to bottom with `reverse_rows=true`.
Whether the data will be padded and joined to a single `Vector` is controlled with
`join_pad`.
Whether each element in the sequence will be padded and returned as a `Matrix` is
controlled with `pad_each`.
`join_pad::Val{true}` has a higher priority than `pad_each::Val{true}`.
"""
function preprocess(row::NamedTuple, per_tile::Bool, reverse_rows::Bool,
                    #=join_pad=#::Val{false}, pad_each::Val)
    constantinput = getconstantinput(row)
    decompressed_data = decompress(row.data)
    if reverse_rows && ndims(decompressed_data) >= 2
        rawdata = reverse(decompressed_data, dims=1)
    else
        rawdata = decompressed_data
    end
    if per_tile
        if ndims(rawdata) == 3
            data = reduce(vcat, eachslice(rawdata, dims=3))
        else
            data = vec(rawdata)
        end
    else
        data = rawdata
    end

    # TODO avoid double vcat; concat both hasnotended-bit and constantinput at once
    if pad_each isa Val{true}
        if per_tile || data isa AbstractVector{<:Number}
            # 1D or scalar sequence
            minlength = 1
        else
            # 2D or 3D
            minlength = LevelStatistics.screenrowshori
        end
        if data isa AbstractVector{<:AbstractSparseMatrix}
            return reduce(hcat, vcat(constantinput, rpad(col, minlength, 0))
                          for col in slicedata(data, Val(reverse_rows)))
        else
            return reduce(hcat, vcat(constantinput, rpad(col, minlength, 0))
                          for col in slicedata(data))
        end
    elseif data isa AbstractVector{<:AbstractSparseMatrix}
        return SparseVector{Float32, defaultindextype}[
                vcat(constantinput, col) for col in slicedata(data, Val(reverse_rows))]
    else
        return Vector{Float32}[vcat(constantinput, col) for col in slicedata(data)]
    end
end

function preprocess(row::NamedTuple, per_tile::Bool, reverse_rows::Bool,
                    #=join_pad=#::Val{true}, #=pad_each=#::Val)
    constantinput = getconstantinput(row)
    decompressed_data = decompress(row.data)
    if reverse_rows && ndims(decompressed_data) >= 2
        rawdata = reverse(decompressed_data, dims=1)
    else
        rawdata = decompressed_data
    end
    if ndims(rawdata) == 3
        data = reduce(vcat, eachslice(rawdata, dims=3))
    else
        data = vec(rawdata)
    end
    return vcat(constantinput, rpad(data, LevelStatistics.maxtileshori, 0))
end

"""
    normalize!(sequence, mean, variance)

Normalize the given sequence using the given mean and variance of its underlying
distribution to mean 0 and variance 1.
If `mean` is `nothing`, the mean is not normalized. This is useful to keep sparsity in data.
"""
function normalize!(sequence, mean, variance)
    invvariance = inv(variance)
    if isnothing(mean)
        sequence .*= invvariance
    else
        sequence .= (sequence .- mean) .* invvariance
    end
end

function normalize!(sequence::AbstractVector{<:AbstractArray}, mean, variance)
    normalize!.(sequence, mean, variance)
end

function makebatch(dataiterator::AbstractChannel, batchsize::Integer)
    error("not implemented.")
    firstelem = take!(dataiterator)
    seqtype = typeof(firstelem)
    sequences = Vector{}(undef, ) # TODO hier stehen geblieben
    batch = cat((reduce(hcat, gen)
                 for gen in Iterators.take(dataiterator, batchsize))..., dims=3)
end


function gan_dataiteratortask(channel::AbstractChannel, db::IndexedTable,
                              trainindices::AbstractVector, batch_size::Integer)
    try
        screendims = db[1].data isa AbstractVector{<:Number} ? 3 : 4
        screen_buffer = [Array{Float32, screendims}(undef, ntuple(_ -> 0, screendims))
                         for _ in 1:batch_size]
        constantinput_buffer = [Vector{Float32}(undef, 0) for _ in 1:batch_size]
        for indices in Iterators.partition(trainindices, batch_size)
            for (i, index) in enumerate(indices)
                @inbounds row = db[index]
                screenview = gan_preprocess(row)
                screen_buffer[i] = reshape(screenview, size(screenview)..., 1, 1)
                constantinput_buffer[i] = getconstantinput(row)
            end
            if length(indices) == batch_size
                screen_batch = reduce((x, y) -> cat(x, y, dims=screendims),
                                      screen_buffer)
                constantinput_batch = reduce(hcat, constantinput_buffer)
            else
                screen_batch = reduce((x, y) -> cat(x, y, dims=screendims),
                                      view(screen_buffer, 1:length(indices)))
                constantinput_batch = reduce(hcat,
                                             view(constantinput_buffer, 1:length(indices)))
            end
            put!(channel, (screen_batch, constantinput_batch))
        end
    catch e
        print("Error in data iterator task: ")
        showerror(stdout, e)
    end
end

function gan_dataiteratorchannel(db::IndexedTable, buffersize)
    if db[1].data isa AbstractVector{<:Number}
        Channel{Tuple{Array{Float32, 3}, Matrix{Float32}}}(buffersize)
    else
        Channel{Tuple{Array{Float32, 4}, Matrix{Float32}}}(buffersize)
    end
end

function gan_dataiterator!(channel::AbstractChannel, db::IndexedTable,
                           trainindices::AbstractVector, batch_size::Integer,
                           num_threads::Integer=0)
    @static if VERSION >= v"1.3-"
        if num_threads > 0
            per_thread_split_length = cld(length(trainindices), num_threads)
            for indices in Iterators.partition(trainindices, per_thread_split_length)
                task = Threads.@spawn gan_dataiteratortask(channel, db,
                                                           trainindices, batch_size)
            end
        else
            task = @async gan_dataiteratortask(channel, db, trainindices, batch_size)
        end
    else
        task = @async gan_dataiteratortask(channel, db, trainindices, batch_size)
    end
    return channel
end

function gan_dataiterator(channel::AbstractChannel, db::IndexedTable,
                          trainindices::AbstractVector, batch_size::Integer,
                          num_threads::Integer=0)
    channel = gan_dataiteratorchannel(3)
    return gan_dataiterator!(channel, db, trainindices, batch_size, num_threads)
end

firstscreen(data::AbstractVector) = view(data, 1:LevelStatistics.screencols)
firstscreen(data::AbstractMatrix) = view(data, :, 1:LevelStatistics.screencols)

function firstscreen(data::AbstractArray{T, 3}) where T
    view(data, :, 1:LevelStatistics.screencols, :)
end

function firstscreen(data::AbstractVector{<:AbstractSparseMatrix})
    reduce((x, y) -> cat(x, y, dims=3), map(firstscreen, data))
end

function gan_preprocess(row::NamedTuple)
    data = decompress(row.data)
    return firstscreen(data)
end

end # module

