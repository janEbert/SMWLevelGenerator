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

function dataiteratorchannel(db::IndexedTable, buffersize, join_pad::Val, as_matrix::Val)
    channeltype = dataiterator_channeltype(db[1].data, join_pad, as_matrix)
    return Channel{channeltype}(buffersize)
end

function dataiterator_channeltype(data, #=join_pad=#::Val{true}, #=as_matrix=#::Val)
    dataiterator_channeltype(data, Val(false))
end

function dataiterator_channeltype(data, #=join_pad=#::Val{false}, as_matrix::Val{true})
    dataiterator_channeltype(data, as_matrix)
end

function dataiterator_channeltype(data, #=join_pad=#::Val{false}, as_matrix::Val{false})
    Vector{dataiterator_channeltype(data, as_matrix)}
end

function dataiterator_channeltype(data::Union{AbstractVector{<:Pair},
                                              AbstractVector{<:SparseMatrixCSC}},
                                  as_matrix::Val{true})
    SparseMatrixCSC{Float32, defaultindextype}
end

function dataiterator_channeltype(data::Union{AbstractVector{<:Pair},
                                              AbstractVector{<:SparseMatrixCSC}},
                                  as_matrix::Val{false})
    SparseVector{Float32, defaultindextype}
end

dataiterator_channeltype(data, as_matrix::Val{true})  = Matrix{Float32}
dataiterator_channeltype(data, as_matrix::Val{false}) = Vector{Float32}

function dataiteratortask(channel::AbstractChannel, db::IndexedTable,
                          splitindices::AbstractVector, per_tile::Bool, reverse_rows::Val,
                          join_pad::Val, as_matrix::Val)
    # try
        while true
            for index in splitindices
                @inbounds row = db[index]
                put!(channel, preprocess(row, per_tile, reverse_rows, join_pad, as_matrix))
            end
        end
    # catch e
    #     print("Error in data iterator task: ")
    #     showerror(stdout, e)
    # end
end

function dataiterator!(channel::AbstractChannel, db::IndexedTable,
                       splitindices::AbstractVector, num_threads::Integer=0,
                       per_tile=false, reverse_rows=false; join_pad=false, as_matrix=false)
    @static if VERSION >= v"1.3-"
        if num_threads > 0
            per_thread_split_length = cld(length(splitindices), num_threads)
            for indices in Iterators.partition(splitindices, per_thread_split_length)
                task = Threads.@spawn dataiteratortask(channel, db, indices, per_tile,
                        Val(reverse_rows), Val(join_pad), Val(as_matrix))
            end
        else
            task = @async dataiteratortask(channel, db, splitindices, per_tile,
                                           Val(reverse_rows), Val(join_pad), Val(as_matrix))
        end
    else
        task = @async dataiteratortask(channel, db, splitindices, per_tile,
                                       Val(reverse_rows), Val(join_pad), Val(as_matrix))
    end
    return channel
end

# TODO speed test for data iterator num_threads in trainingutils
function dataiterator(db::IndexedTable, buffersize::Integer, splitindices::AbstractVector,
                      num_threads::Integer=0, per_tile=false, reverse_rows=false;
                      join_pad=false, as_matrix=false)
    channel = dataiteratorchannel(db, buffersize, Val(join_pad), Val(as_matrix))
    return dataiterator!(channel, db, splitindices, num_threads, per_tile, reverse_rows;
                         join_pad=join_pad, as_matrix=as_matrix)
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

slicedata(data::AbstractVector, ::Val) = (col for col in data)

slicedata(data::AbstractMatrix, ::Val) = (col for col in eachcol(data))

function slicedata(data::AbstractArray{T, 3}, ::Val) where T
    (vec(allcols) for allcols in eachslice(data, dims=2))
end

function slicedata(data::AbstractVector{<:AbstractSparseMatrix},
                   #=reverse_rows=#::Val{true}) where T
    (vec(reverse(allcols, dims=1)) for allcols in data)
end

function slicedata(data::AbstractVector{<:AbstractSparseMatrix},
                   #=reverse_rows=#::Val{false}) where T
    (vec(allcols) for allcols in data)
end

"""
Return a `Vector` where each element is the constant part of an input and a part of
input data.
Whether the part of input data is a whole column or just one tile is controlled
by `per_tile`.
Data will be read from bottom to top instead of top to bottom with
`reverse_rows=Val(true)`.
Whether the data will be padded and joined to a single `Vector` is controlled with
`join_pad`.
Whether each element in the sequence will be padded and returned as a `Matrix` is
controlled with `as_matrix`.
`join_pad=Val(true)` has a higher priority than `as_matrix=Val(true)`.
"""
function preprocess(row::NamedTuple, per_tile::Bool, reverse_rows::Val,
                    #=join_pad=#::Val{false}, as_matrix::Val)
    constantinput = getconstantinput(row)
    decompressed_data = decompress(row.data)
    if reverse_rows isa Val{true} && ndims(decompressed_data) >= 2
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

    build_result(data, constantinput, reverse_rows, as_matrix)
end

function preprocess(row::NamedTuple, per_tile::Bool, reverse_rows::Val,
                    #=join_pad=#::Val{true}, #=as_matrix=#::Val)
    constantinput = getconstantinput(row)
    decompressed_data = decompress(row.data)
    if reverse_rows isa Val{true} && ndims(decompressed_data) >= 2
        rawdata = reverse(decompressed_data, dims=1)
    else
        rawdata = decompressed_data
    end
    # TODO preallocate, then fill
    if ndims(rawdata) == 3
        data = mapreduce(vec, vcat, eachslice(rawdata, dims=3))
        minlength = LevelStatistics.maxtileshori * size(rawdata, 3)
    else
        data = vec(rawdata)
        minlength = LevelStatistics.maxtileshori
    end
    return vcat(constantinput, rpad(data, minlength, 0))
end


function build_result(data, constantinput, reverse_rows::Val, as_matrix::Val{true})
    constantinputsize = length(constantinput) + 1
    result_matrix = construct_empty_result(data, constantinputsize, as_matrix)

    # Writing it this way makes more allocations but is faster.
    result_matrix[1:constantinputsize - 1, :]   .= constantinput
    result_matrix[constantinputsize, 1:end - 1] .= 1
    for (res_col, data_col) in zip(eachcol(result_matrix), slicedata(data, reverse_rows))
        # res_col[1:constantinputsize - 1]   = constantinput
        # res_col[constantinputsize]         = 1
        res_col[constantinputsize + 1:end] = data_col
    end
    result_matrix isa AbstractSparseArray || (result_matrix[constantinputsize, end] = 0)
    return result_matrix
end

function build_result(data, constantinput, reverse_rows::Val, as_matrix::Val{false})
    constantinputsize = length(constantinput) + 1
    result_vector = construct_empty_result(data, constantinputsize, as_matrix)
    for (res_col, data_col) in zip(result_vector, slicedata(data, reverse_rows))
        res_col[1:constantinputsize - 1]   = constantinput
        res_col[constantinputsize]         = 1
        res_col[constantinputsize + 1:end] = data_col
    end
    if result_vector[end] isa AbstractSparseArray
        result_vector[end][constantinputsize] = 0
    end
    return result_vector
end

function construct_empty_result(data::AbstractVector,
                                constantinputsize, #=as_matrix=#::Val{true})
    Matrix(Float32, constantinputsize + 1, length(data))
end

function construct_empty_result(data::AbstractMatrix,
                                constantinputsize, #=as_matrix=#::Val{true})
    Matrix{Float32}(undef, constantinputsize + size(data, 1), size(data, 2))
end

function construct_empty_result(data::AbstractArray{T, 3},
                                constantinputsize, #=as_matrix=#::Val{true}) where T
    spzeros(Float32, constantinputsize + size(data, 1) * size(data, 3), size(data, 2))
end

function construct_empty_result(data::AbstractVector{<:AbstractSparseMatrix},
                                constantinputsize, #=as_matrix=#::Val{true})
    spzeros(Float32, constantinputsize + length(first(data)), length(data))
end

function construct_empty_result(data::AbstractVector,
                                constantinputsize, #=as_matrix=#::Val{false})
    Vector{Float32}[Vector{Float32}(undef, constantinputsize + 1) for _ in 1:length(data)]
end

function construct_empty_result(data::AbstractMatrix,
                                constantinputsize, #=as_matrix=#::Val{false})
    Vector{Float32}[Vector{Float32}(undef, constantinputsize + size(data, 1))
                    for _ in 1:size(data, 2)]
end

function construct_empty_result(data::AbstractArray{T, 3},
                                constantinputsize, #=as_matrix=#::Val{false}) where T
    Vector{Float32}[Vector{Float32}(undef,
                                    constantinputsize + size(data, 1) * size(data, 3))
                    for _ in 1:size(data, 2)]
end

function construct_empty_result(data::AbstractVector{<:AbstractSparseMatrix},
                                constantinputsize, #=as_matrix=#::Val{false})
    SparseVector{Float32, defaultindextype}[
        spzeros(Float32, constantinputsize + length(first(data))) for _ in 1:length(data)
    ]
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
                              splitindices::AbstractVector, batch_size::Integer)
    # try
        screendims = db[1].data isa AbstractVector{<:Number} ? 3 : 4
        screen_buffer = [Array{Float32, screendims}(undef, ntuple(_ -> 0, screendims))
                         for _ in 1:batch_size]
        constantinput_buffer = [Vector{Float32}(undef, 0) for _ in 1:batch_size]
        while true
            for indices in Iterators.partition(splitindices, batch_size)
                for (i, index) in enumerate(indices)
                    @inbounds row = db[index]
                    screenview = gan_preprocess(row)
                    screen_buffer[i] = reshape(screenview, size(screenview)..., 1, 1)
                    constantinput_buffer[i] = getconstantinput(row)
                end
                # TODO pre-allocate and fill array instead of reduction
                if length(indices) == batch_size
                    screen_batch = reduce((x, y) -> cat(x, y, dims=screendims),
                                          @view screen_buffer[1:end])
                    constantinput_batch = reduce(hcat, @view constantinput_buffer[1:end])
                else
                    screen_batch = reduce((x, y) -> cat(x, y, dims=screendims),
                                          view(screen_buffer, 1:length(indices)))
                    constantinput_batch = reduce(hcat,
                                                 view(constantinput_buffer, 1:length(indices)))
                end
                put!(channel, (screen_batch, constantinput_batch))
            end
        end
    # catch e
    #     print("Error in data iterator task: ")
    #     showerror(stdout, e)
    # end
end

function gan_dataiteratorchannel(db::IndexedTable, buffersize)
    if db[1].data isa AbstractVector{<:Number}
        Channel{Tuple{Array{Float32, 3}, Matrix{Float32}}}(buffersize)
    else
        Channel{Tuple{Array{Float32, 4}, Matrix{Float32}}}(buffersize)
    end
end

function gan_dataiterator!(channel::AbstractChannel, db::IndexedTable,
                           splitindices::AbstractVector, batch_size::Integer,
                           num_threads::Integer=0)
    @static if VERSION >= v"1.3-"
        if num_threads > 0
            per_thread_split_length = cld(length(splitindices), num_threads)
            for indices in Iterators.partition(splitindices, per_thread_split_length)
                task = Threads.@spawn gan_dataiteratortask(channel, db,
                                                           splitindices, batch_size)
            end
        else
            task = @async gan_dataiteratortask(channel, db, splitindices, batch_size)
        end
    else
        task = @async gan_dataiteratortask(channel, db, splitindices, batch_size)
    end
    return channel
end

function gan_dataiterator(db::IndexedTable, buffersize::Integer,
                          splitindices::AbstractVector, batch_size::Integer,
                          num_threads::Integer=0)
    channel = gan_dataiteratorchannel(db, buffersize)
    return gan_dataiterator!(channel, db, splitindices, batch_size, num_threads)
end

firstscreen(data::AbstractVector) = view(data, 1:LevelStatistics.screencols)
firstscreen(data::AbstractMatrix) = view(data, :, 1:LevelStatistics.screencols)

function firstscreen(data::AbstractArray{T, 3}) where T
    view(data, :, 1:LevelStatistics.screencols, :)
end

function firstscreen(data::AbstractVector{<:AbstractSparseMatrix})
    mapreduce(firstscreen, (x, y) -> cat(x, y, dims=3), data)
end

function gan_preprocess(row::NamedTuple)
    data = decompress(row.data)
    return firstscreen(data)
end

end # module

