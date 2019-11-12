module DataCompressor

using SparseArrays
import SparseArrays: sparse

export unsparse, compressindices, decompressindices, compressdata, decompressdata

const defaultindextype = Int
const compressedindextype = UInt16

"""
    sparse(array::AbstractArray{T, 3}) where T

Return the given `AbstractArray{T, 3}` as a `Vector{SparseMatrixCSC{T}}` where each
element in the `Vector` is a slice along the second dimension of the given array.
"""
function sparse(array::AbstractArray{T, 3}; dims=2) where T
    map(sparse, eachslice(array, dims=dims))
end

"Return the given `AbstractVector{SparseMatrixCSC{T}}` as an `Array{T, 3}`."
function unsparse(sparsearray::AbstractVector{<:SparseMatrixCSC}; dims=2)
    # TODO is that `map` really necessary?
    reduce((a, b) -> cat(a, b, dims=dims), map(Array, sparsearray))
end

function compressindices(data::SparseMatrixCSC{T, defaultindextype}) where T
    convert(SparseMatrixCSC{T, compressedindextype}, data)
end

function compressindices(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}}) where T
    convert(AbstractVector{SparseMatrixCSC{T, compressedindextype}}, data)
end

function decompressindices(data::SparseMatrixCSC{T, compressedindextype}) where T
    convert(SparseMatrixCSC{T, defaultindextype}, data)
end

function decompressindices(data::AbstractVector{
                                         SparseMatrixCSC{T, compressedindextype}}) where T
    convert(AbstractVector{SparseMatrixCSC{T, defaultindextype}}, data)
end

# TODO possibly use BitArrays non-dim3 compression; they are much smaller but will probably
#      need conversion afterwards for speed (maybe; test this)

"""
Return the given `AbstractVector` of sparse data as a `Vector` of `Pair`s containing the
index and data of each non-empty (`count(!iszero, x) > 0`) entry.
The last entry is always saved so the length of the original array is saved as well.
"""
function compressdata(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}}) where T
    compressed = Vector{Pair{UInt16, SparseMatrixCSC{T, compressedindextype}}}()
    for (i, x) in enumerate(@view data[firstindex(data):end - 1])
        count(!iszero, x) > 0 || continue
        push!(compressed, i => x)
    end
    # We deliberately use `length` instead of `lastindex` for robustness against different
    # indexing types. During decompression, we are creating a standard `Vector` anyway.
    push!(compressed, length(data) => data[end])
    return compressed
end

"""
Return the given `AbstractVector` of `Pair`s converted to its representation prior
to compression.
If `tosparse` is `true`, return a `Vector{SparseMatrixCSC}`; otherwise an `Array{T, 3}`.
"""
function decompressdata(compressed::AbstractVector{
                            Pair{UInt16, SparseMatrixCSC{T, compressedindextype}}},
                        tosparse::Bool) where T
    if tosparse
        data = Vector{SparseMatrixCSC{T, defaultindextype}}(undef, compressed[end].first)
    else
        data = Array{T, 3}(undef, size(compressed[1].second)..., compressed[end].first)
    end
    j = 0x001
    for (i, x) in compressed
        while j < i
            emptyat!(data, j, size(x))
            j += 0x001
        end
        fillat!(data, i, x)
        j += 0x001
    end
    if tosparse
        return sparse(unsparse(data, dims=3))
    else
        return data
    end
end

function emptyat!(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}},
                  index::Integer, emptysize::Tuple) where T
    data[index] = spzeros(T, defaultindextype, emptysize)
end

function emptyat!(data::AbstractArray{T, 3}, index::Integer, ::Any) where T
    data[:, :, index] .= 0
end

function fillat!(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}}, index::Integer,
                 x::SparseMatrixCSC{T, compressedindextype}) where T
    data[index] = x
end

function fillat!(data::AbstractArray{T, 3}, index::Integer,
                 x::SparseMatrixCSC{T, compressedindextype}) where T
    data[:, :, index] = x
end

end # module

