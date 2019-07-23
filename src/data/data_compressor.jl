module DataCompressor

using SparseArrays

export compressdata, decompressdata

const defaultindextype = Int
const compressedindextype = UInt16

"""
Return the given `AbstractVector` of sparse data as a `Vector` of `Pair`s containing the
index and data of each non-empty (`count(!iszero, x) > 0`) entry.
The last entry is always saved so the length of the original array is saved as well.
"""
function compressdata(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}}) where T
    compressed = Vector{Pair{UInt16, SparseMatrixCSC{T, compressedindextype}}}()
    for (i, x) in enumerate(data[firstindex(data):end - 1])
        count(!iszero, x) > 0 || continue
        push!(compressed, i => x)
    end
    # We deliberately use `length` instead of `lastindex` for robustness against different
    # indexing types. During decompression, we are creating a standard `Vector` anyway.
    push!(compressed, length(data) => data[end])
    return compressed
end

# TODO decompress (actually, whole back-to-level pipeline)
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
            emptyat!(data, j)
            j += 0x001
        end
        fillat!(data, i, x)
        j += 0x001
    end
    return data
end

function emptyat!(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}}, index::Integer)
    data[index] = sparse(defaultindextype[], defaultindextype[], T[])
end

function emptyat!(data::AbstractArray{T, 3}, index::Integer)
    data[:, :, index] .= 0
end

function fillat!(data::AbstractVector{SparseMatrixCSC{T, defaultindextype}}, index::Integer,
                 x::SparseMatrixCSC{T, compressedindextype})
    data[index] = x
end

function fillat!(data::AbstractArray{T, 3}, index::Integer,
                 x::SparseMatrixCSC{T, compressedindextype})
    data[:, :, index] = x
end

end # module

