"Contains a dictionary class that returns a default value if a key is not present."
module DefaultDictionary

"A dictionary returning a default value if a key is not present."
struct DefaultDict{S, T} <: AbstractDict{S, T}
    "Internal dictionary."
    dict::Dict{S, T}
    "Default value returned if a key is not present in `dict`."
    default::T

    """
        DefaultDict{S, T}(default::T, args...)
        DefaultDict(default::T, args...)

    Construct a `DefaultDict` with the given default value. Other arguments are passed to
    the constructor of [`Dict`](@ref).
    """
    DefaultDict{S, T}(default::T, args...) where {S, T} = new(Dict{S, T}(args...), default)
    DefaultDict{S, T}(default, args...) where {S, T} = new(Dict{S, T}(args...),
                                                           convert(T, default))
    function DefaultDict(default::T, args...) where T
        dict = Dict(args...)
        S = keytype(dict)
        U = valtype(dict)
        @assert T <: U "incompatible types $T and $U; try the parameterized constructor"
        new{S, U}(dict, default)
    end
end


function Base.show(io::IO, ddict::DefaultDict)
    print(io, "Default", ddict.dict, "\n  Default value: ", ddict.default)
end

function Base.show(io::IO, mime::MIME"text/plain", ddict::DefaultDict)
    print(io, "Default")
    show(io, mime, ddict.dict)
    print(io, "\n  Default value: ", ddict.default)
end

# Iteration

Base.iterate(ddict::DefaultDict) = iterate(ddict.dict)
Base.iterate(ddict::DefaultDict, state) = iterate(ddict.dict, state)

function Base.iterate(revddict::Iterators.Reverse{DefaultDict})
    iterate(Iterators.Reverse{revddict.dict})
end
function Base.iterate(revddict::Iterators.Reverse{DefaultDict}, state)
    iterate(Iterators.Reverse{revddict.dict}, state)
end

Base.empty!(ddict::DefaultDict) = empty!(ddict.dict)

Base.eltype(ddict::Type{DefaultDict{S, T}}) where {S, T} = Pair{S, T}
Base.length(ddict::DefaultDict) = length(ddict.dict)

function Base.in(value::Pair, ddict::DefaultDict, valcmp=:(==))
    if haskey(ddict.dict, value.first)
        in(value, ddict.dict, valcmp)
    else
        valcmp(value.second, ddict.default)
    end
end

# Indexing

Base.getindex(ddict::DefaultDict, key) = get(ddict.dict, key, ddict.default)
Base.setindex!(ddict::DefaultDict, key, value) = setindex!(ddict.dict, key, value)

# Dictionaries

Base.get(ddict::DefaultDict, key, default) = get(ddict.dict, key, default)
Base.get(f::Function, ddict::DefaultDict, key) = get(f, ddict.dict, key)

Base.get!(ddict::DefaultDict, key, default) = get!(ddict.dict, key, default)
Base.get!(f::Function, ddict::DefaultDict, key) = get!(f, ddict.dict, key)

Base.getkey(ddict::DefaultDict, key, default) = getkey(ddict.dict, key, default)

Base.delete!(ddict::DefaultDict, key) = delete!(ddict.dict, key)

Base.pop!(ddict::DefaultDict, key, default=ddict.default) = pop!(ddict.dict, key, default)

Base.merge(ddict::DefaultDict, others::AbstractDict...) = DefaultDict(ddict.default,
        merge(ddict.dict, others...))

Base.merge(combine, ddict::DefaultDict, others::AbstractDict...) = DefaultDict(
        ddict.default, merge(combine, ddict.dict, others...))

Base.merge!(ddict::DefaultDict, others::AbstractDict...) = merge!(ddict.dict, others...)

Base.merge!(combine, ddict::DefaultDict, others::AbstractDict...) = merge!(
        combine, ddict.dict, others...)

Base.sizehint!(ddict::DefaultDict, n) = sizehint!(ddict.dict, n)

end # module

