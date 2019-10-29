module TrainingUtils

using Base.CoreLogging: @_sourceinfo, logmsg_code,
                        _min_enabled_level, current_logger_for_env, shouldlog,
                        handle_message, logging_error
using Dates: now
using Logging

using ..InputStatistics: constantinputsize
using ..ModelUtils: togpu

export mae, @tblog, logprint, newexpdir, maketarget

mae(y_hat, y) = sum(abs.(y_hat .- y)) * 1 // length(y)

"""
    @tblog(logger, exs...)

Log the given expressions `exs` using the given logger with an empty message.

# Examples
```jldoctest
julia> using Logging
julia> l = SimpleLogger()
julia> with_logger(l) do
           @info "" a = 0
       end
┌ Info:
│   a = 0
└ @ [...]

julia> @tblog l a = 0
┌ Info:
│   a = 0
└ @ [...]
"""
macro tblog(logger, exs...)
    f = logmsg_code((@_sourceinfo)..., :(Logging.Info), "", exs...)
    quote
        with_logger($(esc(logger))) do
            $f
        end
    end
end

function logprint(logger::AbstractLogger, msg::AbstractString,
                  loglevel::LogLevel=Logging.Info)
    with_logger(logger) do
        @logmsg loglevel msg
    end
    @logmsg loglevel msg
end

newexpdir(prefix="exp") = replace("$(prefix)_$(now())", ':' => '-')

"""
    maketarget(seq, is_joined_padded::Val{Bool}, each_is_padded::Val{Bool})

Return the target (prediction) sequence of the given sequence.
The result is obtained by removing the first element of `seq` and appending zeros of the
same size instead. Also, the constant input part (except the "has not ended"-bit) is removed
from each element.
In addition, the result is appropriately moved to the GPU.
`Val{Bool}` means a `Val` type of a `Bool` instance.

```jldoctest; setup = :(ENV["SMWLG_IGNORE_GPU"] = true)
julia> maketarget([[1:10;], [11:20;]], Val(false), Val(false))
       # Removes the constant part as well!
2-element Array{SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},true},1}:
 [16, 17, 18, 19, 20]
 [0, 0, 0, 0, 0]
```
"""
function maketarget(seq, #=is_joined_padded=#::Val{false},
                    each_is_padded::Val{false})::AbstractVector{<:AbstractVector}
    # TODO check 1d and 2d case
    # It's much faster to view even sparse arrays than copy them
    # even though their calculation afterwards _should_ be faster.
    @inbounds target = @views map(x -> (x[constantinputsize:end]),
                                  seq[firstindex(seq) + 1:end])
    # Need to do indexing this way, otherwise we get a type error.
    @inbounds push!(target, @views zero(target[end])[firstindex(target[end]):end])
    togpu.(target)
end

function maketarget(seq, #=is_joined_padded=#::Val{false},
                    each_is_padded::Val{true})::AbstractMatrix
    # TODO check 1d and 2d case
    # It's much faster to view even sparse arrays than copy them
    # even though their calculation afterwards _should_ be faster.
    @inbounds target = @view seq[constantinputsize:end, firstindex(seq, 2) + 1:end]
    # Need to do indexing this way, otherwise we get a type error.
    target = hcat(target, @views zero(target[:, end])[firstindex(target[:, end]):end])
    togpu.(target)
end

# TODO need to change DataIterator.preprocess to something that makes sense; then fix this.
function maketarget(seq, #=is_joined_padded=#::Val{true},
                    #=each_is_padded=#::Val)::AbstractVector
    error("not correctly implemented yet")
    # It's much faster to view even sparse arrays than copy them
    # even though their calculation afterwards _should_ be faster.
    @inbounds target = @view seq[firstindex(seq) + 1:end]
    # Need to do indexing this way, otherwise we get a type error.
    @inbounds push!(target, @views zero(target[end])[firstindex(target[end]):end])
    togpu.(target)
end

end # module

