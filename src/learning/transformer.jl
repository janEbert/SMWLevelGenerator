module Transformer

import Flux
import Transformers

using ..LevelStatistics: maxcolshori
using ..InputStatistics
using ..ModelUtils
import ..ModelUtils: makeloss, dataiteratorparams, soft_criterion

export TransformerModel, transformer1d, transformer2d, transformer3dtiles, transformer3d

struct TransformerModel{M} <: LearningModel
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike TransformerModel

(model::TransformerModel)(input) = model.model(input)

function makeloss(model::TransformerModel, criterion)
    function loss(x, y)
        y_hat = model(x)
        @inbounds l = sum(@views criterion(y_hat[:, i], y[:, i]) for i in axes(y, 2))
        return l
    end
end

dataiteratorparams(::TransformerModel) = (join_pad=false, as_matrix=true)


# A possible criterion to reduce over-generalizing. Loss function must be changed for this
# as this criterion shouldn't be broadcasted.
"""
Return a reduced loss from predicting a sequence element differently when the two previous
sequence elements were the same.
"""
function soft_criterion(::TransformerModel, y_hat, y, criterion)
    total = 0.0f0
    last_elem = nothing
    second_to_last = nothing
    for i in axes(y, 2)
        @inbounds @views e_hat, e = y_hat[:, i], y[:, i]
        if last_elem != second_to_last
            total += criterion(e_hat, e)
        else
            total += 0.1f0 * criterion(e_hat, e)
        end
        second_to_last = last_elem
        last_elem = e
    end
    return total
end


struct GPT2Block{A<:Transformers.Basic.MultiheadAttention, L<:Flux.LayerNorm,
                 F<:Transformers.Basic.PwFFN,
                 D<:Flux.Dropout} <: Transformers.Basic.AbstractTransformer
    pre_norm::L
    attention::A
    attn_norm::L
    piecewise::F
    dropout::D
end

Flux.@treelike GPT2Block

function GPT2Block(inputsize::Integer, num_heads::Integer, ffhiddensize::Integer;
                   look_ahead::Bool=false, activation=Flux.gelu,
                   p_dropout_attn=0.1f0)
    if inputsize % num_heads != 0
        throw(ArgumentError("`inputsize` not divisible by `num_heads`"))
    end
    GPT2Block(inputsize, num_heads, div(inputsize, num_heads), ffhiddensize;
              look_ahead=look_ahead, activation=activation, p_dropout_attn=p_dropout_attn)
end

function GPT2Block(inputsize::Integer, heads::Integer, attnhiddensize::Integer,
                   ffhiddensize::Integer;
                   look_ahead::Bool=false, activation=Flux.gelu, p_dropout_attn=0.1f0)
    GPT2Block(
        Flux.LayerNorm(inputsize),
        Transformers.Basic.MultiheadAttention(heads, inputsize, attnhiddensize, inputsize;
                                        future=look_ahead,
                                        pdrop=p_dropout_attn),
        Flux.LayerNorm(inputsize),
        Transformers.Basic.PwFFN(inputsize, ffhiddensize, activation),
        Flux.Dropout(p_dropout_attn),
    )
end

function (b::GPT2Block)(input::AbstractArray{T, N}, mask=nothing) where {T, N}
    input = b.pre_norm(input)::typeof(input)
    attn = b.attention(input, input, input; mask=mask)
    attn = b.dropout(attn)
    res_attn = input .+ attn
    if N == 3
        insize = size(res_attn)
        res_attn = reshape(res_attn, insize[1], :)
    end
    res_attn = b.attn_norm(res_attn)::typeof(res_attn)
    pwff = b.piecewise(res_attn)
    pwff = b.dropout(pwff)
    res_pwff = res_attn .+ pwff
    if N == 3
        res_pwff = reshape(res_pwff, :, Base.tail(insize)...)
    end
    res_pwff
end

struct GPT2{D<:Flux.Dropout, S<:Transformers.Stack,
            L<:Flux.LayerNorm} <: Transformers.Basic.AbstractTransformer
    dropout::D
    blocks::S
    final_norm::L
end

Flux.@treelike GPT2

function GPT2(inputsize::Integer, num_heads::Integer, ffhiddensize::Integer,
              num_layers::Integer;
              activation=Flux.gelu, p_dropout_ff=0.1f0, p_dropout_attn=0.1f0)
    if inputsize % num_heads != 0
        throw(ArgumentError("`inputsize` not divisible by `num_heads`"))
    end
    GPT2(inputsize, num_heads, div(inputsize, num_heads), ffhiddensize, num_layers;
            activation=activation, p_dropout_ff=p_dropout_ff, p_dropout_attn=p_dropout_attn)
end

function GPT2(inputsize::Integer, num_heads::Integer, attnhiddensize::Integer,
              ffhiddensize::Integer, num_layers::Integer;
              activation=Flux.gelu, p_dropout_ff=0.1f0, p_dropout_attn=0.1f0)
    GPT2(
        Flux.Dropout(p_dropout_ff),
        Transformers.Stack(
            Transformers.@nntopo_str("x':x => $num_layers"),
            [
                GPT2Block(inputsize, num_heads, attnhiddensize, ffhiddensize;
                          look_ahead=false, activation=activation,
                          p_dropout_attn=p_dropout_attn)
                for i = 1:num_layers
            ]...
        ),
        Flux.LayerNorm(inputsize)
    )
end

function (gpt::GPT2)(input, mask=nothing; return_all_outputs::Val{false}=Val(false))
    input = gpt.dropout(input)
    output = gpt.blocks(input)[1]
    output = gpt.final_norm(output)
    isnothing(mask) || (output = output .* mask)
    return output  # size(output) == (inputsize, seq_len, batch_len)
end

# TODO Warning: Method definition overwritten ???
# function (gpt::GPT2)(input, mask=nothing; return_all_outputs::Val{true})
#     input = gpt.dropout(input)
#     output, all_outputs = gpt.blocks(input)
#     output = gpt.final_norm(output)
#     isnothing(mask) || (output = output .* mask)
#     return output, all_outputs
# end

struct GPT2Predictor{E<:Transformers.Basic.PositionEmbedding,
                     G<:Transformers.Basic.AbstractTransformer, O}
    embedding::E
    gpt::G
    output::O
end

Flux.@treelike GPT2Predictor

function GPT2Predictor(inputsize::Integer, outputsize::Integer, num_heads::Integer,
                       attnhiddensize::Integer, ffhiddensize::Integer, num_layers::Integer,
                       maxinputsize::Integer; activation=Flux.gelu,
                       p_dropout_ff=0.1f0, p_dropout_attn=0.1f0,
                       output_activation=identity)
    GPT2Predictor(
        Transformers.PositionEmbedding(inputsize, maxinputsize; trainable=true),
        GPT2(
            inputsize, num_heads, attnhiddensize, ffhiddensize, num_layers;
            activation=Flux.gelu, p_dropout_ff=p_dropout_ff, p_dropout_attn=p_dropout_attn
        ),
        Flux.Dense(inputsize, outputsize, output_activation)
    )
end

function (gpt::GPT2Predictor)(input, mask=nothing;
                              return_all_outputs::Val{false}=Val(false))
    input = gpt.embedding(input)
    output = gpt.gpt(input, mask, return_all_outputs=return_all_outputs)
    return gpt.output(output)
end

# TODO correctly initialize layers

# Original parameters:
# embed = Embed(768, vocab_size)
# pe = PositionEmbedding(768, 512; trainable=true)
# gpt = Gpt(768, 12, 768*4, 12; act=gelu, pdrop=0.1, attn_pdrop=0.1)

function maketransformer(num_heads::Integer, attnhiddensize::Integer,
                         ffhiddensize::Integer, num_layers::Integer,
                         inputsize::Integer, outputsize::Integer, dimensionality::Symbol;
                         activation=Flux.gelu, p_dropout_ff=0.1f0, p_dropout_attn=0.1f0,
                         output_activation=Flux.sigmoid)
    model = GPT2Predictor(
        inputsize, outputsize, num_heads, attnhiddensize, ffhiddensize, num_layers,
        convert(Int, maxcolshori);
        activation=activation, p_dropout_ff=p_dropout_ff, p_dropout_attn=p_dropout_attn,
        output_activation=output_activation
    ) |> togpu
    TransformerModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :num_heads         => num_heads,
        :attnhiddensize    => attnhiddensize,
        :ffhiddensize      => ffhiddensize,
        :num_layers        => num_layers,
        :inputsize         => inputsize,
        :outputsize        => outputsize,
        :activation        => activation,
        :p_dropout_ff      => p_dropout_ff,
        :p_dropout_attn    => p_dropout_attn,
        :output_activation => output_activation,
    ))
end

function transformer1d(num_heads::Integer=8, attnhiddensize::Integer=4,
                       ffhiddensize::Integer=16, num_layers::Integer=2,
                       inputsize::Integer=inputsize1d,
                       outputsize::Integer=outputsizeof(inputsize);
                       p_dropout_ff=0.05f0, p_dropout_attn=0.05f0, kwargs...)
    maketransformer(num_heads, attnhiddensize, ffhiddensize, num_layers,
                    inputsize, outputsize, Symbol("1d");
                    p_dropout_ff=p_dropout_ff, p_dropout_attn=p_dropout_attn, kwargs...)
end

function transformer2d(num_heads::Integer=8, attnhiddensize::Integer=8,
                       ffhiddensize::Integer=32, num_layers::Integer=2,
                       inputsize::Integer=inputsize2d,
                       outputsize::Integer=outputsizeof(inputsize);
                       p_dropout_ff=0.05f0, p_dropout_attn=0.05f0, kwargs...)
    maketransformer(num_heads, attnhiddensize, ffhiddensize, num_layers,
                    inputsize, outputsize, Symbol("2d");
                    p_dropout_ff=p_dropout_ff, p_dropout_attn=p_dropout_attn, kwargs...)
end

function transformer3dtiles(num_heads::Integer=8, attnhiddensize::Integer=16,
                            ffhiddensize::Integer=64, num_layers::Integer=3,
                            inputsize::Integer=inputsize3dtiles,
                            outputsize::Integer=outputsizeof(inputsize); kwargs...)
    maketransformer(num_heads, attnhiddensize, ffhiddensize, num_layers,
                    inputsize, outputsize, Symbol("3dtiles"); kwargs...)
end

function transformer3d(num_heads::Integer=8, attnhiddensize::Integer=32,
                       ffhiddensize::Integer=128, num_layers::Integer=3,
                       inputsize::Integer=inputsize3d,
                       outputsize::Integer=outputsizeof(inputsize); kwargs...)
    maketransformer(num_heads, attnhiddensize, ffhiddensize, num_layers,
                    inputsize, outputsize, Symbol("3d"); kwargs...)
end

end # module

