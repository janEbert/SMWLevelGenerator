module LSTM

import Flux
using Flux.Tracker: gradient

using ..InputStatistics
using ..ModelUtils
import ..ModelUtils: step!

export LSTMModel, lstm1d, lstm2d, lstm3dtiles, lstm3d

struct LSTMModel{M} <: LearningModel
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike LSTMModel

(model::LSTMModel)(input) = model.model(input)

function step!(model::LSTMModel, parameters, optimizer, loss, input, target)
    l = calculate_loss(model, loss, input, target)

    # Take gradients
    Flux.reset!(model)
    grads = gradient(() -> l, parameters)

    # Update weights
    Flux.Optimise.update!(optimizer, parameters, grads)
    return l
end


# TODO check if docstring lines up; if not, convert to columns
"""
    function makehiddenlayers(hiddensize::Integer, hiddenlayers::Integer,
                              skipconnections::Val{false}, p_dropout)
    function makehiddenlayers(hiddensize::Integer, hiddenlayers::Integer,
                              skipconnections::Val{true}, p_dropout)

Create a sequence of `hiddenlayers` `LSTM` layers beginning with an `LSTM` with `Dropout`.
Then, every other `LSTM`, add another `Dropout` layer.
The `Dropout` layers are not included in `hiddenlayers`!

Calling the function with different `hiddenlayers` returns a sequence up to and including
the marker indicating the value of `hiddenlayers`.
`LSTM`, `Dropout`, `LSTM`, `LSTM`, `Dropout`, `LSTM`, `LSTM`, `Dropout`, `LSTM`, ...
        1          2               3          4               5          6       ...
"""
function makehiddenlayers(hiddensize::Integer, hiddenlayers::Integer,
                          ::Val{false}, p_dropout)
    return ((i - 2) % 3 == 0 ? Flux.LSTM(hiddensize, hiddensize) : Flux.Dropout(p_dropout)
            for i in 1:hiddenlayers + cld(hiddenlayers, 2))
end

function makehiddenlayers(hiddensize::Integer, hiddenlayers::Integer,
                          ::Val{true}, p_dropout)
    if hiddenlayers > 1
        if hiddenlayers % 2 == 0
            skipped_layer = (Flux.LSTM(hiddensize, hiddensize),)
        else
            skipped_layer = (Flux.LSTM(hiddensize, hiddensize), Flux.Dropout(p_dropout))
        end
        return Flux.SkipConnection(Flux.Chain(skipped_layer...,
                                              makehiddenlayers(hiddensize,
                                                               hiddenlayers - 1)),
                                   (a, b) -> a + b)
    else
        return Flux.LSTM(hiddensize, hiddensize)
    end
end

function makelstm(hiddensize::Integer, hiddenlayers::Integer, inputsize::Integer,
                  outputsize::Integer, dimensionality;
                  skipconnections::Bool=false, p_dropout=0.1f0,
                  output_activation=Flux.sigmoid)
    hiddenlayers = makehiddenlayers(hiddensize, hiddenlayers,
                                    Val(skipconnections), p_dropout)
    skipconnections && (hiddenlayers = (hiddenlayers,))
    model = Flux.Chain(
        Flux.LSTM(inputsize, hiddensize),
        hiddenlayers...,
        Flux.Dense(hiddensize, outputsize, output_activation)
    ) |> togpu
    LSTMModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :hiddensize        => hiddensize,
        :hiddenlayers      => hiddenlayers,
        :inputsize         => inputsize,
        :outputsize        => outputsize,
        :skipconnections   => skipconnections,
        :p_dropout         => p_dropout,
        :output_activation => output_activation,
    ))
end

function lstm1d(hiddensize::Integer=32, hiddenlayers::Integer=1,
                inputsize::Integer=inputsize1d,
                outputsize::Integer=outputsizeof(inputsize); p_dropout=0.05f0, kwargs...)
    makelstm(hiddensize, hiddenlayers, inputsize, outputsize, Symbol("1d");
             p_dropout=p_dropout, kwargs...)
end

function lstm2d(hiddensize::Integer=32, hiddenlayers::Integer=2,
                inputsize::Integer=inputsize2d,
                outputsize::Integer=outputsizeof(inputsize); p_dropout=0.05f0, kwargs...)
    makelstm(hiddensize, hiddenlayers, inputsize, outputsize, Symbol("2d");
             p_dropout=p_dropout, kwargs...)
end

function lstm3dtiles(hiddensize::Integer=64, hiddenlayers::Integer=2,
                     inputsize::Integer=inputsize3dtiles,
                     outputsize::Integer=outputsizeof(inputsize);
                     kwargs...)
    makelstm(hiddensize, hiddenlayers, inputsize, outputsize, Symbol("3dtiles"); kwargs...)
end

function lstm3d(hiddensize::Integer=128, hiddenlayers::Integer=2,
                inputsize::Integer=inputsize3d,
                outputsize::Integer=outputsizeof(inputsize); kwargs...)
    makelstm(hiddensize, hiddenlayers, inputsize, outputsize, Symbol("3d"); kwargs...)
end


end # module

