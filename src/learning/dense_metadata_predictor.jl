"A model that tries to predict a level's metadata from the first screen."
module DenseMetadataPredictor

import Flux

using ..InputStatistics
using ..ModelUtils
import ..ModelUtils: makeloss
using ..LSTM: makehiddenlayers

export DenseMetadataModel
export densemetapredictor1d, densemetapredictor2d
export densemetapredictor3dtiles, densemetapredictor3d

struct DenseMetadataModel{M} <: LearningModel
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike DenseMetadataModel

(model::DenseMetadataModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the metadata predictor to the batch of input
images `x` and return the loss of the predictions in relation to the actual data.
"""
function makeloss(model::DenseMetadataModel, criterion)
    function loss(x, y)
        y_hat = model(x)
        @inbounds l = sum(@views criterion(vec(y_hat[:, i]), vec(y[:, i]))
                          for i in axes(y_hat, 2))
        return l
    end
end

function buildmodel(hiddensize::Integer, num_hiddenlayers::Integer, imgsize,
                    outputsize::Integer, dimensionality; skipconnections::Bool=false,
                    p_dropout=0.1f0, activation=Flux.relu,
                    output_activation=Flux.leakyrelu)
    inputsize = prod(Int, imgsize)
    hiddenlayers = makehiddenlayers(hiddensize, num_hiddenlayers, Val(skipconnections),
                                    p_dropout, (x, y) -> Flux.Dense(x, y, activation))
    model = Flux.Chain(
        BatchToMatrix(inputsize),
        Flux.Dense(inputsize, hiddensize, activation),
        hiddenlayers...,
        Flux.Dense(hiddensize, outputsize, output_activation)
    ) |> togpu
    DenseMetadataModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :hiddensize        => hiddensize,
        :num_hiddenlayers  => num_hiddenlayers,
        :imgsize           => imgsize,
        :outputsize        => outputsize,
        :skipconnections   => skipconnections,
        :p_dropout         => p_dropout,
        :activation        => activation,
        :output_activation => output_activation,
    ))
end

# TODO try out gelu

function densemetapredictor1d(hiddensize::Integer=32, num_hiddenlayers::Integer=1,
                              inputsize=imgsize1d, outputsize=constantinputsize;
                              p_dropout=0.05f0, kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, inputsize, outputsize, Symbol("1d");
               p_dropout=p_dropout, kwargs...)
end

function densemetapredictor2d(hiddensize::Integer=32, num_hiddenlayers::Integer=2,
                              inputsize=imgsize2d, outputsize=constantinputsize;
                              p_dropout=0.1f0, kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, inputsize, outputsize, Symbol("2d");
               p_dropout=p_dropout, kwargs...)
end

function densemetapredictor3dtiles(hiddensize::Integer=64, num_hiddenlayers::Integer=2,
                                   inputsize=imgsize3dtiles, outputsize=constantinputsize;
                                   p_dropout=0.1f0, kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, inputsize, outputsize, Symbol("3dtiles");
               p_dropout=p_dropout, kwargs...)
end

function densemetapredictor3d(hiddensize::Integer=128, num_hiddenlayers::Integer=2,
                              inputsize=imgsize3d, outputsize=constantinputsize;
                              p_dropout=0.1f0, kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, inputsize, outputsize, Symbol("3d");
               p_dropout=p_dropout, kwargs...)
end

end # module

