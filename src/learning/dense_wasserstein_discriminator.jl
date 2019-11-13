module DenseWassersteinDiscriminator

using Statistics: mean

import Flux

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss
using ....LSTM: makehiddenlayers

export DenseWassersteinDiscriminatorModel
export densewsdiscriminator1d, densewsdiscriminator2d
export densewsdiscriminator3dtiles, densewsdiscriminator3d

# TODO Use constant input part as well! Maybe as parallel connection?

struct DenseWassersteinDiscriminatorModel{M} <: AbstractDiscriminator
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike DenseWassersteinDiscriminatorModel

(model::DenseWassersteinDiscriminatorModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the discriminator to the batch of input images
`x` and return the loss of the predictions in relation to whether the images were real
(`y[i] == 1`).
"""
function makeloss(model::DenseWassersteinDiscriminatorModel, ::Any)
    function loss(x, ::Any)
        # Discriminator loss
        y_hat = model(x)
        l = mean(y_hat)
        return l
    end
end

function buildmodel(hiddensize::Integer, num_hiddenlayers::Integer, imgsize, dimensionality;
                    skipconnections::Bool=false, p_dropout=0.1f0, activation=Flux.relu,
                    output_activation=identity)
    inputsize = prod(imgsize)
    hiddenlayers = makehiddenlayers(hiddensize, num_hiddenlayers, Val(skipconnections),
                                    p_dropout, (x, y) -> Flux.Dense(x, y, activation))
    matrixtobatch = dimensionality === Symbol("1d") ? MatrixTo2DBatch : MatrixTo3DBatch
    model = Flux.Chain(
        BatchToMatrix(inputsize),
        Flux.Dense(inputsize, hiddensize, activation),
        hiddenlayers...,
        Flux.Dense(hiddensize, 1, output_activation),
        matrixtobatch(map(_ -> 1, imgsize))
    ) |> togpu
    DenseWassersteinDiscriminatorModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :hiddensize          => hiddensize,
        :num_hiddenlayers    => num_hiddenlayers,
        :imgsize             => imgsize,
        :skipconnections     => skipconnections,
        :p_dropout           => p_dropout,
        :activation          => activation,
        :output_activation   => output_activation,
    ))
end

function densewsdiscriminator1d(hiddensize=32, num_hiddenlayers=3, imgsize=imgsize1d;
                                kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("1d"); kwargs...)
end

function densewsdiscriminator2d(hiddensize=64, num_hiddenlayers=3, imgsize=imgsize2d;
                                kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("2d"); kwargs...)
end

function densewsdiscriminator3dtiles(hiddensize=128, num_hiddenlayers=4,
                                     imgsize=imgsize3dtiles; kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("3dtiles"); kwargs...)
end

function densewsdiscriminator3d(hiddensize=256, num_hiddenlayers=4, imgsize=imgsize3d;
                                kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("3d"); kwargs...)
end

end # module

