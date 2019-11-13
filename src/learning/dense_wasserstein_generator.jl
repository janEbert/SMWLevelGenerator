module DenseWassersteinGenerator

using Statistics: mean

import Flux

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss
using ....LSTM: makehiddenlayers

export DenseWassersteinGeneratorModel
export densewsgenerator1d, densewsgenerator2d, densewsgenerator3dtiles, densewsgenerator3d

struct DenseWassersteinGeneratorModel{M} <: AbstractGenerator
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike DenseWassersteinGeneratorModel

(model::DenseWassersteinGeneratorModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the generator to the random input `x` and return
the loss in relation to the discriminator's loss.
"""
function makeloss(g_model::DenseWassersteinGeneratorModel, d_model, ::Any)
    function loss(x, ::Any)
        # Generator loss
        fakes = g_model(x)
        y_hat = d_model(fakes)
        # The paper says `mean` but the code has nothing (implying `sum` for us)
        l = mean(y_hat)
        return l
    end
end

function buildmodel(hiddensize::Integer, num_hiddenlayers::Integer, imgsize,
                    generator_inputsize::Integer, dimensionality;
                    skipconnections::Bool=false, p_dropout=0.1f0, activation=Flux.relu,
                    output_activation=identity)
    hiddenlayers = makehiddenlayers(hiddensize, num_hiddenlayers, Val(skipconnections),
                                    p_dropout, (x, y) -> Flux.Dense(x, y, activation))
    matrixtobatch = dimensionality === Symbol("1d") ? MatrixTo2DBatch : MatrixTo3DBatch
    model = Flux.Chain(
        BatchToMatrix(generator_inputsize),
        Flux.Dense(generator_inputsize, hiddensize, activation),
        hiddenlayers...,
        Flux.Dense(hiddensize, prod(imgsize), output_activation),
        matrixtobatch(imgsize)
    ) |> togpu
    DenseWassersteinGeneratorModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,
        :inputsize      => generator_inputsize,

        :hiddensize        => hiddensize,
        :num_hiddenlayers  => num_hiddenlayers,
        :imgsize           => imgsize,
        :skipconnections   => skipconnections,
        :p_dropout         => p_dropout,
        :activation        => activation,
        :output_activation => output_activation,
    ))
end

function densewsgenerator1d(hiddensize=32, num_hiddenlayers=3, generator_inputsize=32,
                            imgsize=imgsize1d; kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, generator_inputsize, Symbol("1d");
               kwargs...)
end

function densewsgenerator2d(hiddensize=64, num_hiddenlayers=3, generator_inputsize=96,
                            imgsize=imgsize2d; kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, generator_inputsize, Symbol("2d");
               kwargs...)
end

function densewsgenerator3dtiles(hiddensize=128, num_hiddenlayers=4,
                                 generator_inputsize=256, imgsize=imgsize3dtiles; kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, generator_inputsize,
               Symbol("3dtiles"); kwargs...)
end

function densewsgenerator3d(hiddensize=256, num_hiddenlayers=4, generator_inputsize=512,
                            imgsize=imgsize3d; kwargs...)
    buildmodel(hiddensize, num_hiddenlayers, imgsize, generator_inputsize, Symbol("3d");
               kwargs...)
end

end # module

