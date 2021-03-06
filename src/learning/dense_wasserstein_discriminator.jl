module DenseWassersteinDiscriminator

using Statistics: mean

import Flux

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss, step!
using ....LSTM: makehiddenlayers
using ....WassersteinGAN.WassersteinDiscriminator: wsmakeloss, wsstep!

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
makeloss(model::DenseWassersteinDiscriminatorModel, ::Any) = wsmakeloss(model)

function step!(d_model::DenseWassersteinDiscriminatorModel, d_params, d_optim, d_loss,
               real_batch, real_target, g_model, fake_target, curr_batch_size)
    wsstep!(d_model, d_params, d_optim, d_loss,
            real_batch, real_target, g_model, fake_target, curr_batch_size)
end

function buildmodel(hiddensize::Integer, num_hiddenlayers::Integer, imgsize, dimensionality;
                    skipconnections::Bool=false, p_dropout=0.1f0, activation=Flux.relu,
                    output_activation=identity)
    inputsize = prod(Int, imgsize)
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
                                clamp_value=0.01f0, kwargs...)
    model = buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("1d"); kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

function densewsdiscriminator2d(hiddensize=64, num_hiddenlayers=4, imgsize=imgsize2d;
                                clamp_value=0.01f0, kwargs...)
    model = buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("2d"); kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

function densewsdiscriminator3dtiles(hiddensize=128, num_hiddenlayers=4,
                                     imgsize=imgsize3dtiles; clamp_value=0.01f0, kwargs...)
    model = buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("3dtiles"); kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

function densewsdiscriminator3d(hiddensize=256, num_hiddenlayers=4, imgsize=imgsize3d;
                                clamp_value=0.01f0, kwargs...)
    model = buildmodel(hiddensize, num_hiddenlayers, imgsize, Symbol("3d"); kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

end # module

