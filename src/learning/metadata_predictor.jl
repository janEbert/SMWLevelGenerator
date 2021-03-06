"A model that tries to predict a level's metadata from the first screen."
module MetadataPredictor

import Flux
using Flux.Tracker: gradient
#using Zygote

using ..InputStatistics
using ..ModelUtils
import ..ModelUtils: makeloss, step!

export MetadataModel
export metapredictor1d, metapredictor2d, metapredictor3dtiles, metapredictor3d

struct MetadataModel{M} <: LearningModel
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike MetadataModel

(model::MetadataModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the metadata predictor to the batch of input
images `x` and return the loss of the predictions in relation to the actual data.
"""
makeloss(model::MetadataModel, criterion) = metamakeloss(model, criterion)

function step!(model::MetadataModel, meta_params, meta_optim, meta_loss,
               real_batch, meta_batch)
    metastep!(model, meta_params, meta_optim, meta_loss, real_batch, meta_batch)
end

function metamakeloss(model::LearningModel, criterion)
    function loss(x, y)
        y_hat = model(x)
        @inbounds l = sum(@views criterion(vec(y_hat[:, i]), vec(y[:, i]))
                          for i in axes(y_hat, 2))
        return l
    end
end

function metastep!(model::LearningModel, meta_params, meta_optim, meta_loss,
                   real_batch, meta_batch)
    l = calculate_loss(model, meta_loss, real_batch, meta_batch)
    grads = gradient(() -> l, meta_params)
    Flux.Optimise.update!(meta_optim, meta_params, grads)
    return l
end

function manual1dmodel(num_features, imgsize, outputsize, dimensionality=Symbol("1d");
                       p_dropout=0.1f0, kernelsize=(3,),
                       output_activation=Flux.leakyrelu)
    imgchannels = imgsize[end]
    model = Flux.Chain(
        Flux.Conv(kernelsize, imgchannels => num_features, Flux.relu, pad=2, dilation=2),
        Flux.Conv(kernelsize, num_features => num_features * 2, Flux.relu,
                  pad=2, dilation=2),
        Flux.MaxPool(kernelsize, pad=1),
        Flux.Dropout(p_dropout),
        Flux.Conv(kernelsize, num_features * 2 => num_features * 4, Flux.relu,
                  pad=2, dilation=2),
        Flux.Conv(kernelsize, num_features * 4 => num_features * 8, Flux.relu, pad=0),
        Flux.MeanPool(kernelsize),
        Flux.Dropout(p_dropout),
        BatchToMatrix(num_features * 8),
        Flux.Dense(num_features * 8, outputsize, output_activation)
    ) |> togpu
    MetadataModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :num_features      => num_features,
        :imgsize           => imgsize,
        :outputsize        => outputsize,
        :p_dropout         => p_dropout,
        :kernelsize        => kernelsize,
        :output_activation => output_activation,
    ))
end

function manualmodel(num_features::Integer, imgsize, outputsize::Integer, dimensionality;
                     p_dropout=0.1f0, kernelsize=(3, 3),
                     output_activation=Flux.leakyrelu)
    imgchannels = imgsize[end]
    model = Flux.Chain(
        Flux.Conv(kernelsize, imgchannels => num_features, Flux.relu, pad=2, dilation=2),
        Flux.Conv(kernelsize, num_features => num_features * 2, Flux.relu,
                  pad=2, dilation=2),
        Flux.MaxPool(kernelsize, pad=1),
        Flux.Dropout(p_dropout),
        Flux.Conv(kernelsize, num_features * 2 => num_features * 4, Flux.relu,
                  pad=2, dilation=2),
        Flux.Conv(kernelsize, num_features * 4 => num_features * 8, Flux.relu,
                  pad=2, dilation=2),
        Flux.MaxPool(kernelsize, pad=1),
        Flux.Dropout(p_dropout),
        Flux.Conv(kernelsize, num_features * 8 => num_features * 8, Flux.relu,
                  pad=2, dilation=2),
        Flux.Conv(kernelsize, num_features * 8 => num_features * 4, Flux.relu,
                  pad=2, dilation=2),
        Flux.MeanPool(kernelsize, pad=1),
        BatchToMatrix(num_features * 4),
        Flux.Dense(num_features * 4, outputsize, output_activation)
    ) |> togpu
    MetadataModel(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :num_features      => num_features,
        :imgsize           => imgsize,
        :outputsize        => outputsize,
        :p_dropout         => p_dropout,
        :kernelsize        => kernelsize,
        :output_activation => output_activation,
    ))
end

# TODO try out gelu

function metapredictor1d(num_features::Integer=32, inputsize=imgsize1d,
                         outputsize=constantinputsize; p_dropout=0.1f0,
                         output_activation=Flux.leakyrelu)
    manual1dmodel(num_features, inputsize, outputsize, p_dropout=p_dropout,
                  output_activation=output_activation)
end

function metapredictor2d(num_features::Integer=64, inputsize=imgsize2d,
                         outputsize=constantinputsize; p_dropout=0.1f0,
                         output_activation=Flux.leakyrelu)
    manualmodel(num_features, inputsize, outputsize, Symbol("2d"),
                p_dropout=p_dropout, output_activation=output_activation)
end

function metapredictor3dtiles(num_features::Integer=128, inputsize=imgsize3dtiles,
                              outputsize=constantinputsize; p_dropout=0.1f0,
                              output_activation=Flux.leakyrelu)
    manualmodel(num_features, inputsize, outputsize, Symbol("3dtiles"),
                p_dropout=p_dropout, output_activation=output_activation)
end

function metapredictor3d(num_features::Integer=256, inputsize=imgsize3d,
                         outputsize=constantinputsize; p_dropout=0.1f0,
                         output_activation=Flux.leakyrelu)
    manualmodel(num_features, inputsize, outputsize, Symbol("3d"),
                p_dropout=p_dropout, output_activation=output_activation)
end

end # module

