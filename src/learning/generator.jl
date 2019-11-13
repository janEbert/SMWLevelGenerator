module Generator

import Flux

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss

export GeneratorModel
export generator1d, generator2d, generator3dtiles, generator3d

# TODO implement virtual batch norm

struct GeneratorModel{M} <: AbstractGenerator
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike GeneratorModel

(model::GeneratorModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the generator to the random input `x` and return
the loss in relation to the discriminator's loss.
"""
function makeloss(g_model::GeneratorModel, d_model, criterion)
    function loss(x, y)
        # Generator loss
        fakes = g_model(x)
        y_hat = d_model(fakes)
        l = sum(criterion.(y_hat, y))
        return l
    end
end

function buildmodel(num_features, imgsize, generator_inputsize, dimensionality;
                    modeltype=GeneratorModel, normalization=Flux.BatchNorm,
                    activation=Flux.relu, output_activation=sigmoid, kernelsize=(4, 4))
    imgchannels = imgsize[end]
    j = 2 ^ (convert(Int, log2(imgsize[1])) - 3)
    layers = [
        ConvTransposeNoBias(kernelsize, generator_inputsize => num_features * j,
                            stride=1, pad=0),
        normalization(num_features * j, activation)
    ]
    while j > 1
        i = j
        j ÷= 2
        append!(layers, (
            ConvTransposeNoBias(kernelsize, num_features * i => num_features * j,
                                stride=2, pad=1),
            normalization(num_features * j, activation)
        ))
    end
    push!(layers, ConvTransposeNoBias(kernelsize, num_features => imgchannels,
                                      output_activation, stride=2, pad=1))
    model = Flux.Chain(layers...) |> togpu
    modeltype(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,
        :inputsize      => generator_inputsize,

        :num_features      => num_features,
        :imgsize           => imgsize,
        :normalization     => normalization,
        :activation        => activation,
        :output_activation => output_activation,
        :kernelsize        => kernelsize,
    ))
end

function manualmodel(num_features, imgsize, generator_inputsize, dimensionality;
                     modeltype=GeneratorModel, normalization=Flux.BatchNorm,
                     activation=Flux.relu, output_activation=sigmoid, kernelsize=(4, 4))
    imgchannels = imgsize[end]
    model = Flux.Chain(
        # last dimension (amount of batches) is ignored in these comments
        # input size is 1 x 1 x (generator_inputsize)
        ConvTransposeNoBias(kernelsize, generator_inputsize => num_features * 4,
                            stride=1, pad=2, dilation=2),
        normalization(num_features * 4, activation),
        # state size is 3 x 3 x (num_features * 4)
        ConvTransposeNoBias(kernelsize, num_features * 4 => num_features * 2,
                            stride=2, pad=1),
        normalization(num_features * 2, activation),
        # state size is 6 x 7 x (num_features * 2)
        ConvTransposeNoBias(kernelsize, num_features * 2 => num_features,
                            stride=2, pad=1, dilation=2),
        normalization(num_features, activation),
        # state size is 13 x 15 x (num_features)
        ConvTransposeNoBias(kernelsize, num_features => imgchannels, output_activation,
                            stride=(2,1), pad=4, dilation=(2, 3)),
        # output size is 27 x 16 x (imgchannels)
    ) |> togpu
    modeltype(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,
        :inputsize      => generator_inputsize,

        :num_features      => num_features,
        :imgsize           => imgsize,
        :normalization     => normalization,
        :activation        => activation,
        :output_activation => output_activation,
        :kernelsize        => kernelsize,
    ))
end

function generator1d(num_features=16, generator_inputsize=32, imgsize=imgsize1d; kwargs...)
    buildmodel(num_features, imgsize, generator_inputsize, Symbol("1d");
               kernelsize=(4,), kwargs...)
end

function generator2d(num_features=64, generator_inputsize=96, imgsize=imgsize2d; kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("2d"); kwargs...)
end

function generator3dtiles(num_features=128, generator_inputsize=256, imgsize=imgsize3dtiles;
                          kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("3dtiles"); kwargs...)
end

function generator3d(num_features=256, generator_inputsize=512, imgsize=imgsize3d;
                     kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("3d"); kwargs...)
end

end # module

