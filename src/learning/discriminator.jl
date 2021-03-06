module Discriminator

import Flux

using ....InputStatistics
using ....ModelUtils

export DiscriminatorModel
export discriminator1d, discriminator2d, discriminator3dtiles, discriminator3d

# TODO Use constant input part as well! Maybe as parallel connection?

struct DiscriminatorModel{M} <: AbstractDiscriminator
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike DiscriminatorModel

(model::DiscriminatorModel)(input) = model.model(input)

function powers(n, b=2)
    res = []
    x = 1
    for i in 1:n
        x_old = x
        x *= b
        push!(res, (x_old, x))
    end
    return res
end

function buildmodel(num_features, imgsize, dimensionality; modeltype=DiscriminatorModel,
                    normalization=Flux.BatchNorm, activation=leakyrelu,
                    output_activation=Flux.sigmoid, kernelsize=(4, 4), first_stride=(2, 1))
    imgchannels = imgsize[end]
    layers = Any[ConvNoBias(kernelsize, imgchannels => num_features, activation,
                         stride=first_stride, pad=1)]
    local last_j
    for (i, j) in powers(convert(Int, log2(imgsize[1])) - 3)
        append!(layers, (
            ConvNoBias(kernelsize, num_features * i => num_features * j, stride=2, pad=1),
            normalization(num_features * j, activation)
        ))
        last_j = j
    end
    push!(layers, ConvNoBias(kernelsize, num_features * last_j => 1, output_activation,
                             stride=1, pad=0))
    model = Flux.Chain(layers...) |> togpu
    modeltype(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :num_features      => num_features,
        :imgsize           => imgsize,
        :normalization     => normalization,
        :activation        => activation,
        :output_activation => output_activation,
        :kernelsize        => kernelsize,
    ))
end

function manualmodel(num_features, imgsize, dimensionality; modeltype=DiscriminatorModel,
                     normalization=Flux.BatchNorm, activation=leakyrelu,
                     output_activation=Flux.sigmoid, kernelsize=(4, 4))
    imgchannels = imgsize[end]
    model = Flux.Chain(
        # last dimension (amount of batches) is ignored in these comments
        # input size is 27 x 16 x (imgchannels) (actually `imgsize`)
        ConvNoBias(kernelsize, imgchannels => num_features, activation, stride=(2, 1),
                   pad=1),
        # state size is 13 x 15 x (num_features)
        ConvNoBias(kernelsize, num_features => num_features * 2, stride=2, pad=1),
        normalization(num_features * 2, activation),
        # state size is 6 x 7 x (num_features * 2)
        ConvNoBias(kernelsize, num_features * 2 => num_features * 4, stride=2, pad=1),
        normalization(num_features * 4, activation),
        # state size is 3 x 3 x (num_features * 4)
        ConvNoBias(kernelsize, num_features * 4 => 1, output_activation, stride=2, pad=1)
        # output size is 1 x 1 x 1 (scalar value)
    ) |> togpu
    modeltype(model, Dict{Symbol, Any}(
        :dimensionality => dimensionality,

        :num_features      => num_features,
        :imgsize           => imgsize,
        :normalization     => normalization,
        :activation        => activation,
        :output_activation => output_activation,
        :kernelsize        => kernelsize,
    ))
end

function discriminator1d(num_features=16, imgsize=imgsize1d; kwargs...)
    buildmodel(num_features, imgsize, Symbol("1d");
               kernelsize=(4,), first_stride=2, kwargs...)
end

function discriminator2d(num_features=64, imgsize=imgsize2d; kwargs...)
    manualmodel(num_features, imgsize, Symbol("2d"); kwargs...)
end

function discriminator3dtiles(num_features=128, imgsize=imgsize3dtiles; kwargs...)
    manualmodel(num_features, imgsize, Symbol("3dtiles"); kwargs...)
end

function discriminator3d(num_features=256, imgsize=imgsize3d; kwargs...)
    manualmodel(num_features, imgsize, Symbol("3d"); kwargs...)
end

end # module

