module WassersteinDiscriminator

using Statistics: mean

import Flux

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss
using ....GAN.Discriminator: buildmodel, manualmodel

export WassersteinDiscriminatorModel
export wsdiscriminator1d, wsdiscriminator2d, wsdiscriminator3dtiles, wsdiscriminator3d

# TODO Use constant input part as well! Maybe as parallel connection?

struct WassersteinDiscriminatorModel{M} <: AbstractDiscriminator
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike WassersteinDiscriminatorModel

(model::WassersteinDiscriminatorModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the discriminator to the batch of input images
`x` and return the loss of the predictions in relation to whether the images were real
(`y[i] == 1`).
"""
function makeloss(model::WassersteinDiscriminatorModel, ::Any)
    function loss(x, ::Any)
        # Discriminator loss
        y_hat = model(x)
        l = mean(y_hat)
        return l
    end
end

function wsdiscriminator1d(num_features=16, imgsize=imgsize1d; kwargs...)
    buildmodel(num_features, imgsize, Symbol("1d"); modeltype=WassersteinDiscriminatorModel,
               output_activation=identity, kernelsize=(4,), first_stride=2, kwargs...)
end

function wsdiscriminator2d(num_features=64, imgsize=imgsize2d; kwargs...)
    manualmodel(num_features, imgsize, Symbol("2d");
                modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                kwargs...)
end

function wsdiscriminator3dtiles(num_features=128, imgsize=imgsize3dtiles; kwargs...)
    manualmodel(num_features, imgsize, Symbol("3dtiles");
                modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                kwargs...)
end

function wsdiscriminator3d(num_features=256, imgsize=imgsize3d; kwargs...)
    manualmodel(num_features, imgsize, Symbol("3d");
                modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                kwargs...)
end

end # module

