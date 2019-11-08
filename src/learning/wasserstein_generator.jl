module WassersteinGenerator

using Statistics: mean

import Flux

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss
using ....GAN.Generator: buildmodel, manualmodel

export WassersteinGeneratorModel
export wsgenerator1d, wsgenerator2d, wsgenerator3dtiles, wsgenerator3d

struct WassersteinGeneratorModel{M} <: AbstractGenerator
    model::M
    hyperparams::Dict{Symbol, Any}
end

Flux.@treelike WassersteinGeneratorModel

(model::WassersteinGeneratorModel)(input) = model.model(input)

"""
Return a 2-argument loss function applying the generator to the random input `x` and return
the loss in relation to the discriminator's loss.
"""
function makeloss(g_model::WassersteinGeneratorModel, d_model, ::Any)
    function loss(x, ::Any)
        # Generator loss
        fakes = g_model(x)
        y_hat = d_model(fakes)
        # The paper says `mean` but the code has nothing (implying `sum` for us)
        l = mean(y_hat)
        return l
    end
end

function wsgenerator1d(num_features=16, generator_inputsize=32, imgsize=imgsize1d; kwargs...)
    buildmodel(num_features, imgsize, generator_inputsize, Symbol("1d");
               modeltype=WassersteinGeneratorModel, kernelsize=(4,), kwargs...)
end

function wsgenerator2d(num_features=64, generator_inputsize=64, imgsize=imgsize2d; kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("2d");
                modeltype=WassersteinGeneratorModel, kwargs...)
end

function wsgenerator3dtiles(num_features=128, generator_inputsize=128, imgsize=imgsize3dtiles;
                          kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("3dtiles");
                modeltype=WassersteinGeneratorModel, kwargs...)
end

function wsgenerator3d(num_features=256, generator_inputsize=128, imgsize=imgsize3d;
                     kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("3d");
                modeltype=WassersteinGeneratorModel, kwargs...)
end

end # module

