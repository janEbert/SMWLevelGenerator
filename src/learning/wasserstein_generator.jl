module WassersteinGenerator

using Statistics: mean

import Flux
using Flux.Tracker: gradient
#using Zygote

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss, step!
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
makeloss(g_model::WassersteinGeneratorModel, d_model, ::Any) = wsmakeloss(g_model, d_model)

function step!(g_model::WassersteinGeneratorModel, g_params, g_optim, g_loss,
               real_target, curr_batch_size)
    wsstep!(g_model, g_params, g_optim, g_loss, real_target, curr_batch_size)
end

function wsmakeloss(g_model::AbstractGenerator, d_model)
    function loss(x, ::Any)
        # Generator loss
        fakes = g_model(x)
        y_hat = d_model(fakes)
        # The paper says `mean` but the code has nothing (implying `sum` for us)
        l = mean(y_hat)
        return l
    end
end

function wsstep!(g_model::AbstractGenerator, g_params, g_optim, g_loss,
                 real_target, curr_batch_size)
    GC.enable(false)
    noise_batch = makenoisebatch(g_model, curr_batch_size)

    # Use real labels for modified loss function
    g_l = calculate_loss(g_model, g_loss, noise_batch, real_target)
    g_l = -g_l

    grads = gradient(() -> g_l, g_params)

    # Update
    Flux.Optimise.update!(g_optim, g_params, grads)
    GC.enable(true)
    return g_l
end

function wsgenerator1d(num_features=16, generator_inputsize=32, imgsize=imgsize1d; kwargs...)
    buildmodel(num_features, imgsize, generator_inputsize, Symbol("1d");
               modeltype=WassersteinGeneratorModel, kernelsize=(4,), kwargs...)
end

function wsgenerator2d(num_features=64, generator_inputsize=96, imgsize=imgsize2d; kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("2d");
                modeltype=WassersteinGeneratorModel, kwargs...)
end

function wsgenerator3dtiles(num_features=128, generator_inputsize=256, imgsize=imgsize3dtiles;
                          kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("3dtiles");
                modeltype=WassersteinGeneratorModel, kwargs...)
end

function wsgenerator3d(num_features=256, generator_inputsize=512, imgsize=imgsize3d;
                     kwargs...)
    manualmodel(num_features, imgsize, generator_inputsize, Symbol("3d");
                modeltype=WassersteinGeneratorModel, kwargs...)
end

end # module

