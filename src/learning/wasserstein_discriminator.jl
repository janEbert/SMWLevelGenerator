module WassersteinDiscriminator

using Statistics: mean

import Flux
using Flux.Tracker: gradient
#using Zygote

using ....InputStatistics
using ....ModelUtils
import ....ModelUtils: makeloss, step!
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
makeloss(model::WassersteinDiscriminatorModel, ::Any) = wsmakeloss(model)

function step!(d_model::WassersteinDiscriminatorModel, d_params, d_optim, d_loss,
               real_batch, real_target, g_model, fake_target, curr_batch_size)
    wsstep!(d_model, d_params, d_optim, d_loss,
            real_batch, real_target, g_model, fake_target, curr_batch_size)
end

function wsmakeloss(model::AbstractDiscriminator)
    function loss(x, ::Any)
        # Discriminator loss
        y_hat = model(x)
        l = mean(y_hat)
        return l
    end
end

function wsstep!(d_model::AbstractDiscriminator, d_params, d_optim, d_loss,
                 real_batch, real_target, g_model, fake_target, curr_batch_size)
    # Code clamps weights here

    # Train on real batch
    d_l_real = calculate_loss(d_model, d_loss, real_batch, real_target)

    # Train on fake batch
    noise_batch = makenoisebatch(g_model, curr_batch_size)
    fake_batch = g_model(noise_batch)

    d_l_fake = calculate_loss(d_model, d_loss, fake_batch, fake_target)
    d_l = d_l_fake - d_l_real

    grads = gradient(() -> d_l, d_params)

    # Update
    Flux.Optimise.update!(d_optim, d_params, grads)

    # Paper clamps weights here
    # Clamp weights
    clamp_value = d_model.hyperparams[:clamp_value]
    for p in d_params
        clamp!(p.data, -clamp_value, clamp_value)
    end

    return d_l, d_l_real, d_l_fake
end

function wsdiscriminator1d(num_features=16, imgsize=imgsize1d;
                           clamp_value=0.01f0, kwargs...)
    model = buildmodel(num_features, imgsize, Symbol("1d");
                       modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                       kernelsize=(4,), first_stride=2, kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

function wsdiscriminator2d(num_features=64, imgsize=imgsize2d;
                           clamp_value=0.01f0, kwargs...)
    model = manualmodel(num_features, imgsize, Symbol("2d");
                        modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                        kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

function wsdiscriminator3dtiles(num_features=128, imgsize=imgsize3dtiles;
                                clamp_value=0.01f0, kwargs...)
    model = manualmodel(num_features, imgsize, Symbol("3dtiles");
                        modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                        kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

function wsdiscriminator3d(num_features=256, imgsize=imgsize3d;
                           clamp_value=0.01f0, kwargs...)
    model = manualmodel(num_features, imgsize, Symbol("3d");
                        modeltype=WassersteinDiscriminatorModel, output_activation=identity,
                        kwargs...)
    model.hyperparams[:clamp_value] = clamp_value
    return model
end

end # module

