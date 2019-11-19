module GANTraining

using Dates: now
using Logging: SimpleLogger, Info
using Printf: @sprintf
using Statistics: mean, var
using Random: seed!, shuffle!

using BSON
import Flux
using Flux.Tracker: gradient
using JLD2
import JSON
using TensorBoardLogger

using ..DataIterator
using ..GAN
using ..WassersteinGAN
using ..MetadataPredictor
using ..ModelUtils
using ..TrainingUtils

export gan_trainingloop!, GANTrainingParameters, GTPs

Base.@kwdef struct GANTrainingParameters
    epochs::Integer = 10

    lr::Float64                           = 0.0002
    betas::Tuple{Float64, Float64}        = (0.5, 0.999)
    batch_size::Integer                   = 32
    use_wasserstein_loss::Bool            = true
    d_warmup_steps::Integer               = 0
    d_steps_per_g_step::Integer           = 1
    logevery::Integer                     = 200
    saveevery::Integer                    = 1000
    use_bson::Bool                        = false
    buffer_size::Integer                  = 3
    dataiter_threads::Integer             = 0
    logdir::AbstractString                = joinpath("exps", newexpdir("gan"))
    params_logfile::AbstractString        = "params.json"
    logfile::AbstractString               = "training.log"
    earlystoppingwaitepochs::Integer      = 10
    earlystoppingthreshold::AbstractFloat = Inf32
    criterion::Function                   = Flux.binarycrossentropy
    overfit_on_batch::Bool                = false
    meta_lr::Float64                      = 0.001
    meta_criterion::Function              = Flux.mse
    meta_testratio::AbstractFloat         = 0.1
    seed::Integer                         = 0
end

"Shorthand for GANTrainingParameters for interactive use."
const GTPs = GANTrainingParameters

# TODO probably remove optional meta_model training

function gan_trainingloop!(d_model::Union{AbstractDiscriminator, AbstractString},
                           g_model::Union{AbstractGenerator, AbstractString},
                           dbpath::AbstractString,
                           params::GANTrainingParameters=GANTrainingParameters(),
                           meta_model::Union{LearningModel, AbstractString, Nothing}=nothing)
    seed!(params.seed)
    set_zero_subnormals(true)

    paramdict = Dict{Symbol, Any}(field => getproperty(params, field) |>
                                  x -> x isa Function ? Symbol(x) : x
                                  for field in propertynames(params))
    paramdict[:dbpath] = dbpath

    db = loaddb(dbpath)
    overfit_on_batch = params.overfit_on_batch
    if overfit_on_batch
        if overfit_on_batch isa Bool
            trainindices = collect(one(UInt):convert(UInt, 16))
        else
            trainindices = collect(one(UInt):convert(UInt, overfit_on_batch))
        end
        testindices = @view trainindices[firstindex(trainindices):end]
    else
        trainindices = collect(one(UInt):convert(UInt, length(db)))
        testindices = view(trainindices,
                           firstindex(trainindices):round(UInt, length(trainindices)
                                                          * (1 - params.meta_testratio)))
    end
    shuffle!(trainindices)

    lr    = params.lr
    betas = params.betas
    length(lr) > 1 || (lr = (lr, lr))
    length(betas[1]) > 1 || (betas = (betas, betas))

    use_bson = Val(params.use_bson)

    if d_model isa AbstractString
        paramdict[:d_modelpath] = d_model
        (d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses,
                d_steps) = load_d_cp(d_model, use_bson)
    else
        d_optim = Flux.ADAM(lr[1], betas[1])

        d_trainlosses_real = Float32[]
        d_trainlosses_fake = eltype(d_trainlosses_real)[]
        d_testlosses = eltype(d_trainlosses_real)[]

        d_steps = UInt64(0)
    end
    paramdict[:d_modelparams] = Dict{Symbol, Any}(k => v isa Function ? Symbol(v) : v
                                                  for (k, v) in d_model.hyperparams)

    local testfakes
    if g_model isa AbstractString
        paramdict[:g_modelpath] = g_model
        (g_model, g_optim, g_trainlosses, testfakes, const_noise, past_steps) = load_g_cp(
                g_model, use_bson)
        generator_inputsize = g_model.hyperparams[:inputsize]
    else
        g_optim = Flux.ADAM(lr[2], betas[2])
        generator_inputsize = g_model.hyperparams[:inputsize]

        g_trainlosses = eltype(d_trainlosses_real)[]
        if g_model.hyperparams[:dimensionality] === Symbol("1d")
            const_noise = togpu(randn(1, generator_inputsize, 16))
        else
            const_noise = togpu(randn(1, 1, generator_inputsize, 16))
        end

        past_steps = d_steps
    end
    paramdict[:g_modelparams] = Dict{Symbol, Any}(k => v isa Function ? Symbol(v) : v
                                                  for (k, v) in g_model.hyperparams)

    meta_steps = UInt64(0)
    if !isnothing(meta_model)
        if meta_model isa AbstractString
            paramdict[:meta_modelpath] = meta_model
            (meta_model, meta_optim, meta_trainlosses, meta_meanlosses,
                    meta_varlosses, meta_steps) = load_meta_cp(meta_model, use_bson)
        else
            meta_optim = Flux.ADAM(params.meta_lr)

            meta_trainlosses = eltype(d_trainlosses_real)[]
            meta_meanlosses  = eltype(meta_trainlosses)[]
            meta_varlosses   = eltype(meta_trainlosses)[]
        end
        meta_loss = makeloss(meta_model, params.meta_criterion)
        meta_params = Flux.params(meta_model)

        maxmeanloss = typemin(eltype(meta_trainlosses))
        maxvarloss = typemin(eltype(meta_trainlosses))

        paramdict[:meta_modelparams] = Dict{Symbol, Any}(
            k => v isa Function ? Symbol(v) : v for (k, v) in meta_model.hyperparams)
    end

    epochs               = params.epochs
    batch_size           = params.batch_size
    use_wasserstein_loss = params.use_wasserstein_loss
    d_warmup_steps       = params.d_warmup_steps
    d_steps_per_g_step   = params.d_steps_per_g_step
    logdir               = params.logdir
    logevery             = params.logevery
    saveevery            = params.saveevery

    max_d_loss = typemin(eltype(d_trainlosses_real))
    max_d_testloss = typemin(eltype(d_testlosses))
    max_g_loss = typemin(eltype(g_trainlosses))
    maxvarlossdigits = 0  # Predefined in case `varloss` is NaN.
    local g_l
    local max_d_lossdigits
    local max_d_testlossdigits
    local max_g_lossdigits
    local maxmeanlossdigits
    const_fake_target = togpu(zeros(size(const_noise)[end]))
    steps = UInt64(0)

    trainiter = gan_dataiterator(db, params.buffer_size, trainindices, batch_size,
                                 params.dataiter_threads)
    testiter  = gan_dataiterator(db, params.buffer_size, testindices,  batch_size,
                                 params.dataiter_threads)
    curr_batch_size = 0
    curr_batch_size_changed = false
    local real_target
    local fake_target

    d_loss = makeloss(d_model, params.criterion)
    g_loss = makeloss(g_model, d_model, params.criterion)
    d_params = Flux.params(d_model)
    g_params = Flux.params(g_model)

    earlystoppingthreshold  = params.earlystoppingthreshold
    earlystoppingthreshold < 1 && (earlystoppingthreshold += 1)
    earlystoppingwaitepochs = params.earlystoppingwaitepochs

    mkpath(logdir)
    # Save parameters
    open(joinpath(logdir, params.params_logfile), "w") do io
        JSON.print(io, paramdict, 4)
    end
    # Free `paramdict`
    paramdict = nothing
    log_io = open(joinpath(logdir, params.logfile), "w")

    @fastmath try
        logger = SimpleLogger(log_io)
        tblogger = TBLogger(joinpath(logdir, "tensorboard"), min_level=Info)
        d_steps > 0 && logprint(logger, "Loaded discriminator with $d_steps steps.")
        meta_steps > 0 && logprint(logger, "Loaded meta predictor with $meta_steps steps.")

        # To get local time instead of UTC for printing and filenames:
        starttimestr = replace(string(now()), ':' => '-')
        starttime = time()
        logprint(logger, "Starting GAN training at $starttimestr for $epochs epochs. "
                 * "Seed: $(params.seed).")

        if past_steps == 0
            testfakes = [Flux.data(g_model(const_noise))]
        else
            push!(testfakes, Flux.data(g_model(const_noise)))
        end
        testloss = testmodel(d_model, d_loss, testfakes[end], const_fake_target)
        if past_steps > 0
            @tblog tblogger log_step_increment=convert(Int, past_steps)
            pop!(testfakes)
        end
        if d_steps == 0
            @tblog tblogger d_testloss=testloss log_step_increment=0
            push!(d_testlosses, testloss)
        end

        if !isnothing(meta_model)
            testlosses = testmodel(meta_model, testiter, testindices, batch_size, meta_loss)
            meanloss = mean(testlosses)
            varloss  = var(testlosses, mean=meanloss)
            if meta_steps == 0
                @tblog(tblogger, meta_predictor_meanloss=meanloss,
                       meta_predictor_varloss=varloss, log_step_increment=0)
                push!(meta_meanlosses, meanloss)
                push!(meta_varlosses,  varloss)
            end
        end

        timediff = time() - starttime
        if !isnothing(meta_model)
            logprint(logger, "Initial discriminator test loss: "
                     * "$(@sprintf("%.4f", testloss)); metamodel mean test loss: "
                     * "$(@sprintf("%.4f", meanloss)) "
                     * "(variance: $(@sprintf("%.3f", varloss))); "
                     * "total time: $(@sprintf("%.2f", timediff / 60)) min.")
        else
            logprint(logger, "Initial discriminator test loss: "
                     * "$(@sprintf("%.4f", testloss)); "
                     * "total time: $(@sprintf("%.2f", timediff / 60)) min.")
        end

        for epoch in 1:epochs
            for (i, j) in zip(1:cld(length(trainindices), batch_size),
                              Iterators.countfrom(1, batch_size))
                if steps < past_steps
                    take!(trainiter)
                    steps += 1
                    continue
                end

                real_batch, meta_batch = map(togpu, take!(trainiter))
                curr_batch_size_changed && (curr_batch_size_changed = false)
                curr_batch_size == size(meta_batch, 2) || (curr_batch_size_changed = true)
                curr_batch_size = size(meta_batch, 2)

                # Only generate these anew if necessary.
                if curr_batch_size_changed
                    real_target = togpu(ones(curr_batch_size))
                    fake_target = togpu(zeros(curr_batch_size))
                end


                # Discriminator

                d_training_step!(d_model, d_params, d_optim, d_loss,
                                 real_batch, real_target, g_model, fake_target,
                                 generator_inputsize, curr_batch_size,
                                 d_trainlosses_real, d_trainlosses_fake, tblogger)


                if steps % d_steps_per_g_step == 0 && (steps > d_warmup_steps || steps == 0)
                    # Generator
                    g_training_step!(g_model, g_params, g_optim, g_loss,
                                     real_target, curr_batch_size, g_trainlosses, tblogger)
                end


                # Metadata predictor
                if !isnothing(meta_model) && (overfit_on_batch || j > length(testindices))
                    meta_training_step!(meta_model, meta_params, meta_optim, meta_loss,
                                        real_batch, meta_batch, meta_trainlosses, tblogger)
                end


                steps += 1

                if logevery != 0 && steps % logevery == 0
                    push!(testfakes, Flux.data(g_model(const_noise)))
                    testloss = testmodel(d_model, d_loss, testfakes[end], const_fake_target)
                    @tblog tblogger d_testloss=testloss log_step_increment=0
                    if d_l > max_d_loss
                        max_d_loss = d_l
                        max_d_lossdigits = ndigits(trunc(Int, max_d_loss)) + 5
                    end
                    if testloss > max_d_testloss
                        max_d_testloss = testloss
                        max_d_testlossdigits = ndigits(trunc(Int, max_d_testloss)) + 5
                    end
                    if g_l > max_g_loss
                        max_g_loss = Flux.data(g_l)
                        max_g_lossdigits = ndigits(trunc(Int, max_g_loss)) + 5
                    end
                    push!(d_testlosses, testloss)
                    if length(d_testlosses) > 1
                        lossdiff  = testloss - d_testlosses[end - 1]
                        lossratio = testloss / d_testlosses[end - 1]
                    else
                        lossdiff = 0
                    end

                    if !isnothing(meta_model)
                        testlosses = testmodel(meta_model, testiter, testindices,
                                               batch_size, meta_loss)
                        meanloss = mean(testlosses)
                        varloss  = var(testlosses, mean=meanloss)
                        @tblog(tblogger, meta_predictor_meanloss=meanloss,
                               meta_predictor_varloss=varloss, log_step_increment=0)
                        if meanloss > maxmeanloss
                            maxmeanloss = meanloss
                            maxmeanlossdigits = ndigits(trunc(Int, maxmeanloss)) + 5
                        end
                        if varloss > maxvarloss
                            maxvarloss = varloss
                            maxvarlossdigits = ndigits(trunc(Int, maxvarloss)) + 4
                        end
                        push!(meta_meanlosses, meanloss)
                        push!(meta_varlosses,  varloss)
                    end

                    timediff = time() - starttime
                    logprint(logger, "Epoch $(lpad(epoch, ndigits(epochs))) / "
                             * "$epochs; sequence "
                             * "$(lpad(j, ndigits(length(trainindices)))) / "
                             * "$(length(trainindices)); discriminator loss: "
                             * "$(lpad(@sprintf("%.4f", testloss), max_d_testlossdigits)) "
                             * "test, $(lpad(@sprintf("%.4f", d_l), max_d_lossdigits)) "
                             * "train; generator loss: "
                             * "$(lpad(@sprintf("%.4f", Flux.data(g_l)), max_g_lossdigits))"
                             * "; mean time per step: "
                             * "$(@sprintf("%.3f", timediff / steps)) s; "
                             * "total time: $(@sprintf("%.2f", timediff / 60)) min.")
                    if !isnothing(meta_model)
                        logprint(logger, "Metamodel mean test loss: "
                                 * "$(lpad(@sprintf("%.4f", meanloss), maxmeanlossdigits)) "
                                 * "(variance: "
                                 * "$(lpad(@sprintf("%.3f", varloss), maxvarlossdigits))).")
                    end

                    # Early stopping
                    if (epoch > earlystoppingwaitepochs && lossdiff > 0
                            && lossratio >= earlystoppingthreshold)
                        save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake,
                                  d_testlosses, steps, logdir, starttimestr, use_bson)
                        save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                                  steps, testloss, logdir, starttimestr, use_bson)
                        if !isnothing(meta_model)
                            save_meta_cp(meta_model, meta_optim, meta_trainlosses,
                                         meta_meanlosses, meta_varlosses, steps,
                                         logdir, starttimestr, use_bson)
                        end
                        logprint(logger, "Early stopping activated after $steps training "
                                 * "steps ($epoch epochs, $j sequences in current epoch). "
                                 * "Loss increase: $testloss - $(d_testlosses[end - 1]) = "
                                 * "$(round(lossdiff, digits=3)) "
                                 * "($(round((lossratio - 1) * 100, digits=2)) %). "
                                 * "Total time: $(round(timediff / 60, digits=2)) min.")
                        cleanupall(trainiter, testiter, log_io)
                        return (d_model, g_model, d_trainlosses_real,
                                d_trainlosses_fake, d_testlosses, g_trainlosses, testfakes,
                                db, trainindices, const_noise)
                    end
                end
                if saveevery != 0 && steps % saveevery == 0
                    if logevery != 0 && steps % logevery != 0
                        push!(testfakes, Flux.data(g_model(const_noise)))
                        testloss = testmodel(d_model, d_loss, testfakes[end],
                                             const_fake_target)
                        @tblog tblogger d_testloss=testloss log_step_increment=0
                        push!(d_testlosses, testloss)

                        if !isnothing(meta_model)
                            testlosses = testmodel(meta_model, testiter, testindices,
                                                   batch_size, meta_loss)
                            meanloss = mean(testlosses)
                            varloss  = var(testlosses, mean=meanloss)
                            @tblog(tblogger, meta_predictor_meanloss=meanloss,
                                   meta_predictor_varloss=varloss, log_step_increment=0)
                            push!(meta_meanlosses, meanloss)
                            push!(meta_varlosses,  varloss)
                        end
                    end
                    save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake,
                              d_testlosses, steps, logdir, starttimestr, use_bson)
                    save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                              steps, testloss, logdir, starttimestr, use_bson)
                    if !isnothing(meta_model)
                        save_meta_cp(meta_model, meta_optim, meta_trainlosses,
                                     meta_meanlosses, meta_varlosses, steps,
                                     logdir, starttimestr, use_bson)
                    end
                    logprint(logger, "Saved checkpoints after $steps training steps.")
                end
            end
        end
        save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake,
                  d_testlosses, steps, logdir, starttimestr, use_bson)
        save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                  steps, testloss, logdir, starttimestr, use_bson)
        if !isnothing(meta_model)
            save_meta_cp(meta_model, meta_optim, meta_trainlosses, meta_meanlosses,
                         meta_varlosses, steps, logdir, starttimestr, use_bson)
        end
        logprint(logger, "Training finished after $steps training steps and "
                 * "$(round((time() - starttime) / 60, digits=2)) minutes.")
    finally
        cleanupall(trainiter, testiter, log_io)
    end
    return (d_model, g_model, d_trainlosses_real, d_trainlosses_fake,
            d_testlosses, g_trainlosses, testfakes, db, trainindices, const_noise)
end


function d_training_step!(d_model, d_params, d_optim, d_loss,
                          real_batch, real_target, g_model, fake_target, curr_batch_size,
                          d_trainlosses_real, d_trainlosses_fake, tblogger)
    d_l_real, d_l_fake = map(Flux.data,
                             step!(d_model, d_params, d_optim, d_loss,
                                   real_batch, real_target, g_model, fake_target,
                                   curr_batch_size))

    push!(d_trainlosses_real, d_l_real)
    @tblog tblogger d_loss_real=d_l_real
    push!(d_trainlosses_fake, d_l_fake)
    @tblog tblogger d_loss_fake=d_l_fake log_step_increment=0

    return d_l_real, d_l_fake
end

function g_training_step!(g_model, g_params, g_optim, g_loss, real_target, curr_batch_size,
                          g_trainlosses, tblogger)
    g_l = Flux.data(step!(g_model, g_params, g_optim, g_loss, real_target, curr_batch_size))
    push!(g_trainlosses, g_l)
    @tblog tblogger g_loss=g_l log_step_increment=0
    return g_l
end

function meta_training_step!(meta_model, meta_params, meta_optim, meta_loss,
                             real_batch, meta_batch, meta_trainlosses, tblogger)
    l = Flux.data(step!(meta_model, meta_params, meta_optim, meta_loss,
                        real_batch, meta_batch))
    push!(meta_trainlosses, l)
    @tblog tblogger trainloss=l log_step_increment=0
    return l
end

function testmodel(d_model, d_loss, fake_batch, const_fake_target)
    Flux.testmode!(d_model)
    d_l = Flux.data(d_loss(fake_batch, const_fake_target))
    Flux.testmode!(d_model, false)
    return d_l
end

function testmodel(meta_model, testiter, testindices, batch_size, meta_loss)
    Flux.testmode!(meta_model)
    testlosses = Float32[]

    for i in 1:cld(length(testindices), batch_size)
        real_batch, meta_batch = map(togpu, take!(testiter))
        l = Flux.data(meta_loss(real_batch, meta_batch))
        push!(testlosses, l)
    end
    Flux.testmode!(meta_model, false)
    return testlosses
end

# Numbered Vararg so we can make sure we didn't miss one without having to list them all.
cleanupall(args::Vararg{Any, 3}) = foreach(cleanup, args)

function save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses,
                   steps, logdir, starttimestr, use_bson::Val{true})
    bson(joinpath(logdir, "discriminator-cp_$steps-steps_loss-"
                  * "$(d_testlosses[end])_$starttimestr.bson"),
         d_model=tocpu(d_model), d_optim=tocpu(d_optim),
         d_trainlosses_real=d_trainlosses_real,
         d_trainlosses_fake=d_trainlosses_fake,
         d_testlosses=d_testlosses, steps=steps)
end

function save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses,
                   steps, logdir, starttimestr, use_bson::Val{false})
    # We use this signature so we don't use `mmap`, trading speed for stability.
    # See https://github.com/JuliaIO/JLD2.jl/issues/55
    jldopen(joinpath(logdir, "discriminator-cp_$steps-steps_loss-"
                     * "$(d_testlosses[end])_$starttimestr.jld2"),
            true, true, true, IOStream) do io
        # addrequire(io, :Flux)
        write(io, "d_model", tocpu(d_model))
        write(io, "d_optim", tocpu(d_optim))
        write(io, "d_trainlosses_real", d_trainlosses_real)
        write(io, "d_trainlosses_fake", d_trainlosses_fake)
        write(io, "d_testlosses", d_testlosses)
        write(io, "steps", steps)
    end
end

function save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                   steps, testloss, logdir, starttimestr, use_bson::Val{true})
    bson(joinpath(logdir, "generator-cp_$steps-steps_d-loss-$(testloss)_"
                  * "$starttimestr.bson"),
         g_model=tocpu(g_model), g_optim=tocpu(g_optim),
         g_trainlosses=g_trainlosses, testfakes=tocpu.(testfakes),
         const_noise=tocpu(const_noise), steps=steps)
end

function save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                   steps, testloss, logdir, starttimestr, use_bson::Val{false})
    # We use this signature so we don't use `mmap`, trading speed for stability.
    # See https://github.com/JuliaIO/JLD2.jl/issues/55
    jldopen(joinpath(logdir, "generator-cp_$steps-steps_d-loss-$(testloss)_"
                     * "$starttimestr.jld2"), true, true, true, IOStream) do io
        # addrequire(io, :Flux)
        write(io, "g_model", tocpu(g_model))
        write(io, "g_optim", tocpu(g_optim))
        write(io, "g_trainlosses", g_trainlosses)
        write(io, "testfakes", tocpu.(testfakes))
        write(io, "const_noise", tocpu(const_noise))
        write(io, "steps", steps)
    end
end

function load_d_cp(cppath::AbstractString, use_bson::Val{true})
    cp = BSON.load(cppath)
    load_d_cp(cp, Symbol)
end

function load_d_cp(cppath::AbstractString, use_bson::Val{false})
    cp = load(cppath)
    load_d_cp(cp, String)
end

function load_d_cp(cp::AbstractDict, cpkeytype::Type)
    d_model = togpu(cp[cpkeytype("d_model")]::Flux.Chain)
    d_optim::Flux.ADAM = cp[cpkeytype("d_optim")]

    d_trainlosses_real::Vector{Float32} = cp[cpkeytype("d_trainlosses_real")]
    d_trainlosses_fake::Vector{eltype(d_trainlosses_real)} = cp[
            cpkeytype("d_trainlosses_fake")]
    d_testlosses::Vector{eltype(d_trainlosses_real)} = cp[cpkeytype("d_testlosses")]
    steps::UInt64 = cp[cpkeytype("steps")]

    return (d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses, steps)
end

function load_g_cp(cppath::AbstractString, use_bson::Val{true})
    cp = BSON.load(cppath)
    load_g_cp(cp, Symbol)
end

function load_g_cp(cppath::AbstractString, use_bson::Val{false})
    cp = load(cppath)
    load_g_cp(cp, String)
end

function load_g_cp(cp::AbstractDict, cpkeytype::Type)
    g_model = togpu(cp[cpkeytype("g_model")]::Flux.Chain)
    g_optim::Flux.ADAM = cp[cpkeytype("g_optim")]

    g_trainlosses::Vector{Float32} = cp[cpkeytype("g_trainlosses")]
    testfakes::Vector = cp[cpkeytype("testfakes")]
    const_noise::AbstractArray = togpu(cp[cpkeytype("const_noise")])
    steps::UInt64 = cp[cpkeytype("steps")]

    return (g_model, g_optim, g_trainlosses, testfakes, const_noise, steps)
end


function save_meta_cp(meta_model, meta_optim, meta_trainlosses, meta_meanlosses,
                      meta_varlosses, steps, logdir, starttimestr, use_bson::Val{true})
    # TODO Due to having to use a different BSON PR branch, this fails.
    #      When the branch is merged, update the package and remove the try-catch.
    try
        bson(joinpath(logdir, "meta-cp_$steps-steps_loss-"
                      * "$(meta_trainlosses[end])_$starttimestr.bson"),
             meta_model=tocpu(meta_model), meta_optim=tocpu(meta_optim),
             meta_trainlosses=meta_trainlosses,
             meta_meanlosses=meta_meanlosses,
             meta_varlosses=meta_varlosses, steps=steps)
    catch e
        bson(joinpath(logdir, "meta-cp_$steps-steps_loss-"
                      * "$(meta_trainlosses[end])_$starttimestr.bson"),
             meta_model=tocpu(meta_model), meta_optim=nothing,
             meta_trainlosses=meta_trainlosses,
             meta_meanlosses=meta_meanlosses,
             meta_varlosses=meta_varlosses, steps=steps)
    end
end

function save_meta_cp(meta_model, meta_optim, meta_trainlosses, meta_meanlosses,
                      meta_varlosses, steps, logdir, starttimestr, use_bson::Val{false})
    # TODO Due to having to use a different BSON PR branch, this fails.
    #      When the branch is merged, update the package and remove the try-catch.
    # We use this signature so we don't use `mmap`, trading speed for stability.
    # See https://github.com/JuliaIO/JLD2.jl/issues/55
    jldopen(joinpath(logdir, "meta-cp_$steps-steps_loss-"
                     * "$(meta_trainlosses[end])_$starttimestr.jld2"),
            true, true, true, IOStream) do io
        # addrequire(io, :Flux)
        write(io, "meta_model", tocpu(meta_model))
        write(io, "meta_optim", tocpu(meta_optim))
        write(io, "meta_trainlosses", meta_trainlosses)
        write(io, "meta_meanlosses", meta_meanlosses)
        write(io, "meta_varlosses", meta_varlosses)
        write(io, "steps", steps)
    end
end

function load_meta_cp(cppath::AbstractString, use_bson::Val{true})
    cp = BSON.load(cppath)
    load_meta_cp(cp, Symbol)
end

function load_meta_cp(cppath::AbstractString, use_bson::Val{false})
    cp = load(cppath)
    load_meta_cp(cp, String)
end

function load_meta_cp(cp::AbstractDict, cpkeytype::Type)
    meta_model = togpu(cp[cpkeytype("meta_model")]::Flux.Chain)
    meta_optim::Flux.ADAM = cp[cpkeytype("meta_optim")]

    meta_trainlosses::Vector{Float32} = cp[cpkeytype("meta_trainlosses")]
    meta_meanlosses::Vector{eltype(meta_trainlosses)} = cp[cpkeytype("meta_meanlosses")]
    meta_varlosses::Vector{eltype(meta_trainlosses)} = cp[cpkeytype("meta_varlosses")]
    meta_steps::UInt64 = cp[cpkeytype("steps")]

    return (meta_model, meta_optim, meta_trainlosses,
            meta_meanlosses, meta_varlosses, meta_steps)
end

end # module

