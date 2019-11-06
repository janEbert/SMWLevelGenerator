module GANTraining

using Dates: now
using Logging: SimpleLogger, Info
using Printf: @sprintf
using Statistics: mean, var
using Random: seed!, shuffle!

using BSON
import Flux
using Flux.Tracker: gradient
import JSON
using TensorBoardLogger

using ..DataIterator
using ..GAN
using ..MetadataPredictor
using ..ModelUtils
using ..TrainingUtils

export gan_trainingloop!, GANTrainingParameters, GTPs

Base.@kwdef struct GANTrainingParameters
    epochs::Integer = 10

    lr::Float64                               = 0.0002
    betas::Tuple{Float64, Float64}            = (0.5, 0.999)
    batch_size::Integer                       = 32
    d_warmup_steps::Integer                   = 0
    logevery::Integer                         = 200
    saveevery::Integer                        = 1000
    dataiter_threads::Integer                 = 0
    logdir::AbstractString                    = joinpath("exps", newexpdir("gan"))
    params_logfile::AbstractString            = "params.json"
    logfile::AbstractString                   = "training.log"
    earlystoppingwaitepochs::Integer          = 10
    earlystoppingthreshold::AbstractFloat     = Inf32
    criterion::Function                       = Flux.binarycrossentropy
    overfit_on_batch::Bool                    = false
    meta_lr::Float64                          = 0.001
    meta_criterion::Function                  = Flux.mse
    meta_testratio::AbstractFloat             = 0.1
    seed::Integer                             = 0
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

    if d_model isa AbstractString
        paramdict[:d_modelpath] = d_model
        (d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses,
                d_steps) = load_d_cp(d_model)
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
                g_model)
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
                    meta_varlosses, meta_steps) = load_meta_cp(meta_model)
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

    epochs     = params.epochs
    batch_size = params.batch_size
    logdir     = params.logdir
    logevery   = params.logevery
    saveevery  = params.saveevery
    d_warmup_steps  = params.d_warmup_steps

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

    dataiter_threads = params.dataiter_threads
    trainiter = gan_dataiteratorchannel(db, 3)
    testiter  = gan_dataiteratorchannel(db, 3)
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
            testfakes = [g_model(const_noise).data]
        else
            push!(testfakes, g_model(const_noise).data)
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
            testlosses = testmodel(meta_model, testiter, db, testindices,
                                   batch_size, dataiter_threads, meta_loss)
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
            gan_dataiterator!(trainiter, db, trainindices, batch_size, dataiter_threads)
            for i in 1:cld(length(trainindices), batch_size)
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
                # Train on real batch
                d_l_real = d_loss(real_batch, real_target)
                d_l = d_l_real.data
                push!(d_trainlosses_real, d_l)
                @tblog tblogger d_loss_real=d_l

                # Train on fake batch
                if g_model.hyperparams[:dimensionality] === Symbol("1d")
                    noise_batch = togpu(randn(1, generator_inputsize, curr_batch_size))
                else
                    noise_batch = togpu(randn(1, 1, generator_inputsize, curr_batch_size))
                end
                fake_batch = g_model(noise_batch)

                d_l_fake = d_loss(fake_batch, fake_target)
                d_l += d_l_fake.data
                push!(d_trainlosses_fake, d_l_fake.data)
                @tblog tblogger d_loss_fake=d_l_fake.data log_step_increment=0

                grads = gradient(() -> d_l_real + d_l_fake, d_params)

                # Update
                Flux.Optimise.update!(d_optim, d_params, grads)


                if steps > d_warmup_steps || steps == 0
                    # Generator
                    # Use real labels for modified loss function
                    g_l = g_loss(noise_batch, real_target)
                    push!(g_trainlosses, g_l.data)
                    @tblog tblogger g_loss=g_l.data log_step_increment=0
                    # Use real labels for modified loss function
                    grads = gradient(() -> g_l, g_params)

                    # Update
                    Flux.Optimise.update!(g_optim, g_params, grads)
                end


                # Metadata predictor
                if !isnothing(meta_model) && (overfit_on_batch || i > length(testindices))
                    l = meta_loss(real_batch, meta_batch)
                    push!(meta_trainlosses, l.data)
                    @tblog tblogger meta_predictor_loss=l.data log_step_increment=0
                    grads = gradient(() -> l, meta_params)

                    Flux.Optimise.update!(meta_optim, meta_params, grads)
                end


                steps += 1

                if logevery != 0 && steps % logevery == 0
                    push!(testfakes, g_model(const_noise).data)
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
                        max_g_loss = g_l.data
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
                        testlosses = testmodel(meta_model, testiter, db, testindices,
                                               batch_size, dataiter_threads, meta_loss)
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
                             * "$(lpad(i, ndigits(length(trainindices)))) / "
                             * "$(length(trainindices)); discriminator loss: "
                             * "$(lpad(@sprintf("%.4f", testloss), max_d_testlossdigits)) "
                             * "test, $(lpad(@sprintf("%.4f", d_l), max_d_lossdigits)) "
                             * "train; generator loss: "
                             * "$(lpad(@sprintf("%.4f", g_l.data), max_g_lossdigits))); "
                             * "mean time per step: "
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
                                  d_testlosses, steps, logdir, starttimestr)
                        save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                                  steps, testloss, logdir, starttimestr)
                        if !isnothing(meta_model)
                            save_meta_cp(meta_model, meta_optim, meta_trainlosses,
                                         meta_meanlosses, meta_varlosses, steps,
                                         logdir, starttimestr)
                        end
                        logprint(logger, "Early stopping activated after $steps training "
                                 * "steps ($epoch epochs, $i sequences in current epoch). "
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
                        push!(testfakes, g_model(const_noise).data)
                        testloss = testmodel(d_model, d_loss, testfakes[end],
                                             const_fake_target)
                        @tblog tblogger d_testloss=testloss log_step_increment=0
                        push!(d_testlosses, testloss)

                        if !isnothing(meta_model)
                            testlosses = testmodel(meta_model, testiter, db, testindices,
                                                   batch_size, dataiter_threads, meta_loss)
                            meanloss = mean(testlosses)
                            varloss  = var(testlosses, mean=meanloss)
                            @tblog(tblogger, meta_predictor_meanloss=meanloss,
                                   meta_predictor_varloss=varloss, log_step_increment=0)
                            push!(meta_meanlosses, meanloss)
                            push!(meta_varlosses,  varloss)
                        end
                    end
                    save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake,
                              d_testlosses, steps, logdir, starttimestr)
                    save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                              steps, testloss, logdir, starttimestr)
                    if !isnothing(meta_model)
                        save_meta_cp(meta_model, meta_optim, meta_trainlosses,
                                     meta_meanlosses, meta_varlosses, steps,
                                     logdir, starttimestr)
                    end
                    logprint(logger, "Saved checkpoints after $steps training steps.")
                end
            end
        end
        save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake,
                  d_testlosses, steps, logdir, starttimestr)
        save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                  steps, testloss, logdir, starttimestr)
        if !isnothing(meta_model)
            save_meta_cp(meta_model, meta_optim, meta_trainlosses, meta_meanlosses,
                         meta_varlosses, steps, logdir, starttimestr)
        end
        logprint(logger, "Training finished after $steps training steps and "
                 * "$(round((time() - starttime) / 60, digits=2)) minutes.")
    finally
        cleanupall(trainiter, testiter, log_io)
    end
    return (d_model, g_model, d_trainlosses_real, d_trainlosses_fake,
            d_testlosses, g_trainlosses, testfakes, db, trainindices, const_noise)
end


function testmodel(d_model, d_loss, fake_batch, const_fake_target)
    Flux.testmode!(d_model)
    d_l = d_loss(fake_batch, const_fake_target).data
    Flux.testmode!(d_model, false)
    return d_l
end

function testmodel(meta_model, testiter, db, testindices, batch_size, dataiter_threads,
                   meta_loss)
    gan_dataiterator!(testiter, db, testindices, batch_size, dataiter_threads)
    Flux.testmode!(meta_model)
    testlosses = Float32[]

    for i in 1:cld(length(testindices), batch_size)
        real_batch, meta_batch = map(togpu, take!(testiter))
        l = meta_loss(real_batch, meta_batch).data
        push!(testlosses, l)
    end
    Flux.testmode!(meta_model, false)
    return testlosses
end

# Numbered Vararg so we can make sure we didn't miss one without having to list them all.
cleanupall(args::Vararg{Any, 3}) = foreach(cleanup, args)

function save_d_cp(d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses,
                   steps, logdir, starttimestr)
    bson(joinpath(logdir, "discriminator-cp_$steps-steps_loss-$(d_testlosses[end])_"
                  * "$starttimestr.bson"),
         d_model=Flux.cpu(d_model), d_optim=d_optim, d_trainlosses_real=d_trainlosses_real,
         d_trainlosses_fake=d_trainlosses_fake, d_testlosses=d_testlosses, steps=steps)
end

function save_g_cp(g_model, g_optim, g_trainlosses, testfakes, const_noise,
                   steps, testloss, logdir, starttimestr)
    bson(joinpath(logdir, "generator-cp_$steps-steps_d-loss-$(testloss)_"
                  * "$starttimestr.bson"),
         g_model=Flux.cpu(g_model), g_optim=g_optim, g_trainlosses=g_trainlosses,
         testfakes=testfakes, const_noise=Flux.cpu(const_noise), steps=steps)
end

function load_d_cp(cppath::AbstractString)
    cp = BSON.load(cppath)
    d_model = togpu(cp[:d_model]::Flux.Chain)
    d_optim::Flux.ADAM = cp[:d_optim]

    d_trainlosses_real::Vector{Float32} = cp[:d_trainlosses_real]
    d_trainlosses_fake::Vector{eltype(d_trainlosses_real)} = cp[:d_trainlosses_fake]
    d_testlosses::Vector{eltype(d_trainlosses_real)} = cp[:d_testlosses]
    steps::UInt64 = cp[:steps]

    return (d_model, d_optim, d_trainlosses_real, d_trainlosses_fake, d_testlosses, steps)
end

function load_g_cp(cppath::AbstractString)
    cp = BSON.load(cppath)
    g_model = togpu(cp[:g_model]::Flux.Chain)
    g_optim::Flux.ADAM = cp[:g_optim]

    g_trainlosses::Vector{Float32} = cp[:g_trainlosses]
    testfakes::Vector = cp[:testfakes]
    const_noise::AbstractArray = togpu(cp[:const_noise])
    steps::UInt64 = cp[:steps]

    return (g_model, g_optim, g_trainlosses, testfakes, const_noise, steps)
end


function save_meta_cp(meta_model, meta_optim, meta_trainlosses, meta_meanlosses,
                      meta_varlosses, steps, logdir, starttimestr)
    # TODO Due to having to use a different BSON PR branch, this fails.
    #      When the branch is merged, update the package and remove the try-catch.
    try
        bson(joinpath(logdir, "meta-cp_$steps-steps_loss-$(meta_trainlosses[end])_"
                      * "$starttimestr.bson"),
             meta_model=Flux.cpu(meta_model), meta_optim=meta_optim,
             meta_trainlosses=meta_trainlosses, meta_meanlosses=meta_meanlosses,
             meta_varlosses=meta_varlosses, steps=steps)
    catch e
        e isa ErrorException || rethrow()
        bson(joinpath(logdir, "meta-cp_$steps-steps_loss-$(meta_trainlosses[end])_"
                      * "$starttimestr.bson"),
             meta_model=Flux.cpu(meta_model), meta_optim=nothing,
             meta_trainlosses=meta_trainlosses, meta_meanlosses=meta_meanlosses,
             meta_varlosses=meta_varlosses, steps=steps)
    end
end

function load_meta_cp(cppath::AbstractString)
    cp = BSON.load(cppath)
    meta_model = togpu(cp[:meta_model]::Flux.Chain)
    meta_optim::Flux.ADAM = cp[:meta_optim]

    meta_trainlosses::Vector{Float32} = cp[:meta_trainlosses]
    meta_meanlosses::Vector{eltype(meta_trainlosses)} = cp[:meta_meanlosses]
    meta_varlosses::Vector{eltype(meta_trainlosses)} = cp[:meta_varlosses]
    meta_steps::UInt64 = cp[:steps]

    return (meta_model, meta_optim, meta_trainlosses,
            meta_meanlosses, meta_varlosses, meta_steps)
end

end # module

