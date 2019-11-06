module MetaTraining

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
using ..MetadataPredictor
using ..ModelUtils
using ..TrainingUtils

export meta_trainingloop!, MetaTrainingParameters, MTPs

Base.@kwdef struct MetaTrainingParameters
    epochs::Integer = 10

    lr::Float64                               = 0.0002
    batch_size::Integer                       = 32
    logevery::Integer                         = 300
    saveevery::Integer                        = 1500
    testratio::AbstractFloat                  = 0.1
    dataiter_threads::Integer                 = 0
    logdir::AbstractString                    = joinpath("exps", newexpdir("meta"))
    params_logfile::AbstractString            = "params.json"
    logfile::AbstractString                   = "training.log"
    earlystoppingwaitepochs::Integer          = 10
    earlystoppingthreshold::AbstractFloat     = Inf32
    criterion::Function                       = Flux.mse
    overfit_on_batch::Bool                    = false
    seed::Integer                             = 0
end

"Shorthand for MetaTrainingParameters for interactive use."
const MTPs = MetaTrainingParameters


function meta_trainingloop!(model::Union{LearningModel, AbstractString},
                            dbpath::AbstractString,
                            params::MetaTrainingParameters=MetaTrainingParameters())
    seed!(params.seed)
    set_zero_subnormals(true)

    paramdict = Dict{Symbol, Any}(field => getproperty(params, field) |>
                                  x -> x isa Function ? Symbol(x) : x
                                  for field in propertynames(params))
    paramdict[:dbpath] = dbpath

    db = loaddb(dbpath)
    trainindices, testindices = traintestsplit(db, params.testratio)
    if params.overfit_on_batch
        if params.overfit_on_batch isa Bool
            trainindices = view(trainindices, firstindex(trainindices):16)
        else
            trainindices = view(trainindices,
                                firstindex(trainindices):params.overfit_on_batch)
        end
        testindices = trainindices
    end

    if model isa AbstractString
        paramdict[:modelpath] = model
        (model, optim, trainlosses, meanlosses, varlosses, past_steps) = load_cp(model)
    else
        optim = Flux.ADAM(params.lr)

        trainlosses = Float32[]
        "Means of test losses."
        meanlosses  = eltype(trainlosses)[]
        "Variances of test losses."
        varlosses   = eltype(trainlosses)[]

        past_steps = UInt64(0)
    end
    model::LearningModel
    paramdict[:modelparams] = Dict{Symbol, Any}(k => v isa Function ? Symbol(v) : v
                                                for (k, v) in model.hyperparams)

    epochs     = params.epochs
    batch_size = params.batch_size
    logdir     = params.logdir
    logevery   = params.logevery
    saveevery  = params.saveevery

    maxmeanloss = typemin(eltype(trainlosses))
    maxvarloss = typemin(eltype(trainlosses))
    maxvarlossdigits = 0  # Predefined in case `varloss` is NaN.
    local maxmeanlossdigits
    local testlosses
    steps = UInt64(0)

    dataiter_threads = params.dataiter_threads
    trainiter = gan_dataiteratorchannel(db, 3)
    testiter  = gan_dataiteratorchannel(db, 3)

    loss = makeloss(model, params.criterion)
    parameters = Flux.params(model)

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
        past_steps > 0 && logprint(logger, "Loaded meta predictor with $past_steps steps.")

        # To get local time instead of UTC for printing and filenames:
        starttimestr = replace(string(now()), ':' => '-')
        starttime = time()
        logprint(logger, "Starting meta predictor training at $starttimestr for $epochs "
                 * "epochs. Seed: $(params.seed).")

        # Initial test
        testlosses = testmodel(model, testiter, db, testindices,
                               batch_size, dataiter_threads, loss)
        meanloss = mean(testlosses)
        varloss  = var(testlosses, mean=meanloss)
        if past_steps == 0
            @tblog(tblogger, meantestloss=meanloss, vartestloss=varloss,
                   log_step_increment=0)
            push!(meanlosses, meanloss)
            push!(varlosses,  varloss)
        else
            @tblog(tblogger, log_step_increment=convert(Int, past_steps))
        end

        timediff = time() - starttime
        logprint(logger, "Initial mean test loss: $(@sprintf("%.4f", meanloss)) "
                 * "(variance: $(@sprintf("%.3f", varloss))); "
                 * "total time: $(@sprintf("%.2f", timediff / 60)) min.")

        for epoch in 1:epochs
            gan_dataiterator!(trainiter, db, trainindices, batch_size, dataiter_threads)
            for i in 1:cld(length(trainindices), batch_size)
                if steps < past_steps
                    take!(trainiter)
                    steps += 1
                    continue
                end

                real_batch, meta_batch = map(togpu, take!(trainiter))

                l = loss(real_batch, meta_batch)
                push!(trainlosses, l.data)
                @tblog tblogger trainloss=l.data log_step_increment=0
                grads = gradient(() -> l, parameters)

                Flux.Optimise.update!(optim, parameters, grads)
                steps += 1

                if logevery != 0 && steps % logevery == 0
                    testlosses = testmodel(model, testiter, db, testindices,
                                           batch_size, dataiter_threads, loss)
                    meanloss = mean(testlosses)
                    varloss  = var(testlosses, mean=meanloss)
                    @tblog(tblogger, meantestloss=meanloss, vartestloss=varloss,
                           log_step_increment=0)
                    if meanloss > maxmeanloss
                        maxmeanloss = meanloss
                        maxmeanlossdigits = ndigits(trunc(Int, maxmeanloss)) + 5
                    end
                    if varloss > maxvarloss
                        maxvarloss = varloss
                        maxvarlossdigits = ndigits(trunc(Int, maxvarloss)) + 4
                    end
                    push!(meanlosses, meanloss)
                    push!(varlosses,  varloss)
                    if length(meanlosses) > 1
                        lossdiff  = meanloss - meanlosses[end - 1]
                        lossratio = meanloss / meanlosses[end - 1]
                    else
                        lossdiff = 0
                    end

                    timediff = time() - starttime
                    logprint(logger, "Epoch $(lpad(epoch, ndigits(epochs))) / "
                             * "$epochs; sequence "
                             * "$(lpad(i, ndigits(length(trainindices)))) / "
                             * "$(length(trainindices)); mean test loss: "
                             * "$(lpad(@sprintf("%.4f", meanloss), maxmeanlossdigits)) "
                             * "(variance: "
                             * "$(lpad(@sprintf("%.3f", varloss), maxvarlossdigits))); "
                             * "mean time per step: "
                             * "$(@sprintf("%.3f", timediff / (steps - past_steps))) s; "
                             * "total time: $(@sprintf("%.2f", timediff / 60)) min.")

                    # Early stopping
                    if (epoch > earlystoppingwaitepochs && lossdiff > 0
                            && lossratio >= earlystoppingthreshold)
                        save_cp(model, optim, trainlosses, meanlosses, varlosses,
                                steps, logdir, starttimestr)
                        logprint(logger, "Early stopping activated after $steps training "
                                 * "steps ($epoch epochs, $i sequences in current epoch). "
                                 * "Loss increase: $meanloss - $(meanlosses[end - 1]) = "
                                 * "$(round(lossdiff, digits=3)) "
                                 * "($(round((lossratio - 1) * 100, digits=2)) %). "
                                 * "Total time: $(round(timediff / 60, digits=2)) min.")
                        cleanupall(trainiter, testiter, log_io)
                        return (model, trainlosses, testlosses, meanlosses, varlosses,
                                db, trainindices, testindices)
                    end
                end
                if saveevery != 0 && steps % saveevery == 0
                    if logevery != 0 && steps % logevery != 0
                        testlosses = testmodel(model, testiter, db, testindices,
                                               batch_size, dataiter_threads, loss)
                        meanloss = mean(testlosses)
                        varloss  = var(testlosses, mean=meanloss)
                        @tblog(tblogger, meantestloss=meanloss, vartestloss=varloss,
                               log_step_increment=0)
                        push!(meanlosses, meanloss)
                        push!(varlosses,  varloss)
                    end
                    save_cp(model, optim, trainlosses, meanlosses, varlosses,
                            steps, logdir, starttimestr)
                    logprint(logger, "Saved checkpoint after $steps training steps.")
                end
            end
        end
        save_cp(model, optim, trainlosses, meanlosses, varlosses,
                steps, logdir, starttimestr)
        logprint(logger, "Training finished after $steps training steps and "
                 * "$(round((time() - starttime) / 60, digits=2)) minutes.")
    finally
        cleanupall(trainiter, testiter, log_io)
    end
    return (model, trainlosses, testlosses, meanlosses, varlosses,
            db, trainindices, testindices)
end


function testmodel(model, testiter, db, testindices, batch_size, dataiter_threads,
                   loss)
    gan_dataiterator!(testiter, db, testindices, batch_size, dataiter_threads)
    Flux.testmode!(model)
    testlosses = Float32[]

    for i in 1:cld(length(testindices), batch_size)
        real_batch, meta_batch = map(togpu, take!(testiter))
        l = loss(real_batch, meta_batch).data
        push!(testlosses, l)
    end
    Flux.testmode!(model, false)
    return testlosses
end

# Numbered Vararg so we can make sure we didn't miss one without having to list them all.
cleanupall(args::Vararg{Any, 3}) = foreach(cleanup, args)

function save_cp(model, optim, trainlosses, meanlosses,
                      varlosses, steps, logdir, starttimestr)
    # TODO Due to having to use a different BSON PR branch, this fails.
    #      When the branch is merged, update the package and remove the try-catch.
    try
        bson(joinpath(logdir, "meta-cp_$steps-steps_loss-$(trainlosses[end])_"
                      * "$starttimestr.bson"),
             meta_model=Flux.cpu(model), meta_optim=optim,
             meta_trainlosses=trainlosses, meta_meanlosses=meanlosses,
             meta_varlosses=varlosses, steps=steps)
    catch e
        e isa ErrorException || rethrow()
        bson(joinpath(logdir, "meta-cp-no-optim_$steps-steps_loss-$(trainlosses[end])_"
                      * "$starttimestr.bson"),
             meta_model=Flux.cpu(model), meta_optim=nothing,
             meta_trainlosses=trainlosses, meta_meanlosses=meanlosses,
             meta_varlosses=varlosses, steps=steps)
    end
end

function load_cp(cppath::AbstractString)
    cp = BSON.load(cppath)
    model = togpu(cp[:meta_model]::Flux.Chain)
    optim::Flux.ADAM = cp[:meta_optim]

    trainlosses::Vector{Float32} = cp[:meta_trainlosses]
    meanlosses::Vector{eltype(trainlosses)} = cp[:meta_meanlosses]
    varlosses::Vector{eltype(trainlosses)} = cp[:meta_varlosses]
    past_steps::UInt64 = cp[:steps]

    return (model, optim, trainlosses, meanlosses, varlosses, past_steps)
end

end # module

