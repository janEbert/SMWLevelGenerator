module TrainingLoop

using Dates: now
using Logging: SimpleLogger, Info
using Printf: @sprintf
using Statistics: mean, var
using Random: seed!

using BSON  # we currently use a fork (pr #47) due to issue #3
import Flux
import JSON
using TensorBoardLogger

using ..DataIterator
using ..InputStatistics
using ..ModelUtils
using ..TrainingUtils
using ..LSTM
using ..Transformer
# using ..ESN
using ..RandomPredictor

export trainingloop!, TrainingParameters, TPs

Base.@kwdef struct TrainingParameters
    epochs::Integer = 10

    lr::Float64                                      = 0.0002
    warmupepochs::Integer                            = 1  # TODO unused
    warmuplr::Float64                                = 0.0001  # TODO unused
    logevery::Integer                                = 300
    saveevery::Integer                               = 1500
    testratio::AbstractFloat                         = 0.1
    dataiter_threads::Integer                        = 0
    per_tile::Bool                                   = false
    reverse_rows::Bool                               = false
    logdir::AbstractString                           = joinpath("exps", newexpdir())
    params_logfile::AbstractString                   = "params.json"
    logfile::AbstractString                          = "training.log"
    earlystoppingwaitepochs::Integer                 = 10
    earlystoppingthreshold::AbstractFloat            = Inf32
    criterion::Function                              = Flux.mse
    use_soft_criterion::Bool                         = false
    overfit_on_batch::Bool                           = false
    modeltype::Union{Type{<:LearningModel}, Nothing} = nothing
    seed::Integer                                    = 0
end

"Shorthand for TrainingParameters for interactive use."
const TPs = TrainingParameters


# TODO save losses in own file to save space but support branching from them
#      possibly by storing the losses file in the model cp and creating a new one if loading
#      a model
#= TODO
Create struct for experimental parameters including model to:
   1: simplify saving
   2: dispatch training step and loss functions (maybe?)
   3: simplify data iterator interface: Instead of passing stuff like `per_tile` and
      friends, instead have them in the struct
   4: using the struct, automatically load the correct db, data iterator and pass the
      correct arguments to the SequenceGenerator module.
=#
function trainingloop!(model::Union{LearningModel, AbstractString}, dbpath::AbstractString,
                       params::TrainingParameters=TrainingParameters())
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
        (model, optimizer, trainlosses, meanlosses, varlosses, past_steps) = loadcp(
            model, params.modeltype)
    else
        optimizer = Flux.ADAM(params.lr)

        trainlosses = Float32[]
        "Means of test losses."
        meanlosses = eltype(trainlosses)[]
        "Variances of test losses."
        varlosses = eltype(trainlosses)[]

        past_steps = UInt64(0)
    end
    model::LearningModel
    paramdict[:modelparams] = Dict{Symbol, Any}(k => v isa Function ? Symbol(v) : v
                                                for (k, v) in model.hyperparams)

    epochs    = params.epochs
    logdir    = params.logdir
    logevery  = params.logevery
    saveevery = params.saveevery

    maxmeanloss = typemin(eltype(trainlosses))
    maxvarloss = typemin(eltype(trainlosses))
    maxvarlossdigits = 0  # Predefined in case `varloss` is NaN.
    local maxmeanlossdigits
    local testlosses
    steps = UInt64(0)

    dataiterparams = dataiteratorparams(model)
    dataiter_threads = params.dataiter_threads
    per_tile         = params.per_tile
    reverse_rows     = params.reverse_rows
    trainiter = dataiteratorchannel(db, 4, Val(dataiterparams.join_pad),
                                    Val(dataiterparams.as_matrix))
    testiter  = dataiteratorchannel(db, 4, Val(dataiterparams.join_pad),
                                    Val(dataiterparams.as_matrix))
    # TODO store max loss and log (as in logging) normalized loss (maybe).
    #      only applicable when sequences are padded to have the same length
    # TODO make function for that so it can be calculated online; then maybe normalize loss
    #      for training

    if params.use_soft_criterion
        loss = makesoftloss(model, params.criterion)
    else
        loss = makeloss(model, params.criterion)
    end
    parameters = Flux.params(model)

    earlystoppingthreshold = params.earlystoppingthreshold
    earlystoppingthreshold < 1 && (earlystoppingthreshold += 1)
    earlystoppingwaitepochs = params.earlystoppingwaitepochs

    mkpath(logdir)
    # Save parameters
    open(joinpath(logdir, params.params_logfile), "w") do io
        JSON.print(io, paramdict, 4)
    end
    # Free `paramdict`
    paramdict = nothing
    log_io = open(joinpath(logdir, params.logfile), "a")

    @fastmath try
        logger = SimpleLogger(log_io)
        tblogger = TBLogger(joinpath(logdir, "tensorboard"), min_level=Info)
        past_steps > 0 && logprint(logger, "Loaded model with $past_steps steps.")

        # To get local time instead of UTC for printing and filenames:
        starttimestr = replace(string(now()), ':' => '-')
        starttime = time()
        logprint(logger, "Starting training at $starttimestr for $epochs epochs. "
                 * "Seed: $(params.seed).")

        # Initial test
        testlosses = testmodel(model, testiter, db, testindices, dataiter_threads,
                               per_tile, reverse_rows, dataiterparams, loss)
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
            dataiterator!(trainiter, db, trainindices, dataiter_threads,
                          per_tile, reverse_rows; dataiterparams...)
            # Cannot iterate directly over a RemoteChannel.
            for i in 1:length(trainindices)
                # Continue exactly where training stopped.
                if steps < past_steps
                    take!(trainiter)
                    steps += 1
                    continue
                end

                training_step!(model, parameters, optimizer, trainiter, dataiterparams,
                               loss, trainlosses, tblogger)
                steps += 1

                if logevery != 0 && steps % logevery == 0
                    testlosses = testmodel(model, testiter, db, testindices,
                                           dataiter_threads, per_tile, reverse_rows,
                                           dataiterparams, loss)
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
                    push!(varlosses, varloss)
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
                        savecp(model, optimizer, trainlosses, testlosses, meanlosses,
                               varlosses, steps, logdir, starttimestr)
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
                                               dataiter_threads, per_tile, reverse_rows,
                                               dataiterparams, loss)
                        meanloss = mean(testlosses)
                        varloss = var(testlosses, mean=meanloss)
                        @tblog(tblogger, meantestloss=meanloss, vartestloss=varloss,
                               log_step_increment=0)
                        push!(meanlosses, meanloss)
                        push!(varlosses,  varloss)
                    end
                    savecp(model, optimizer, trainlosses, testlosses, meanlosses,
                           varlosses, steps, logdir, starttimestr)
                    logprint(logger, "Saved checkpoint after $steps training steps.")
                end
            end
        end
        savecp(model, optimizer, trainlosses, testlosses, meanlosses,
               varlosses, steps, logdir, starttimestr)
        logprint(logger, "Training finished after $steps training steps and "
                 * "$(round((time() - starttime) / 60, digits=2)) minutes.")
    finally
        cleanupall(trainiter, testiter, log_io)
    end
    return (model, trainlosses, testlosses, meanlosses, varlosses,
            db, trainindices, testindices)
end


"""
Update the given model with a single training step on the next data point in the
given `trainiter`. Return the loss.
"""
function training_step!(model, parameters, optimizer, trainiter, dataiterparams,
                        loss, trainlosses, tblogger)
    seq = togpu.(take!(trainiter))
    # Construct target
    target = maketarget(seq, Val(dataiterparams.join_pad), Val(dataiterparams.as_matrix))

    l = step!(model, parameters, optimizer, loss, seq, target).data
    push!(trainlosses, l)
    @tblog tblogger trainloss=l
    return l
end

"""
Return a `Vector{Float32}` of losses obtained by apply the given model to all data in the
to be reset `testiter`.
"""
function testmodel(model, testiter, db, testindices, dataiter_threads,
                   per_tile, reverse_rows, dataiterparams, loss)
    dataiterator!(testiter, db, testindices, dataiter_threads,
                  per_tile, reverse_rows; dataiterparams...)
    Flux.testmode!(model)
    testlosses = Float32[]

    for i in 1:length(testindices)
        seq = togpu.(take!(testiter))
        # Construct target
        target = maketarget(seq, Val(dataiterparams.join_pad),
                            Val(dataiterparams.as_matrix))

        l = calculate_loss(model, loss, seq, target).data
        push!(testlosses, l)
    end
    Flux.testmode!(model, false)
    return testlosses
end

function getmaxloss(seq, dataiterparams, loss)
    target = maketarget(seq, Val(dataiterparams.join_pad, Val(dataiterparams.as_matrix)))
    loss(ones(eltype(target), size(target)) .- target, target)
end

# Numbered Vararg so we can make sure we didn't miss one without having to list them all.
function cleanupall(args::Vararg{Any, 3})
    map(cleanup, args)
    return
end

function loadcp(cppath::AbstractString, modeltype::Nothing)
    cp = BSON.load(cppath)
    model = togpu(cp[:model]::LearningModel)

    optimizer, trainlosses, meanlosses, varlosses, past_steps = loadother(cp)
    return model, optimizer, trainlosses, meanlosses, varlosses, past_steps
end

function loadcp(cppath::AbstractString, modeltype::Type{<:LearningModel})
    cp = BSON.load(cppath)
    model = togpu(cp[:model]::modeltype)

    optimizer, trainlosses, meanlosses, varlosses, past_steps = loadother(cp)
    return model, optimizer, trainlosses, meanlosses, varlosses, past_steps
end

function loadother(cp)
    optimizer::Flux.ADAM = cp[:optimizer]

    trainlosses::Vector{Float32} = cp[:trainlosses]
    meanlosses::Vector{eltype(trainlosses)} = cp[:meanlosses]
    varlosses::Vector{eltype(trainlosses)} = cp[:varlosses]

    past_steps::UInt64 = cp[:steps]
    return optimizer, trainlosses, meanlosses, varlosses, past_steps
end

function savecp(model, optimizer, trainlosses, testlosses, meanlosses,
                varlosses, steps, logdir, starttimestr)
    bson(joinpath(logdir,
                  "model-cp_$steps-steps_loss-$(meanlosses[end])_$starttimestr.bson"),
         model=Flux.cpu(model), optimizer=optimizer, steps=steps, trainlosses=trainlosses,
         testlosses=testlosses, meanlosses=meanlosses, varlosses=varlosses)
end

function test_trainingloop(dbpath::AbstractString="levels_1d_flags_t.jdb")
    model = LSTM.lstm1d()
    @time trainingloop!(model, dbpath, 1)
end

end # module

