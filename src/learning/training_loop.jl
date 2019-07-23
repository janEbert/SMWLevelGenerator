module TrainingLoop

using Base.CoreLogging: @_sourceinfo, logmsg_code,
                        _min_enabled_level, current_logger_for_env, shouldlog,
                        handle_message, logging_error
using Dates: now
using Logging
using Statistics: mean
using Random: seed!

using BSON  # we currently use a fork (pr #47) due to issue #3
import Flux
using Flux.Tracker: gradient
import JSON
using TensorBoardLogger
#using Zygote

using ..DataIterator
using ..InputStatistics
using ..Transformer
using ..LSTM
using ..ESN
using ..RandomModel


function trainingloop!(model, dbpath::AbstractString, epochs::Integer; lr=0.001,
                       warmupepochs=1, warmuplr=0.0001,
                       logevery=500, saveevery=2000, testratio=0.1,
                       logdir=joinpath("exps", "exp_$(now())"), logfile="training.log",
                       earlystoppingwaitepochs=3, earlystoppingthreshold=0.05,
                       seed=0)
    seed!(seed)
    set_zero_subnormals(true)

    db = loaddb(dbpath)
    trainindices, testindices = traintestsplit(db, testratio)
    testiter = dataiterator(db, testindices)

    function loss(x, y)
        # TODO try L1 norm
        l = Flux.mse(model(x), y)
        Flux.truncate!(model)
        return l
    end

    parameters = Flux.params(model)
    optimizer = Flux.ADAM(lr)

    trainlosses = Float32[]
    meanlosses = eltype(trainlosses)[]
    maxmeanloss = typemin(eltype(trainlosses))
    local maxmeanlossdigits
    steps = UInt64(0)

    earlystoppingthreshold < 1 && (earlystoppingthreshold += 1)

    mkpath(logdir)
    log_io = open(joinpath(logdir, logfile), "w")
    try
        logger = SimpleLogger(log_io)
        tblogger = TBLogger(joinpath(logdir, "tensorboard"), min_level=Logging.Info)

        starttime = time()
        starttimestr = now()  # to get local time instead of UTC
        logprint(logger, "Starting training at $starttimestr for $epochs epochs. "
                 * "Seed: $seed.")
        for epoch in 1:epochs
            trainiter = dataiterator(db, trainindices)
            for (i, seqgen) in enumerate(trainiter)
                firstcol = true
                # Reset hidden state
                Flux.reset!(model)

                local prevcol
                for (j, col) in enumerate(map(togpu, seqgen))
                    if !firstcol
                        # Predict next column (target: `col`)

                        # Calculate loss
                        l = loss(prevcol, col).data
                        push!(trainlosses, l)

                        # Take gradients
                        grads = gradient(() -> loss(prevcol, col), parameters)

                        # Update weights
                        Flux.Optimise.update!(optimizer, parameters, grads)
                        steps += 1
                    else
                        firstcol = false
                    end

                    prevcol = col
                end

                if i % logevery == 0
                    testlosses = testmodel(model, db, testindices, loss, optimizer)
                    meanloss = mean(testlosses)
                    if meanloss > maxmeanloss
                        maxmeanloss = meanloss
                        maxmeanlossdigits = ndigits(trunc(Int, maxmeanloss)) + 4
                    end
                    push!(meanlosses, meanloss)
                    if length(meanlosses) > 1
                        lossdiff  = meanloss - meanlosses[end - 1]
                        lossratio = meanloss / meanlosses[end - 1]
                    end

                    timediff = time() - starttime
                    logprint(logger, "Epoch $(lpad(epoch, ndigits(epochs))) / "
                             * "$epochs; sequence "
                             * "$(lpad(i, ndigits(length(trainindices)))) / "
                             * "$(length(trainindices)); mean test loss: "
                             * "$(lpad(round(meanloss, digits=3), maxmeanlossdigits)); "
                             * "mean time per step: "
                             * "$(round(timediff / steps * 1000, digits=3)) ms; "
                             * "total time: $(round(timediff / 60, digits=2)) min.")

                    # Early stopping
                    if (epoch > earlystoppingwaitepochs && lossdiff > 0
                            && lossratio >= earlystoppingthreshold)
                        bson(joinpath(logdir, "model-cp_$steps-steps_loss-$(meanloss)_"
                                      * "$starttimestr.bson"),
                             model=model, optimizer=optimizer, trainlosses=trainlosses,
                             testlosses=testlosses, meanlosses=meanlosses, steps=steps)
                        logprint(logger,"Early stopping activated after $steps training "
                                 * "steps ($epoch epochs, $i sequences in current epoch). "
                                 * "Loss increase: $meanloss - $(meanlosses[end - 1]) = "
                                 * "$(round(lossdiff, digits=3)) "
                                 * "($(round((lossratio - 1) * 100, digits=2)) %). "
                                 * "Total time: $(round(timediff / 60, digits=2)) min.")
                        close(log_io)
                        return (model, trainlosses, testlosses, meanlosses)
                    end
                end
                if i % saveevery == 0
                    if i % logevery != 0
                        testlosses = testmodel(model, db, testindices, loss, optimizer)
                        push!(meanlosses, mean(testlosses))
                    end
                    logprint(logger, "Saving checkpoint after $steps training steps.")
                    bson(joinpath(logdir, "model-cp_$steps-steps_loss-$(meanlosses[end])_"
                                  * "$starttimestr.bson"),
                         model=model, optimizer=optimizer, trainlosses=trainlosses,
                         testlosses=testlosses, meanlosses=meanlosses, steps=steps)
                end
            end
        end
        logprint(logger, "Training finished after $steps training steps and "
                 * "$(round((time() - starttime) / 60, digits=2)) minutes.")
    finally
        close(log_io)
    end
    return (model, trainlosses, testlosses, meanlosses)
end

# function processsequence!(model, losses, seqgen, # TODO)

function testmodel(model, db, testindices, loss, optimizer)
    testiter = dataiterator(db, testindices)
    testlosses = Float32[]

    for (i, seqgen) in enumerate(testiter)
        firstcol = true
        # Reset hidden state
        Flux.reset!(model)

        local prevcol
        for (j, col) in enumerate(map(Flux.cpu, seqgen))
            if !firstcol
                # Calculate loss
                l = loss(prevcol, col).data
                push!(testlosses, l)
            else
                firstcol = false
            end

            prevcol = col
        end
    end
    return testlosses
end

#=function earlystopping(losses::AbstractArray, prevmeanloss::Number,
    if meanloss > prevmeanloss
        logprint(logger,"Early stopping activated after $steps train steps "
                 * "($epoch epochs, $i steps in current epoch). "
                 * "Loss increase: $(meanloss - prevmeanloss)")
        return meanloss, true
    end
    return meanloss, false
end=#

function logprint(logger::AbstractLogger, msg::AbstractString,
                  loglevel::LogLevel=Logging.Info)
    with_logger(logger) do
        @logmsg loglevel msg
    end
    @logmsg loglevel msg
end

"""
    @tblog(logger, exs...)

Log the given expressions `exs` using the given logger with an empty message.

# Examples
```jldoctest
julia> using Logging
julia> l = SimpleLogger()
julia> with_logger(l) do
           @info "" a = 0
       end
┌ Info:
│   a = 0
└ @ [...]

julia> @tblog l a = 0
┌ Info:
│   a = 0
└ @ [...]
"""
macro tbllog(logger, level, exs...)
    f = logmsg_code((@_sourceinfo)..., :(Logging.Info), "", exs...)
    quote
        with_logger($(esc(logger))) do
            $f
        end
    end
end

"""
    run(modelfunction::Function, dbpath, epochs; modelargs,
        modelparams::AbstractDict, trainingparams::AbstractDict,
        logdir=joinpath("exps", "exp_$(now())"), logfile="params.json")

Return a model created using `model = modelfunction(modelargs...; modelparams...)` and
trained on `trainingloop!(model, db, epochs; logdir=logdir, trainingparams...)`.
The parameters are all saved in JSON format in the given `logfile` in `logdir`.
"""
function run(modelfunction, dbpath, epochs; modelargs::Union{Tuple, AbstractArray},
             modelparams::AbstractDict, trainingparams::AbstractDict,
             logdir=joinpath("exps", "exp_$(now())"), logfile="params.json")
    type = keytype(modelparams)
    @assert type === keytype(trainingparams) "different key types in `Dict`s."
    modelparams[type(:args)] = modelargs

    trainingparams[type(:dbpath)] = dbpath
    trainingparams[type(:epochs)] = epochs
    trainingparams[type(:logdir)] = logdir

    open(joinpath(logdir, logfile), "w") do io
        JSON.print(io, Dict(
            type(:model) => modelparams, type(:training) => trainingparams
        ), 4)
    end
    # Remove values we cannot pass.
    delete!(modelparams,    type(:args))
    delete!(trainingparams, type(:db))
    delete!(trainingparams, type(:epochs))

    model = modelfunction(modelargs...; modelparams...)
    trainingloop!(model, db, epochs; trainingparams...)
end

function test_trainingloop(dbpath::AbstractString="levels_1d_flags_t.jdb")
    model = LSTM.lstm1d()
    @time trainingloop!(model, dbpath, 1)
end

end # module

