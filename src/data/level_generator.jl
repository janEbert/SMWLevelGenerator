module LevelGenerator

using BSON
using Flux
using JuliaDB

using ..DataIterator
using ..InputStatistics
using ..ModelUtils: LearningModel, AbstractGenerator, togpu
using ..SequenceGenerator
using ..ScreenGenerator
using ..LevelFormatReverter
using ..LevelDeconstructor
using ..LevelWriter

export predict_level, generatelevel

const lmpath = joinpath(@__DIR__, "..", "..", "tools",
                                  "lm304ebert21892usix", "Lunar Magic.exe")

const level105dbid_1d_singleline_t = 0x2b7d
const level105dbid_2d_t = 0x2b7d
const level105dbid_3d_t = 0x2b7d
const level105dbid_3d_tesx = 0x2b7d

function setup_generation(modelpath::AbstractString, dbpath::AbstractString)
    cp = BSON.load(modelpath)
    model = togpu(cp[:model])
    db = DataIterator.loaddb(dbpath)
    dimensionality = model.hyperparams[:dimensionality]
    from_method = getfield(LevelFormatReverter,
                           Symbol("from" * String(dimensionality)[1:2]))
    return model, db, from_method
end

# TODO better function names (indicate that the levels are written to a file)
# TODO more views

# TODO a hack full of levels. generate 512 levels all with different numberered file names
function predict_hack()
end

"""
    predict_levels(amount, modelpath, dbpath="levels_3d_flags_tesx.jdb",
                   first_screen=true, seed=nothing; write_rom=nothing)

Generate the given amount of levels from random database entries and write each to a file.

If `first_screen` is `true`, use the level's first screen as prediction input instead
of only the first part of the sequence.
If `write_rom` is not `nothing`, the given path to a ROM will be used to reconstruct the
generated level in the ROM (overwriting the previous contents).
A random seed can optionally be specified with `seed`.
"""
function predict_levels(amount, modelpath::AbstractString,
                        dbpath::AbstractString="levels_3d_flags_tesx.jdb",
                        first_screen=true, seed=nothing;
                        write_rom::Union{AbstractString, Nothing}=nothing)
    model, db, from_method = setup_generation(modelpath, dbpath)
    indices = collect(one(UInt):convert(UInt, length(db)))
    if !isnothing(seed)
        shuffle!(MersenneTwister(seed), indices)
    else
        shuffle!(indices)
    end
    indices = view(indices, 1:amount)

    for index in indices
        sequence, constantinput = predict_from_index(model, db, index, from_method,
                                                     first_screen)
        lmwrite(write_rom, constantinput.number)
    end
end

"""
    predict_level(modelpath, dbpath, dbid, first_screen; write_rom)

Predict the rest of the level with the given database ID and write it to a file.
If `first_screen` is `true`, use the level's first screen as prediction input instead
of only the first part of the sequence.
If `write_rom` is not `nothing`, the given path to a ROM will be used to reconstruct the
generated level in the ROM (overwriting the previous contents).
"""
function predict_level(modelpath::AbstractString,
                       dbpath::AbstractString="levels_3d_flags_tesx.jdb",
                       dbid::UInt64=UInt64(SequenceGenerator.level105dbid_3d_tesx),
                       first_screen=true; write_rom=nothing)
    model, db, from_method = setup_generation(modelpath, dbpath)
    sequence, constantinput = predict_from_id(model, db, dbid, from_method, first_screen)
    writeback(sequence, constantinput)
    lmwrite(write_rom, constantinput.number)
    return sequence, constantinput
end

function predict_from_id(model, db, dbid, from_method, first_screen)
    row = filter(r -> r.id == dbid, db)[1]
    predict_from_row(model, row, from_method, first_screen)
end

function predict_from_index(model, db, index, from_method, first_screen)
    row = db[index]
    predict_from_row(model, row, from_method, first_screen)
end

function predict_from_row(model, row, from_method, first_screen)
    input = DataIterator.preprocess(row, false, false, Val(false), Val(false))
    if first_screen
        constantinput, sequence = generatesequence(model, input[1:screencols])
    else
        constantinput, sequence = generatesequence(model, input[1])
    end
    sequence, constantinput = deconstructall(model, sequence, constantinput, from_method)
end

function deconstructall(model, sequence, constantinput, from_method)
    # TODO get default flags for dimension!!!!
    sequence = from_method(sequence,
                           dimensionality_defaultflags[model.hyperparams[:dimensionality]])
    sequence = deconstructlevel(sequence)
    constantinput = deconstructconstantinput(constantinput)
    return sequence, constantinput
end


"""
    generatelevel(predictor::LearningModel, g_model::AbstractGenerator,
                  meta_model::LearningModel, input=randinput(g_model);
                  return_intermediate=false)

Return the output of the given level predictor, generator and metadata predictor models
applied sequentially (order: `g_model`, `meta_model`, `predictor`) to the given input.
The output is a `Tuple` containing the predicted level and the level's metadata.
Whether the whole first screen or only a column will be used is controlled
by `first_screen`.

If `return_intermediate` is `true`, the output will be a `Tuple` of all intermediate
results (in the order listed above).
"""
function generatelevel(predictor::LearningModel, g_model::AbstractGenerator,
                       meta_model::LearningModel, first_screen=true,
                       input=randinput(g_model); return_intermediate=false)
    first_screen = generatescreen(g_model, input)
    constantinput = generatemetadata(meta_model, first_screen)
    if first_screen
        initialinput = reduce(hcat, map(col -> vcat(constantinput, col),
                                        eachcol(first_screen)))
    else
        initialinput = vcat(constantinput,
                            vec(view(first_screen, :, firstindex(first_screen, 2))))
    end
    level = generatesequence(predictor, initialinput)
    if return_intermediate
        return (first_screen, constantinput, level)
    else
        return level, constantinput
    end
end

function writelevel(predictor::LearningModel, g_model::AbstractGenerator,
                    meta_model::LearningModel, first_screen=true, input=randinput(g_model);
                    write_rom=nothing)
    level, constantinput = generatelevel(predictor, g_model, meta_model,
                                         first_screen, input)
    from_method = getfield(LevelFormatReverter,
                           Symbol("from" * String(dimensionality)[1:2]))
    level, constantinput = deconstructall(level, constantinput, from_method)

    writeback(level, constantinput)
    lmwrite(write_rom, constantinput.number)
    return level, constantinput
end

"""
    writelevels(predictor::LearningModel, g_model::AbstractGenerator,
                meta_model::LearningModel, inputs=randinput(g_model, 512),
                first_screen=true; write_rom=nothing)

Generate levels from the given inputs. `inputs` can also be an integer – the amount of
levels to generate.
"""
function writelevels(predictor::LearningModel, g_model::AbstractGenerator,
                     meta_model::LearningModel, inputs=randinput(g_model, 512),
                     first_screen=true; write_rom=nothing)
    inputs isa Number && (inputs = randinput(g_model, inputs))
    for input in inputs
        writelevel(predictor, g_model, meta_model, first_screen, input; write_rom=write_rom)
    end
end

# TODO a hack full of levels. generate 512 levels all with different numberered file names
function writehack()
end

function lmwrite(::Nothing, ::Any) end

function lmwrite(write_rom, number)
    levelnumber = string(number, base=16)
    @assert isfile(lmpath) ("cannot find Lunar Magic in the given path. Please download it "
                            * "(see setup scripts) and/or change the given path.")
    run(`wine $lmpath -ReconstructLevel $write_rom $levelnumber`)
end

end # module

