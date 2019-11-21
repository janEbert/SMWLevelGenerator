module LevelGenerator

using BSON
using Flux
using JuliaDB

using ..LevelFormatter: dimensionality_defaultflags
using ..DataIterator
using ..InputStatistics
using ..ModelUtils: LearningModel, AbstractGenerator, dataiteratorparams, togpu
using ..SequenceGenerator
using ..ScreenGenerator
using ..LevelFormatReverter
using ..LevelDeconstructor
using ..LevelWriter

export predict_hack, predict_vanilla, predict_levels, predict_level
export generate_reshaped_screen, generatelevel, writescreen, writelevel, writelevels

const lmpath = joinpath(@__DIR__, "..", "..", "tools",
                                  "lm304ebert21892usix", "Lunar Magic.exe")

const level105dbid_1d_singleline_t = 0x2b7d
const level105dbid_2d_t = 0x2b7d
const level105dbid_3d_t = 0x2b7d
const level105dbid_3d_tesx = 0x2b7d

const vanilladbids_2d_t = 0x2b0a:0x2bf4
# TODO Take `reverse_rows` into account!
# TODO and `each_tile`

function get_from_method(model::LearningModel)
    dimensionality = model.hyperparams[:dimensionality]
    from_method = getfield(LevelFormatReverter,
                           Symbol("from" * String(dimensionality)[1:2]))
end

function setup_prediction(modelpath::AbstractString, dbpath::AbstractString)
    cp = BSON.load(modelpath)
    model = togpu(cp[:model])
    db = DataIterator.loaddb(dbpath)
    from_method = get_from_method(model)
    return model, db, from_method
end

# TODO better function names (indicate that the levels are written to a file)
# TODO more views

# TODO a hack full of levels. generate 512 levels all with different numberered file names
function predict_hack()
    error("not implemented.")
end

function predict_vanilla(modelpath::AbstractString,
                         dbpath::AbstractString="levels_3d_flags_tesx.jdb",
                         indices=vanilladbids_2d_t; first_screen=true,
                         write_rom::Union{AbstractString, Nothing}=nothing)
    model, db, from_method = setup_prediction(modelpath, dbpath)
    # Create in parallel, write to ROM sequentially.
    for row in filter(r -> r.id in indices, db)
        sequence, constantinput = predict_from_row(model, row, from_method, first_screen)
        writeback(sequence, constantinput)
        lmwrite(write_rom, constantinput.number)
    end
end

function predict_vanilla(model::LearningModel, db::IndexedTable,
                         indices=vanilladbids_2d_t; first_screen=true,
                         write_rom::Union{AbstractString, Nothing}=nothing)
end

"""
    predict_levels(amount, modelpath, dbpath="levels_3d_flags_tesx.jdb";
                   first_screen=true, seed=nothing, write_rom=nothing)

Generate the given amount of levels from random database entries and write each to a file.

If `first_screen` is `true`, use the level's first screen as prediction input instead
of only the first part of the sequence.
If `write_rom` is not `nothing`, the given path to a ROM will be used to reconstruct the
generated level in the ROM (overwriting the previous contents).
A random seed can optionally be specified with `seed`.
"""
function predict_levels(amount, modelpath::AbstractString,
                        dbpath::AbstractString="levels_3d_flags_tesx.jdb";
                        first_screen=true, seed=nothing,
                        write_rom::Union{AbstractString, Nothing}=nothing)
    model, db, from_method = setup_prediction(modelpath, dbpath)
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
        writeback(sequence, constantinput)
        lmwrite(write_rom, constantinput.number)
    end
end

"""
    predict_level(modelpath, dbpath, dbid; first_screen=true, write_rom=nothing)

Predict the rest of the level with the given database ID and write it to a file.
If `first_screen` is `true`, use the level's first screen as prediction input instead
of only the first part of the sequence.
If `write_rom` is not `nothing`, the given path to a ROM will be used to reconstruct the
generated level in the ROM (overwriting the previous contents).
"""
function predict_level(modelpath::AbstractString,
                       dbpath::AbstractString="levels_3d_flags_tesx.jdb",
                       dbid::UInt64=UInt64(SequenceGenerator.level105dbid_3d_tesx);
                       first_screen=true, write_rom=nothing)
    model, db, from_method = setup_prediction(modelpath, dbpath)
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
    dataiterparams = dataiteratorparams(model)
    input = DataIterator.preprocess(row, false, Val(false), Val(dataiterparams.join_pad),
                                    Val(dataiterparams.as_matrix))
    if first_screen
        constantinput, sequence = generatesequence(model, getrange(input, 1:screencols))
    else
        constantinput, sequence = generatesequence(model, getrange(input, 1:1))
    end
    sequence, constantinput = deconstructall(model, sequence, constantinput, from_method)
end

function getrange(input::AbstractVector, range)
    range == 1:1 && return input[1]
    return input[range]
end

function getrange(input::AbstractMatrix, range)
    return input[:, range]
end

function deconstructall(model, sequence, constantinput, from_method)
    # TODO get default flags for dimension!!!!
    # if Symbol(from_method) === :from3d
    sequence = from_method(sequence,
                           dimensionality_defaultflags[model.hyperparams[:dimensionality]])
    # else
    #     # get default ground tile
    #     sequence = from_method(sequence)
    # end
    sequence = deconstructlevel(sequence)
    constantinput = deconstructconstantinput(constantinput)
    return sequence, constantinput
end


"""
    generate_reshaped_screen(g_model::AbstractGenerator, input=randinputs(g_model))

Return the output of the given generator applied to the given input.
"""
function generate_reshaped_screen(g_model::AbstractGenerator, input=randinputs(g_model))
    first_screen = generatescreen(g_model, input)
    reshape_first_screen(g_model, first_screen)
end

function reshape_first_screen(g_model, first_screen)
    if g_model.hyperparams[:dimensionality] === Symbol("1d")
        return vec(first_screen)
    elseif g_model.hyperparams[:dimensionality] === Symbol("2d")
        return reshape(first_screen, size(first_screen)[1:2])
    else
        return first_screen
    end
end

function build_first_screen(model::LearningModel, gen_first_screen, constantinput)
    dataiterparams = dataiteratorparams(model)
    if dataiterparams.join_pad
        error("not implemented.")
    else
        if dataiterparams.as_matrix
            maybevec = reduce(hcat, (vcat(constantinput[1:end - 1], 1, col)
                                     for col in DataIterator.slicedata(gen_first_screen)))
            resultmatrix = reshape(maybevec, size(maybevec, 1), size(maybevec, 2))
            resultmatrix[constantinputsize, end] = constantinput[end]
            return resultmatrix
        else
            result = [vcat(constantinput[1:end - 1], 1, col)
                      for col in DataIterator.slicedata(gen_first_screen)]
            result[end][constantinputsize] = constantinput[end]
        end
    end
end

function build_one_input(model::LearningModel, gen_first_screen, constantinput)
    dataiterparams = dataiteratorparams(model)
    if dataiterparams.join_pad
        error("not implemented.")
    else
        if dataiterparams.as_matrix
            reshape(vcat(constantinput,
                         first(DataIterator.slicedata(gen_first_screen))), :, 1)
        else
            vcat(constantinput, first(DataIterator.slicedata(gen_first_screen)))
        end
    end
end

"""
    generatelevel(predictor::LearningModel, g_model::AbstractGenerator,
                  meta_model::LearningModel; first_screen=true,
                  input=randinputs(g_model), return_intermediate=false)

Return the output of the given level predictor, generator and metadata predictor models
applied sequentially (order: `g_model`, `meta_model`, `predictor`) to the given input.
The output is a `Tuple` containing the predicted level and the level's metadata.
Whether the whole first screen or only a column will be used is controlled
by `first_screen`.

If `return_intermediate` is `true`, the output will be a `Tuple` of all intermediate
results (in the order listed above).
"""
function generatelevel(predictor::LearningModel, g_model::AbstractGenerator,
                       meta_model::LearningModel; first_screen=true,
                       input=randinputs(g_model), return_intermediate=false)
    gen_first_screen = generatescreen(g_model, input)
    constantinput = generatemetadata(meta_model, gen_first_screen)
    gen_first_screen = reshape_first_screen(g_model, gen_first_screen)
    if first_screen
        initialinput = build_first_screen(predictor, gen_first_screen, constantinput)
    else
        initialinput = build_one_input(predictor, gen_first_screen, constantinput)
    end
    level = generatesequence(predictor, initialinput)
    if return_intermediate
        return (gen_first_screen, constantinput, level)
    else
        return level, constantinput
    end
end


function writescreen(g_model::AbstractGenerator; input=randinputs(g_model),
                     write_rom=nothing, number::UInt16=0x105)
    first_screen = generate_reshaped_screen(g_model)
    from_method = get_from_method(g_model)
    if Symbol(get(g_model.hyperparams, :output_activation, Flux.sigmoid)) === :tanh
        # Normalize: (-1, 1) -> (0, 1)
        first_screen .+= 1
        first_screen ./= 2
    end
    first_screen = from_method(first_screen,
                               dimensionality_defaultflags[
                                   g_model.hyperparams[:dimensionality]])
    first_screen = deconstructlevel(first_screen)
    writemap(first_screen, "Level" * string(number, base=16, pad=3))
    lmwrite(write_rom, number)
end

function writelevel(predictor::LearningModel, g_model::AbstractGenerator,
                    meta_model::LearningModel; first_screen=true, input=randinputs(g_model),
                    write_rom=nothing)
    level, constantinput = generatelevel(predictor, g_model, meta_model,
                                         first_screen=first_screen, input=input)
    from_method = get_from_method(predictor)
    level, constantinput = deconstructall(level, constantinput, from_method)

    writeback(level, constantinput)
    lmwrite(write_rom, constantinput.number)
    return level, constantinput
end

"""
    writelevels(predictor::LearningModel, g_model::AbstractGenerator,
                meta_model::LearningModel, inputs=randinputs(g_model, 512),
                first_screen=true; write_rom=nothing)

Generate levels from the given inputs. `inputs` can also be an integer – the amount of
levels to generate.
"""
function writelevels(predictor::LearningModel, g_model::AbstractGenerator,
                     meta_model::LearningModel; inputs=randinputs(g_model, 512),
                     first_screen=true, write_rom=nothing)
    inputs isa Number && (inputs = randinputs(g_model, inputs))
    for input in inputs
        writelevel(predictor, g_model, meta_model,
                   first_screen=first_screen, input=input, write_rom=write_rom)
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
    @static if Sys.iswindows()
        run(`$lmpath -ReconstructLevel $write_rom $levelnumber`)
    else
        run(`wine $lmpath -ReconstructLevel $write_rom $levelnumber`)
    end
end

end # module

