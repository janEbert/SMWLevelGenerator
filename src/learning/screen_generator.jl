module ScreenGenerator

import Flux

using ..InputStatistics
using ..ModelUtils: LearningModel, AbstractGenerator, togpu, tocpu
using ..LevelStatistics

export generatescreen, generatemetadata, randinputs

function randinputs(g_model::AbstractGenerator, amount=1)
    if g_model.hyperparams[:dimensionality] === Symbol("1d")
        togpu(rand(1, g_model.hyperparams[:inputsize], amount))
    else
        togpu(rand(1, 1, g_model.hyperparams[:inputsize], amount))
    end
end

"""
    generatescreen(g_model::AbstractGenerator, input=randinputs(g_model))

Return the generator's output for the given input.
"""
function generatescreen(g_model::AbstractGenerator, input=randinputs(g_model))
    Flux.testmode!(g_model)
    output = tocpu(Flux.data(g_model(togpu(input))))
    Flux.testmode!(g_model, false)
    return output
end

"""
    generatemetadata(meta_model::LearningModel, screen)

Return the meta model's output when applied to the given (first) screen.
"""
function generatemetadata(meta_model::LearningModel, screen)
    Flux.testmode!(meta_model)
    output = tocpu(Flux.data(meta_model(togpu(screen))))
    Flux.testmode!(meta_model, false)
    return output
end

end

