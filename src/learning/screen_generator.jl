module ScreenGenerator

using ..InputStatistics
using ..ModelUtils: LearningModel, AbstractGenerator, togpu
using ..LevelStatistics

export generatescreen, generatemetadata, randinput

function randinput(g_model::AbstractGenerator, amount=1)
    if g_model.hyperparams[:dimensionality] === Symbol("1d")
        togpu(rand(1, g_model.hyperparams[:inputsize], amount))
    else
        togpu(rand(1, 1, g_model.hyperparams[:inputsize], amount))
    end
end

"""
    generatescreen(g_model::AbstractGenerator, input=randinput(g_model))

Return the generator's output for the given input.
"""
generatescreen(g_model::AbstractGenerator, input=randinput(g_model)) = g_model(togpu(input))

"""
    generatemetadata(meta_model::LearningModel, screen)

Return the meta model's output when applied to the given (first) screen.
"""
generatemetadata(meta_model::LearningModel, screen) = meta_model(screen)

end

