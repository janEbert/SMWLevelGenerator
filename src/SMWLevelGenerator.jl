module SMWLevelGenerator

include("data/xytables.jl")
include("data/default_dictionary.jl")

include("data/level_statistics.jl")
include("data/tiles.jl")
include("data/secondary_level_stats.jl")
include("data/sprites.jl")
include("data/level_builder.jl")

include("data/level_formatter.jl")
include("data/data_compressor.jl")
include("data/database.jl")
include("data/data_iterator.jl")


include("learning/input_statistics.jl")
include("learning/transformer.jl")
include("learning/lstm.jl")
include("learning/esn.jl")
include("learning/random_model.jl")

include("learning/training_loop.jl")

end # module

