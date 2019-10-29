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

include("data/level_format_reverter.jl")
include("data/level_deconstructor.jl")


include("learning/input_statistics.jl")

# Models
include("learning/model_utils.jl")
include("learning/transformer.jl")
include("learning/lstm.jl")
include("learning/random_predictor.jl")

include("learning/training_utils.jl")
include("learning/training_loop.jl")

include("learning/gan.jl")
include("learning/metadata_predictor.jl")
include("learning/gan_training.jl")

# Generation
include("learning/sequence_generator.jl")
include("learning/screen_generator.jl")
include("data/level_writer.jl")
include("data/level_generator.jl")

end # module

