module SMWLevelGenerator

# Data preparation
export generatedb, generate_default_databases
# These are primarily for testing and benchmarking.
export loaddb, dataiterator, dataiterator!, dataiteratorchannel

# Model utilities
export toggle_gpu

# Sequence prediction models
export lstm1d, lstm2d, lstm3dtiles, lstm3d
export transformer1d, transformer2d, transformer3dtiles, transformer3d
export random1d, random2d, random3dtiles, random3d

# GANs
export discriminator1d, discriminator2d, discriminator3dtiles, discriminator3d
export generator1d, generator2d, generator3dtiles, generator3d
export wsdiscriminator1d, wsdiscriminator2d, wsdiscriminator3dtiles, wsdiscriminator3d
export wsgenerator1d, wsgenerator2d, wsgenerator3dtiles, wsgenerator3d
# Metadata predictors
export metapredictor1d, metapredictor2d, metapredictor3dtiles, metapredictor3d
export densemetapredictor1d, densemetapredictor2d
export densemetapredictor3dtiles, densemetapredictor3d

# Training loops
export TPs, trainingloop!
export GTPs, gan_trainingloop!
export MTPs, meta_trainingloop!

# Level generation
export predict_hack, predict_vanilla, predict_levels, predict_level
export generatelevel, writelevel, writelevels


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
include("learning/wasserstein_gan.jl")
include("learning/metadata_predictor.jl")
include("learning/dense_metadata_predictor.jl")
include("learning/gan_training.jl")
include("learning/meta_training.jl")

# Generation
include("learning/sequence_generator.jl")
include("learning/screen_generator.jl")
include("data/level_writer.jl")
include("data/level_generator.jl")


# Import exports
using .Database
using .DataIterator
using .ModelUtils
using .LSTM
using .Transformer
using .RandomPredictor
using .GAN
using .WassersteinGAN
using .MetadataPredictor
using .DenseMetadataPredictor
using .TrainingLoop
using .GANTraining
using .MetaTraining
using .LevelGenerator

end # module

