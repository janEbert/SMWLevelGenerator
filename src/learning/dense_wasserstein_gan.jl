module DenseWassersteinGAN

include("dense_wasserstein_discriminator.jl")
include("dense_wasserstein_generator.jl")

using .DenseWassersteinDiscriminator
using .DenseWassersteinGenerator

export DenseWassersteinDiscriminatorModel
export densewsdiscriminator1d, densewsdiscriminator2d
export densewsdiscriminator3dtiles, densewsdiscriminator3d
export DenseWassersteinGeneratorModel
export densewsgenerator1d, densewsgenerator2d, densewsgenerator3dtiles, densewsgenerator3d

end # module

