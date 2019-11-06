module WassersteinGAN

include("wasserstein_discriminator.jl")
include("wasserstein_generator.jl")

using .WassersteinDiscriminator
using .WassersteinGenerator

export WassersteinDiscriminatorModel
export wsdiscriminator1d, wsdiscriminator2d, wsdiscriminator3dtiles, wsdiscriminator3d
export WassersteinGeneratorModel
export wsgenerator1d, wsgenerator2d, wsgenerator3dtiles, wsgenerator3d

end # module

