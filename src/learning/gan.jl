module GAN

include("discriminator.jl")
include("generator.jl")

using .Discriminator
using .Generator

export DiscriminatorModel
export discriminator1d, discriminator2d, discriminator3dtiles, discriminator3d
export GeneratorModel
export generator1d, generator2d, generator3dtiles, generator3d

end # module

