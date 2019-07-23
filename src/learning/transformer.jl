module Transformer

import Transformers

using ..InputStatistics

export transformer1d, transformer2d, transformer3dtiles, transformer3d


function transformer1d(heads::Integer=8, attnhiddensize::Integer=4,
                       ffhiddensize::Integer=16)
    Transformers.Transformer(inputsize1d, heads, attnhiddensize, ffhiddensize)
end

function transformer2d(heads::Integer=8, attnhiddensize::Integer=8,
                       ffhiddensize::Integer=32, inputsize::Integer=inputsize2d)
    Transformers.Transformer(inputsize, heads, attnhiddensize, ffhiddensize)
end

function transformer3dtiles(heads::Integer=8, attnhiddensize::Integer=16,
                            ffhiddensize::Integer=64, inputsize::Integer=inputsize3dtile)
    Transformers.Transformer(inputsize, heads, attnhiddensize, ffhiddensize)
end

function transformer3d(heads::Integer=8, attnhiddensize::Integer=32,
                       ffhiddensize::Integer=128, inputsize::Integer=inputsize3d)
    Transformers.Transformer(inputsize, heads, attnhiddensize, ffhiddensize)
end

end # module

