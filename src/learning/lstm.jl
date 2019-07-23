module LSTM

import Flux

using ..InputStatistics

export lstm1d, lstm2d, lstm3dtiles, lstm3d


function makehiddenlayers(hiddensize::Integer, hiddenlayers::Integer, ::Val{false})
    (Flux.LSTM(hiddensize, hiddensize) for _ in 1:hiddenlayers)
end

function makehiddenlayers(hiddensize::Integer, hiddenlayers::Integer, ::Val{true})
    if hiddenlayers > 1
        return (Flux.SkipConnection(Flux.Chaing(Flux.LSTM(hiddensize, hiddensize),
                                                makehiddenlayers(hiddensize,
                                                                 hiddenlayers - 1)),
                                    (a, b) -> a + b))
    else
        return Flux.LSTM(hiddensize, hiddensize)
    end
end

function makelstm(hiddensize::Integer, hiddenlayers::Integer, inputsize::Integer,
                  skipconnections::Bool=false)
    hiddenlayers = makehiddenlayers(hiddensize, hiddenlayers, Val(skipconnections))
    Flux.Chain(
        Flux.LSTM(inputsize, hiddensize),
        hiddenlayers...,
        Flux.Dense(hiddensize, inputsize)
    ) |> togpu
end

function lstm1d(hiddensize::Integer=32, hiddenlayers::Integer=1,
                skipconnections::Bool=false)
    makelstm(hiddensize, hiddenlayers, inputsize1d, skipconnections)
end

function lstm2d(hiddensize::Integer=64, hiddenlayers::Integer=1,
                inputsize::Integer=inputsize2d, skipconnections::Bool=false)
    makelstm(hiddensize, hiddenlayers, inputsize, skipconnections)
end

function lstm3dtiles(hiddensize::Integer=128, hiddenlayers::Integer=1,
                inputsize::Integer=inputsize3dtiles, skipconnections::Bool=false)
    makelstm(hiddensize, hiddenlayers, inputsize, skipconnections)
end

function lstm3d(hiddensize::Integer=128, hiddenlayers::Integer=2,
                inputsize::Integer=inputsize3d, skipconnections::Bool=false)
    makelstm(hiddensize, hiddenlayers, inputsize, skipconnections)
end


end # module

