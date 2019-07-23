module ESN

using EchoStateNetworks

using ..InputStatistics

export esn1d, esn2d, esn3dtiles, esn3d

function makeesn(inputsize, reservoirsize, sparsity, spectralradius, noiselevel,
                 leakingrate, teacherforcing)
    EchoStateNetwork(; Ni=inputsize, No=inputsize, Nr=reservoirsize, sparsity=sparsity,
                     spectral_radius=spectralradius, noise_level=noiselevel,
                     leaking_rate=leakingrate, teacher_forcing=teacherforcing)
end

function esn1d(reservoirsize::Integer=2048, sparsity::Real=0.95, spectralradius::Real=0.95,
               noiselevel::Real=0.001, leakingrate::Real=0.99, teacherforcing::Bool=true)
    makeesn(inputsize1d, reservoirsize, sparsity, spectralradius, noiselevel, leakingrate,
            teacherforcing)
end

function esn2d(reservoirsize::Integer=4096, sparsity::Real=0.95, spectralradius::Real=1.2,
               noiselevel::Real=0.001, leakingrate::Real=0.99, teacherforcing::Bool=true,
               inputsize::Integer=inputsize2d)
    makeesn(inputsize, reservoirsize, sparsity, spectralradius, noiselevel, leakingrate,
            teacherforcing)
end

function esn3dtiles(reservoirsize::Integer=8192, sparsity::Real=0.95,
                    spectralradius::Real=1.2, noiselevel::Real=0.001,
                    leakingrate::Real=0.99, teacherforcing::Bool=true,
                    inputsize::Integer=inputsize3dtiles)
    makeesn(inputsize, reservoirsize, sparsity, spectralradius, noiselevel, leakingrate,
            teacherforcing)
end

function esn3d(reservoirsize::Integer=16384, sparsity::Real=0.95, spectralradius::Real=1.4,
               noiselevel::Real=0.001, leakingrate::Real=0.99, teacherforcing::Bool=true,
               inputsize::Integer=inputsize3d)
    makeesn(inputsize, reservoirsize, sparsity, spectralradius, noiselevel, leakingrate,
            teacherforcing)
end

end # module

