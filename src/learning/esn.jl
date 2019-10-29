module ESN

using ..InputStatistics

export esn1d, esn2d, esn3dtiles, esn3d

struct EchoStateNetwork{F <: AbstractFloat, I <: Integer}
    in_size::I
    out_size::I
    reservoir_size::I
    leaking_rate::F
    weights_in::AbstractMatrix{F}
    weights_reservior::AbstractMatrix{F}
    weights_out::AbstractMatrix{F}
    spectral_radius::F
    warmup_length::I
    # This could be a fixed size matrix for efficiency.
    states::AbstractVector{AbstractVector{F}}
    regularization_coefficient::F
    # TODO put on hold
end

function makeesn(inputsize, outputsize, reservoirsize, sparsity, spectralradius, noiselevel,
                 leakingrate, teacherforcing)
    EchoStateNetwork{Float32}(; Ni=inputsize, No=outputsize, Nr=reservoirsize,
                              sparsity=sparsity, spectral_radius=spectralradius,
                              noise_level=noiselevel, leaking_rate=leakingrate,
                              teacher_forcing=teacherforcing)
end

function esn1d(reservoirsize::Integer=2048, sparsity::Real=0.95, spectralradius::Real=0.95,
               noiselevel::Real=0.001, leakingrate::Real=0.99, teacherforcing::Bool=true,
               inputsize::Integer=inputsize1d,
               outputsize::Integer=inputsize - constantinputsize + 1)
    makeesn(inputsize, outputsize, reservoirsize, sparsity, spectralradius, noiselevel,
            leakingrate, teacherforcing)
end

function esn2d(reservoirsize::Integer=4096, sparsity::Real=0.95, spectralradius::Real=1.2,
               noiselevel::Real=0.001, leakingrate::Real=0.99, teacherforcing::Bool=true,
               inputsize::Integer=inputsize2d,
               outputsize::Integer=inputsize - constantinputsize + 1)
    makeesn(inputsize, outputsize, reservoirsize, sparsity, spectralradius, noiselevel,
            leakingrate, teacherforcing)
end

function esn3dtiles(reservoirsize::Integer=8192, sparsity::Real=0.95,
                    spectralradius::Real=1.2, noiselevel::Real=0.001,
                    leakingrate::Real=0.99, teacherforcing::Bool=true,
                    inputsize::Integer=inputsize3dtiles,
                    outputsize::Integer=inputsize - constantinputsize + 1)
    makeesn(inputsize, outputsize, reservoirsize, sparsity, spectralradius, noiselevel,
            leakingrate, teacherforcing)
end

function esn3d(reservoirsize::Integer=16384, sparsity::Real=0.95, spectralradius::Real=1.4,
               noiselevel::Real=0.001, leakingrate::Real=0.99, teacherforcing::Bool=true,
               inputsize::Integer=inputsize3d,
               outputsize::Integer=inputsize - constantinputsize + 1)
    makeesn(inputsize, outputsize, reservoirsize, sparsity, spectralradius, noiselevel,
            leakingrate, teacherforcing)
end

end # module

