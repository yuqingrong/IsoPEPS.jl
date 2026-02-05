using IsoPEPS
using Test

@testset "gates" begin
    include("gates.jl")
end

@testset "transfer_matrix" begin
    include("transfer_matrix.jl")
end

@testset "observables" begin
    include("observables.jl")
end


@testset "quantum_channels" begin
    include("quantum_channels.jl")
end

@testset "reference" begin
    include("reference.jl")
end

@testset "training" begin
    include("training.jl")
end

@testset "visualization" begin
    include("visualization.jl")
end
