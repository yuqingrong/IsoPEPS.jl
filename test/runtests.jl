using IsoPEPS
using Test

@testset "exact" begin
    include("exact.jl")
end

@testset "gate_and_cost" begin
    include("gate_and_cost.jl")
end

@testset "quantum_channels" begin
    include("quantum_channels.jl")
end

@testset "refer" begin
    include("refer.jl")
end