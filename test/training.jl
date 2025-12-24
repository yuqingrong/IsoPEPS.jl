using Test
using IsoPEPS
using LinearAlgebra

@testset "optimize_circuit" begin
    # Small test: 2 iterations only
    g, J = 2.0, 1.0
    p, row, nqubits = 1, 2, 3
    params = rand(2 * nqubits * p)
    
    result = optimize_circuit(params, J, g, p, row, nqubits; 
                              maxiter=2, conv_step=10, samples=50)
    
    @test result isa CircuitOptimizationResult
    @test result.energy isa Float64
    @test length(result.energy_history) > 0
    @test length(result.gates) == row
    @test result.params isa Vector{Float64}
    @test result.Z_samples isa Vector{Float64}
    @test result.X_samples isa Vector{Float64}
    @test result.converged isa Bool
end

@testset "optimize_exact" begin
    # Small test: 2 iterations only
    g, J = 2.0, 1.0
    p, row, nqubits = 1, 2, 3
    params = rand(2 * nqubits * p)
    
    result = optimize_exact(params, J, g, p, row, nqubits; maxiter=2)
    
    @test result isa ExactOptimizationResult
    @test result.energy isa Float64
    @test length(result.energy_history) > 0
    @test length(result.gates) == row
    @test result.gap > 0
    @test result.eigenvalues isa Vector{Float64}
    @test result.X_expectation isa Float64
    @test result.ZZ_vertical isa Float64
    @test result.ZZ_horizontal isa Float64
    @test result.converged isa Bool
end

@testset "optimize_manifold" begin
    using Manifolds
    
    g, J = 2.0, 1.0
    row, nqubits = 1, 3
    dim = 2^nqubits
    
    # Create unitary manifold
    M = Stiefel(dim, dim, ℂ)
    gate = rand(M)
    
    result = optimize_manifold(gate, row, nqubits, M, J, g; maxiter=2)
    
    @test result isa ManifoldOptimizationResult
    @test result.energy isa Float64
    @test length(result.energy_history) > 0
    @test size(result.gate) == (dim, dim)
    @test result.gap_history isa Vector{Float64}
    @test result.converged isa Bool
end
