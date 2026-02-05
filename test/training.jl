using Test
using IsoPEPS
using LinearAlgebra
using Random

@testset "initialize_tfim_params modes" begin
    g = 1.0
    p, row, nqubits = 2, 2, 3
    
    # Test all modes return correct size
    for mode in [:meanfield, :entangled, :random]
        params = initialize_tfim_params(p, nqubits, g; mode=mode)
        @test length(params) == 2 * nqubits * p
    end
    
    # Test entangled mode creates GHZ/Bell state
    # Only layer 1, last qubit has Rx(π/2); all others are identity
    entangled_params = initialize_tfim_params(p, nqubits, g; mode=:entangled)
    for layer in 1:p
        for q in 1:nqubits
            idx = 2*nqubits*(layer-1) + 2*(q-1) + 1
            if layer == 1 && q == nqubits
                # Layer 1, last qubit: creates superposition
                @test entangled_params[idx] ≈ π/2      # Rx(π/2)
                @test entangled_params[idx+1] ≈ 0.0    # Rz(0)
            else
                # All others: identity to preserve GHZ
                @test entangled_params[idx] ≈ 0.0      # Rx(0)
                @test entangled_params[idx+1] ≈ 0.0    # Rz(0)
            end
        end
    end
    
    # Test meanfield mode uses mean-field angle
    meanfield_params = initialize_tfim_params(p, nqubits, g; mode=:meanfield)
    θ_mf = atan(1.0 / g)
    for layer in 1:p
        for q in 1:nqubits
            idx = 2*nqubits*(layer-1) + 2*(q-1) + 1
            @test abs(meanfield_params[idx] - θ_mf) < 0.5  # Rx near mean-field
        end
    end
end

@testset "optimize_circuit" begin
    # Small test: 2 iterations only
    g, J = 2.0, 1.0
    p, row, nqubits = 1, 2, 3
    # Uses 3 params per qubit per layer (Rz-Ry-Rz decomposition)
    params = rand(3 * nqubits * p)
    
    result = optimize_circuit(params, J, g, p, row, nqubits; 
                              maxiter=2, conv_step=10, samples=50)
    
    @test result isa CircuitOptimizationResult
    @test result.final_cost isa Float64
    @test length(result.energy_history) > 0
    @test length(result.final_gates) == row
    @test result.final_params isa Vector{Float64}
    @test result.final_Z_samples isa Vector{Float64}
    @test result.final_X_samples isa Vector{Float64}
    @test result.converged isa Bool
end

@testset "optimize_exact" begin
    # Small test: 2 iterations only
    g, J = 2.0, 1.0
    p, row, nqubits = 1, 2, 3
    # Uses 3 params per qubit per layer (Rz-Ry-Rz decomposition)
    params = rand(3 * nqubits * p)
    
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
