using Test
using IsoPEPS
using LinearAlgebra

@testset "mps_bond_entanglement" begin
    # Test with a simple diagonal MPS tensor
    A_mps = zeros(Float64, 2, 2, 2)  # (physical, left, right)
    λ = 0.5
    A_mps[1, 1, 1] = 1.0
    A_mps[1, 2, 2] = 1.0
    A_mps[2, 1, 1] = λ
    A_mps[2, 2, 2] = λ
    
    S_bond, schmidt = mps_bond_entanglement(A_mps)
    @test S_bond >= 0.0
    @test all(schmidt .>= 0.0)
    @test all(schmidt .<= 1.0)
end

@testset "mps_physical_entanglement_product_state" begin
    # Test that a product state has zero physical entanglement
    # The tensor A[s,l,r,u,d] with l=r=u=d constraint represents a diagonal MPS.
    # M^s = diag(A[s,1,1,1,1], A[s,2,2,2,2]) = diag(1,1) for s=1, diag(λ,λ) for s=2
    # This gives a product state |ψ⟩ = (|0⟩ + λ|1⟩)^⊗N with S = 0.
    
    λ = 0.5
    A_mps = zeros(Float64, 2, 2, 2)  # (physical, left, right)
    A_mps[1, 1, 1] = 1.0
    A_mps[1, 2, 2] = 1.0
    A_mps[2, 1, 1] = λ
    A_mps[2, 2, 2] = λ
    
    S_physical, σ_physical = mps_physical_entanglement(A_mps, 10)
    S_physical_infinite, σ_physical_infinite = mps_physical_entanglement_infinite(A_mps)
    
    println("\n=== MPS Entanglement Entropy (Product State) ===")
    println("Physical entanglement (N=10): S = ", S_physical)
    println("Physical entanglement infinite: S = ", S_physical_infinite)
    println("Expected physical entropy: S = 0 (product state)")
    
    # The physical entanglement is 0 because |ψ⟩ = (|0⟩ + λ|1⟩)^⊗N is a product state
    @test abs(S_physical) < 1e-10
    @test abs(S_physical_infinite) < 1e-10
end

@testset "mps_physical_entanglement_entangled_state" begin
    # Test with a GHZ-like MPS that has maximal entanglement
    # |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    # A^0 = [[1, 0], [0, 0]], A^1 = [[0, 0], [0, 1]]
    
    A_ghz = zeros(ComplexF64, 2, 2, 2)  # (physical, left, right)
    A_ghz[1, 1, 1] = 1.0  # A^0[0,0] = 1
    A_ghz[2, 2, 2] = 1.0  # A^1[1,1] = 1
    
    # For GHZ state, entanglement entropy = log(2) ≈ 0.693
    S_physical, σ_physical = mps_physical_entanglement(A_ghz, 10)
    S_physical_infinite, σ_physical_infinite = mps_physical_entanglement_infinite(A_ghz)
    
    println("\n=== MPS Entanglement Entropy (GHZ-like State) ===")
    println("Physical entanglement (N=10): S = ", S_physical)
    println("Physical entanglement infinite: S = ", S_physical_infinite)
    println("Expected: S = log(2) ≈ ", log(2))
    
    @test isapprox(S_physical, log(2), atol=1e-8)
    @test isapprox(S_physical_infinite, log(2), atol=1e-8)
end

@testset "mps_entanglement_consistency" begin
    # Test that finite-N entanglement converges to infinite-N result
    # Use a random MPS tensor
    
    A_random = randn(ComplexF64, 2, 3, 3)  # bond_dim = 3
    
    # Compute at different N values
    S_N6, _ = mps_physical_entanglement(A_random, 6)
    S_N8, _ = mps_physical_entanglement(A_random, 8)
    S_N10, _ = mps_physical_entanglement(A_random, 10)
    S_inf, _ = mps_physical_entanglement_infinite(A_random)
    
    println("\n=== Finite-N Convergence Test ===")
    println("S(N=6)  = ", S_N6)
    println("S(N=8)  = ", S_N8)
    println("S(N=10) = ", S_N10)
    println("S(∞)    = ", S_inf)
    
    # The sequence should be converging towards S_inf
    # Check that S_N10 is closer to S_inf than S_N6
    @test abs(S_N10 - S_inf) ≤ abs(S_N6 - S_inf) + 1e-6
end

@testset "entanglement_bounds" begin
    # Entanglement entropy should be non-negative and bounded by log(bond_dim)
    for bond_dim in [2, 3, 4]
        A = randn(ComplexF64, 2, bond_dim, bond_dim)
        
        S_bond, _ = mps_bond_entanglement(A)
        S_phys, _ = mps_physical_entanglement(A, 6)
        S_inf, _ = mps_physical_entanglement_infinite(A)
        
        @test S_bond >= -1e-10
        @test S_phys >= -1e-10
        @test S_inf >= -1e-10
        
        # Upper bound: log(bond_dim)
        max_entropy = log(bond_dim)
        @test S_bond ≤ max_entropy + 1e-6
        @test S_inf ≤ max_entropy + 1e-6
    end
end

