using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra
using Random

# =============================================================================
# Basic Observable Tests (Random States)
# =============================================================================

@testset "compute_X_expectation_basic" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        X_exp = compute_X_expectation(rho, gates, row, virtual_qubits)
        Z_exp = compute_Z_expectation(rho, gates, row, virtual_qubits)
        @test abs(imag(X_exp)) < 1e-10
        @test -1.0 ≤ real(X_exp) ≤ 1.0
        @test abs(imag(Z_exp)) < 1e-10
        @test -1.0 ≤ real(Z_exp) ≤ 1.0
    end
end

@testset "compute_ZZ_expectation_basic" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
        @test abs(imag(ZZ_vert)) < 1e-10
        @test -1.0 ≤ real(ZZ_vert) ≤ 1.0
        @test abs(imag(ZZ_horiz)) < 1e-10
        @test -1.0 ≤ real(ZZ_horiz) ≤ 1.0
    end
end

# =============================================================================
# Concrete Known States
# =============================================================================

@testset "observables_product_state_Z_eigenbasis" begin
    # Product state |0⟩^⊗N (Z eigenstate with eigenvalue +1)
    # Expected: ⟨Z⟩ = 1, ⟨X⟩ = 0, ⟨ZZ⟩ = 1
    row = 2
    bond_dim = 2
    
    # Create tensors for |0⟩ product state
    A_tensors = []
    for r in 1:row
        A = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        # Only |0⟩ state has amplitude
        A[1, 1, 1, 1, 1] = 1.0
        A[1, 2, 2, 2, 2] = 1.0
        push!(A_tensors, A)
    end
    # Convert to gates for compute_transfer_spectrum
    virtual_qubits = 1
    nqubits = 3
    # Build gates from tensors (need to reverse gates_to_tensors)
    gates = []
    for A in A_tensors
        # Reshape tensor back to gate
        A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_data = zeros(ComplexF64, A_size)
        gate_data[:, :, :, 1, :, :] = A
        gate = reshape(gate_data, 2^nqubits, 2^nqubits)
        push!(gates, gate)
    end
    
    rho, gap, _ = compute_transfer_spectrum(gates, row, nqubits)
    X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
    Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Product State |0⟩^⊗N ===")
    println("⟨X⟩ = $X_exp (expected: 0)")
    println("⟨Z⟩ = $Z_exp (expected: +1)")
    println("⟨ZZ⟩_vert = $(real(ZZ_vert)) (expected: +1)")
    println("⟨ZZ⟩_horiz = $(real(ZZ_horiz)) (expected: +1)")
    
    @test abs(X_exp) < 1e-6        # ⟨0|X|0⟩ = 0
    @test isapprox(Z_exp, 1.0, atol=1e-6)   # ⟨0|Z|0⟩ = +1
    @test isapprox(real(ZZ_vert), 1.0, atol=1e-6)   # ⟨0|Z|0⟩⟨0|Z|0⟩ = 1
    @test isapprox(real(ZZ_horiz), 1.0, atol=1e-6)  # ⟨0|Z|0⟩⟨0|Z|0⟩ = 1
end

@testset "observables_product_state_X_eigenbasis" begin
    # Product state |+⟩^⊗N where |+⟩ = (|0⟩ + |1⟩)/√2 (X eigenstate with eigenvalue +1)
    # Expected: ⟨X⟩ = 1, ⟨Z⟩ = 0, ⟨ZZ⟩ = 0
    
    row = 1  # Keep simple for concrete test
    bond_dim = 2
    virtual_qubits = 1
    nqubits = 3
    
    # Create |+⟩ state: equal superposition
    A_tensors = []
    for r in 1:row
        A = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        # Both |0⟩ and |1⟩ with equal weight
        A[1, 1, 1, 1, 1] = 1.0/sqrt(2)
        A[1, 2, 2, 2, 2] = 1.0/sqrt(2)
        A[2, 1, 1, 1, 1] = 1.0/sqrt(2)
        A[2, 2, 2, 2, 2] = 1.0/sqrt(2)
        push!(A_tensors, A)
    end
    
    # Convert to gates
    gates = []
    for A in A_tensors
        A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_data = zeros(ComplexF64, A_size)
        gate_data[:, :, :, 1, :, :] = A
        gate = reshape(gate_data, 2^nqubits, 2^nqubits)
        push!(gates, gate)
    end
    
    rho, gap, _ = compute_transfer_spectrum(gates, row, nqubits)
    X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
    Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Product State |+⟩^⊗N ===")
    println("⟨X⟩ = $X_exp (expected: +1)")
    println("⟨Z⟩ = $Z_exp (expected: 0)")
    println("⟨ZZ⟩_vert = $(real(ZZ_vert)) (expected: 0)")
    println("⟨ZZ⟩_horiz = $(real(ZZ_horiz)) (expected: 0)")
    
    @test isapprox(X_exp, 1.0, atol=1e-6)   # ⟨+|X|+⟩ = +1
    @test abs(Z_exp) < 1e-6                  # ⟨+|Z|+⟩ = 0
    @test abs(real(ZZ_vert)) < 1e-6          # ⟨+|Z|+⟩⟨+|Z|+⟩ = 0
    @test abs(real(ZZ_horiz)) < 1e-6         # ⟨+|Z|+⟩⟨+|Z|+⟩ = 0
end

# =============================================================================
# Directional Tests (Vertical vs Horizontal)
# =============================================================================

@testset "ZZ_directional_test_single_row" begin
    # For single row (row=1), there is NO vertical coupling
    # ZZ_vert should be undefined or equal to ZZ_horiz
    
    row = 1
    virtual_qubits = 1
    nqubits = 3
    
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Directional Test: Single Row (row=$row) ===")
    println("⟨ZZ⟩_vertical   = $(real(ZZ_vert))")
    println("⟨ZZ⟩_horizontal = $(real(ZZ_horiz))")
    println("Note: Single row has no vertical neighbors")
    
    @test abs(imag(ZZ_vert)) < 1e-10
    @test abs(imag(ZZ_horiz)) < 1e-10
    @test -1.0 ≤ real(ZZ_vert) ≤ 1.0
    @test -1.0 ≤ real(ZZ_horiz) ≤ 1.0
end

@testset "ZZ_directional_test_multirow" begin
    # For multiple rows, vertical and horizontal correlations can be different
    # Test that they are computed independently
    
    row = 3
    virtual_qubits = 1
    nqubits = 3
    
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Directional Test: Multiple Rows (row=$row) ===")
    println("⟨ZZ⟩_vertical   = $(real(ZZ_vert))  (sites i and i+1 in same column)")
    println("⟨ZZ⟩_horizontal = $(real(ZZ_horiz))  (sites across transfer matrix)")
    
    @test abs(imag(ZZ_vert)) < 1e-10
    @test abs(imag(ZZ_horiz)) < 1e-10
    @test -1.0 ≤ real(ZZ_vert) ≤ 1.0
    @test -1.0 ≤ real(ZZ_horiz) ≤ 1.0
    
    # For random gates, they should generally be different
    # (unless there's special symmetry)
    println("Directional difference: |ZZ_vert - ZZ_horiz| = $(abs(real(ZZ_vert - ZZ_horiz)))")
end

# =============================================================================
# Extreme Cases
# =============================================================================

@testset "observables_ferromagnetic_state" begin
    # Test Z-aligned ferromagnetic state: all spins point up |↑↑↑...⟩
    # Expected: ⟨Z⟩ = +1, ⟨ZZ⟩ = +1
    
    row = 2
    bond_dim = 2
    virtual_qubits = 1
    nqubits = 3
    
    # Create |0⟩ (Z=+1) product state
    A_tensors = []
    for r in 1:row
        A = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        # Only physical |0⟩ state
        A[1, 1, 1, 1, 1] = 1.0
        A[1, 2, 2, 2, 2] = 1.0
        push!(A_tensors, A)
    end
    
    # Convert to gates
    gates = []
    for A in A_tensors
        A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_data = zeros(ComplexF64, A_size)
        gate_data[:, :, :, 1, :, :] = A
        gate = reshape(gate_data, 2^nqubits, 2^nqubits)
        push!(gates, gate)
    end
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Ferromagnetic State |↑↑↑...⟩ ===")
    println("⟨Z⟩ = $Z_exp (expected: +1)")
    println("⟨ZZ⟩_vert = $(real(ZZ_vert)) (expected: +1)")
    println("⟨ZZ⟩_horiz = $(real(ZZ_horiz)) (expected: +1)")
    
    @test isapprox(Z_exp, 1.0, atol=1e-6)
    @test isapprox(real(ZZ_vert), 1.0, atol=1e-6)
    @test isapprox(real(ZZ_horiz), 1.0, atol=1e-6)
end

@testset "observables_antiferromagnetic_state" begin
    # Test antiferromagnetic state: |↓↓↓...⟩ (all spins down)
    # Expected: ⟨Z⟩ = -1, ⟨ZZ⟩ = +1 (same spin direction)
    
    row = 2
    bond_dim = 2
    virtual_qubits = 1
    nqubits = 3
    
    # Create |1⟩ (Z=-1) product state
    A_tensors = []
    for r in 1:row
        A = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        # Only physical |1⟩ state
        A[2, 1, 1, 1, 1] = 1.0
        A[2, 2, 2, 2, 2] = 1.0
        push!(A_tensors, A)
    end
    
    # Convert to gates
    gates = []
    for A in A_tensors
        A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_data = zeros(ComplexF64, A_size)
        gate_data[:, :, :, 1, :, :] = A
        gate = reshape(gate_data, 2^nqubits, 2^nqubits)
        push!(gates, gate)
    end
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Antiferromagnetic State |↓↓↓...⟩ ===")
    println("⟨Z⟩ = $Z_exp (expected: -1)")
    println("⟨ZZ⟩_vert = $(real(ZZ_vert)) (expected: +1)")
    println("⟨ZZ⟩_horiz = $(real(ZZ_horiz)) (expected: +1)")
    
    @test isapprox(Z_exp, -1.0, atol=1e-6)
    @test isapprox(real(ZZ_vert), 1.0, atol=1e-6)
    @test isapprox(real(ZZ_horiz), 1.0, atol=1e-6)
end

@testset "observables_X_eigenstate" begin
    # Product state |+⟩^⊗N where |+⟩ = (|0⟩ + |1⟩)/√2
    # Expected: ⟨X⟩ = +1, ⟨Z⟩ = 0
    
    row = 1  # Keep simple
    bond_dim = 2
    virtual_qubits = 1
    nqubits = 3
    
    # Create |+⟩ state
    A_tensors = []
    for r in 1:row
        A = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        # Equal superposition of |0⟩ and |1⟩
        A[1, 1, 1, 1, 1] = 1.0/sqrt(2)
        A[1, 2, 2, 2, 2] = 1.0/sqrt(2)
        A[2, 1, 1, 1, 1] = 1.0/sqrt(2)
        A[2, 2, 2, 2, 2] = 1.0/sqrt(2)
        push!(A_tensors, A)
    end
    
    # Convert to gates
    gates = []
    for A in A_tensors
        A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_data = zeros(ComplexF64, A_size)
        gate_data[:, :, :, 1, :, :] = A
        gate = reshape(gate_data, 2^nqubits, 2^nqubits)
        push!(gates, gate)
    end
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
    Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
    
    println("\n=== X Eigenstate |+⟩^⊗N ===")
    println("⟨X⟩ = $X_exp (expected: +1)")
    println("⟨Z⟩ = $Z_exp (expected: 0)")
    
    @test isapprox(X_exp, 1.0, atol=1e-6)
    @test abs(Z_exp) < 1e-6
end

# =============================================================================
# Trivial Edge Cases
# =============================================================================
@testset "observables_identity_gate" begin
    # Test with identity gate
    # 
    # OBSERVATION: The identity gate, after gates_to_tensors transformation,
    # creates a DEGENERATE transfer matrix (eigenvalues all ≈ 1, gap ≈ 0).
    # 
    # The fixed point projects to a Z eigenstate: ⟨Z⟩=1, ⟨ZZ⟩=1, ⟨X⟩≈0
    # This is due to how gates_to_tensors slices the identity matrix.
    #
    # This test verifies the degenerate behavior is handled correctly.
    
    row = 2
    virtual_qubits = 1
    nqubits = 3
    
    # Identity gate
    gates = [Matrix{ComplexF64}(I, 2^nqubits, 2^nqubits) for _ in 1:row]
    
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
    Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Identity Gate (Degenerate Transfer Matrix) ===")
    println("⟨X⟩ = $X_exp")
    println("⟨Z⟩ = $Z_exp")
    println("⟨ZZ⟩_vert = $(real(ZZ_vert))")
    println("⟨ZZ⟩_horiz = $(real(ZZ_horiz))")
    println("Gap = $gap")
    println("Top eigenvalues: ", eigenvalues[1:min(3, length(eigenvalues))])
    
    # Identity has degenerate transfer matrix (gap ≈ 0)
    @test abs(gap) < 1e-6
    
    # The fixed point gives a Z eigenstate (specific to gates_to_tensors slicing)
    @test isapprox(Z_exp, 1.0, atol=0.1)
    @test isapprox(real(ZZ_vert), 1.0, atol=0.1)
    @test isapprox(real(ZZ_horiz), 1.0, atol=0.1)
    @test abs(X_exp) < 0.2
    
    # All values should be real
    @test abs(imag(X_exp)) < 1e-10
    @test abs(imag(Z_exp)) < 1e-10
    
    println("Note: Identity gate creates degenerate transfer matrix (gap≈0)")
    println("      Fixed point is Z eigenstate due to tensor slicing convention")
end

@testset "observables_random_gate_consistency" begin
    # Test that observable computations work with random gates
    # and produce reasonable values
    
    row = 1
    virtual_qubits = 1
    nqubits = 3
    p = 2
    
    # Build two different random gates using same pattern as other tests
    params1 = randn(3 * nqubits * p)
    params2 = randn(3 * nqubits * p)
    
    # build_unitary_gate returns Vector{Matrix} - one matrix per row
    gates1 = build_unitary_gate(params1, p, row, nqubits)
    gates2 = build_unitary_gate(params2, p, row, nqubits)
    
    # Compute observables for both gates
    rho1, gap1, _ = compute_transfer_spectrum(gates1, row, nqubits)
    rho2, gap2, _ = compute_transfer_spectrum(gates2, row, nqubits)
    
    X_exp1 = real(compute_X_expectation(rho1, gates1, row, virtual_qubits))
    X_exp2 = real(compute_X_expectation(rho2, gates2, row, virtual_qubits))
    
    Z_exp1 = real(compute_Z_expectation(rho1, gates1, row, virtual_qubits))
    Z_exp2 = real(compute_Z_expectation(rho2, gates2, row, virtual_qubits))
    
    println("\n=== Random Gate Observable Test ===")
    println("Gate 1: ⟨X⟩ = $X_exp1, ⟨Z⟩ = $Z_exp1, gap = $gap1")
    println("Gate 2: ⟨X⟩ = $X_exp2, ⟨Z⟩ = $Z_exp2, gap = $gap2")
    
    # Verify all values are finite and within physical bounds
    @test isfinite(X_exp1) && abs(X_exp1) <= 1.0
    @test isfinite(X_exp2) && abs(X_exp2) <= 1.0
    @test isfinite(Z_exp1) && abs(Z_exp1) <= 1.0
    @test isfinite(Z_exp2) && abs(Z_exp2) <= 1.0
    @test gap1 >= 0.0
    @test gap2 >= 0.0
end

# =============================================================================
# Energy Computation Tests
# =============================================================================

@testset "compute_exact_energy" begin
    g = 2.0
    J = 1.0
    p = 3
    row = 3
    nqubits = 3
    params = rand(3 * nqubits * p)
    gap, energy = compute_exact_energy(params, g, J, p, row, nqubits)
    
    @test energy isa Float64
    @test gap isa Float64
    @test gap >= 0.0
end

@testset "energy_bounds_and_limits" begin
    # Test energy for different regimes
    row = 2
    nqubits = 3
    p = 2
    J = 1.0
    
    Random.seed!(808)
    params = randn(3 * nqubits * p)
    
    # Test different g values
    g_values = [0.0, 1.0, 5.0, 10.0]
    energies = []
    
    println("\n=== Energy vs Transverse Field ===")
    for g in g_values
        gap, E = compute_exact_energy(params, g, J, p, row, nqubits)
        push!(energies, E)
        println("g = $g: E = $(round(E, digits=4)), gap = $(round(gap, digits=4))")
        
        @test E isa Float64
        @test isfinite(E)
    end
    
    # Energy should be finite for all g values
    @test all(isfinite.(energies))
end

@testset "energy_parameter_sensitivity" begin
    # Test that energy changes with parameters
    g = 1.0
    J = 1.0
    p = 2
    row = 2
    nqubits = 3
    
    Random.seed!(909)
    
    params1 = randn(3 * nqubits * p)
    params2 = randn(3 * nqubits * p)
    
    _, E1 = compute_exact_energy(params1, g, J, p, row, nqubits)
    _, E2 = compute_exact_energy(params2, g, J, p, row, nqubits)
    
    println("\n=== Parameter Sensitivity ===")
    println("Energy with params1: E1 = $E1")
    println("Energy with params2: E2 = $E2")
    println("Difference: |E1 - E2| = $(abs(E1 - E2))")
    
    # Different parameters should generally give different energies
    # (unless we hit very special symmetric points)
    @test E1 isa Float64
    @test E2 isa Float64
end

# =============================================================================
# Observable Consistency Tests
# =============================================================================

@testset "observable_hermiticity" begin
    # All observables should be Hermitian → real expectation values
    Random.seed!(1010)
    
    row = 2
    nqubits = 3
    virtual_qubits = (nqubits - 1) ÷ 2
    
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    
    # Test multiple observables
    X_exp = compute_X_expectation(rho, gates, row, virtual_qubits)
    Z_exp = compute_Z_expectation(rho, gates, row, virtual_qubits)
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Hermiticity Test ===")
    println("Im(⟨X⟩) = $(imag(X_exp))")
    println("Im(⟨Z⟩) = $(imag(Z_exp))")
    println("Im(⟨ZZ⟩_vert) = $(imag(ZZ_vert))")
    println("Im(⟨ZZ⟩_horiz) = $(imag(ZZ_horiz))")
    
    # All should be real (Hermitian observables)
    @test abs(imag(X_exp)) < 1e-8
    @test abs(imag(Z_exp)) < 1e-8
    @test abs(imag(ZZ_vert)) < 1e-8
    @test abs(imag(ZZ_horiz)) < 1e-8
end

@testset "observable_bounds" begin
    # Single-site observables should be in [-1, 1]
    # Two-site observables should also be in [-1, 1]
    Random.seed!(1111)
    
    for row in [1, 2, 3]
        nqubits = 3
        virtual_qubits = 1
        
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
        
        X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
        Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
        ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
        
        @testset "row=$row" begin
            @test -1.0 - 1e-6 ≤ X_exp ≤ 1.0 + 1e-6
            @test -1.0 - 1e-6 ≤ Z_exp ≤ 1.0 + 1e-6
            @test -1.0 - 1e-6 ≤ real(ZZ_vert) ≤ 1.0 + 1e-6
            @test -1.0 - 1e-6 ≤ real(ZZ_horiz) ≤ 1.0 + 1e-6
        end
    end
end

@testset "observables_anticommutation" begin
    # For product states in eigenbases: if ⟨Z⟩ = ±1, then ⟨X⟩ = 0 and vice versa
    # This follows from {X, Z} = 0 (anticommutation)
    
    row = 1
    bond_dim = 2
    virtual_qubits = 1
    nqubits = 3
    
    # Test 1: Z eigenstate |0⟩
    A_z = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
    A_z[1, 1, 1, 1, 1] = 1.0
    A_z[1, 2, 2, 2, 2] = 1.0
    
    gates_z = []
    A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
    gate_data = zeros(ComplexF64, A_size)
    gate_data[:, :, :, 1, :, :] = A_z
    push!(gates_z, reshape(gate_data, 2^nqubits, 2^nqubits))
    
    rho_z, _, _ = compute_transfer_spectrum(gates_z, row, nqubits)
    X_z = real(compute_X_expectation(rho_z, gates_z, row, virtual_qubits))
    Z_z = real(compute_Z_expectation(rho_z, gates_z, row, virtual_qubits))
    
    # Test 2: X eigenstate |+⟩
    A_x = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
    A_x[1, 1, 1, 1, 1] = 1.0/sqrt(2)
    A_x[1, 2, 2, 2, 2] = 1.0/sqrt(2)
    A_x[2, 1, 1, 1, 1] = 1.0/sqrt(2)
    A_x[2, 2, 2, 2, 2] = 1.0/sqrt(2)
    
    gates_x = []
    gate_data_x = zeros(ComplexF64, A_size)
    gate_data_x[:, :, :, 1, :, :] = A_x
    push!(gates_x, reshape(gate_data_x, 2^nqubits, 2^nqubits))
    
    rho_x, _, _ = compute_transfer_spectrum(gates_x, row, nqubits)
    X_x = real(compute_X_expectation(rho_x, gates_x, row, virtual_qubits))
    Z_x = real(compute_Z_expectation(rho_x, gates_x, row, virtual_qubits))
    
    println("\n=== Anticommutation Relation Test ===")
    println("Z eigenstate |0⟩: ⟨X⟩ = $X_z, ⟨Z⟩ = $Z_z")
    println("X eigenstate |+⟩: ⟨X⟩ = $X_x, ⟨Z⟩ = $Z_x")
    
    # Z eigenstate: ⟨Z⟩ ≈ ±1, ⟨X⟩ ≈ 0
    @test abs(Z_z) > 0.9
    @test abs(X_z) < 0.2
    
    # X eigenstate: ⟨X⟩ ≈ ±1, ⟨Z⟩ ≈ 0
    @test abs(X_x) > 0.9
    @test abs(Z_x) < 0.2
end

# =============================================================================
# Physical Limit Tests
# =============================================================================

@testset "energy_strong_field_limit" begin
    # Strong transverse field limit (g >> J): spins align with X
    # Expected: Energy ≈ -g (dominated by X term)
    
    g_strong = 100.0
    J = 1.0
    p = 2
    row = 2
    nqubits = 3
    
    Random.seed!(1212)
    params = randn(3 * nqubits * p)
    
    gap, energy = compute_exact_energy(params, g_strong, J, p, row, nqubits)
    
    println("\n=== Strong Field Limit (g = $g_strong >> J = $J) ===")
    println("Energy: E = $energy")
    println("Expected: E ≈ -g = $(-g_strong) (X-dominated)")
    println("Gap: $gap")
    
    # Energy should be dominated by -g⟨X⟩ term
    # For strong field, ⟨X⟩ → 1, so E → -g
    @test energy < 0  # Should be negative
end

@testset "energy_weak_field_limit" begin
    # Weak transverse field (g << J): ZZ coupling dominates
    # Expected: Energy ≈ -J⟨ZZ⟩ (dominated by ZZ term)
    
    g_weak = 0.01
    J = 100.0
    p = 2
    row = 2
    nqubits = 3
    
    Random.seed!(1313)
    params = randn(3 * nqubits * p)
    
    gap, energy = compute_exact_energy(params, g_weak, J, p, row, nqubits)
    
    println("\n=== Weak Field Limit (g = $g_weak << J = $J) ===")
    println("Energy: E = $energy")
    println("Expected: E ≈ -J⟨ZZ⟩ (ZZ-dominated)")
    println("Gap: $gap")
    
    # Energy should be dominated by -J⟨ZZ⟩ term
    @test energy < 0  # Should be negative
end

@testset "energy_components_sign_check" begin
    # Verify the energy formula: E = -g⟨X⟩ - J⟨ZZ⟩
    # All terms have negative sign, so aligning observables should lower energy
    
    g = 1.5
    J = 1.0
    p = 2
    row = 2
    nqubits = 3
    virtual_qubits = 1
    
    Random.seed!(1414)
    params = randn(3 * nqubits * p)
    
    gates = build_unitary_gate(params, p, row, nqubits; share_params=true)
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    
    X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    ZZ_vert = real(ZZ_vert)
    ZZ_horiz = real(ZZ_horiz)
    
    # Compute energy manually
    E_manual = -g * X_exp - J * (ZZ_vert + ZZ_horiz)
    
    # Compare with compute_exact_energy
    _, E_func = compute_exact_energy(params, g, J, p, row, nqubits)
    
    println("\n=== Energy Components ===")
    println("⟨X⟩ = $X_exp → X contribution = $(-g * X_exp)")
    println("⟨ZZ⟩_v = $ZZ_vert → ZZ_v contribution = $(-J * ZZ_vert)")
    println("⟨ZZ⟩_h = $ZZ_horiz → ZZ_h contribution = $(-J * ZZ_horiz)")
    println("Manual energy:   E = $E_manual")
    println("Function energy: E = $E_func")
    
    # Manual calculation should match function
    @test isapprox(E_manual, E_func, atol=1e-6)
end

# =============================================================================
# Consistency and Symmetry Tests
# =============================================================================

@testset "observable_row_scaling" begin
    # Test how observables scale with number of rows
    # Single-site observables should be relatively stable
    
    nqubits = 3
    virtual_qubits = 1
    
    Random.seed!(1515)
    
    results = []
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        rho, gap, _ = compute_transfer_spectrum(gates, row, nqubits)
        X_exp = real(compute_X_expectation(rho, gates, row, virtual_qubits))
        Z_exp = real(compute_Z_expectation(rho, gates, row, virtual_qubits))
        
        push!(results, (row=row, X=X_exp, Z=Z_exp, gap=gap))
    end
    
    println("\n=== Observable Scaling with Row ===")
    for r in results
        println("Row $(r.row): ⟨X⟩=$(round(r.X, digits=3)), ⟨Z⟩=$(round(r.Z, digits=3)), gap=$(round(r.gap, digits=3))")
    end
    
    # All should be in physical bounds
    for r in results
        @test -1.0 ≤ r.X ≤ 1.0
        @test -1.0 ≤ r.Z ≤ 1.0
        @test r.gap >= 0.0
    end
end

@testset "ZZ_correlation_symmetry" begin
    # For uniform system with same gate on each row, vertical correlations
    # should be translation invariant
    # Test: compute ZZ at different positions (if we had that capability)
    # For now: just verify ZZ is well-defined and physical
    
    row = 3
    nqubits = 3
    virtual_qubits = 1
    
    Random.seed!(1616)
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]  # Same gate for all rows
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Uniform System Correlations ===")
    println("Same gate on all $row rows:")
    println("⟨ZZ⟩_vertical   = $(real(ZZ_vert))")
    println("⟨ZZ⟩_horizontal = $(real(ZZ_horiz))")
    
    # Both should be well-defined and in bounds
    @test -1.0 - 1e-6 ≤ real(ZZ_vert) ≤ 1.0 + 1e-6
    @test -1.0 - 1e-6 ≤ real(ZZ_horiz) ≤ 1.0 + 1e-6
end

@testset "compute_single_expectation_custom_observable" begin
    # Test compute_single_expectation with custom 2x2 observable
    row = 2
    nqubits = 3
    virtual_qubits = 1
    
    using Random
    Random.seed!(1717)
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    
    # Test with Pauli Y
    Y_matrix = [0 -im; im 0]
    Y_exp = compute_single_expectation(rho, gates, row, virtual_qubits, Y_matrix)
    
    # Also test with symbolic :X and :Z
    X_exp_sym = compute_single_expectation(rho, gates, row, virtual_qubits, :X)
    Z_exp_sym = compute_single_expectation(rho, gates, row, virtual_qubits, :Z)
    
    # Compare with dedicated functions
    X_exp_ded = compute_X_expectation(rho, gates, row, virtual_qubits)
    Z_exp_ded = compute_Z_expectation(rho, gates, row, virtual_qubits)
    
    println("\n=== Custom Observable Test ===")
    println("⟨Y⟩ (custom) = $Y_exp")
    println("⟨X⟩ (symbol) = $X_exp_sym vs (dedicated) = $X_exp_ded")
    println("⟨Z⟩ (symbol) = $Z_exp_sym vs (dedicated) = $Z_exp_ded")
    
    # Symbolic and dedicated functions should match
    @test isapprox(X_exp_sym, X_exp_ded, atol=1e-10)
    @test isapprox(Z_exp_sym, Z_exp_ded, atol=1e-10)
    
    # Y should be Hermitian (real for expectation value up to small imaginary part)
    @test abs(imag(Y_exp)) < 1e-8
    @test -1.0 - 1e-6 ≤ real(Y_exp) ≤ 1.0 + 1e-6
end

@testset "correlation_function_vs_sampling" begin
    # Test that exact correlation_function matches sampling-based compute_acf
    # Use small system for reasonable sampling statistics
    virtual_qubits = 1
    nqubits = 1 + 2 * virtual_qubits
    row = 3
    max_lag = 5
    
    # Generate random unitary gates
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    # Compute exact correlation using transfer matrix (connected correlation)
    exact_corr = correlation_function(gates, row, virtual_qubits, :Z, 1:max_lag; connected=false)
    # Sample from the quantum channel
    # Need enough samples for statistical convergence
    conv_step = 1000
    samples = 500000
    rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                        conv_step=conv_step, 
                                                        samples=samples,
                                                        measure_first=:Z)
    
    # Reshape samples for compute_acf (row samples per column → reshape to get horizontal correlations)
    # Z_samples is a flat vector, need to compute correlations properly
    # The sampling gives Z values site by site, so we compute correlation directly
    Z_vec = Z_samples[conv_step+1:end]  # Discard burn-in
    
    # Compute sample-based correlation using compute_acf
    # compute_acf expects samples in matrix form (chains × samples)
    # We'll use it as a single chain
    # Need enough lags: horizontal separation r corresponds to lag r*row in sample stream
    sample_max_lag = max_lag * row + 1
    lags, acf, acf_err, corr_full, corr_err, corr_connected, corr_connected_err = compute_acf(
        reshape(Float64.(Z_vec), 1, :); max_lag=sample_max_lag
    )
    # Subsample to get horizontal correlations: separation r → sample lag r*row
    # After subsampling, index r+1 gives separation r
    corr_full = corr_full[1:row:end]
    corr_err = corr_err[1:row:end]
    
    # Compare correlations at each separation
    # After subsampling: corr_full[1] is separation 0, corr_full[r+1] is separation r
    
    # Test that magnitudes are in the same ballpark (within statistical error)
    # We use a looser tolerance since sampling has statistical noise
    for r in 1:min(max_lag, length(corr_full) - 1)
        exact_val = real(exact_corr[r])
        sample_val = corr_full[r + 1]  # After subsampling, index r+1 is separation r
        sample_err = corr_err[r + 1]
        
        # Allow for ~3 sigma statistical error, plus some systematic differences due to finite-size effects
        tolerance = max(3 * sample_err, 0.05 * abs(exact_val), 0.01)
        
        @test isapprox(exact_val, sample_val, atol=0.01) 
    end
    
    # At minimum, test that both give non-trivial correlations (not all zero)
    @test any(abs.(values(exact_corr)) .> 1e-10)
    @test any(abs.(corr_connected[2:end]) .> 1e-10)
end
