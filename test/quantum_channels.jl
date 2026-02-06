using Test
using IsoPEPS
using Yao, YaoBlocks
using Statistics
using Random

@testset "sample_quantum_channel" begin
    Random.seed!(4567)
    nqubits = 3
    row = 3
    virtual_qubits = 1  # Bond qubits per side (nqubits = 1 physical + 2*virtual_qubits)
    g = 0.0
    J = 1.0
    
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    rho_iter, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits)
    rho_exact, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    X_exact = compute_X_expectation(rho_exact, gates, row, virtual_qubits)
    @test mean(X_samples) ≈ X_exact atol = 1e-1

    energy = compute_energy(X_samples, Z_samples, g, J, row)
    @test energy isa Float64
end
