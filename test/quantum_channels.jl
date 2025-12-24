using Test
using IsoPEPS
using Yao, YaoBlocks
using Statistics

@testset "sample_quantum_channel" begin
    nqubits = 3
    row = 3
    g = 0.0
    J = 1.0
    
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    rho_iter, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits)
    rho_exact, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    X_exact = compute_X_expectation(rho_exact, gates, row, nqubits)
    @test mean(X_samples) ≈ X_exact atol = 1e-2

    energy = compute_energy(X_samples, Z_samples, g, J, row)
    @test energy isa Float64
end
