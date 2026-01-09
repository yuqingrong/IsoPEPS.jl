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


@testset "check samples" begin
    using Statistics
    
    nqubits = 3
    row = 1
    n_channels = 100000
    
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate) for _ in 1:row]
    
    # Collect samples from many runs
    Z_for_track = Float64[]
    Z_for_expect = Float64[]
    for _ in 1:n_channels
        _, Z_samples,_= sample_quantum_channel(gates, row, nqubits; conv_step=10, samples=100, measure_first=:Z)
        append!(Z_for_track, Z_samples[1:60])
        append!(Z_for_expect, Z_samples[60:end])
    end
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    Z1 = mean(Z_for_track[1:60:end])
    Z2 = mean(Z_for_track[2:60:end])
    Z3 = mean(Z_for_track[3:60:end])
    Z4 = mean(Z_for_track[4:60:end])
    Z5 = mean(Z_for_track[5:60:end])
    Z50 = mean(Z_for_track[60:60:end])
    Z_expect = mean(Z_for_expect)
    @show Z1, Z2, Z3, Z4, Z5, Z50, Z_expect
    @show gap
end