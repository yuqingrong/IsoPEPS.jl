using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra

@testset "compute_X_expectation" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        X_exp = compute_X_expectation(rho, gates, row, virtual_qubits)
        @test abs(imag(X_exp)) < 1e-10
        @test -1.0 ≤ real(X_exp) ≤ 1.0
    end
end

@testset "compute_ZZ_expectation" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
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

@testset "compute_exact_energy" begin
    g = 2.0
    J = 1.0
    p = 3
    row = 3
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
    # Uses 3 params per qubit per layer (Rz-Ry-Rz decomposition)
    params = rand(3 * nqubits * p)
    _, energy = compute_exact_energy(params, g, J, p, row, virtual_qubits)
    @test energy isa Float64
end

