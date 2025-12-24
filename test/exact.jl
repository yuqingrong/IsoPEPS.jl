using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra

@testset "contract_transfer_matrix" begin
    A = randn(ComplexF64, 2, 2, 2, 2, 2)
    for row in 1:3
        code, result = contract_transfer_matrix([A for _ in 1:row], [conj(A) for _ in 1:row], row)
        @test result isa Array{ComplexF64, 4*(row + 1)}
    end
end

@testset "compute_transfer_spectrum" begin
    nqubits = 3
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        @test LinearAlgebra.tr(rho) ≈ 1.0
        @test all(eigenvalues[1:end-1] .< 1.0)
    end
end

@testset "compute_X_expectation" begin
    nqubits = 3
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        X_exp = compute_X_expectation(rho, gates, row, nqubits)
        @test abs(imag(X_exp)) < 1e-10
        @test -1.0 ≤ real(X_exp) ≤ 1.0
    end
end

@testset "compute_ZZ_expectation" begin
    nqubits = 3
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, nqubits)
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
    nqubits = 3
    params = rand(2 * nqubits * p)
    _, energy = compute_exact_energy(params, g, J, p, row, nqubits)
    @test energy isa Float64
end
