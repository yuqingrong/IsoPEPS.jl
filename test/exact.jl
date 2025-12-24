using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra

@testset "contract_Elist" begin
    A = randn(ComplexF64, 2, 2, 2, 2, 2)
    for row in 1:3
        code, result = contract_Elist([A for _ in 1:row], [conj(A) for _ in 1:row], row)
        @test result isa Array{ComplexF64,4*(row + 1)}
    end
end

@testset "exact_left_eigen" begin
    nqubits = 3
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        A_matrix = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
        @test LinearAlgebra.tr(rho) ≈ 1.0
        @test all(eigenvalues[1:end-1] .< 1.0)
    end
end

@testset "cost_X" begin
    nqubits = 3
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        A_matrix = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
        X_exp = cost_X(rho, A_matrix, row, nqubits)
        @test abs(imag(X_exp)) < 1e-10
        @test -1.0 ≤ real(X_exp) ≤ 1.0
    end
end

@testset "cost_ZZ" begin
    nqubits = 3
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        A_matrix = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
        ZZ_ver, ZZ_hor = cost_ZZ(rho, A_matrix, row, nqubits)
        @test abs(imag(ZZ_ver)) < 1e-10
        @test -1.0 ≤ real(ZZ_ver) ≤ 1.0
        @test abs(imag(ZZ_hor)) < 1e-10
        @test -1.0 ≤ real(ZZ_hor) ≤ 1.0
    end
end

@testset "exact_E_from_params" begin
    g = 2.0
    J = 1.0
    p = 3
    row = 3
    nqubits = 3
    params = rand(2 * nqubits * p)
    _, energy = exact_E_from_params(params, g, J, p, row, nqubits)
    @test energy isa Float64
end

