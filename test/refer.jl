using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra
using MPSKit

@testset "result_MPSKit" begin
    J = 1.0
    g = 4.0
    row = 2
    d = 2
    D = 8
    E, len_gapped, entrop_gapped, spectrum = result_MPSKit(d, D, g, row)
    @test E isa Real
    @test len_gapped > 0
end

@testset "transfer matrix match MPSKit" begin
    J = 1.0
    g = 1.0
    d = 2
    D = 4
    nqubits = 3
    row = 1
    E, len_gapped, entrop_gapped, spectrum, psi = result_1d(d, D, g)
    exact_A = Array{ComplexF64}(undef, D, 2, D)
    for (i, j, k) in Iterators.product(1:D, 1:2, 1:D)
        exact_A[i, j, k] = psi.AR.data[1][i, j, k]
    end
    exact_A = reshape(permutedims(exact_A, (2, 1, 3)), (d * D, D))
    nullspace_A = LinearAlgebra.nullspace(exact_A')
    A_matrix = vcat(exact_A, nullspace_A)
    A_matrix_list = [A_matrix for _ in 1:row]
    rho, gap, eigenval = exact_left_eigen(A_matrix_list, row, nqubits)
    @test gap ≈ 1 / len_gapped atol = 1e-5
end

