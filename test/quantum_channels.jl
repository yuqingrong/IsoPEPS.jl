using Test
using IsoPEPS
using Yao, YaoBlocks
using Statistics

@testset "iterate_channel_PEPS" begin
    nqubits = 3
    row = 3
    g = 0.0
    J = 1.0
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    A_matrix = [Matrix(gate) for _ in 1:row]
    rho_iter, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits)
    rho_eigen, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
    cost_x = IsoPEPS.cost_X(rho_eigen, A_matrix, row, nqubits)
    @test mean(X_list) ≈ cost_x atol = 1e-2

    # Use Z_list directly from iterate_channel_PEPS
    energy = IsoPEPS.energy_measure(X_list, Z_list, g, J, row)
    @test energy isa Float64
end

