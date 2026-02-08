using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra
using MPSKit

@testset "mpskit_ground_state" begin
    g = 4.0
    row = 2
    d = 2
    D = 8
    result = mpskit_ground_state(d, D, g, row)
    @test result.energy isa Real
    @test result.correlation_length > 0
end

@testset "transfer matrix match MPSKit" begin
    g = 1.0
    d = 2
    D = 4
    nqubits = 3
    row = 1
    virtual_qubits = 1 
    result = mpskit_ground_state_1d(d, D, g)
    
    exact_A = Array{ComplexF64}(undef, D, 2, D)
    for (i, j, k) in Iterators.product(1:D, 1:2, 1:D)
        exact_A[i, j, k] = result.psi.AR.data[1][i, j, k]
    end
    exact_A = permutedims(exact_A, (2,3,1))
    exact_A2 = reshape(exact_A, (d * D, D))
    
    nullspace_A = LinearAlgebra.nullspace(exact_A2')
    A_matrix = hcat(exact_A2, nullspace_A)
    A_mod = similar(A_matrix)

    # Put exact_A2 columns into odd positions so that [:,:,1,:] picks them
    A_mod[:, 1] = A_matrix[:, 1]
    A_mod[:, 3] = A_matrix[:, 2]
    A_mod[:, 5] = A_matrix[:, 3]
    A_mod[:, 7] = A_matrix[:, 4]

    # Put the remaining 4 columns anywhere in even positions (e.g. 2,4,6,8)
    A_mod[:, 2] = A_matrix[:, 5]
    A_mod[:, 4] = A_matrix[:, 6]
    A_mod[:, 6] = A_matrix[:, 7]
    A_mod[:, 8] = A_matrix[:, 8]
    A_matrix_list = [A_mod for _ in 1:row]
    
    rho, gap, eigenval = compute_transfer_spectrum(A_matrix_list, row, nqubits)
    @test gap ≈ 1 / result.correlation_length atol = 1e-5
end