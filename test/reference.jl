using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra
using MPSKit
using MPSKitModels
using TensorKit

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
    exact_A = reshape(permutedims(exact_A, (2, 1, 3)), (d * D, D))
    nullspace_A = LinearAlgebra.nullspace(exact_A')
    A_matrix = vcat(exact_A, nullspace_A)
    A_matrix_list = [A_matrix for _ in 1:row]
    
    rho, gap, eigenval = compute_transfer_spectrum(A_matrix_list, row, nqubits)
    @test gap ≈ 1 / result.correlation_length atol = 1e-5
end

@testset "j1-j2 infinite cylinder bond counting" begin
    row = 4
    unit_cell_cols = 2
    n_sites = row * unit_cell_cols
    lattice = InfiniteCylinder(row, n_sites)

    nn = collect(nearest_neighbours(lattice))
    nnn = collect(next_nearest_neighbours(lattice))

    @test length(nn) == 2 * n_sites
    @test length(nnn) == 2 * n_sites

    linear_bond(pair) = begin
        i, j = pair
        ii = Int(MPSKitModels.linearize_index(i))
        jj = Int(MPSKitModels.linearize_index(j))
        ii < jj ? (ii, jj) : (jj, ii)
    end

    @test length(unique(linear_bond.(nn))) == 2 * n_sites
    @test length(unique(linear_bond.(nnn))) == 2 * n_sites
end
