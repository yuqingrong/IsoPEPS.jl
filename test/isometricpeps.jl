using IsoPEPS
using Test
using Graphs
using Manifolds, OMEinsumContractionOrders
import RecursiveArrayTools: ArrayPartition
@testset "IsometricPEPS" begin
    peps = rand_isometricpeps(ComplexF64, 2, 3, 3)
    @test peps.col == 3
    @test peps.vertex_tensors[1][1] isa AbstractArray{ComplexF64, 5}
end


@testset "mose_move_single_column!" begin
    # initialize a peps
    peps = rand_isometricpeps(ComplexF64, 2, 4, 4)
    p1 = mose_move_right_step!(peps, 1)
    @test peps_fidelity(p1, peps) ≈ 1

    p1 = mose_move_right_step!(copy(peps), 2)
    @test peps_fidelity(p1, peps) ≈ 1

    p1 = mose_move_right_step!(copy(peps), 3)
    @test peps_fidelity(p1, peps) ≈ 1
    
end


@testset "mose_move_all_columns!" begin
    peps = rand_isometricpeps(ComplexF64, 2, 4, 4)
    p1 = mose_move_right!(copy(peps))
    @test peps_fidelity(p1, peps) ≈ 1
end 


@testset "isometric_condition" begin

end

@testset "isometric_peps" begin
    g = SimpleDiGraph(9)
    g2 = grid([3,3])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps,_ = isometric_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    @test isapprox(reshape(peps.vertex_tensors[5], 8, 4)'*reshape(peps.vertex_tensors[5], 8, 4), Matrix(I, 4, 4),atol=1e-10)  
end


@testset "vector2point" begin
    g = SimpleDiGraph(4)
    g2 = grid([2,2])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps, matrix_dims = isometric_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    x0 = variables(peps)
    p0 = vector2point(x0, matrix_dims)
    x1 = point2vector(p0, matrix_dims)
   @test isapprox(x1, x0)
end


@testset "isometric_peps_optimize" begin
    g = SimpleDiGraph(4)
    g2 = Graphs.grid([2,2])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps, matrix_dims = isometric_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    
    M = ProductManifold([Manifolds.Stiefel(n, p, ℂ) for (n, p) in matrix_dims]...)
    
    J, h = 1.0, 0.2

    #x = variables(peps)
    #p0 = Tuple(vector2point(x, matrix_dims)) 

    #@test all(Manifolds.is_point.(M.manifolds, p0))
    #@test is_point(M, ArrayPartition(p0...))

    #G = iso_g_ising!(zeros(eltype(x),size(x)), p0, matrix_dims, peps, g, J, h, GreedyMethod(), MergeGreedy())
    result,energy = isopeps_optimize_ising(peps, M, matrix_dims, g, J, h, GreedyMethod(), MergeGreedy())
    @show energy
    hami = ising_hamiltonian_2d(2,2,J,h)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(hami), 1, :SR; ishermitian=true)
    @show eigenval[1]
    @test isapprox(energy, eigenval[1], rtol=1e-3)
end

@testset "isometric_peps_to_unitary" begin
    g = SimpleDiGraph(4)
    g2 = grid([2,2])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps, matrix_dims = isometric_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    ugates = isometric_peps_to_unitary(peps, g)
    @test isapprox((reshape(ugates.vertex_tensors[1], 8, 8)'*reshape(ugates.vertex_tensors[1], 8, 8)), Matrix(I, 8, 8), atol=1e-10)
    @test isapprox((reshape(ugates.vertex_tensors[1], 8, 8)*reshape(ugates.vertex_tensors[1], 8, 8)'), Matrix(I, 8, 8), atol=1e-10)
end