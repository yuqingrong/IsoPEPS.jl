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
    g1 = dgrid(3,3)
    g2 = dtorus(3,3)
    peps1,_ = isometric_peps(ComplexF64, g1, 2, 2, TreeSA(), MergeGreedy())
    peps2,_ = isometric_peps(ComplexF64, g2, 2, 2, TreeSA(), MergeGreedy())

    @test isapprox(reshape(peps1.vertex_tensors[1], 8, 1)'*reshape(peps1.vertex_tensors[1], 8, 1), Matrix(I, 1, 1),atol=1e-10)  
    @test isapprox(reshape(peps2.vertex_tensors[1], 8, 4)'*reshape(peps2.vertex_tensors[1], 8, 4), Matrix(I, 4, 4),atol=1e-10)  

    @test isapprox(reshape(peps1.vertex_tensors[2], 8, 2)'*reshape(peps1.vertex_tensors[2], 8, 2), Matrix(I, 2, 2),atol=1e-10)  
    @test isapprox(reshape(peps2.vertex_tensors[2], 8, 4)'*reshape(peps2.vertex_tensors[2], 8, 4), Matrix(I, 4, 4),atol=1e-10)  

    @test isapprox(reshape(peps1.vertex_tensors[end], 2, 4)*reshape(peps1.vertex_tensors[end], 2, 4)', Matrix(I, 2, 2),atol=1e-10)  
    @test isapprox(reshape(peps2.vertex_tensors[end], 8, 4)'*reshape(peps2.vertex_tensors[end], 8, 4), Matrix(I, 4, 4),atol=1e-10)  

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
    g = SimpleDiGraph(9)
    g2 = Graphs.grid([3,3])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps, matrix_dims = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    
    M = ProductManifold([Manifolds.Stiefel(n, p) for (n, p) in matrix_dims]...)

    J, h = 1.0, 0.2

    x = variables(peps)
    p0 = Tuple(vector2point(x, matrix_dims)) 

    @test all(Manifolds.is_point.(M.manifolds, p0))


    result, energy, optimized_peps, record= isopeps_optimize_ising(peps, M, matrix_dims, g, J, h, GreedyMethod(), MergeGreedy(), 100)
    
    hami = ising_hamiltonian_2d(3,3,J,h)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(hami), 1, :SR; ishermitian=true)
    @show energy
    @show eigenval[1]
    @show optimized_peps  
    @show record, typeof(record)
    @test isapprox(energy, eigenval[1], rtol=1e-3)
end


@testset "isometric_peps_to_unitary" begin
    g1 = dgrid(2,2)
    peps1, matrix_dims1 = isometric_peps(ComplexF64, g1, 2, 2, TreeSA(), MergeGreedy())
    ugates1 = isometric_peps_to_unitary(peps1, g1)
    @test isapprox(ugates1.vertex_tensors[2]*ugates1.vertex_tensors[2]', Matrix(I, 4, 4), atol=1e-10)
    @test isapprox(ugates1.vertex_tensors[2]'*ugates1.vertex_tensors[2], Matrix(I, 4, 4), atol=1e-10)

    @test isapprox(ugates1.vertex_tensors[4]*ugates1.vertex_tensors[4]', Matrix(I, 4, 4), atol=1e-10)
    @test isapprox(ugates1.vertex_tensors[4]'*ugates1.vertex_tensors[4], Matrix(I, 4, 4), atol=1e-10)

    g2 = dtorus(3,3)
    peps2, matrix_dims2 = isometric_peps(ComplexF64, g2, 2, 2, TreeSA(), MergeGreedy())
    ugates2 = isometric_peps_to_unitary(peps2, g2)
    @test isapprox(ugates2.vertex_tensors[1]*ugates2.vertex_tensors[1]', Matrix(I, 8, 8), atol=1e-10)
    @test isapprox(ugates2.vertex_tensors[1]'*ugates2.vertex_tensors[1], Matrix(I, 8, 8), atol=1e-10)

    @test isapprox(ugates2.vertex_tensors[9]*ugates2.vertex_tensors[9]', Matrix(I, 8, 8), atol=1e-10)
    @test isapprox(ugates2.vertex_tensors[9]'*ugates2.vertex_tensors[9], Matrix(I, 8, 8), atol=1e-10)
end



using Profile

Profile.clear()

Profile.init(n=100_000_000, delay=0.001)  # 100M samples, 1ms interval

@profile begin
    g = SimpleDiGraph(9)
    g2 = Graphs.grid([3,3])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps, matrix_dims = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    
    M = ProductManifold([Manifolds.Stiefel(n, p) for (n, p) in matrix_dims]...)

    J, h = 1.0, 0.2

    x = variables(peps)
    result, energy, optimized_peps, record= isopeps_optimize_ising(peps, M, matrix_dims, g, J, h, TreeSA(), MergeGreedy(), 100)
    open("profile_results1.txt", "w") do io
        Profile.print(io; format=:tree, mincount=2000)
    end
end