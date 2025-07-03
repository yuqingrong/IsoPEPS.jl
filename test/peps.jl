using IsoPEPS
using Test
using ForwardDiff
import DifferentiationInterface
import OMEinsumContractionOrders
import Yao
@testset "zero_peps and inner_product" begin
    g = dgrid(2,2)
    peps = zero_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    @test peps.physical_labels == [1,2,3,4]
    @test peps.virtual_labels == [5,6,7,8]
    @test peps.vertex_tensors[2][2] == 0 && peps.vertex_tensors[2][3]==0
    @test real(inner_product(peps,peps)[]) == 1.0
end

@testset "rand_peps and inner_product" begin
    g = dgrid(2,2)
    peps1 = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    peps2 = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    @test peps1.physical_labels == [1,2,3,4]
    @test peps1.virtual_labels == [5,6,7,8]
    @test inner_product(peps1, peps1)[] ≈ statevec(peps1)' * statevec(peps1)
    @test inner_product(peps1, peps2)[] ≈ statevec(peps1)' * statevec(peps2)
end

@testset "apply_onsite" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    reg = Yao.ArrayReg(vec(peps))
    m = Matrix(Yao.X)
    apply_onsite!(peps, 1, m)
    reg |> put(4, (1,)=>Yao.matblock(m))
    @test vec(peps) ≈ statevec(reg)
end

@testset "single_sandwich" begin
    g = dgrid(2,2)
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
   
    h = put(4,(1,)=>Yao.X)
    expect, gradient = single_sandwich(peps, peps, 1, Matrix(Yao.X), TreeSA(), MergeGreedy())
    exact = (statevec(peps)' * Yao.mat(h) * statevec(peps))[1]
    @test expect ≈ exact
end

@testset "two_sandwich" begin
    g = dgrid(2,2)
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())

    h = put(4,(1,2)=>kron(Yao.Z,Yao.Z))
    expect, _ = two_sandwich(peps, peps, 1, 2, reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2), TreeSA(), MergeGreedy())
    exact = (statevec(peps)' * Yao.mat(h) * statevec(peps))[1]
    @test expect ≈ exact
end


@testset "gradient" begin
    g = SimpleGraph(2)
    for (i,j) in [(1,2)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())

    

    x = variables(peps)
    G1 = zeros(eltype(x),size(x))
    G2 = zeros(eltype(x),size(x))

    gradient1 = g1!(G1, peps, x, 1, Matrix(Yao.X), TreeSA(), MergeGreedy())
    f_closure1(x) =  f1(peps, x, 1, Matrix(Yao.X), TreeSA(), MergeGreedy())
    expect_gradient1 = grad(central_fdm(12, 1), f_closure1, x)[1]
    @test isapprox(gradient1, expect_gradient1, rtol = 1e-2)

    gradient2 = g2!(G2, peps, x, 1, 2, reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2), TreeSA(), MergeGreedy())
    f_closure2(x) =  f2(peps, x, 1, 2, reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2), TreeSA(), MergeGreedy())
    expect_gradient2 = grad(central_fdm(12, 1), f_closure2, x)[1]
    @test isapprox(gradient2, expect_gradient2, rtol = 1e-4)

    G1 = zeros(eltype(x),size(x))
    G2 = zeros(eltype(x),size(x))
    G = zeros(eltype(x),size(x))
    gradient3 = g_ising!(G, peps, x, g, 1.0, 0.2, GreedyMethod(), MergeGreedy())
    f_closure_ising(x) =  f_ising(peps, x, g, 1.0, 0.2, GreedyMethod(), MergeGreedy())
    expect_gradient3 = grad(central_fdm(12, 1), f_closure_ising, x)[1]
    @test isapprox(gradient3, expect_gradient3, rtol = 1e-4)
end
 

 @testset "single variational optimization" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    result = peps_optimize1(peps, 1, real(Matrix(Yao.X)), GreedyMethod(), MergeGreedy())
    h = put(4,(1,)=>Yao.X)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(h), 1, :SR; ishermitian=true)
    @test isapprox(result , eigenval[1], rtol = 1e-2)
end


#TODO: draw energy vs. iteration step
@testset "ZZ variational optimization" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    result = peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2)), GreedyMethod(), MergeGreedy())
    #result = peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)), TreeSA(), MergeGreedy())
    @show result
    #result = cached_peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)), TreeSA(), MergeGreedy())
    h = kron(4, 1=>Yao.Z, 2=>Yao.Z)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(h), 1, :SR; ishermitian=true)
    @test isapprox(result , eigenval[1], rtol=1e-2)
end

@testset "ising variational optimization" begin  
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    J, h = 1.0, 0.2
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    result = peps_optimize_ising(peps, g, J, h, GreedyMethod(), MergeGreedy())
  
    hami = ising_hamiltonian_2d(2,2,J,h)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(hami), 1, :SR; ishermitian=true)
    @test isapprox(result , eigenval[1], rtol=1e-4)
end





@testset "expect using Yao" begin
    h1 = put(4,(1,)=>Yao.X)
    eigenval1,eigenvec1 = IsoPEPS.eigsolve(IsoPEPS.mat(h1), 1, :SR; ishermitian=true)
    @test eigenval1[1] ≈ -1.0
    h2 = kron(4, 1=>Yao.Z, 2=>Yao.Z)
    eigenval2,eigenvec2 = IsoPEPS.eigsolve(IsoPEPS.mat(h2), 1, :SR; ishermitian=true)
    @test eigenval2[1] ≈ -1.0
    h3 = ising_hamiltonian_2d(2,2,1.0,0.2)
    eigenval2,eigenvec2 = IsoPEPS.eigsolve(IsoPEPS.mat(h3), 1, :SR; ishermitian=true)
    @test eigenval2[1] ≈ -4.040593699203847
end



@testset "long_range_coherence_peps" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    corr = long_range_coherence_peps(peps, 2, 3)
    @show corr
    @test long_range_coherence_peps(peps, 1, 1) ≈ 1.0
end

@testset "dtorus" begin
    g = dtorus(2,2)
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    @test peps.physical_labels == [1,2,3,4]
    @test peps.virtual_labels == [5,6,7,8,9,10,11,12]
    @show peps.vertex_labels
    @test inner_product(peps,peps)[] ≈ statevec(peps)' * statevec(peps)

    h1 = put(4,(1,)=>Yao.X)
    exact1 = (statevec(peps)' * Yao.mat(h1) * statevec(peps))[1]
    @test single_sandwich(peps, peps, 1, Matrix(Yao.X), TreeSA(), MergeGreedy())[1] ≈ exact1

    h2 = put(4,(1,2)=>Yao.kron(Yao.Z,Yao.Z))
    exact2 = (statevec(peps)' * Yao.mat(h2) * statevec(peps))[1]
    @test two_sandwich(peps, peps, 1, 2, reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2), TreeSA(), MergeGreedy())[1] ≈ exact2
end



using OMEinsum
Mooncake.tangent_type(::Type{<:AbstractEinsum}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:Vector{<:AbstractEinsum}}) = Mooncake.NoTangent

g = SimpleGraph(4)
for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
    add_edge!(g, i, j)
end
peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
using OMEinsumContractionOrders
result = optimized_peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2)), TreeSA(), MergeGreedy())
@time optimized_peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2)), TreeSA(), MergeGreedy())
@time peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2)), TreeSA(), MergeGreedy())
Profile.clear() 
@profile peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Yao.Z),Matrix(Yao.Z)),2,2,2,2)), TreeSA(), MergeGreedy())
Profile.print(format=:flat)
