using IsoPEPS
using Test
import Mooncake
import ForwardDiff
import DifferentiationInterface
@testset "zero_peps and inner_product" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = zero_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    @test peps.physical_labels == [1,2,3,4]
    @test peps.virtual_labels == [5,6,7,8]
    @test peps.vertex_tensors[2][2] == 0 && peps.vertex_tensors[2][3]==0
    @test real(inner_product(peps,peps)[]) == 1.0
end

@testset "rand_peps and inner_product" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
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
    m = Matrix(X)
    apply_onsite!(peps, 1, m)
    reg |> put(4, (1,)=>matblock(m))
    @test vec(peps) ≈ statevec(reg)
end

@testset "single_sandwich" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    
    h = put(4,(1,)=>X)
    @show mat
    expect, gradient = single_sandwich(peps, peps, 1, Matrix(X), TreeSA(), MergeGreedy())
    exact = (statevec(peps)' * mat(h) * statevec(peps))[1]
    @test expect ≈ exact
end

@testset "two_sandwich" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())

    h = put(4,(1,2)=>kron(Z,Z))
    expect, _ = two_sandwich(peps, peps, 1, 2, reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2), TreeSA(), MergeGreedy())
    exact = (statevec(peps)' * mat(h) * statevec(peps))[1]
    @test expect ≈ exact
end


@testset "gradient" begin

    g = SimpleGraph(2)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    x = variables(peps)
    
    G1 = zeros(eltype(x),size(x))
    G2 = zeros(eltype(x),size(x))
    @show G1

    gradient1 = g1!(G1, peps, x, 1, real(Matrix(X)), TreeSA(), MergeGreedy())
    f_closure1(x) =  f1(peps, x, 1, real(Matrix(X)), TreeSA(), MergeGreedy())
    expect_gradient1 = grad(central_fdm(12, 1), f_closure1, x)[1]
    @test isapprox(gradient1, expect_gradient1, rtol = 1e-4)

    gradient2 = g2!(G2, peps, x, 1, 2, real(reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)), TreeSA(), MergeGreedy())
    f_closure2(x) =  f2(peps, x, 1, 2, real(reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)), TreeSA(), MergeGreedy())
    expect_gradient2 = grad(central_fdm(12, 1), f_closure2, x)[1]
    @test isapprox(gradient2, expect_gradient2, rtol = 1e-4)

    G1 = zeros(eltype(x),size(x))
    G2 = zeros(eltype(x),size(x))
    G = zeros(eltype(x),size(x))
    gradient3 = g_ising!(G, peps, x, g, 1.0, 0.2, TreeSA(), MergeGreedy())
    f_closure_ising(x) =  f_ising(peps, x, g, 1.0, 0.2, TreeSA(), MergeGreedy())
    expect_gradient3 = grad(central_fdm(2, 1), f_closure_ising, x)[1]
    @test isapprox(gradient3, expect_gradient3, rtol = 1e-4)
end
 

@testset "single variational optimization" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    result = peps_optimize1(peps, 1, real(Matrix(X)), TreeSA(), MergeGreedy())
    h = put(4,(1,)=>X)
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
    result = peps_optimize2(peps, 1, 2, real(reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)), TreeSA(), MergeGreedy())
    h = kron(4, 1=>Z, 2=>Z)
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
    result = peps_optimize_ising(peps, g, J, h, TreeSA(), MergeGreedy())
  
    hami = ising_hamiltonian_2d(2,2,J,h)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(hami), 1, :SR; ishermitian=true)
    @test isapprox(result , eigenval[1], rtol=1e-4)
end

@testset "expect using Yao" begin
    h1 = put(4,(1,)=>X)
    eigenval1,eigenvec1 = IsoPEPS.eigsolve(IsoPEPS.mat(h1), 1, :SR; ishermitian=true)
    @test eigenval1[1] ≈ -1.0
    h2 = kron(4, 1=>Z, 2=>Z)
    eigenval2,eigenvec2 = IsoPEPS.eigsolve(IsoPEPS.mat(h2), 1, :SR; ishermitian=true)
    @test eigenval2[1] ≈ -1.0
    h3 = ising_hamiltonian_2d(2,2,1.0,0.2)
    eigenval2,eigenvec2 = IsoPEPS.eigsolve(IsoPEPS.mat(h3), 1, :SR; ishermitian=true)
    @test eigenval2[1] ≈ -4.040593699203847
end

@testset "AD gradient" begin
   
    g = SimpleGraph(2)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    psi = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    
    J, h = 1.0, 0.2
    x = variables(psi)
    backend = AutoMooncake(; config=nothing)
    f_closure_ising(x) = f_ising(IsoPEPS.convert_type(eltype(x), psi), x, g, 1.0, 0.2, TreeSA(), MergeGreedy())
    #grad = ForwardDiff.gradient(f_closure_ising, x)
    prep = prepare_gradient(f_closure_ising, backend, x) 
    grad = gradient(f_closure_ising, prep, backend, x)
end


@testset "long_range_coherence_peps" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    corr = long_range_coherence_peps(peps, 2, 3)
    @show corr
    #@test long_range_coherence_peps(peps, 1, 2) ≈ 0.0
end


using OMEinsum
Mooncake.tangent_type(::Type{<:AbstractEinsum}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:Vector{<:AbstractEinsum}}) = Mooncake.NoTangent

