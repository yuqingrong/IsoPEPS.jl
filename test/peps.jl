using IsoPEPS
using Test

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
    expect = single_sandwich(peps, peps, 1, Matrix(X), TreeSA(), MergeGreedy())
    exact = (statevec(peps)' * mat(h) * statevec(peps))[1]
    @show expect
    @test expect ≈ exact
end

@testset "two_sandwich" begin
    g = SimpleGraph(4)
    for (i,j) in [(1,2), (1,3), (2,4), (3,4)]
        add_edge!(g, i, j)
    end
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())

    h = put(4,(1,2)=>kron(Z,Z))
    expect = two_sandwich(peps, peps, 1, 2, reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2), TreeSA(), MergeGreedy())
    exact = (statevec(peps)' * mat(h) * statevec(peps))[1]
    @test expect ≈ exact
end









@testset "truncate1" begin #product + noise
    mps=generate_mps(8,10; d=2)
    mps0=deepcopy(mps)
    mps0.tensors[3] = cat(mps0.tensors[3], zeros(8, 2, 4), dims=3)
    mps0.tensors[4] = cat(mps0.tensors[4], zeros(4, 2, 8), dims=1)
    mps1=IsoPEPS.truncated_bmps(mps0.tensors,8)
    @test size(mps.tensors[3],3)==size(mps1[4],1)
end

@testset "truncate2" begin #local random unitary 
    mps=generate_mps(8,10; d=2)
    mps0=deepcopy(mps)
    A=randn(12,12)
    Q,R=IsoPEPS.qr(A)
    Q=Q[1:8, :]
    mps0.tensors[3]=IsoPEPS.ein"ijk,kl->ijl"(mps0.tensors[3],Q)
    mps0.tensors[4]=IsoPEPS.ein"li,ijk->ljk"(Q',mps0.tensors[4])
    mps1=IsoPEPS.truncated_bmps(mps0.tensors,8)
    @test size(mps.tensors[3],3)==size(mps1[4],1)
end

@testset "truncate3" begin #TFIM ground state
    eigenval,eigenvec=IsoPEPS.eigsolve(IsoPEPS.mat(sum([kron(10, i=>(-IsoPEPS.X),  mod1(i+1, 10)=>IsoPEPS.X) for i in 1:10])
                                                 + sum([-0.5 * kron(10, i => IsoPEPS.Z) for i in 1:10])), 1, :SR; ishermitian=true)

    H=IsoPEPS.mat(sum([kron(10, i=>(-IsoPEPS.X),  mod1(i+1, 10)=>IsoPEPS.X) for i in 1:10])
                  + sum([-0.5 * kron(10, i => IsoPEPS.Z) for i in 1:10]))

    eigenmps=vec2mps(Array(eigenvec[1]))
    eigenmps1=IsoPEPS.truncated_bmps(eigenmps.tensors,16)
    eigenvec1=code_mps2vec(eigenmps1)
    eigenval1=real((eigenvec1'*(H*eigenvec1))/(eigenvec1'*eigenvec1))
    @show eigenval[1],eigenval1
    @test isapprox(eigenval[1],eigenval1,atol=1e-3)
end

@testset "peps" begin
    bra_ket=IsoPEPS.contract_2peps(peps,peps)
    @show size(bra_ket[1][1])
    @show size(bra_ket[2][1])
    dmax=4
    #bmps=IsoPEPS.truncated_bmps(bra_ket[1],dmax)
    
    result=IsoPEPS.overlap_peps(bra_ket,dmax)
    @show result
    @test  isless(0.0,result)
end 

@testset "peps_variation" begin
    Ly=2
    Lx=2
    nsites=Ly*Lx
    bond_dim=2
    #h=0.2
    result= peps_variation(Ly,Lx,bond_dim)
    @test isapprox(result, -4.040593699203846, atol=1e-4)
end




@testset "hamiltonian expectation value" begin
    h = ising_hamiltonian_2d(2,2, 1.0, 0.2)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(h), 1, :SR; ishermitian=true)
    @test eigenval[1]≈  -4.040593699203846
end

@testset "gradient of hamiltonian with FiniteDifference" begin
    Ly=2
    Lx=2
    nsites=Ly*Lx
    bond_dim=2
    #h=0.2
    #result,f,g!= peps_variation(Ly,Lx,bond_dim)
    a=IsoPEPS.grad(central_fdm(5, 1), f, params)[1] - g!(params)
    @test a==0.0
end

@testset "optimization" begin
    
end