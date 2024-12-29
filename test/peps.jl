using IsoPEPS
using Test


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
    peps=generate_peps(4,3,3)
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
    Ly=4
    Lx=4
    nsites=Ly*Lx
    bond_dim=2
    h=0.2
    result,f,g! = peps_variation(Ly,Lx,bond_dim,h)
    @test isapprox(result.minimum, -9.120354170186685, atol=1e-4)
end