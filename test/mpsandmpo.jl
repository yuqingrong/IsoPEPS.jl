using IsoPEPS
using Test


@testset "mps_dot_mpo_dot_mps" begin #TFIM ground state
    nsites=10
    eigenval,eigenvec=IsoPEPS.eigsolve(IsoPEPS.mat(sum([kron(nsites, i=>(-IsoPEPS.Z),  mod1(i+1, nsites)=>IsoPEPS.Z) for i in 1:nsites])
                                                 + sum([-0.5 * kron(nsites, i => IsoPEPS.X) for i in 1:nsites])), 1, :SR; ishermitian=true)
    
    H=Matrix(IsoPEPS.mat(sum([kron(nsites, i=>(-IsoPEPS.Z),  mod1(i+1, nsites)=>IsoPEPS.Z) for i in 1:nsites])
                       + sum([-0.5 * kron(nsites, i => IsoPEPS.X) for i in 1:nsites])))
    eigenmps=vec2mps(Array(eigenvec[1]))
    #eigenmps=MPS(IsoPEPS.truncated_bmps(vec2mps(Array(eigenvec[1])).tensors,16))
        
    #mpo=IsoPEPS.transverse_ising_mpo(nsites,0.5)
    mpo=mat2mpo(H)
    new_mps=mps_dot_mpo(eigenmps::MPS,mpo::MPO)
    #new_mps=MPS(IsoPEPS.truncated_bmps(new_mps.tensors,16))
    @show code_dot(eigenmps::MPS, eigenmps::MPS)
    eigenval1=code_dot(eigenmps::MPS,new_mps::MPS)
    result=code_sandwich(eigenmps,mpo,eigenmps) 
    @show result,eigenval1
    @test isapprox(eigenval[1],result,atol=1e-3)
    @test isapprox(eigenval[1],eigenval1,atol=1e-3)  
end
 