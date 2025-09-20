using IsoPEPS
using Test
import Yao, YaoBlocks
using LinearAlgebra
@testset "contract_Elist" begin
    A = randn(ComplexF64, 2,2,2,2,2)
    nsites = 1
    code, result = contract_Elist(A, nsites; optimizer=IsoPEPS.GreedyMethod())
    @show code
    @test result isa Array{ComplexF64, 4*(nsites+1)}
end

@testset "left_eigen" begin
    nsites = 3; row = 1
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
    rho = exact_left_eigen(gate, row)
    @test tr(rho) ≈ 1.
end

@testset "iterate_channel_PEPS" begin
    nsites = 3; row = 3; niters = 30
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
    rho_iter = iterate_channel_PEPS(gate, niters, row)
    rho_eigen = exact_left_eigen(gate, row)
    @test norm(rho_iter - rho_eigen) < 1e-8
    @test rho_iter ≈ rho_eigen 
end



# try GHZ + Toffoli
Toffoli = Matrix{ComplexF64}(I, 8, 8)  
Toffoli[7,7] = 0
Toffoli[8,8] = 0
Toffoli[7,8] = 1
Toffoli[8,7] = 1

gate = YaoBlocks.matblock(Toffoli)
rho_iter = iterate_channel_PEPS(gate, 1, 1)
