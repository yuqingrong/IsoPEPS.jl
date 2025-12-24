using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra

@testset "compute_single_transfer" begin
    nqubits = 3
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    gates = [Matrix(gate)]
    
    rho, gap, eigenvalues = compute_single_transfer(gates, nqubits)
    
    @test LinearAlgebra.tr(rho) ≈ 1.0
    @test gap > 0
    @test all(eigenvalues[1:end-1] .< 1.0)
    @test eigenvalues[end] ≈ 1.0
end

