using IsoPEPS
using Test
import Yao, YaoBlocks
using LinearAlgebra
using Optim
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
    rho_eigen, gap = exact_left_eigen(gate, row)
    @show gap
    @test gap > 0
    @test norm(rho_iter.state - rho_eigen) < 1e-8
    @test rho_iter.state ≈ rho_eigen 
end

@testset "exact_energy_PEPS" begin
    d = 2; D = 2; J =1.0; g = 0.00; row=3
    x = 4; y = 4
    E = exact_energy_PEPS(d, D, g, row)
    h = ising_ham_periodic2d(x, y, J, g)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(h), 1, :SR; ishermitian=true)
    @show E
    @test eigenval[1]/(x*y) ≈ E atol=1e-3
end

@testset "train_measure" begin
    J=1.0; g=0.00; row=3
    d=D=2
    p=6
    E = exact_energy_PEPS(d, D, g, row)
    params = rand(3*p)
    X_history, final_A, final_params, final_cost = train_energy_circ(params, J, g, p, row)
    _, gap = exact_left_eigen(final_A, row)
    @show gap
    @test X_history[end] ≈ E atol=1e-2
end



# try GHZ + Toffoli
Toffoli = Matrix{ComplexF64}(I, 8, 8)  
Toffoli[7,7] = 0
Toffoli[8,8] = 0
Toffoli[7,8] = 1
Toffoli[8,7] = 1

gate = YaoBlocks.matblock(Toffoli)
rho_iter = iterate_channel_PEPS(gate, 1, 1)

using Plots
draw() 

J=1.0; g=0.00; row=3
d=D=2
p=8
E = exact_energy_PEPS(d, D, g, row)
params = rand(3*p)
X_history, final_A, final_params, final_cost = train_energy_circ(params, J, g, p, row)
final_A == Matrix(I, 8, 8)
_, gap = exact_left_eigen(final_A, row)
@show gap
@show X_history[end]