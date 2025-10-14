using IsoPEPS
using Test
import Yao, YaoBlocks
using LinearAlgebra
using Optim
import Manifolds
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
    nsites = 3; row = 3; niters = 50
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
    rho_iter = iterate_channel_PEPS(gate, row)
    rho_eigen, gap = exact_left_eigen(gate, row)
    re1 = cost_ZZ(rho_iter, row, gate); re2 = cost_X(rho_iter, row, gate)
    @show re1, re2
    @test gap > 0
    @test norm(rho_iter.state - rho_eigen) < 1e-8
    @test rho_iter.state ≈ rho_eigen 
end

@testset "exact_energy_PEPS" begin
    d = 2; D = 2; J =1.0; g = 1.00; row=3
    x = 4; y = 4
    E = exact_energy_PEPS(d, D, g, row)
    h = ising_ham_periodic2d(x, y, J, g)
    eigenval,eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(h), 1, :SR; ishermitian=true)
    @show E, eigenval[1]/(x*y)
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

@testset "cost_X" begin
    J=1.0; g=0.00; row=3
    d=D=2
    p=6
    E = exact_energy_PEPS(d, D, g, row)
    params = rand(3*p)
    X_history, final_A, final_params, final_cost = train_energy_circ(params, J, g, p, row)

    @test X_history[end] ≈ E atol=1e-2
end

@testset "no compile" begin
    J = 1.0; g=0.0
    nsites = 3; row = 3; niters = 50
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
    M = Manifolds.Unitary(2^nsites, Manifolds.ℂ)
    result, final_energy,gate = train_nocompile(gate, row, M, J, g)
    @show result
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
using Random
J=1.0; g=0.00; row=3
d=D=2
p=3
#E = exact_energy_PEPS(d, D, g, row)
Random.seed!(12)
params = rand(6*p)
X_history, final_A, final_params, final_cost = train_energy_circ(params, J, g, p, row)
_, gap = exact_left_eigen(final_A, row)
@show gap
@show X_history[end]
@show final_A
@show final_params
final_params ≈ params


J = 1.0; g=0.00
nsites = 3; row = 3; niters = 100
gate1 = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
A =  Matrix(gate1)
M = Manifolds.Unitary(2^nsites, Manifolds.ℂ)
result, final_energy, final_p = train_nocompile(gate1, row, M, J, g)

@show final_energy
_, gap = exact_left_eigen(gate1, nsites)
@show final_p
gate = YaoBlocks.matblock(final_p)
_, gap = exact_left_eigen(gate, row)
@show gap
final_A ≈ final_p
tr(final_A' * final_p)

d=D=2; J=1.0; g=1.00
E = exact_iPEPS(d, D, J, g)
