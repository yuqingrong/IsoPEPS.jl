using IsoPEPS
using Test
using Yao, YaoBlocks
using LinearAlgebra
using Optim
using Manifolds
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
    nsites = 3; row = 3; niters = 40
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
    rho_iter,_ = iterate_channel_PEPS(gate, row; niters=niters)
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


using Optimization, OptimizationCMAEvolutionStrategy
using Random
J=1.0; g=0.0; row=3
d=D=2
p=3
#E = exact_energy_PEPS(d, D, g, row)
Random.seed!(12)
params = rand(6*p)
X_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history = train_energy_circ(params, J, g, p, row)
_, gap = exact_left_eigen(final_A, row)
@show gap

@show X_history[end]
@show final_params
@show mean(X_history[end-100:end])
@show mean(gap_list[end-1000:end])


show(IOContext(stdout, :limit => false), "text/plain", final_params)
# Save the training datagap
save_training_data(X_list_list, Z_list_list, gap_list)
gap_file="data/params_list_g=1.75.dat"
open(gap_file, "w") do io
    println(io, "# energy_list data from iPEPS optimization")
    println(io, "# Number of iterations: $(length(Z_list_list))")
    if !isempty(gap_list)
        println(io, "# Length of each Z_list: $(length(Z_list_list[1]))")
    end
    println(io, "# Format: Each line contains one gaZp (space-separated values)")
    println(io, "#")
    for energy in params_history
        println(io, join(energy, " "))
    end
end


vcat(Z_list_list...)
if !isempty(X_history)
    Plots.plot(
        vcat(gap_list),
        xlabel="Iteration",
        ylabel="gap",
        title="gap vs parameter iteration",
        ylims=(-0.0,10.0),
        legend=false,
        size=(900,300)  
    )
else
    @warn "Z_list is empty, nothing to plot."
end

g=0.0
final_params = [0.5853313274828805, 1.3221182398953806, 0.5870239199007706, 1.6493164670031948, 0.006467861307395547, 1.5440504250265632, 0.0028286948235898834, 0.7491018190850874, 0.0016330941515845588, 0.37818070322012504, 0.0002828854638718699, 0.43774879243693443, 0.010527141466183934, 0.0596061656172594, 0.00012856349358425536, 0.5729865546127543, 0.004902198156040871, 0.03810612518307743]
draw_X_from_file(g,[1,10,15,20,26])

check_gap_sensitivity(final_params, 13, g, row, p)

check_all_gap_sensitivity_combined(final_params, g, row, p)

draw_gap()