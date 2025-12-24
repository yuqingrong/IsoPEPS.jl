using IsoPEPS.InfPEPS
using Optimization, OptimizationCMAEvolutionStrategy
using Random
using Plots
using Yao, Manifolds
using LinearAlgebra, OMEinsum

function simulation(J::Float64, g::Float64, row::Int, p::Int, nqubits::Int; maxiter=5000, measure_first=:X)
    Random.seed!(12)
    params = rand(2*nqubits*p)
    energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, final_gap, final_eigenvalues, params_history = train_energy_circ(params, J, g, p, row, nqubits; measure_first=measure_first, share_params=true, maxiter=maxiter)
    #energy_history, final_A, final_params, final_cost, X_list, ZZ_list1, ZZ_list2, gap_list, eigenvalues_list = train_exact(params, J, g, p, row, nqubits; maxiter=maxiter, measure_first=measure_first)
    #gate = Yao.matblock(rand_unitary(ComplexF64, 2^nqubits))
    #M = Manifolds.Unitary(2^nqubits, Manifolds.ℂ)
    #result, final_energy, final_p, X_list, ZZ_list1, ZZ_list2, energy_history, gap_list, eigenvalues_list = train_nocompile(gate, row, nqubits,M, J, g; maxiter=maxiter)
    #return result, final_energy, final_p, X_list, ZZ_list1, ZZ_list2, energy_history, gap_list, eigenvalues_list
    #energy_history, final_A, final_params, final_cost, X_list, ZZ_list1, ZZ_list2, gap_list, params_history, eigenvalues_list = train_exact(params, J, g, p, row, nqubits; maxiter=maxiter, measure_first=measure_first)
    @show final_params
    return nothing
end

"""
    parallel_simulation_threaded(J::Float64, g_values::Vector{Float64}, row::Int, p::Int; maxiter=5000, measure_first=:X)

Run simulations for multiple g values in parallel using multi-threading.
Returns a dictionary with g values as keys and simulation results as values.

"""
function parallel_simulation_threaded(J::Float64, g_values::Vector{Float64}, row::Int, p::Int, nqubits::Int; maxiter=5000, measure_first=:Z)
    n = length(g_values)
    results = Vector{Any}(undef, n)
    
    println("Running $(n) simulations in parallel with $(Threads.nthreads()) threads...")
    #==
    Threads.@threads for i in 1:n
        g = g_values[i]
        println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        # Each thread gets its own random seed for thread safety
        Random.seed!(1234 + i)
        gate = Yao.matblock(rand_unitary(ComplexF64, 2^row))
        M = Manifolds.Unitary(2^row, Manifolds.ℂ)
        
        result, final_energy, final_p, X_list, ZZ_list, energy_history, gap_list, eigenvalues_list = 
            train_nocompile(gate, row, M, J, g; maxiter=maxiter)
        
        results[i] = (
            g = g,
            result = result,
            final_energy = final_energy,
            final_p = final_p,
            X_list = X_list,
            ZZ_list = ZZ_list,
            energy_history = energy_history,
            gap_list = gap_list,
            eigenvalues_list = eigenvalues_list
        )
        
        println("Thread $(Threads.threadid()): Completed simulation for g = $(g), final_energy = $(final_energy)")
    end
    ==#
    #Old version using train_energy_circ
    Threads.@threads for i in 1:n
        g = g_values[i]
        println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        Random.seed!(12)
        params = rand(2*nqubits*p)
        
        energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, eigenvalues_list, params_history, final_gap = train_energy_circ(params, J, g, p, row, nqubits; maxiter=maxiter, measure_first=measure_first)
        
        results[i] = (
            g = g,
            energy_history = energy_history,
            final_A = final_A,
            final_params = final_params,
            final_cost = final_cost,
            Z_list_list = Z_list_list,
            X_list_list = X_list_list,
            gap_list = gap_list,
            params_history = params_history
        )
        
        println("Thread $(Threads.threadid()): Completed simulation for g = $(g)")
    end
    
    
    # Convert to dictionary for easier access
    return Dict(results[i].g => results[i] for i in 1:n)
end

using CairoMakie
J=1.0; g=3.0; g_values=[0.5, 1.0, 1.5,2.0, 2.5, 3.0, 3.5, 4.0]; row=8
d=2; D=2; nqubits=3
p=3
 
#E, ξ_h, ξ_v, λ_h, λ_v = result_PEPSKit(d, D, J, g; χ=20, ctmrg_tol=1e-10, grad_tol=1e-4, maxiter=1000)
#E, len_gapped, entrop_gapped = result_MPSKit(d, D, g, row)
simulation(J, g, row, p, nqubits; maxiter=5000, measure_first=:Z)
parallel_simulation_threaded(J, g_values, row, p, nqubits; maxiter=2000, measure_first=:Z)
ACF(J, g, p,row, nqubits; max_lag=10, hor=false)
gap, energy = exact_E_from_params(g, J, p, row, nqubits; data_dir="data", optimizer=GreedyMethod())
draw()
draw_energy_error()
spin_correlation(J, g, p, row, nqubits; measure_first=:Z, conv_step=1000, samples=1000000, data_dir="data")
C1C2(g_values, J, p, row, nqubits;samples=1000000)
magnectization(g_values, row)
dynamics(g, J, p, row, nqubits; conv_step=100, samples=0, nshots=100000, data_dir="data")
energy_con([0.5, 1.0, 1.5, 2.0], row)
#=
gap, energy = exact_E_from_params(g, J, p, row, nqubits; data_dir="data", optimizer=GreedyMethod())
@show energy
eigenvalues(g_values; data_dir="data_exact")
gap(g_values; data_dir="data_exact")

correlation(g; measure_first=:Z, data_dir="data",max_lag=2)

#dynamics_observables(g; measure_first=:Z)
#dynamics_observables_all(g_values; measure_first=:Z)
#block_variance(g,[1,5000])
draw()
chain_result(J::Float64, g::Float64, row::Int, d::Int, D::Int)
energy_converge([0.25, 0.5, 1.25, 1.5])
=#

function analyze_trained_gate(g::Float64, row::Int, p::Int; 
                              measure_first=:Z, data_dir="data")
    # Construct filename
    prefix = measure_first == :X ? "X" : "Z"
    filename = joinpath(data_dir, "$(prefix)_first_params_history_g=$(g).dat")
    
    if !isfile(filename)
        error("File not found: $filename")
    end
    
    # Read the last line
    lines = readlines(filename)
    # Filter out empty lines
    non_empty_lines = filter(line -> !isempty(strip(line)), lines)
    
    if isempty(non_empty_lines)
        error("File is empty: $filename")
    end
    
    last_line = non_empty_lines[end]
    
    # Parse parameters
    params = parse.(Float64, split(last_line))
    
    # Verify we have the right number of parameters
    expected_params = 6 * p
    if length(params) != expected_params
        @warn "Expected $expected_params parameters but got $(length(params)). Using first $expected_params."
        params = params[1:expected_params]
    end
    
    @info "Loaded $(length(params)) parameters from $filename"
    @info "Parameter range: [$(minimum(params)), $(maximum(params))]"
    
    # Build gate from parameters
    A_matrix = build_gate_from_params(params, p)
    gate = Yao.matblock(A_matrix)
    
    # Compute spectral properties
    rho, gap,gap_h, eigenvalues = exact_left_eigen(gate, row)
    
    @info "Spectral gap: $gap"
    @info "Horizontal spectral gap: $gap_h"
    @info "Largest eigenvalue: $(maximum(abs.(eigenvalues)))"
    
    return gate, rho, gap, gap_h, eigenvalues, params
end


params = rand(2*nqubits*p)
A_matrix = build_gate_from_params(params, p, row, nqubits)
rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=1000, samples=0,measure_first=:Z)
rho = join(rho, zero_state(1))
rho = Yao.apply!(rho, put(Int((nqubits-1)/2)*(row+1)+1,(1, 2, 3,4,5)=>matblock(A_matrix[1]))) 
von_neumann_entropy(rho, (1))