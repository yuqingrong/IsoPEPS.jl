using IsoPEPS
using Optimization, OptimizationCMAEvolutionStrategy
using Random
using CairoMakie
using Yao, Manifolds
using LinearAlgebra, OMEinsum

function simulation(J::Float64, g::Float64, row::Int, p::Int, nqubits::Int; maxiter=5000, measure_first=:X)
    Random.seed!(12)
    params = rand(2*nqubits*p)
    
    # Use new API with result struct
    result = optimize_circuit(params, J, g, p, row, nqubits; 
                              measure_first=measure_first, 
                              share_params=true, 
                              maxiter=maxiter)
    
    @show result.params
    @show result.energy
    @show result.converged
    
    # Alternative: exact optimization
    # result = optimize_exact(params, J, g, p, row, nqubits; maxiter=maxiter)
    
    # Alternative: manifold optimization
    # gate = Yao.matblock(rand_unitary(ComplexF64, 2^nqubits))
    # M = Manifolds.Unitary(2^nqubits, Manifolds.ℂ)
    # result = optimize_manifold(gate, row, nqubits, M, J, g; maxiter=maxiter)
    
    return result
end

"""
    parallel_simulation_threaded(J::Float64, g_values::Vector{Float64}, row::Int, p::Int; maxiter=5000, measure_first=:X)

Run simulations for multiple g values in parallel using multi-threading.
Returns a dictionary with g values as keys and simulation results as values.
"""
function parallel_simulation_threaded(J::Float64, g_values::Vector{Float64}, row::Int, p::Int, nqubits::Int; 
                                       maxiter=5000, measure_first=:Z)
    n = length(g_values)
    results = Vector{CircuitOptimizationResult}(undef, n)
    
    println("Running $(n) simulations in parallel with $(Threads.nthreads()) threads...")

    Threads.@threads for i in 1:n
        g = g_values[i]
        println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        Random.seed!(12 + i)  # Different seed for each thread
        params = rand(2*nqubits*p)
        
        result = optimize_circuit(params, J, g, p, row, nqubits; 
                                  maxiter=maxiter, 
                                  measure_first=measure_first)
        
        results[i] = result
        
        println("Thread $(Threads.threadid()): Completed simulation for g = $(g), energy = $(result.energy)")
    end
    
    # Convert to dictionary for easier access (with g values as keys)
    return Dict(g_values[i] => results[i] for i in 1:n)
end

# Example usage (commented out to prevent execution on include)
#=
using CairoMakie
J = 1.0
g = 3.0
g_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
row = 8
d = 2
D = 2
nqubits = 3
p = 3

# Reference calculations
# result_ref = pepskit_ground_state(d, D, J, g; χ=20, ctmrg_tol=1e-10, grad_tol=1e-4, maxiter=1000)
# result_ref = mpskit_ground_state(d, D, g, row)

# Single simulation
result = simulation(J, g, row, p, nqubits; maxiter=5000, measure_first=:Z)

# Parallel simulations
results_dict = parallel_simulation_threaded(J, g_values, row, p, nqubits; maxiter=2000, measure_first=:Z)

# Plot results
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel="g", ylabel="Energy", title="TFIM Energy vs g")
energies = [results_dict[g].energy for g in g_values]
lines!(ax, g_values, energies, marker=:circle, linewidth=2)
display(fig)

# Compute exact energy for comparison
params = rand(2*nqubits*p)
gap, energy = compute_exact_energy(params, g, J, p, row, nqubits)
@show gap, energy
=#

"""
    analyze_result(filename::String)

Analyze a saved training result from JSON file.
"""
function analyze_result(filename::String)
    result, input_args = load_result(filename)
    
    println("=== Training Result Analysis ===")
    println("Type: ", typeof(result))
    println("Final energy: ", result.energy)
    println("Converged: ", result.converged)
    println("Iterations: ", length(result.energy_history))
    
    if haskey(input_args, :g)
        println("\nModel parameters:")
        println("  g = ", input_args[:g])
        println("  J = ", get(input_args, :J, "N/A"))
        println("  row = ", get(input_args, :row, "N/A"))
        println("  p = ", get(input_args, :p, "N/A"))
        println("  nqubits = ", get(input_args, :nqubits, "N/A"))
    end
    
    # Plot training history
    fig = plot_training_history(result; title="Training History")
    display(fig)
    
    return result, input_args
end

"""
    compare_methods(J::Float64, g::Float64, row::Int, p::Int, nqubits::Int; maxiter=1000)

Compare circuit optimization vs exact optimization.
"""
function compare_methods(J::Float64, g::Float64, row::Int, p::Int, nqubits::Int; maxiter=1000)
    Random.seed!(42)
    params = rand(2*nqubits*p)
    
    println("Running circuit optimization...")
    result_circuit = optimize_circuit(copy(params), J, g, p, row, nqubits; 
                                      maxiter=maxiter, 
                                      conv_step=100, 
                                      samples=1000)
    
    println("Running exact optimization...")
    result_exact = optimize_exact(copy(params), J, g, p, row, nqubits; 
                                   maxiter=maxiter)
    
    println("\n=== Results Comparison ===")
    println("Circuit optimization:")
    println("  Energy: ", result_circuit.energy)
    println("  Converged: ", result_circuit.converged)
    
    println("\nExact optimization:")
    println("  Energy: ", result_exact.energy)
    println("  Gap: ", result_exact.gap)
    println("  Converged: ", result_exact.converged)
    
    # Plot comparison
    fig = Figure(size=(1200, 500))
    
    ax1 = Axis(fig[1, 1], xlabel="Iteration", ylabel="Energy", title="Circuit Optimization")
    lines!(ax1, 1:length(result_circuit.energy_history), result_circuit.energy_history, linewidth=2)
    
    ax2 = Axis(fig[1, 2], xlabel="Iteration", ylabel="Energy", title="Exact Optimization")
    lines!(ax2, 1:length(result_exact.energy_history), result_exact.energy_history, linewidth=2)
    
    display(fig)
    
    return result_circuit, result_exact
end

"""
    reconstruct_gates(result, p::Int, row::Int, nqubits::Int)

Reconstruct gates from optimization result parameters.
"""
function reconstruct_gates(result::Union{CircuitOptimizationResult, ExactOptimizationResult}, 
                          p::Int, row::Int, nqubits::Int; share_params=true)
    gates = build_unitary_gate(result.params, p, row, nqubits; share_params=share_params)
    
    # Compute transfer spectrum
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    println("=== Gate Analysis ===")
    println("Spectral gap: ", gap)
    println("Largest eigenvalue: ", maximum(abs.(eigenvalues)))
    println("Second largest eigenvalue: ", sort(abs.(eigenvalues))[end-1])
    
    return gates, rho, gap, eigenvalues
end

"""
    visualize_correlation(result::CircuitOptimizationResult, row::Int)

Visualize spin-spin correlation from circuit optimization result.
"""
function visualize_correlation(result::CircuitOptimizationResult, row::Int)
    if isempty(result.Z_samples)
        @warn "No Z samples in result"
        return nothing
    end
    
    fig = plot_correlation_heatmap(result.Z_samples, row; 
                                    title="Spin-Spin Correlation")
    display(fig)
    
    return fig
end

"""
    analyze_acf(samples::Vector{Float64}; max_lag=100)

Analyze autocorrelation function of samples.
"""
function analyze_acf(samples::Vector{Float64}; max_lag=100)
    lags, acf, acf_err = compute_acf(samples; max_lag=max_lag, n_bootstrap=50)
    
    try
        A, ξ = fit_acf_exponential(lags, acf)
        println("Autocorrelation length: ξ = ", ξ)
        
        fig = plot_acf(lags, acf; 
                      acf_err=acf_err,
                      fit_params=(A, ξ),
                      title="Autocorrelation Function (ξ = $(round(ξ, digits=2)))")
        display(fig)
        
        return lags, acf, ξ
    catch e
        @warn "ACF fit failed: $e"
        fig = plot_acf(lags, acf; acf_err=acf_err)
        display(fig)
        return lags, acf, NaN
    end
end
