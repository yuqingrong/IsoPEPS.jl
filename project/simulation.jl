using IsoPEPS
using Optimization, OptimizationCMAEvolutionStrategy
using Random
using CairoMakie
using Yao, Manifolds
using LinearAlgebra, OMEinsum
"""
    simulation(J::Float64, g_values::Vector{Float64}, row::Int, p::Int, nqubits::Int; 
               maxiter=5000, measure_first=:X, seed=12, verbose=true, 
               output_dir="data", share_params=true)

Run circuit optimization in parallel for multiple g values and save results to JSON files.

# Arguments
- `J::Float64`: Horizontal coupling strength
- `g_values::Vector{Float64}`: Vector of transverse field strengths
- `row::Int`: Number of rows in the PEPS
- `p::Int`: Circuit depth
- `nqubits::Int`: Number of qubits per row
- `maxiter`: Maximum iterations for optimization
- `measure_first`: Which basis to measure first (:X or :Z)
- `seed`: Random seed for reproducibility
- `verbose`: Print progress information
- `output_dir`: Directory to save results (default: "data")
- `share_params`: Share parameters across circuit layers
- `conv_step`: Convergence step for sampling (default: 100)
- `samples`: Number of samples per run (default: 10000)
- `n_runs`: Number of parallel sampling runs (default: 44)
- `abstol`: Absolute tolerance for optimization convergence (default: 0.01)

# Returns
- Dict{Float64, CircuitOptimizationResult} with g values as keys

# Side effects
- Saves each result to `output_dir/circuit_J={J}_g={g}_row={row}.json`

# Example
```julia
g_values = [1.0, 2.0, 3.0, 4.0]
results = simulation(1.0, g_values, 3, 2, 8; maxiter=5000, output_dir="results")
```
"""
function simulation(; J::Float64, g_values::Vector{Float64}, row::Int, p::Int, nqubits::Int, 
                    maxiter::Int, measure_first::Symbol, seed::Int, verbose::Bool,
                    output_dir::String, share_params::Bool, conv_step::Int=100, samples::Int=10000,
                    n_runs::Int=44, abstol::Float64=0.01, sigma0::Float64=1.0, popsize::Union{Int,Nothing}=20, zz_weight::Float64=0.1)
    
    # Create output directory if it doesn't exist
    !isdir(output_dir) && mkpath(output_dir)
    
    n = length(g_values)
    results = Vector{CircuitOptimizationResult}(undef, n)
    
    verbose && println("Running $(n) simulations in parallel with $(Threads.nthreads()) threads...")
    verbose && println("Results will be saved to: $output_dir/")
    
    for i in 1:n
        g = g_values[i]
        verbose && println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        Random.seed!(seed)  # Different seed for each thread
        params = initialize_tfim_params(p, nqubits, g; mode=:entangled)
        result = optimize_circuit(params, J, g, p, row, nqubits; 
                                  maxiter=maxiter, 
                                  measure_first=measure_first,
                                  share_params=share_params,
                                  conv_step=conv_step,
                                  samples=samples,
                                  n_runs=n_runs,
                                  abstol=abstol,sigma0=sigma0, popsize=popsize,zz_weight=zz_weight)
        
        results[i] = result
        
        # Save result to JSON
        filename = joinpath(output_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_nqubits=$(nqubits).json")
        input_args = Dict(
            :J => J, :g => g, :row => row, :p => p, :nqubits => nqubits,
            :maxiter => maxiter, :measure_first => measure_first,
            :share_params => share_params, :seed => seed
        )
        save_result(filename, result, input_args) 
        
        verbose && println("Thread $(Threads.threadid()): Completed g = $(g), energy = $(result.final_cost), saved to $(basename(filename))")
    end
end

simulation(;
    J=1.0,
    g_values=[2.0],
    row=2,
    p=3,
    nqubits=5,
    maxiter=3000,
    measure_first=:Z,
    seed=42,
    verbose=true,
    output_dir=joinpath(@__DIR__, "results"),
    share_params=true,
    conv_step=100,
    samples=1000,
    n_runs=44,
    abstol=1e-10,
    sigma0=1.0,
    popsize=30,
    zz_weight=9.0)