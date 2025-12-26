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
                    output_dir::String, share_params::Bool)
    
    # Create output directory if it doesn't exist
    !isdir(output_dir) && mkpath(output_dir)
    
    n = length(g_values)
    results = Vector{CircuitOptimizationResult}(undef, n)
    
    verbose && println("Running $(n) simulations in parallel with $(Threads.nthreads()) threads...")
    verbose && println("Results will be saved to: $output_dir/")
    
    Threads.@threads for i in 1:n
        g = g_values[i]
        verbose && println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        Random.seed!(seed)  # Different seed for each thread
        params = rand(2*nqubits*p)
        
        result = optimize_circuit(params, J, g, p, row, nqubits; 
                                  maxiter=maxiter, 
                                  measure_first=measure_first,
                                  share_params=share_params)
        
        results[i] = result
        
        # Save result to JSON
        filename = joinpath(output_dir, "circuit_J=$(J)_g=$(g)_row=$(row).json")
        input_args = Dict(
            :J => J, :g => g, :row => row, :p => p, :nqubits => nqubits,
            :maxiter => maxiter, :measure_first => measure_first,
            :share_params => share_params, :seed => seed + i
        )
        save_result(filename, result, input_args)
        
        verbose && println("Thread $(Threads.threadid()): Completed g = $(g), energy = $(result.energy), saved to $(basename(filename))")
    end
end

simulation(;
    J=1.0,
    g_values=[3.0],
    row=4,
    p=4,
    nqubits=5,
    maxiter=10000,
    measure_first=:Z,
    seed=12,
    verbose=true,
    output_dir=joinpath(@__DIR__, "results"),
    share_params=true
)