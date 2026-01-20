using IsoPEPS
using Optimization, OptimizationCMAEvolutionStrategy
using Random
using CairoMakie
using Yao, Manifolds
using LinearAlgebra, OMEinsum
"""
    simulation(; J, g_values, row, p, nqubits, ...)

Run circuit optimization for multiple g values and save results to JSON files.

# Arguments
- `J::Float64`: Horizontal coupling strength
- `g_values::Vector{Float64}`: Vector of transverse field strengths
- `row::Int`: Number of rows in the PEPS
- `p::Int`: Circuit depth
- `nqubits::Int`: Number of qubits per row

# Optimization Settings
- `maxiter::Int`: Maximum CMA-ES iterations (default: 5000)
- `abstol::Float64`: Function tolerance for convergence (default: 0.02)
- `xtol::Float64`: Parameter tolerance for convergence (default: 1e-6)
- `sigma0::Float64`: Initial step size (default: 1.0)
- `popsize::Union{Int,Nothing}`: Population size (default: auto)

# Sampling Settings
- `measure_first::Symbol`: Which basis to measure first, :X or :Z (default: :Z)
- `share_params::Bool`: Share parameters across circuit layers (default: true)
- `samples_per_run::Int`: Samples per chain (default: 1000)
- `n_parallel_runs::Int`: Number of parallel chains (default: 44)
- `conv_step::Int`: Burn-in steps (default: 100)

# Other
- `seed::Int`: Random seed for reproducibility (default: 42)
- `verbose::Bool`: Print progress information (default: true)
- `output_dir::String`: Directory to save results (default: "data")

# Returns
- `Vector{CircuitOptimizationResult}` with results for each g value

# Example
```julia
results = simulation(
    J = 1.0,
    g_values = [0.5, 1.0, 2.0],
    row = 1, p = 4, nqubits = 3,
    maxiter = 5000,
    output_dir = "results"
)
```
"""
function simulation(;
    J::Float64,
    g_values::Vector{Float64},
    row::Int,
    p::Int,
    nqubits::Int,
    # Optimization settings
    maxiter::Int = 5000,
    abstol::Float64 = 0.02,
    # Gap regularization
    gap_regularization::Bool = false,
    gap_weight::Float64 = 1.0,
    gap_threshold::Float64 = 0.1,
    # Sampling settings
    measure_first::Symbol = :Z,
    share_params::Bool = true,
    samples_per_run::Int = 1000,
    n_parallel_runs::Int = 44,
    conv_step::Int = 100,
    # Other
    seed::Int = 42,
    verbose::Bool = true,
    output_dir::String = "data")
    # Validate inputs
    measure_first ∈ (:X, :Z) || throw(ArgumentError("measure_first must be :X or :Z"))
    isempty(g_values) && throw(ArgumentError("g_values cannot be empty"))
    
    # Create output directory if it doesn't exist
    !isdir(output_dir) && mkpath(output_dir)
    
    n = length(g_values)
    n_params = 2 * nqubits * p
    results = Vector{CircuitOptimizationResult}(undef, n)
    
    if verbose
        println("=" ^ 60)
        println("TFIM Circuit Optimization")
        println("=" ^ 60)
        println("Model parameters:")
        println("  J = $J, g values = $g_values")
        println("  row = $row, nqubits = $nqubits, p = $p")
        println("  n_params = $n_params")
        println()
        println("Optimization settings:")
        println("  Stopping: maxiter=$maxiter, abstol=$abstol")
        if gap_regularization
            println("  Gap regularization: weight=$gap_weight, threshold=$gap_threshold")
        end
        println()
        println("Sampling settings:")
        println("  samples_per_run = $samples_per_run, n_parallel_runs = $n_parallel_runs")
        println("  conv_step = $conv_step, measure_first = $measure_first")
        println()
        println("Running $n simulations with $(Threads.nthreads()) threads...")
        println("Results will be saved to: $output_dir/")
        println("=" ^ 60)
    end
    
    for i in 1:n
        g = g_values[i]
        
        verbose && println("\n[$i/$n] Starting simulation for g = $g")
        
        # Set seed for reproducibility
        sim_seed = seed
        Random.seed!(sim_seed)
        
        # Initialize parameters
        params = ones(n_params)
        
        # Run optimization with all settings
        result = optimize_circuit(
            params, J, g, p, row, nqubits;
            # Optimization settings
            maxiter = maxiter,
            abstol = abstol,
            # Gap regularization
            gap_regularization = gap_regularization,
            gap_weight = gap_weight,
            gap_threshold = gap_threshold,
            # Sampling settings
            measure_first = measure_first,
            share_params = share_params,
            samples_per_run = samples_per_run,
            n_parallel_runs = n_parallel_runs,
            conv_step = conv_step
        )
        
        results[i] = result
        
        # Save result to JSON (note: optimize_circuit also saves, but with different filename)
        filename = joinpath(output_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")
        input_args = Dict{Symbol,Any}(
            # Model parameters
            :J => J,
            :g => g,
            :row => row,
            :p => p,
            :nqubits => nqubits,
            :n_params => n_params,
            # Optimization settings
            :maxiter => maxiter,
            :abstol => abstol,
            # Gap regularization
            :gap_regularization => gap_regularization,
            :gap_weight => gap_weight,
            :gap_threshold => gap_threshold,
            # Sampling settings
            :measure_first => String(measure_first),
            :share_params => share_params,
            :samples_per_run => samples_per_run,
            :n_parallel_runs => n_parallel_runs,
            :conv_step => conv_step,
            # Reproducibility
            :seed => sim_seed
        )
        save_result(filename, result, input_args)
        
        if verbose
            status = result.converged ? "converged" : "stopped at maxiter"
            println("[$i/$n] Completed g = $g")
            println("       Best energy: $(round(result.final_cost, digits=6)) ($status)")
            println("       Saved to: $(basename(filename))")
        end
    end
    
    if verbose
        println("\n" * "=" ^ 60)
        println("All simulations completed!")
        println("=" ^ 60)
        println("\nSummary:")
        for (i, g) in enumerate(g_values)
            status = results[i].converged ? "✓" : "○"
            println("  $status g = $g: E = $(round(results[i].final_cost, digits=6))")
        end
    end
    
    return results
end

simulation(
    J = 1.0,
    g_values = [4.0],
    row = 2,
    p = 4,
    nqubits = 3,
    # Optimization settings
    maxiter = 500,
    abstol = 0.01,
    # Gap regularization
    gap_regularization = false,
    gap_weight = 10.0,
    gap_threshold = 0.1,
    # Sampling settings
    measure_first = :Z,
    share_params = true,
    samples_per_run = 1000,
    n_parallel_runs = 11,
    conv_step = 100,
    # Other
    seed = 123,
    verbose = true,
    output_dir = joinpath(@__DIR__, "results")
)