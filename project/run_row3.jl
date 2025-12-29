using IsoPEPS
using Optimization, OptimizationCMAEvolutionStrategy
using Random
using CairoMakie
using Yao, Manifolds
using LinearAlgebra, OMEinsum

# Load simulation function definition (but not execute the call at bottom)
function simulation(; J::Float64, g_values::Vector{Float64}, row::Int, p::Int, nqubits::Int, 
                    maxiter::Int, measure_first::Symbol, seed::Int, verbose::Bool,
                    output_dir::String, share_params::Bool)
    
    !isdir(output_dir) && mkpath(output_dir)
    n = length(g_values)
    results = Vector{CircuitOptimizationResult}(undef, n)
    
    verbose && println("Running $(n) simulations in parallel with $(Threads.nthreads()) threads...")
    verbose && println("Results will be saved to: $output_dir/")
    
    for i in 1:n
        g = g_values[i]
        verbose && println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        Random.seed!(seed)
        params = ones(2*nqubits*p)
        result = optimize_circuit(params, J, g, p, row, nqubits; 
                                  maxiter=maxiter, 
                                  measure_first=measure_first,
                                  share_params=share_params)
        
        results[i] = result
        
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

# Run simulation for row=3
simulation(
    J=1.0,
    g_values=[2.0],
    row=3,
    p=4,
    nqubits=5,
    maxiter=5000,
    measure_first=:Z,
    seed=12,
    verbose=true,
    output_dir=joinpath(@__DIR__, "results"),
    share_params=true
)
