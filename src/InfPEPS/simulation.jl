using IsoPEPS.InfPEPS
using Optimization, OptimizationCMAEvolutionStrategy
using Random
using Plots

function simulation(J::Float64, g::Float64, row::Int, p::Int; maxiter=5000, measure_first=:X)
    Random.seed!(12)
    params = rand(6*p)
    energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history = train_energy_circ(params, J, g, p, row; maxiter=maxiter,measure_first=measure_first)
    return energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history
end

"""
    parallel_simulation_threaded(J::Float64, g_values::Vector{Float64}, row::Int, p::Int; maxiter=5000, measure_first=:X)

Run simulations for multiple g values in parallel using multi-threading.
Returns a dictionary with g values as keys and simulation results as values.

"""
function parallel_simulation_threaded(J::Float64, g_values::Vector{Float64}, row::Int, p::Int; maxiter=5000, measure_first=:X)
    n = length(g_values)
    results = Vector{Any}(undef, n)
    
    println("Running $(n) simulations in parallel with $(Threads.nthreads()) threads...")
    
    Threads.@threads for i in 1:n
        g = g_values[i]
        println("Thread $(Threads.threadid()): Starting simulation for g = $(g)")
        
        Random.seed!(12)
        params = rand(6*p)
        
        energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history = 
            train_energy_circ(params, J, g, p, row; maxiter=maxiter, measure_first=measure_first)
        
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


J=1.0; g=1.0; g_values=[0.0, 0.25,0.5,0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]; row=3
d=D=2
p=3
#simulation(J, g, row, p; maxiter=5000, measure_first=:X)
parallel_simulation_threaded(J, g_values, row, p; maxiter=5000, measure_first=:Z)

dynamics_observables(g; measure_first=:X)

block_variance(g,[1,5000])

Plots.plot(
        vcat(gap_list),
        xlabel="Iteration",
        ylabel="gap",
        title="gap vs parameter iteration",
        ylims=(-0.0,10.0),
        legend=false,
        size=(900,300)  
    )

    