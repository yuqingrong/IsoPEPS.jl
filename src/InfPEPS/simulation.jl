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


J=1.0; g=0.0; row=3
d=D=2
p=3
simulation(J, g, row, p; maxiter=5000, measure_first=:Z)

dynamics_observables(g; measure_first=:Z)

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

    