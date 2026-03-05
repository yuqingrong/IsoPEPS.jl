"""
Minimal working example comparing two CMAES implementations:
1. Optimization.jl with CMAEvolutionStrategyOpt (current implementation)
2. CMAEvolutionStrategy.jl direct usage

Test on a simple 2D TFIM optimization problem.
"""

using IsoPEPS
using Optimization
using OptimizationCMAEvolutionStrategy
using CMAEvolutionStrategy
using Random
using Statistics

println("="^70)
println("CMAES Comparison: Optimization.jl vs CMAEvolutionStrategy.jl")
println("="^70)

# Test problem setup
J = 1.0
g = 2.0
row = 2
circuit_depth = 2  # Renamed from 'p' to avoid confusion with optimization parameter
nqubits = 3
share_params = true
n_params = 2 * nqubits * circuit_depth

# Sampling parameters (small for speed)
conv_step = 100
samples = 5000
n_runs = 4

println("\nProblem setup:")
println("  TFIM: J=$J, g=$g, row=$row")
println("  Circuit: p=$circuit_depth, nqubits=$nqubits")
println("  Parameters: $n_params")
println("  Sampling: $samples samples × $n_runs runs")

# Initialize parameters
Random.seed!(123)
params_init = rand(n_params)

# Objective function for Optimization.jl (takes x and optional p parameter)
function objective_opt(x, opt_p=nothing)
    gates = build_unitary_gate(x, circuit_depth, row, nqubits; share_params=share_params)

    Z_samples_all = Vector{Vector{Float64}}(undef, n_runs)
    X_samples_all = Vector{Vector{Float64}}(undef, n_runs)

    Threads.@threads for run_idx in 1:n_runs
        rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits;
                                          conv_step=conv_step,
                                          samples=samples,
                                          measure_first=:Z)
        discard = conv_step + row - 1 - (conv_step - 1) % row
        start_idx = discard + 1
        Z_samples_all[run_idx] = Z_samples[start_idx:end]
        X_samples_all[run_idx] = X_samples[start_idx:end]
    end

    Z_combined = reduce(vcat, Z_samples_all)
    X_combined = reduce(vcat, X_samples_all)

    energy = compute_energy(X_combined, Z_combined, g, J, row)
    return real(energy)
end

# Objective function for CMAEvolutionStrategy.jl (takes only x)
function objective_cma(x)
    gates = build_unitary_gate(x, circuit_depth, row, nqubits; share_params=share_params)

    Z_samples_all = Vector{Vector{Float64}}(undef, n_runs)
    X_samples_all = Vector{Vector{Float64}}(undef, n_runs)

    Threads.@threads for run_idx in 1:n_runs
        rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits;
                                          conv_step=conv_step,
                                          samples=samples,
                                          measure_first=:Z)
        discard = conv_step + row - 1 - (conv_step - 1) % row
        start_idx = discard + 1
        Z_samples_all[run_idx] = Z_samples[start_idx:end]
        X_samples_all[run_idx] = X_samples[start_idx:end]
    end

    Z_combined = reduce(vcat, Z_samples_all)
    X_combined = reduce(vcat, X_samples_all)

    energy = compute_energy(X_combined, Z_combined, g, J, row)
    return real(energy)
end

println("\n" * "="^70)
println("Method 1: Optimization.jl with CMAEvolutionStrategyOpt")
println("="^70)

# Method 1: Optimization.jl wrapper
opt_func = OptimizationFunction(objective_opt)
prob = OptimizationProblem(opt_func, copy(params_init), nothing;
    lb = zeros(n_params),
    ub = fill(2π, n_params)
)

energy_history_opt = Float64[]
function callback_opt(state, loss_val)
    push!(energy_history_opt, loss_val)
    if length(energy_history_opt) % 5 == 0
        println("  Gen $(length(energy_history_opt)): E = $(round(loss_val, digits=4))")
    end
    return false
end

println("\nRunning optimization (maxiter=20)...")
@time result_opt = solve(
    prob,
    CMAEvolutionStrategyOpt(),
    abstol = 1e-3,
    maxiters = 50,
    callback = callback_opt
)

println("\nResults (Optimization.jl):")
println("  Final energy: $(round(result_opt.objective, digits=6))")
println("  Converged: $(result_opt.retcode)")
println("  Generations: $(length(energy_history_opt))")

println("\n" * "="^70)
println("Method 2: CMAEvolutionStrategy.jl direct usage")
println("="^70)

# Method 2: Direct CMAEvolutionStrategy.jl
energy_history_cma = Float64[]
best_energy_cma = Ref(Inf)  # Use Ref to allow modification inside function
best_params_cma = copy(params_init)

# Wrapper for CMAEvolutionStrategy.jl (it expects fitness function)
function fitness(x)
    energy = objective_cma(x)
    if energy < best_energy_cma[]
        best_energy_cma[] = energy
        best_params_cma .= x
    end
    return energy
end

println("\nRunning optimization (maxiter=20)...")
@time result_cma = CMAEvolutionStrategy.minimize(
    fitness,
    copy(params_init),
    1.0;  # σ₀
    lower = zeros(n_params),
    upper = fill(2π, n_params),
    maxiter = 50,
    ftol = 1e-3,
    callback = (optimizer, params_matrix, fitness_vals, sigma_vec) -> begin
        # fitness_vals is a vector of all population fitnesses
        # Get the best (minimum) fitness
        best_fitness = minimum(fitness_vals)
        push!(energy_history_cma, best_fitness)
        iteration = length(energy_history_cma)
        if iteration % 5 == 0
            println("  Gen $iteration: E = $(round(best_fitness, digits=4))")
        end
        return false  # continue
    end
)

println("\nResults (CMAEvolutionStrategy.jl):")
println("  Final energy: $(round(best_energy_cma[], digits=6))")
println("  Generations: $(length(energy_history_cma))")

println("\n" * "="^70)
println("Comparison Summary")
println("="^70)

println("\nFinal energies:")
println("  Optimization.jl:          $(round(result_opt.objective, digits=6))")
println("  CMAEvolutionStrategy.jl:  $(round(best_energy_cma[], digits=6))")
println("  Difference:               $(round(abs(result_opt.objective - best_energy_cma[]), digits=6))")

println("\nConvergence:")
println("  Optimization.jl:          $(result_opt.retcode)")

println("\nGenerations:")
println("  Optimization.jl:          $(length(energy_history_opt))")
println("  CMAEvolutionStrategy.jl:  $(length(energy_history_cma))")

# Plot comparison
using CairoMakie

fig = Figure(size=(1000, 500))

ax1 = Axis(fig[1, 1],
           xlabel="Generation",
           ylabel="Energy",
           title="Optimization.jl (CMAEvolutionStrategyOpt)")
lines!(ax1, 1:length(energy_history_opt), energy_history_opt,
       color=:blue, linewidth=2)
scatter!(ax1, 1:length(energy_history_opt), energy_history_opt,
         color=:blue, markersize=8)

ax2 = Axis(fig[1, 2],
           xlabel="Generation",
           ylabel="Energy",
           title="CMAEvolutionStrategy.jl (direct)")
lines!(ax2, 1:length(energy_history_cma), energy_history_cma,
       color=:red, linewidth=2)
scatter!(ax2, 1:length(energy_history_cma), energy_history_cma,
         color=:red, markersize=8)

# Link y-axes for easier comparison
linkyaxes!(ax1, ax2)

save("project/results/cmaes_comparison.pdf", fig)
println("\nComparison plot saved to: project/results/cmaes_comparison.pdf")

println("\n" * "="^70)
println("Conclusion:")
println("="^70)
println("Both methods should give similar results.")
println("Differences may arise from:")
println("  - Different random seeds in sampling")
println("  - Different internal CMAES implementations")
println("  - Different default hyperparameters")
println("="^70)
