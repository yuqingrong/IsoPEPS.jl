"""
Result structures for optimization/training functions.
"""

"""
    CircuitOptimizationResult

Result from circuit-based optimization using sampling (optimize_circuit).

# Fields
- `energy_history::Vector{Float64}`: Energy at each iteration
- `gates::Vector{Matrix{ComplexF64}}`: Final optimized gates
- `params::Vector{Float64}`: Final optimized parameters
- `energy::Float64`: Final energy value
- `Z_samples::Vector{Float64}`: Z measurement samples from final iteration
- `X_samples::Vector{Float64}`: X measurement samples from final iteration
- `converged::Bool`: Whether optimization converged
"""
struct CircuitOptimizationResult
    energy_history::Vector{Float64}
    final_gates::Any
    final_params::Vector{Float64}
    final_cost::Float64
    final_Z_samples::Vector{Float64}
    final_X_samples::Vector{Float64}
    final_Y_samples::Vector{Float64}
    converged::Bool
end

"""
    ExactOptimizationResult

Result from exact tensor contraction optimization (optimize_exact).

# Fields
- `energy_history::Vector{Float64}`: Energy at each iteration
- `gates::Vector{Matrix{ComplexF64}}`: Final optimized gates
- `params::Vector{Float64}`: Final optimized parameters
- `energy::Float64`: Final energy value
- `gap::Float64`: Spectral gap
- `eigenvalues::Vector{Float64}`: Transfer matrix eigenvalues
- `X_expectation::Float64`: Final ⟨X⟩ value
- `ZZ_vertical::Float64`: Final ⟨ZZ⟩ vertical
- `ZZ_horizontal::Float64`: Final ⟨ZZ⟩ horizontal
- `converged::Bool`: Whether optimization converged
"""
struct ExactOptimizationResult
    energy_history::Vector{Float64}
    gates::Any
    params::Vector{Float64}
    energy::Float64
    gap::Float64
    eigenvalues::Vector{Float64}
    X_expectation::Float64
    ZZ_vertical::Float64
    ZZ_horizontal::Float64
    converged::Bool
end

"""
    ManifoldOptimizationResult

Result from manifold optimization using particle swarm (optimize_manifold).

# Fields
- `energy_history::Vector{Float64}`: Energy at each iteration
- `gate::Matrix{ComplexF64}`: Final optimized gate
- `energy::Float64`: Final energy value
- `gap_history::Vector{Float64}`: Spectral gap at each iteration
- `converged::Bool`: Whether optimization converged
"""
struct ManifoldOptimizationResult
    energy_history::Vector{Float64}
    gate::Matrix{ComplexF64}
    energy::Float64
    gap_history::Vector{Float64}
    converged::Bool
end

"""
    _construct_model(model_str, kwargs) → AbstractModel

Construct a model type from a string name and keyword arguments.
"""
function _construct_model(model_str::String, kw::Dict{Symbol,Any})
    if model_str == "tfim"
        TFIM(J=Float64(get(kw, :J, 1.0)), g=Float64(get(kw, :g, 1.0)))
    elseif model_str == "heisenberg_j1j2"
        HeisenbergJ1J2(J1=Float64(get(kw, :J1, 1.0)), J2=Float64(get(kw, :J2, 0.0)))
    else
        error("Unknown model: \"$model_str\". Supported: \"tfim\", \"heisenberg_j1j2\"")
    end
end

"""
    _flush_generation!(gen_energies, gen_params, gen_Z, gen_X, gen_Y,
                       energy_hist, params_hist, Z_hist, X_hist, Y_hist)

Select the best individual from the current generation, append to history, and clear generation buffers.
"""
function _flush_generation!(generation_energies, generation_params,
                            generation_Z_samples, generation_X_samples, generation_Y_samples,
                            energy_history, params_history,
                            Z_samples_history, X_samples_history, Y_samples_history)
    isempty(generation_energies) && return
    idx = argmin(generation_energies)
    push!(energy_history, generation_energies[idx])
    push!(params_history, generation_params[idx])
    push!(Z_samples_history, generation_Z_samples[idx])
    push!(X_samples_history, generation_X_samples[idx])
    push!(Y_samples_history, generation_Y_samples[idx])
    empty!(generation_energies); empty!(generation_params)
    empty!(generation_Z_samples); empty!(generation_X_samples); empty!(generation_Y_samples)
end

"""
    optimize_circuit(params, p, row, nqubits; model="tfim", model_kwargs...)

Optimize a quantum circuit using sampling-based CMA-ES.

# Arguments
- `params::Vector{Float64}`: Initial parameter vector
- `p::Int`: Circuit depth
- `row::Int`: Number of rows in PEPS
- `nqubits::Int`: Number of qubits

# Keyword Arguments
- `model`: `"tfim"` or `"heisenberg_j1j2"` (string) or an `AbstractModel` instance
- `share_params::Bool=true`: Whether to share parameters across layers
- `conv_step::Int=100`: Thermalization steps before sampling
- `samples::Int=10000`: Samples per sampling run
- `n_runs::Int=44`: Number of parallel sampling runs (threaded)
- `maxiter::Int=5000`: Maximum CMA-ES generations
- `abstol::Float64=0.01`: Function tolerance (ftol) for CMA-ES convergence
- `unit_cell::Symbol=:single`: Unit cell type (`:single` or `:two_by_two`)
- Model-specific parameters:
  - TFIM: `J` (coupling), `g` (transverse field)
  - Heisenberg J1-J2: `J1` (NN coupling), `J2` (NNN coupling)

# Returns
- `CircuitOptimizationResult`: Optimization results with energy history and final state
"""
function optimize_circuit(params, p::Int, row::Int, nqubits::Int;
    model::Union{String,AbstractModel}="tfim",
    share_params=true, conv_step=100,
    samples=10000, maxiter=5000, abstol=0.01, n_runs=44,
    unit_cell::Symbol=:single,
    model_kwargs...)

    kw = Dict{Symbol,Any}(model_kwargs)

    # Construct model type if string was passed
    m = model isa AbstractModel ? model : _construct_model(model, kw)
    mlabel = model_label(m)
    model_str = model_name(m)

    # Store initial parameters
    initial_params = copy(params)

    energy_history = Float64[]
    params_history = Vector{Float64}[]
    Z_samples_history = Vector{Float64}[]
    X_samples_history = Vector{Float64}[]
    Y_samples_history = Vector{Float64}[]

    # Track energies within each generation
    generation_energies = Float64[]
    generation_params = Vector{Float64}[]
    generation_Z_samples = Vector{Float64}[]
    generation_X_samples = Vector{Float64}[]
    generation_Y_samples = Vector{Float64}[]
    generation_count = Ref(0)
    logged_threads = Ref(false)
    eval_count = Ref(0)

    # Objective function for Optimization.jl (takes x and optional params_opt)
    function objective(x, params_opt=nothing)
        eval_count[] += 1

        # Build gates based on unit cell type
        local gates, gates_odd, gates_even
        if unit_cell == :two_by_two && model_str == "heisenberg_j1j2"
            gates_odd, gates_even = build_unitary_gate_2x2(x, p, row, nqubits)
            gates = gates_odd  # for final_gates compatibility
        else
            gates = build_unitary_gate(x, p, row, nqubits; share_params=share_params)
        end

        # Run circuit with samples
        need_y = needs_y_measurement(m)
        Z_samples_all = Vector{Vector{Float64}}(undef, n_runs)
        X_samples_all = Vector{Vector{Float64}}(undef, n_runs)
        Y_samples_all = need_y ? Vector{Vector{Float64}}(undef, n_runs) : nothing

        # Log threading info once
        if !logged_threads[]
            @info "Using $(Threads.nthreads()) threads for parallel sampling"
            logged_threads[] = true
        end

        Threads.@threads for run_idx in 1:n_runs
        if unit_cell == :two_by_two && model_str == "heisenberg_j1j2"
            result_ch = sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                              conv_step=conv_step,
                                              samples=samples,
                                              model=m)
        else
            result_ch = sample_quantum_channel(gates, row, nqubits;
                                              conv_step=conv_step,
                                              samples=samples,
                                              model=m)
        end
        # Each phase now has conv_step + samples raw measurements; discard same thermalization
        Z_samples_all[run_idx] = result_ch[2][conv_step+1:end]
        X_samples_all[run_idx] = result_ch[3][conv_step+1:end]
        if need_y
            Y_samples_all[run_idx] = result_ch[4][conv_step+1:end]
        end
        end

        # Compute energy per-run to avoid cross-chain boundary artifacts, then average
        energies_per_run = Vector{Float64}(undef, n_runs)
        for run_idx in 1:n_runs
            Y_run = need_y ? Y_samples_all[run_idx] : Float64[]
            energies_per_run[run_idx] = real(compute_energy_from_samples(
                m, X_samples_all[run_idx], Z_samples_all[run_idx], Y_run, row))
        end
        energy = mean(energies_per_run)

        # Combine for history storage (used by visualization for running-mean plots)
        Z_samples_combined = reduce(vcat, Z_samples_all)
        X_samples_combined = reduce(vcat, X_samples_all)
        Y_samples_combined = need_y ? reduce(vcat, Y_samples_all) : Float64[]

        # Store in generation arrays
        push!(generation_energies, real(energy))
        push!(generation_params, copy(x))
        push!(generation_Z_samples, Z_samples_combined)
        push!(generation_X_samples, X_samples_combined)
        push!(generation_Y_samples, Y_samples_combined)

        return real(energy)
    end

    # Callback for Optimization.jl CMA-ES
    # Called after each function evaluation
    function optimization_callback(state, loss_val)
        # Check if we've completed a generation by tracking evaluation count
        # CMA-ES evaluates popsize individuals per generation
        actual_popsize = 4 + floor(Int, 3*log(length(params)))

        if eval_count[] % actual_popsize == 0 && !isempty(generation_energies)
            generation_count[] += 1

            # Log every 10 generations
            if generation_count[] % 10 == 0
                min_energy = minimum(generation_energies)
                @info "$mlabel $(row)×∞ PEPS | Generation $(generation_count[]) | Min Energy: $(round(min_energy, digits=6))"
            end

            _flush_generation!(generation_energies, generation_params,
                               generation_Z_samples, generation_X_samples, generation_Y_samples,
                               energy_history, params_history,
                               Z_samples_history, X_samples_history, Y_samples_history)
        end

        return false  # Continue optimization
    end

    # Create Optimization.jl problem with CMA-ES
    opt_func = OptimizationFunction(objective)
    prob = OptimizationProblem(opt_func, params, nothing;
        lb = zeros(length(params)),
        ub = fill(2π, length(params))
    )

    # Solve using CMA-ES from OptimizationCMAEvolutionStrategy
    opt_result = solve(
        prob,
        CMAEvolutionStrategyOpt(),
        abstol = abstol,
        maxiters = maxiter,
        callback = optimization_callback
    )

    converged = (opt_result.retcode == :Success || opt_result.retcode == :MaxIters ||
                 string(opt_result.retcode) == "Success" || string(opt_result.retcode) == "MaxIters")

    # Process remaining generation if any
    _flush_generation!(generation_energies, generation_params,
                       generation_Z_samples, generation_X_samples, generation_Y_samples,
                       energy_history, params_history,
                       Z_samples_history, X_samples_history, Y_samples_history)

    # Use best parameters found across all generations
    if !isempty(energy_history)
        min_idx = argmin(energy_history)
        final_params = params_history[min_idx]
        final_cost = energy_history[min_idx]
        final_Z_samples = Z_samples_history[min_idx]
        final_X_samples = X_samples_history[min_idx]
        final_Y_samples = Y_samples_history[min_idx]
        @info "Best energy at generation $min_idx: $final_cost"
    else
        final_params = opt_result.u
        final_cost = opt_result.objective
        final_Z_samples = Float64[]
        final_X_samples = Float64[]
        final_Y_samples = Float64[]
    end

    if unit_cell == :two_by_two && model_str == "heisenberg_j1j2"
        final_gates = build_unitary_gate_2x2(final_params, p, row, nqubits)
    else
        final_gates = build_unitary_gate(final_params, p, row, nqubits; share_params=share_params)
    end

    result = CircuitOptimizationResult(
        energy_history,
        final_gates,
        final_params,
        final_cost,
        final_Z_samples,
        final_X_samples,
        final_Y_samples,
        converged
    )

    # Save result with all input parameters
    input_args = Dict{Symbol, Any}(
        :model => model_str,
        :row => row, :p => p, :nqubits => nqubits,
        :initial_params => initial_params,
        :share_params => share_params,
        :conv_step => conv_step,
        :samples => samples,
        :maxiter => maxiter,
        :abstol => abstol,
        :total_generations => generation_count[],
    )
    merge!(input_args, Dict{Symbol,Any}(model_kwargs))

    return result
end

"""
    optimize_exact(params, p, row, nqubits; model="tfim", maxiter=5000, abstol=1e-6, model_kwargs...)

Optimize circuit parameters using exact tensor contraction (no sampling noise).

# Arguments
- `params`: Initial parameter vector
- `p`: Circuit depth
- `row`: Number of rows in PEPS
- `nqubits`: Number of qubits per gate

# Keyword Arguments
- `model`: `"tfim"` or `"heisenberg_j1j2"`
- `maxiter`: Maximum CMA-ES generations
- `abstol`: Convergence tolerance
- Model-specific parameters:
  - TFIM: `J` (coupling), `g` (transverse field)
  - Heisenberg J1-J2: `J1` (NN coupling), `J2` (NNN coupling)

# Returns
`ExactOptimizationResult` with energy history, spectral properties, and expectation values
"""
function optimize_exact(params, p::Int, row::Int, nqubits::Int;
                        model::Union{String,AbstractModel}="tfim", maxiter=5000, abstol=1e-6,
                        unit_cell::Symbol=:single,
                        model_kwargs...)
    kw = Dict{Symbol,Any}(model_kwargs)

    # Construct model type if string was passed
    m = model isa AbstractModel ? model : _construct_model(model isa String ? model : string(model), kw)
    mlabel = model_label(m)
    model_str = model_name(m)

    # Store initial parameters
    initial_params = copy(params)

    # Compute virtual_qubits for expectation value functions
    virtual_qubits = (nqubits - 1) ÷ 2

    energy_history = Float64[]
    params_history = Vector{Float64}[]
    X_history = Float64[]
    ZZ_vert_history = Float64[]
    ZZ_horiz_history = Float64[]
    gap_history = Float64[]
    eigenvalues_history = Vector{Float64}[]
    current_params = copy(params)
    iter_count = Ref(0)
    converged = false

    function objective(x, _)
        iter_count[] += 1

        if iter_count[] > maxiter
            @warn "Reached maximum iterations ($maxiter). Stopping..."
            error("Maximum iterations reached")
        end

        current_params .= x
        push!(params_history, copy(x))

        local energy, X_cost, ZZ_vert, ZZ_horiz, gap, eigenvalues

        if unit_cell == :two_by_two && model_str == "heisenberg_j1j2"
            gates_odd, gates_even = build_unitary_gate_2x2(x, p, row, nqubits)
            _, gap, eigenvalues, _ = compute_transfer_spectrum_2x2(gates_odd, gates_even, row, nqubits)
            energy, X_cost, ZZ_vert, ZZ_horiz = compute_exact_energy_from_gates(
                m, gates_odd, row, virtual_qubits; unit_cell=unit_cell, gates_even=gates_even)
        else
            gates = build_unitary_gate(x, p, row, nqubits)
            _, gap, eigenvalues, _ = compute_transfer_spectrum(gates, row, nqubits)
            energy, X_cost, ZZ_vert, ZZ_horiz = compute_exact_energy_from_gates(
                m, gates, row, virtual_qubits; unit_cell=unit_cell)
        end

        push!(X_history, X_cost)
        push!(ZZ_vert_history, ZZ_vert)
        push!(ZZ_horiz_history, ZZ_horiz)
        push!(gap_history, gap)
        push!(eigenvalues_history, eigenvalues)
        push!(energy_history, real(energy))

        @info "$mlabel $(row)×∞ PEPS (exact) | Iter $(length(energy_history)) | Energy: $(round(energy, digits=6)) | Gap: $(round(gap, digits=4))"

        return real(energy)
    end

    @info "Optimizing $(length(params)) parameters with CMA-ES (exact contraction, model=$model_str)"

    f = OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(f, params,
                                             lb=zeros(length(params)),
                                             ub=fill(2π, length(params)))

    converged = false
    try
        sol = solve(prob, CMAEvolutionStrategyOpt(), maxiters=maxiter, abstol=abstol)
        converged = true
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Optimization stopped at iteration $maxiter"
        else
            rethrow(e)
        end
    end

    # Find the best iteration (lowest energy) from history
    best_idx = argmin(energy_history)
    final_cost = energy_history[best_idx]
    final_params = params_history[best_idx]
    final_gap = gap_history[best_idx]
    final_eigenvalues = eigenvalues_history[best_idx]
    final_X = X_history[best_idx]
    final_ZZ_vert = ZZ_vert_history[best_idx]
    final_ZZ_horiz = ZZ_horiz_history[best_idx]
    if unit_cell == :two_by_two && model_str == "heisenberg_j1j2"
        final_gates = build_unitary_gate_2x2(final_params, p, row, nqubits)
    else
        final_gates = build_unitary_gate(final_params, p, row, nqubits)
    end

    @info "Best energy at iteration $best_idx: $final_cost"

    result = ExactOptimizationResult(
        energy_history,
        final_gates,
        final_params,
        final_cost,
        final_gap,
        final_eigenvalues,
        final_X,
        final_ZZ_vert,
        final_ZZ_horiz,
        converged
    )

    # Save result with all input parameters
    input_args = Dict{Symbol, Any}(
        :model => model_str,
        :row => row, :p => p, :nqubits => nqubits,
        :initial_params => initial_params,
        :maxiter => maxiter,
        :abstol => abstol,
        :best_iteration => best_idx,
        :total_iterations => length(energy_history)
    )
    merge!(input_args, Dict{Symbol,Any}(model_kwargs))
    save_result("data/exact_$(model_str)_row=$(row).json", result, input_args)

    return result
end

# Backward-compatible TFIM convenience method
function optimize_exact(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int;
                        maxiter=5000, abstol=1e-6)
    return optimize_exact(params, p, row, nqubits; model="tfim", J=J, g=g,
                          maxiter=maxiter, abstol=abstol)
end

