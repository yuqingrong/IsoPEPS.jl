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
    gates::Vector{Matrix{ComplexF64}}
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
    initialize_tfim_params(p, nqubits, g; mode=:meanfield)

Generate initial parameters for TFIM optimization that avoid trivial product states.

# Arguments
- `p`: Circuit depth
- `nqubits`: Number of qubits per gate
- `g`: Transverse field strength
- `mode`: Initialization mode
  - `:meanfield`: Parameters tuned to approximate mean-field state
  - `:entangled`: Parameters that create significant entanglement
  - `:random`: Random initialization (default CMA-ES behavior)

# Returns
- Parameter vector of length `2*nqubits*p`
"""
function initialize_tfim_params(p::Int, nqubits::Int, g::Float64; mode::Symbol=:meanfield)
    n_params = 2 * nqubits * p
    
    if mode == :meanfield
        # Mean-field angle: θ such that ⟨Z⟩ = -J/(g) approximately
        # For g >> J: nearly all X, for g << J: nearly all Z
        θ_mf = atan(1.0 / g)  # Approximate mean-field angle
        
        params = zeros(n_params)
        for layer in 1:p
            for q in 1:nqubits
                idx = 2*nqubits*(layer-1) + 2*(q-1) + 1
                # Rx rotation to tilt from |0⟩ toward mean-field direction
                params[idx] = θ_mf + 0.1 * randn()  # Rx angle
                params[idx+1] = 0.1 * randn()        # Rz angle (small)
            end
        end
        return params
        
    elseif mode == :entangled
        # GHZ/Bell state initialization:
        # CNOT structure: cnot(control=i+1, target=i), so last qubit is always control
        # 
        # Layer 1 creates GHZ:
        #   - Last qubit (control): Rx(π/2) creates superposition
        #   - Other qubits (targets): stay in |0⟩
        #   - CNOT cascade creates: (|000⟩ - i|111⟩)/√2
        # 
        # Layer 2+ preserves GHZ:
        #   - All qubits: identity (Rx(0), Rz(0))
        #   - CNOTs on GHZ state: |000⟩↔|000⟩, |111⟩↔|111⟩ (self-inverse)
        params = zeros(n_params)
        for layer in 1:p
            for q in 1:nqubits
                idx = 2*nqubits*(layer-1) + 2*(q-1) + 1
                if layer == 1 && q == nqubits
                    # Layer 1, last qubit (control): create superposition
                    params[idx] = π/2      # Rx(π/2)
                    params[idx+1] = 0.0    # Rz(0)
                else
                    # All other cases: identity to preserve GHZ
                    params[idx] = 0.0      # Rx(0)
                    params[idx+1] = 0.0    # Rz(0)
                end
            end
        end
        return params
        
    else  # :random
        return 2π * rand(n_params)
    end
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
- `model`: `"tfim"` or `"heisenberg_j1j2"`
- `measure_first::Symbol=:Z`: Which observable to measure first (:X or :Z)
- `share_params::Bool=true`: Whether to share parameters across layers
- `conv_step::Int=100`: Burn-in steps before sampling
- `samples::Int=10000`: Samples per sampling run
- `n_runs::Int=44`: Number of parallel sampling runs (threaded)
- `maxiter::Int=5000`: Maximum CMA-ES generations
- `abstol::Float64=0.01`: Function tolerance (ftol) for CMA-ES convergence
- `sigma0::Float64=1.0`: Initial step size (σ₀) for CMA-ES
- `popsize::Union{Int,Nothing}=nothing`: Population size (nothing = CMA-ES default: 4+⌊3ln(n)⌋)
- `zz_weight::Float64=0.0`: Weight for ZZ correlation regularization
- `target_energy::Float64=-Inf`: Stop early if energy drops below this value
- Model-specific parameters:
  - TFIM: `J` (coupling), `g` (transverse field)
  - Heisenberg J1-J2: `J1` (NN coupling), `J2` (NNN coupling)

# Returns
- `CircuitOptimizationResult`: Optimization results with energy history and final state
"""
function optimize_circuit(params, p::Int, row::Int, nqubits::Int;
    model::String="tfim",
    measure_first=:Z, share_params=true, conv_step=100,
    samples=10000, maxiter=5000, abstol=0.01, n_runs=44,
    sigma0::Float64=1.0, popsize::Union{Int,Nothing}=nothing,
    zz_weight::Float64=0.0, target_energy::Float64=-Inf,
    model_kwargs...)

    kw = Dict{Symbol,Any}(model_kwargs)

    # Build model label for logging
    model_label = if model == "tfim"
        "TFIM J=$(get(kw, :J, 1.0)) g=$(get(kw, :g, 1.0))"
    elseif model == "heisenberg_j1j2"
        "Heisenberg J1=$(get(kw, :J1, 1.0)) J2=$(get(kw, :J2, 0.0))"
    else
        error("Unknown model: \"$model\". Supported: \"tfim\", \"heisenberg_j1j2\"")
    end

    # Store initial parameters
    initial_params = copy(params)

    # Flag for early stopping
    should_stop = Ref(false)

    energy_history = Float64[]
    params_history = Vector{Float64}[]
    Z_samples_history = Vector{Float64}[]
    X_samples_history = Vector{Float64}[]
    current_params = copy(params)

    # Track energies within each generation
    generation_energies = Float64[]
    generation_params = Vector{Float64}[]
    generation_Z_samples = Vector{Float64}[]
    generation_X_samples = Vector{Float64}[]
    generation_count = Ref(0)
    logged_threads = Ref(false)
    eval_count = Ref(0)

    # Objective function for Optimization.jl (takes x and optional params_opt)
    function objective(x, params_opt=nothing)
        eval_count[] += 1
        current_params .= x

        gates = build_unitary_gate(x, p, row, nqubits; share_params=share_params)

        # Run circuit with samples
        need_y = (model == "heisenberg_j1j2")
        Z_samples_all = Vector{Vector{Float64}}(undef, n_runs)
        X_samples_all = Vector{Vector{Float64}}(undef, n_runs)
        Y_samples_all = need_y ? Vector{Vector{Float64}}(undef, n_runs) : nothing

        # Log threading info once
        if !logged_threads[]
            @info "Using $(Threads.nthreads()) threads for parallel sampling"
            logged_threads[] = true
        end

        Threads.@threads for run_idx in 1:n_runs
        result_ch = sample_quantum_channel(gates, row, nqubits;
                                          conv_step=conv_step,
                                          samples=samples,
                                          measure_first=measure_first,
                                          measure_y=need_y)
        Z_samples = result_ch[2]
        X_samples = result_ch[3]
        # Z_samples includes burn-in period, X_samples does not (collected in second phase)
        # Align to row boundary so column structure is preserved for ZZ computation
        # Find first index > conv_step that starts a new column: (idx-1) % row == 0
        discard = conv_step + row - 1 - (conv_step - 1) % row  # round up to next row boundary
        start_idx = discard + 1
        Z_samples_all[run_idx] = Z_samples[start_idx:end]
        X_samples_all[run_idx] = X_samples[start_idx:end]
        if need_y
            Y_samples_all[run_idx] = result_ch[4][start_idx:end]
        end
        end

        # Combine all samples
        Z_samples_combined = reduce(vcat, Z_samples_all)
        X_samples_combined = reduce(vcat, X_samples_all)

        # Energy computation depends on model
        local energy
        if model == "tfim"
            J = Float64(get(kw, :J, 1.0))
            g = Float64(get(kw, :g, 1.0))
            energy = compute_energy(X_samples_combined, Z_samples_combined, g, J, row)
        elseif model == "heisenberg_j1j2"
            J1 = Float64(get(kw, :J1, 1.0))
            J2 = Float64(get(kw, :J2, 0.0))
            Y_samples_combined = reduce(vcat, Y_samples_all)
            energy = compute_heisenberg_energy(X_samples_combined, Z_samples_combined, Y_samples_combined, J1, J2, row)
        end

        # Cost = energy + penalty (penalty is for optimization only)
        cost = energy

        # Store in generation arrays (store true energy, not cost with penalty)
        push!(generation_energies, real(energy))
        push!(generation_params, copy(x))
        push!(generation_Z_samples, Z_samples_combined)
        push!(generation_X_samples, X_samples_combined)

        # Return cost (with penalty) for optimization
        return real(cost)
    end

    # Callback for Optimization.jl CMA-ES
    # Called after each function evaluation
    function optimization_callback(state, loss_val)
        # Check if we've completed a generation by tracking evaluation count
        # CMA-ES evaluates popsize individuals per generation
        actual_popsize = isnothing(popsize) ? 4 + floor(Int, 3*log(length(params))) : popsize

        if eval_count[] % actual_popsize == 0 && !isempty(generation_energies)
            # When a generation is complete, find the best from that generation
            min_idx = argmin(generation_energies)
            min_energy = generation_energies[min_idx]
            best_params = generation_params[min_idx]
            best_Z_samples = generation_Z_samples[min_idx]
            best_X_samples = generation_X_samples[min_idx]

            # Store the minimum energy and associated data
            push!(energy_history, min_energy)
            push!(params_history, best_params)
            push!(Z_samples_history, best_Z_samples)
            push!(X_samples_history, best_X_samples)

            generation_count[] += 1

            # Log every 10 generations
            if generation_count[] % 10 == 0
                @info "$model_label $(row)×∞ PEPS | Generation $(generation_count[]) | Min Energy: $(round(min_energy, digits=6))"
            end

            # Clear generation arrays for next generation
            empty!(generation_energies)
            empty!(generation_params)
            empty!(generation_Z_samples)
            empty!(generation_X_samples)
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
    if !isempty(generation_energies)
        min_idx = argmin(generation_energies)
        min_energy = generation_energies[min_idx]
        best_params = generation_params[min_idx]
        best_Z_samples = generation_Z_samples[min_idx]
        best_X_samples = generation_X_samples[min_idx]

        push!(energy_history, min_energy)
        push!(params_history, best_params)
        push!(Z_samples_history, best_Z_samples)
        push!(X_samples_history, best_X_samples)
    end

    # Use best parameters found across all generations
    if !isempty(energy_history)
        min_idx = argmin(energy_history)
        final_params = params_history[min_idx]
        final_cost = energy_history[min_idx]
        final_Z_samples = Z_samples_history[min_idx]
        final_X_samples = X_samples_history[min_idx]
        @info "Best energy at generation $min_idx: $final_cost"
    else
        final_params = opt_result.u
        final_cost = opt_result.objective
        final_Z_samples = Float64[]
        final_X_samples = Float64[]
    end

    final_gates = build_unitary_gate(final_params, p, row, nqubits; share_params=share_params)

    result = CircuitOptimizationResult(
        energy_history,
        final_gates,
        final_params,
        final_cost,
        final_Z_samples,
        final_X_samples,
        converged
    )

    # Save result with all input parameters
    input_args = Dict{Symbol, Any}(
        :model => model,
        :row => row, :p => p, :nqubits => nqubits,
        :initial_params => initial_params,
        :measure_first => String(measure_first),
        :share_params => share_params,
        :conv_step => conv_step,
        :samples => samples,
        :maxiter => maxiter,
        :abstol => abstol,
        :sigma0 => sigma0,
        :zz_weight => zz_weight,
        :target_energy => target_energy,
        :total_generations => generation_count[],
        :early_stopped => should_stop[],
    )
    merge!(input_args, Dict{Symbol,Any}(model_kwargs))

    return result
end

# Backward-compatible TFIM convenience method
function optimize_circuit(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int;
    measure_first=:Z, share_params=true, conv_step=100,
    samples=10000, maxiter=5000, abstol=0.01, n_runs=44,
    sigma0::Float64=1.0, popsize::Union{Int,Nothing}=nothing,
    zz_weight::Float64=0.0, target_energy::Float64=-Inf)
    return optimize_circuit(params, p, row, nqubits;
        model="tfim", J=J, g=g,
        measure_first=measure_first, share_params=share_params,
        conv_step=conv_step, samples=samples, maxiter=maxiter,
        abstol=abstol, n_runs=n_runs, sigma0=sigma0, popsize=popsize,
        zz_weight=zz_weight, target_energy=target_energy)
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
                        model::String="tfim", maxiter=5000, abstol=1e-6,
                        model_kwargs...)
    kw = Dict{Symbol,Any}(model_kwargs)

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

    # Build model label for logging
    model_label = if model == "tfim"
        "TFIM J=$(get(kw, :J, 1.0)) g=$(get(kw, :g, 1.0))"
    elseif model == "heisenberg_j1j2"
        "Heisenberg J1=$(get(kw, :J1, 1.0)) J2=$(get(kw, :J2, 0.0))"
    else
        error("Unknown model: \"$model\". Supported: \"tfim\", \"heisenberg_j1j2\"")
    end

    function objective(x, _)
        iter_count[] += 1

        if iter_count[] > maxiter
            @warn "Reached maximum iterations ($maxiter). Stopping..."
            error("Maximum iterations reached")
        end

        current_params .= x
        push!(params_history, copy(x))

        gates = build_unitary_gate(x, p, row, nqubits)
        _, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)

        local energy, X_cost, ZZ_vert, ZZ_horiz

        if model == "tfim"
            J = Float64(get(kw, :J, 1.0))
            g = Float64(get(kw, :g, 1.0))
            X_cost = real(compute_X_expectation(nothing, gates, row, virtual_qubits))
            ZZ_vert, ZZ_horiz = compute_ZZ_expectation(nothing, gates, row, virtual_qubits)
            ZZ_vert = real(ZZ_vert)
            ZZ_horiz = real(ZZ_horiz)
            energy = -g * X_cost - J * (row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
        elseif model == "heisenberg_j1j2"
            J1 = Float64(get(kw, :J1, 1.0))
            J2 = Float64(get(kw, :J2, 0.0))
            energy = compute_exact_heisenberg_energy(gates, row, virtual_qubits, J1, J2)
            X_cost = 0.0
            ZZ_vert = 0.0
            ZZ_horiz = 0.0
        end

        push!(X_history, X_cost)
        push!(ZZ_vert_history, ZZ_vert)
        push!(ZZ_horiz_history, ZZ_horiz)
        push!(gap_history, gap)
        push!(eigenvalues_history, eigenvalues)
        push!(energy_history, real(energy))

        @info "$model_label $(row)×∞ PEPS (exact) | Iter $(length(energy_history)) | Energy: $(round(energy, digits=6)) | Gap: $(round(gap, digits=4))"

        return real(energy)
    end

    @info "Optimizing $(length(params)) parameters with CMA-ES (exact contraction, model=$model)"

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
    final_gates = build_unitary_gate(final_params, p, row, nqubits)

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
        :model => model,
        :row => row, :p => p, :nqubits => nqubits,
        :initial_params => initial_params,
        :maxiter => maxiter,
        :abstol => abstol,
        :best_iteration => best_idx,
        :total_iterations => length(energy_history)
    )
    merge!(input_args, Dict{Symbol,Any}(model_kwargs))
    save_result("data/exact_$(model)_row=$(row).json", result, input_args)

    return result
end

# Backward-compatible TFIM convenience method
function optimize_exact(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int;
                        maxiter=5000, abstol=1e-6)
    return optimize_exact(params, p, row, nqubits; model="tfim", J=J, g=g,
                          maxiter=maxiter, abstol=abstol)
end

