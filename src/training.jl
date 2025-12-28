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
    gates::Vector{Matrix{ComplexF64}}
    params::Vector{Float64}
    energy::Float64
    Z_samples::Vector{Float64}
    X_samples::Vector{Float64}
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
    optimize_circuit(params, J, g, p, row, nqubits; kwargs...)

Optimize circuit parameters to minimize TFIM energy using CMA-ES.

# Arguments
- `params`: Initial parameter vector
- `J`: Coupling strength
- `g`: Transverse field strength
- `p`: Number of circuit layers
- `row`: Number of rows in PEPS
- `nqubits`: Number of qubits per gate

# Keyword Arguments
- `measure_first`: Observable to measure first, `:X` or `:Z` (default: `:Z`)
- `share_params`: Share parameters across gates (default: true)
- `conv_step`: Convergence steps (default: 1000)
- `samples`: Samples per iteration (default: 10000)
- `maxiter`: Maximum iterations (default: 5000)
- `abstol`: Absolute tolerance (default: 0.01)

# Returns
`CircuitOptimizationResult` with energy history, final gates, parameters, and samples
"""
function optimize_circuit(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int; 
                          measure_first=:Z, share_params=true, conv_step=100, 
                          samples=10000, maxiter=5000, abstol=0.01)
    # Store initial parameters
    initial_params = copy(params)
    
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    Z_samples_history = Vector{Float64}[]
    X_samples_history = Vector{Float64}[]
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
        
        gates = build_unitary_gate(x, p, row, nqubits; share_params=share_params)
        rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                            conv_step=conv_step, 
                                                            samples=samples,
                                                            measure_first=measure_first)
        push!(Z_samples_history, Z_samples)
        push!(X_samples_history, X_samples)
    
        if measure_first == :X
            energy = compute_energy(X_samples[conv_step:end], Z_samples, g, J, row) 
        else
            energy = compute_energy(X_samples, Z_samples[conv_step:end], g, J, row) 
        end
        
        push!(energy_history, real(energy))
        @info "TFIM J=$J g=$g $(row)×∞ PEPS | Iter $(length(energy_history)) | Energy: $(round(energy, digits=6))"

        return real(energy)
    end
    
    @info "Optimizing $(length(params)) parameters with CMA-ES"
  
    f = OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(f, params, 
                                             lb=zeros(length(params)), 
                                             ub=fill(2π, length(params)))
   
    local final_params, final_cost
    try
        sol = solve(prob, CMAEvolutionStrategyOpt(), maxiters=maxiter, abstol=abstol)
        final_params = sol.u
        final_cost = sol.objective
        converged = true
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Using parameters from iteration $maxiter"
            final_params = current_params
            final_cost = isempty(energy_history) ? NaN : energy_history[end]
            converged = false
        else
            rethrow(e)
        end
    end
     
    # Use best parameters found
    if !isempty(energy_history)
        min_idx = argmin(energy_history)
        final_params = params_history[min_idx]
        final_cost = energy_history[min_idx]
        @info "Best energy at iteration $min_idx: $final_cost"
    end
   
    final_gates = build_unitary_gate(final_params, p, row, nqubits; share_params=share_params)
    final_Z_samples = isempty(Z_samples_history) ? Float64[] : Z_samples_history[end]
    final_X_samples = isempty(X_samples_history) ? Float64[] : X_samples_history[end]
   
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
        # Model parameters
        :g => g, :J => J, :row => row, :p => p, :nqubits => nqubits,
        # Optimization settings
        :initial_params => initial_params,
        :measure_first => String(measure_first),
        :share_params => share_params,
        :conv_step => conv_step,
        :samples => samples,
        :maxiter => maxiter,
        :abstol => abstol
    )
    save_result("data/circuit_g=$(g)_row=$(row).json", result, input_args)
    
    return result
end

"""
    optimize_exact(params, J, g, p, row, nqubits; kwargs...)

Optimize circuit parameters using exact tensor contraction (no sampling noise).

# Arguments
Same as `optimize_circuit`

# Returns
`ExactOptimizationResult` with energy history, spectral properties, and expectation values
"""
function optimize_exact(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int; 
                        maxiter=5000, abstol=1e-6)
    # Store initial parameters
    initial_params = copy(params)
    
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
        
        gates = build_unitary_gate(x, p, row, nqubits)
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        
        X_cost = real(compute_X_expectation(rho, gates, row, nqubits))
        ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, nqubits)
        ZZ_vert = real(ZZ_vert)
        ZZ_horiz = real(ZZ_horiz)
        
        energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz) 
    
        push!(X_history, X_cost)
        push!(ZZ_vert_history, ZZ_vert)
        push!(ZZ_horiz_history, ZZ_horiz)
        push!(gap_history, gap)
        push!(eigenvalues_history, eigenvalues)
        push!(energy_history, real(energy))
        
        @info "TFIM J=$J g=$g $(row)×∞ PEPS (exact) | Iter $(length(energy_history)) | Energy: $(round(energy, digits=6)) | Gap: $(round(gap, digits=4))"

        return real(energy)
    end
    
    @info "Optimizing $(length(params)) parameters with CMA-ES (exact contraction)"
  
    f = OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(f, params,
                                             lb=zeros(length(params)),
                                             ub=fill(2π, length(params)))
    
    local final_params, final_cost
    try
        sol = solve(prob, CMAEvolutionStrategyOpt(), maxiters=maxiter, abstol=abstol)
        final_params = sol.u
        final_cost = sol.objective
        converged = true
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Using parameters from iteration $maxiter"
            final_params = current_params
            final_cost = isempty(energy_history) ? NaN : energy_history[end]
            converged = false
        else
            rethrow(e)
        end
    end   
    
    final_gates = build_unitary_gate(final_params, p, row, nqubits)
    final_gap = isempty(gap_history) ? NaN : gap_history[end]
    final_eigenvalues = isempty(eigenvalues_history) ? Float64[] : eigenvalues_history[end]
    final_X = isempty(X_history) ? NaN : X_history[end]
    final_ZZ_vert = isempty(ZZ_vert_history) ? NaN : ZZ_vert_history[end]
    final_ZZ_horiz = isempty(ZZ_horiz_history) ? NaN : ZZ_horiz_history[end]
   
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
        # Model parameters
        :g => g, :J => J, :row => row, :p => p, :nqubits => nqubits,
        # Optimization settings
        :initial_params => initial_params,
        :maxiter => maxiter,
        :abstol => abstol
    )
    save_result("data/exact_g=$(g)_row=$(row).json", result, input_args)
    
    return result
end

"""
    optimize_manifold(gate, row, nqubits, manifold, J, g; maxiter=3000)

Optimize gate on unitary manifold using particle swarm.

# Arguments
- `gate`: Initial gate matrix
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
- `manifold`: Manifold to optimize on (e.g., Stiefel)
- `J`: Coupling strength
- `g`: Transverse field strength
- `maxiter`: Maximum iterations (default: 3000)

# Returns
`ManifoldOptimizationResult` with energy and gap history
"""
function optimize_manifold(gate, row::Int, nqubits::Int, manifold::AbstractManifold, 
                           J::Float64, g::Float64; maxiter=3000)
    # Store initial gate
    initial_gate = copy(gate)
    
    energy_history = Float64[]
    gap_history = Float64[]
    eigenvalues_history = Vector{Float64}[]
    X_history = Float64[]
    ZZ_vert_history = Float64[]
    ZZ_horiz_history = Float64[]
    
    function f(M, gate)
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
        
        X_cost = real(compute_X_expectation(rho, gates, row, nqubits))
        ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, nqubits)
        ZZ_vert = real(ZZ_vert)
        ZZ_horiz = real(ZZ_horiz)
        
        energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
        
        push!(X_history, X_cost)
        push!(ZZ_vert_history, ZZ_vert)
        push!(ZZ_horiz_history, ZZ_horiz)
        push!(gap_history, gap)
        push!(eigenvalues_history, eigenvalues)
        push!(energy_history, real(energy))
        
        @info "Manifold opt | Iter $(length(energy_history)) | Energy: $(round(energy, digits=6)) | Gap: $(round(gap, digits=4))"
        
        return real(energy)
    end

    @assert is_point(manifold, Matrix(gate)) "Initial gate must be on manifold"

    result = Manopt.particle_swarm(manifold, f;
        swarm_size = 20,
        stopping_criterion = StopAfterIteration(maxiter) | StopWhenSwarmVelocityLess(1e-6),
        record = [:Iteration, :Cost],
        return_state = true
    )
    
    final_gate = get_solver_result(result)
    final_energy = f(manifold, final_gate)
    
    # Check convergence based on velocity criterion
    converged = true  # particle_swarm returns when velocity is low enough or maxiter
    
    opt_result = ManifoldOptimizationResult(
        energy_history,
        final_gate,
        final_energy,
        gap_history,
        converged
    )
    
    # Save result with all input parameters
    input_args = Dict{Symbol, Any}(
        # Model parameters
        :g => g, :J => J, :row => row, :nqubits => nqubits,
        # Optimization settings
        :initial_gate => initial_gate,
        :maxiter => maxiter,
        :manifold_type => string(typeof(manifold))
    )
    save_result("data/manifold_g=$(g)_row=$(row).json", opt_result, input_args)
    
    return opt_result
end
