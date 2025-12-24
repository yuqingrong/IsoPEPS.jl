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
Named tuple with:
- `energy_history`: Energy values during optimization
- `gates`: Final optimized gates
- `params`: Final parameters
- `energy`: Final energy
- `gap`: Spectral gap (placeholder)
"""
function optimize_circuit(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int; 
                          measure_first=:Z, share_params=true, conv_step=1000, 
                          samples=10000, maxiter=5000, abstol=0.01)
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    Z_samples_history = Vector{Float64}[]
    X_samples_history = Vector{Float64}[]
    current_params = copy(params)
    iter_count = Ref(0)
    
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
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Using parameters from iteration $maxiter"
            final_params = current_params
            final_cost = isempty(energy_history) ? NaN : energy_history[end]
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
    
    # Save results
    save_results("data/training_g=$(g)_row=$(row).json";
        g=g, J=J, row=row, p=p, nqubits=nqubits,
        energy_history=energy_history,
        final_params=final_params,
        final_energy=final_cost,
        measure_first=String(measure_first)
    )
   
    return (
        energy_history = energy_history,
        gates = final_gates,
        params = final_params,
        energy = final_cost,
        gap = 1.0  # Placeholder
    )
end

"""
    optimize_exact(params, J, g, p, row, nqubits; kwargs...)

Optimize circuit parameters using exact tensor contraction (no sampling noise).

# Arguments
Same as `optimize_circuit`

# Returns
Named tuple with energy_history, gates, params, energy, gap, eigenvalues
"""
function optimize_exact(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int; 
                        maxiter=5000, abstol=1e-6)
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    X_history = Float64[]
    ZZ_vert_history = Float64[]
    ZZ_horiz_history = Float64[]
    gap_history = Float64[]
    eigenvalues_history = Vector{Float64}[]
    current_params = copy(params)
    iter_count = Ref(0)
    
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
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Using parameters from iteration $maxiter"
            final_params = current_params
            final_cost = isempty(energy_history) ? NaN : energy_history[end]
        else
            rethrow(e)
        end
    end   
    
    final_gates = build_unitary_gate(final_params, p, row, nqubits)
    
    # Save results
    save_results("data/training_exact_g=$(g)_row=$(row).json";
        g=g, J=J, row=row, p=p, nqubits=nqubits,
        energy_history=energy_history,
        X_history=X_history,
        ZZ_vert_history=ZZ_vert_history,
        ZZ_horiz_history=ZZ_horiz_history,
        gap_history=gap_history,
        final_params=final_params,
        final_energy=final_cost
    )
   
    return (
        energy_history = energy_history,
        gates = final_gates,
        params = final_params,
        energy = final_cost,
        gap = isempty(gap_history) ? NaN : gap_history[end],
        eigenvalues = isempty(eigenvalues_history) ? Float64[] : eigenvalues_history[end]
    )
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
Named tuple with result, energy, gate, and history
"""
function optimize_manifold(gate, row::Int, nqubits::Int, manifold::AbstractManifold, 
                           J::Float64, g::Float64; maxiter=3000)
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
    
    # Save results
    save_results("data/training_manifold_g=$(g)_row=$(row).json";
        g=g, J=J, row=row, nqubits=nqubits,
        energy_history=energy_history,
        gap_history=gap_history,
        final_energy=final_energy
    )
    
    return (
        result = result,
        energy = final_energy,
        gate = final_gate,
        energy_history = energy_history,
        gap_history = gap_history
    )
end
