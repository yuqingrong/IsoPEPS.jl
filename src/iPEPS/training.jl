"""
Training and optimization routines for variational iPEPS.

Provides high-level optimization functions for finding ground states
of quantum many-body systems using variational iPEPS ansatz.
"""

"""
    train_energy_circ(params, J::Float64, g::Float64, p::Int, row::Int; maxiter=5000)

Train variational iPEPS circuit to minimize energy.

# Arguments
- `params`: Initial parameter vector
- `J`: Coupling strength
- `g`: Transverse field strength  
- `p`: Number of layers in the circuit
- `row`: Number of rows
- `maxiter`: Maximum number of iterations (default: 5000)

# Returns
- `X_history`: Energy history during optimization
- `final_A`: Final optimized gate matrix
- `final_params`: Final parameter values
- `final_cost`: Final energy
- `Z_list_list`: Z measurement history
- `X_list_list`: X measurement history
- `gap_list`: Spectral gap history
- `params_history`: Parameter evolution history

# Description
Optimizes the variational circuit using CMA-ES to minimize the ground state
energy of the transverse field Ising model. Tracks various observables and
convergence metrics during training.
"""
function train_energy_circ(params, J::Float64, g::Float64, p::Int, row::Int; maxiter=5000)
    # Initialize tracking variables
    X_history = Float64[]
    params_history = Vector{Float64}[]
    final_A = Matrix(I, 8, 8)
    Z_list_list = Vector{Float64}[]
    X_list_list = Vector{Float64}[]
    gap_list = Float64[]
    current_params = copy(params)
    iter_count = Ref(0)
    
    """
    Objective function for optimization.
    
    Builds gate from parameters, simulates quantum channel, computes energy.
    """
    function objective(x, _)
        iter_count[] += 1
        
        # Hard stop after maxiter function evaluations
        if iter_count[] > maxiter
            @warn "Reached maximum iterations ($maxiter). Stopping..."
            error("Maximum iterations reached")
        end
        
        current_params .= x
        push!(params_history, copy(x))
        
        # Build gate from parameters
        A_matrix = build_gate_from_params(x, p)
        gate = matblock(A_matrix)
       
        # Simulate channel and compute observables
        rho, Z_list, X_list = iterate_channel_PEPS(gate, row)
        _, gap = exact_left_eigen(gate, row)
        
        # Record observables
        push!(gap_list, gap)
        push!(Z_list_list, Z_list)
        push!(X_list_list, X_list)
        
        # Compute energy from measurements
        Z_configs = extract_Z_configurations(Z_list, row)
        energy = compute_energy_from_measurements(X_list, Z_configs, g, J, row)
        
        push!(X_history, real(energy))
        @info "Iter $(length(X_history)), energy: $energy, gap: $gap"

        return real(energy)
    end
    
    @info "Number of parameters is $(length(params))"
  
    # Setup optimization problem
    f = OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(
        f, params, 
        lb = zeros(length(params)), 
        ub = fill(2*π, length(params))
    )
    
    local final_params, final_cost
    try
        sol = solve(prob, CMAEvolutionStrategyOpt(), 
                   maxiters=5000, reltol=1e-6, abstol=1e-6)
        final_params = sol.u
        final_cost = sol.objective
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Using parameters from iteration $maxiter"
            final_params = current_params
            final_cost = isempty(X_history) ? NaN : X_history[end]
        else
            rethrow(e)
        end
    end   
    
    # Build final gate
    final_A = build_gate_from_params(final_params, p)
    
    @show final_cost
    return X_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history
end

"""
    train_nocompile(gate, row, M::AbstractManifold, J::Float64, g::Float64; maxiter=3000)

Train on a manifold without gate compilation.

# Arguments
- `gate`: Initial gate (as matrix on manifold)
- `row`: Number of rows
- `M`: Manifold to optimize on
- `J`: Coupling strength
- `g`: Transverse field strength
- `maxiter`: Maximum iterations (default: 3000)

# Returns
- `result`: Manopt optimization result
- `final_energy`: Final energy value
- `final_p`: Final point on manifold

# Description
Uses Manopt's Nelder-Mead on a manifold for optimization without
recompiling the gate at each step. Useful for constrained optimization
on unitary groups or other manifolds.
"""
function train_nocompile(gate, row, M::AbstractManifold, J::Float64, g::Float64; maxiter=3000)
    
    function f(M, gate)
        gate = matblock(gate)
        rho = iterate_channel_PEPS(gate, row)
        energy = -g*cost_X(rho, row, gate) - J*cost_ZZ(rho, row, gate)
        return real(energy)
    end

    @assert is_point(M, Matrix(gate))

    result = Manopt.NelderMead(
        M, 
        f,
        population=NelderMeadSimplex(M);
        stopping_criterion = StopAfterIteration(maxiter) | 
                           StopWhenPopulationConcentrated(1e-4, 1e-4),
        record = [:Iteration, :Cost],
        return_state = true
    )
    
    final_p = get_solver_result(result)
    final_energy = f(M, final_p)
   
    @show final_energy
    return result, final_energy, final_p
end

