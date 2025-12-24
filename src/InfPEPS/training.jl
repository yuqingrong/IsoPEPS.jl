
function train_energy_circ(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int; measure_first=:Z, share_params=true, conv_step=1000, samples=10000, maxiter=5000, abstol=0.01)
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    final_A = Matrix(I, 8, 8)
    Z_list_list = Vector{Float64}[]
    X_list_list = Vector{Float64}[]
    eigenvalues_list = Vector{Float64}[]
    current_params = copy(params)
    iter_count = Ref(0)
    function objective(x,_)
        iter_count[] += 1
        
        # Hard stop after maxiter function evaluations
        if iter_count[] > maxiter
            @warn "Reached maximum iterations ($maxiter). Stopping..."
            error("Maximum iterations reached")
        end
        current_params .= x
        push!(params_history, copy(x))
        A_matrix = build_gate_from_params(x, p, row, nqubits; share_params=share_params)
    
        rho, Z_list, X_list = iterate_channel_PEPS(A_matrix, row, nqubits; conv_step=conv_step, samples=samples,measure_first=measure_first)
        push!(Z_list_list, Z_list)
        push!(X_list_list, X_list)
    
        if measure_first == :X
            energy = energy_measure(X_list[conv_step:end], Z_list, g, J, row) 
        else
            energy = energy_measure(X_list, Z_list[conv_step:end], g, J, row) 
        end
        
        push!(energy_history, real(energy))
        @info "TFIM J=$J g=$g $row × ∞ PEPS, $measure_first first, Iter $(length(energy_history)), energy: $energy"

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
                  maxiters=maxiter, abstol=abstol)
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
     
    # Use parameters that achieved minimum energy
    if !isempty(energy_history)
        min_idx = argmin(energy_history)
        final_params = params_history[min_idx]
        final_cost = energy_history[min_idx]
        @info "Best energy found at iteration $min_idx: $final_cost"
    end
   
    final_A = build_gate_from_params(final_params, p, row, nqubits)
    #_,final_gap, final_eigenvalues = exact_left_eigen(final_A, row, nqubits)
    final_gap = 1.0
    final_eigenvalues = [1.0, 0.0]
    @show final_cost, final_gap, final_eigenvalues
    _save_training_data(g, p, row, nqubits, energy_history, params_history, Z_list_list, X_list_list, final_gap, final_eigenvalues, final_params, final_cost; measure_first=measure_first)
   
    return energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, final_gap, final_eigenvalues, params_history
end

function train_exact(params, J::Float64, g::Float64, p::Int, row::Int, nqubits::Int; measure_first=:X, niters=10000, maxiter=5000, abstol=1e-6)
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    final_A = Matrix(I, 8, 8)
    X_list = Float64[]
    ZZ_list1 = Float64[]
    ZZ_list2 = Float64[]
    gap_list = Float64[]
    eigenvalues_list = Vector{Float64}[]
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
        A_matrix = build_gate_from_params(x, p, row, nqubits)
        rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
        X_cost = real(cost_X(rho, A_matrix, row, nqubits))
        ZZ_exp1, ZZ_exp2 = cost_ZZ(rho, A_matrix, row, nqubits)
        ZZ_exp1 = real(ZZ_exp1)
        ZZ_exp2 = real(ZZ_exp2)
        energy = -g*X_cost - J*(row == 1 ? ZZ_exp2 : ZZ_exp1 + ZZ_exp2) 
    
        push!(X_list, X_cost)
        push!(ZZ_list1, ZZ_exp1)
        push!(ZZ_list2, ZZ_exp2)
        push!(gap_list, gap)
        push!(eigenvalues_list, eigenvalues)
        push!(energy_history, real(energy))
        @info "TFIM J=$J g=$g $row × ∞ PEPS, Iter $(length(energy_history)), energy: $energy, gap: $gap"

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
                   maxiters=maxiter, abstol=abstol)
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
    
    # Build final gate
    final_A = build_gate_from_params(final_params, p, row, nqubits)
    
    _save_training_data_exact(g, row, energy_history, X_list, ZZ_list1, ZZ_list2, gap_list, eigenvalues_list, final_params)
   
    return energy_history, final_A, final_params, final_cost, X_list, ZZ_list1, ZZ_list2, gap_list, eigenvalues_list
end

function train_energy_circ_gradient(params, J::Float64, g::Float64, p::Int, row::Int; 
                                   measure_first=:X, niters=10000, maxiter=5000, 
                                   abstol=1e-6, reltol=1e-8, epsilon=0.1)
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    final_A = Matrix(I, 8, 8)
    Z_list_list = Vector{Float64}[]
    X_list_list = Vector{Float64}[]
    gap_list = Float64[]
    current_params = copy(params)
    iter_count = Ref(0)
    
    # Cache for function evaluations to avoid recomputation
    cache = Dict{Vector{Float64}, Tuple{Float64, Vector{Float64}, Vector{Float64}, Float64}}()
    cache_lock = ReentrantLock()  # Thread-safe access to cache
    
    """
    Compute energy for given parameters.
    Returns energy and caches intermediate results.
    Thread-safe implementation using locks.
    """
    function compute_energy(x)
        
        # Compute energy (expensive operation, done outside lock)
        A_matrix = build_gate_from_params(x, p)
        gate = matblock(A_matrix)    
    
        rho, Z_list, X_list = iterate_channel_PEPS(gate, row, nqubits; measure_first=measure_first)
        _, gap = exact_left_eigen(gate, row)
        
        Z_list_trimmed = Z_list[Int(1+end-3/4*niters):end]
        Z_configs = extract_Z_configurations(Z_list_trimmed, row)
        energy = energy_measure(X_list, Z_configs, g, J, row; niters=niters)
        
        result = (real(energy), Z_list, X_list, gap)
        
        # Store in cache (thread-safe)
        lock(cache_lock) do
            cache[copy(x)] = result
        end
        
        return result
    end
    
    """
    Objective function for optimization.
    """
    function objective(x)
        iter_count[] += 1
        
        if iter_count[] > maxiter
            @warn "Reached maximum iterations ($maxiter). Stopping..."
            error("Maximum iterations reached")
        end
        
        current_params .= x
        push!(params_history, copy(x))
        
        energy, Z_list, X_list, gap = compute_energy(x)
        
        push!(gap_list, gap)
        push!(Z_list_list, Z_list)
        push!(X_list_list, X_list)
        push!(energy_history, energy)
        
        @info "TFIM J=$J g=$g $row × ∞ PEPS (Gradient), $measure_first first, Iter $(length(energy_history)), energy: $energy, gap: $gap"
        
        return energy
    end
    
    """
    Compute gradient using central finite differences with parallel execution.
    More accurate than forward differences, uses: f'(x) ≈ [f(x+ε) - f(x-ε)] / (2ε)
    
    Uses multi-threading to parallelize gradient computation across parameters.
    Set JULIA_NUM_THREADS environment variable to control thread count.
    """
    function gradient!(G, x)
        n = length(x)
        
        # Parallel computation of finite differences
        Threads.@threads for i in 1:n
            # Create local copies for thread safety
            x_plus = copy(x)
            x_minus = copy(x)
            
            # Perturb parameter i
            x_plus[i] = x[i] + epsilon
            x_minus[i] = x[i] - epsilon
            
            # Enforce bounds during perturbation
            x_plus[i] = clamp(x_plus[i], 0.0, 2π)
            x_minus[i] = clamp(x_minus[i], 0.0, 2π)
            
            # Compute finite difference
            energy_plus, _, _, _ = compute_energy(x_plus)
            energy_minus, _, _, _ = compute_energy(x_minus)
            
            G[i] = (energy_plus - energy_minus) / (2 * epsilon)
        end
        
        return G
    end
    
    @info "Number of parameters is $(length(params))"
    @info "Using finite difference gradient with ε=$epsilon"
    @info "Using $(Threads.nthreads()) thread(s) for parallel gradient computation"
  
    result = Optim.optimize(objective, gradient!, params, Optim.LBFGS()
                    )
    final_params = result.minimizer
    final_cost = result.minimum
    # Build final gate
    final_A = build_gate_from_params(final_params, p)
    
    _save_training_data(g, energy_history, params_history, Z_list_list, X_list_list, gap_list; measure_first=measure_first)
   
    return energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history
end


function train_hybrid(params, J::Float64, g::Float64, p::Int, row::Int; 
                      measure_first=:X, niters=10000, cma_maxiter=3000, nm_maxiter=3000, 
                      abstol=1e-6)
    
    @info "=" ^ 80
    @info "PHASE 1: CMA-ES Optimization ($cma_maxiter iterations)"
    @info "=" ^ 80
    
    # Phase 1: CMA-ES optimization
    energy_history = Float64[]
    params_history = Vector{Float64}[]
    Z_list_list = Vector{Float64}[]
    X_list_list = Vector{Float64}[]
    gap_list = Float64[]
    current_params = copy(params)
    iter_count = Ref(0)
    
    function objective_phase1(x, _)
        iter_count[] += 1
        
        # Hard stop after cma_maxiter function evaluations
        if iter_count[] > cma_maxiter
            @warn "Reached maximum CMA-ES iterations ($cma_maxiter). Moving to Phase 2..."
            error("Maximum iterations reached")
        end
        
        current_params .= x
        push!(params_history, copy(x))
        
        A_matrix = build_gate_from_params(x, p)
        gate = matblock(A_matrix)    
    
        rho, Z_list, X_list = iterate_channel_PEPS(gate, row, nqubits; measure_first=measure_first)
        _, gap = exact_left_eigen(gate, row)
        
        push!(gap_list, gap)
        push!(Z_list_list, Z_list)
        push!(X_list_list, X_list)
        
        Z_list_trimmed = Z_list[Int(1+end-3/4*niters):end]
        Z_configs = extract_Z_configurations(Z_list_trimmed, row)
        energy = energy_measure(X_list, Z_configs, g, J, row; niters=niters)
        
        push!(energy_history, real(energy))
        @info "Phase 1 (CMA-ES) - Iter $(length(energy_history))/$cma_maxiter: energy = $(real(energy)), gap = $gap"

        return real(energy)
    end
    
    @info "Number of parameters: $(length(params))"
    
    # Setup Phase 1: CMA-ES optimization problem
    f = OptimizationFunction(objective_phase1)
    prob = Optimization.OptimizationProblem(
        f, params, 
        lb = zeros(length(params)), 
        ub = fill(2*π, length(params))
    )
    
    # Run Phase 1: CMA-ES
    local phase1_params
    try
        sol = solve(prob, CMAEvolutionStrategyOpt(), 
                   maxiters=cma_maxiter, abstol=abstol)
        phase1_params = sol.u
        @info "Phase 1 completed normally"
    catch e
        if occursin("Maximum iterations reached", string(e))
            @info "Phase 1 completed after $cma_maxiter iterations"
            phase1_params = current_params
        else
            rethrow(e)
        end
    end
    
    phase1_energy = isempty(energy_history) ? NaN : energy_history[end]
    
    @info "=" ^ 80
    @info "PHASE 2: Nelder-Mead Optimization ($nm_maxiter iterations)"
    @info "Starting energy from Phase 1: $phase1_energy"
    @info "=" ^ 80
    
    # Phase 2: Nelder-Mead in parameter space
    phase2_iter = Ref(0)
    
    function objective_phase2(x)
        phase2_iter[] += 1
        
        push!(params_history, copy(x))
        
        A_matrix = build_gate_from_params(x, p)
        gate = matblock(A_matrix)    
    
        rho, Z_list, X_list = iterate_channel_PEPS(gate, row, nqubits; measure_first=measure_first)
        _, gap = exact_left_eigen(gate, row)
        
        Z_list_trimmed = Z_list[Int(1+end-3/4*niters):end]
        Z_configs = extract_Z_configurations(Z_list_trimmed, row)
        energy = energy_measure(X_list, Z_configs, g, J, row; niters=niters)
        
        push!(Z_list_list, Z_list)
        push!(X_list_list, X_list)
        push!(gap_list, gap)
        push!(energy_history, real(energy))
        
        @info "Phase 2 - Iter $(phase2_iter[])/$nm_maxiter: energy = $(real(energy)), gap = $gap"
        
        return real(energy)
    end
    
    # Run Phase 2: Nelder-Mead optimization
    nm_result = Optim.optimize(objective_phase2, phase1_params, Optim.NelderMead(),
                               Optim.Options(iterations=nm_maxiter, 
                                           f_abstol=abstol,
                                           show_trace=false))
    
    final_params = nm_result.minimizer
    final_cost = nm_result.minimum
    
    # Build final gate
    final_A = build_gate_from_params(final_params, p)
    
    @info "=" ^ 80
    @info "HYBRID TRAINING COMPLETE"
    @info "Phase 1 final energy: $phase1_energy"
    @info "Phase 2 final energy: $final_cost"
    @info "Total improvement: $(phase1_energy - final_cost)"
    @info "=" ^ 80
    
    # Save training data
    _save_training_data(g, energy_history, params_history, Z_list_list, X_list_list, gap_list; 
                       measure_first=measure_first)
    
    return energy_history, final_A, final_params, final_cost, Z_list_list, X_list_list, gap_list, params_history
end

function train_nocompile(gate, row::Int, nqubits::Int, M::AbstractManifold, J::Float64, g::Float64; maxiter=3000)
    energy_history = Float64[]
    gap_list = Float64[]
    eigenvalues_list = Vector{Float64}[]
    X_list = Float64[]
    ZZ_list1 = Float64[]
    ZZ_list2 = Float64[]
    function f(M, gate)
        A_matrix = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
        X_cost = real(cost_X(rho, A_matrix, row, nqubits))
        ZZ_exp1, ZZ_exp2 = cost_ZZ(rho, A_matrix, row, nqubits)
        ZZ_exp1 = real(ZZ_exp1)
        ZZ_exp2 = real(ZZ_exp2)
        energy = -g*X_cost - J*(row == 1 ? ZZ_exp2 : ZZ_exp1 + ZZ_exp2)
        push!(X_list, X_cost)
        push!(ZZ_list1, ZZ_exp1)
        push!(ZZ_list2, ZZ_exp2)
        push!(gap_list, gap)
        push!(eigenvalues_list, eigenvalues)
        push!(energy_history, real(energy))
        @info "Iteration: $(length(energy_history)), energy: $energy, gap: $gap"
        return real(energy)
    end

    @assert is_point(M, Matrix(gate))

    result = Manopt.particle_swarm(
        M, 
        f;
        swarm_size = 20,
        stopping_criterion = StopAfterIteration(maxiter) | 
                           StopWhenSwarmVelocityLess(1e-6),
        record = [:Iteration, :Cost],
        return_state = true
    )
    
    final_p = get_solver_result(result)
    final_energy = f(M, final_p)
   @show result
    _save_training_data_exact(g, energy_history, X_list, ZZ_list1, ZZ_list2, gap_list, eigenvalues_list, final_p)
    
    return result, final_energy, final_p, X_list, ZZ_list1, ZZ_list2, energy_history, gap_list, eigenvalues_list
end

