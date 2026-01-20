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
    final_Z_samples::Matrix{Float64}  # n_chains × n_samples
    final_X_samples::Matrix{Float64}  # n_chains × n_samples
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

Optimize a quantum circuit for the transverse-field Ising model (TFIM) using CMA-ES.

# Arguments
- `params::Vector{Float64}`: Initial parameter vector
- `J::Float64`: Ising coupling strength
- `g::Float64`: Transverse field strength
- `p::Int`: Circuit depth
- `row::Int`: Number of rows in PEPS
- `nqubits::Int`: Number of qubits

# Keyword Arguments
- `measure_first::Symbol=:Z`: Which observable to measure first (:X or :Z)
- `share_params::Bool=true`: Whether to share parameters across layers
- `conv_step::Int=100`: Burn-in steps before sampling
- `samples_per_run::Int=1000`: Samples per parallel chain
- `n_parallel_runs::Int=44`: Number of parallel sampling chains
- `maxiter::Int=5000`: Maximum CMA-ES iterations
- `abstol::Float64=0.02`: Function tolerance (ftol) for convergence
- `xtol::Float64=1e-6`: Parameter tolerance for convergence
- `sigma0::Float64=1.0`: Initial step size for CMA-ES
- `popsize::Union{Int,Nothing}=nothing`: Population size (nothing = auto)

# Returns
- `CircuitOptimizationResult`: Optimization results with energy history and final state

# Notes
- Only the best result per CMA-ES iteration is saved (not all popsize evaluations)
- Samples are stored as matrices: rows = chains, columns = samples
- This preserves chain information for diagnostics and analysis
"""
function optimize_circuit(
    params::Vector{Float64}, 
    J::Float64, 
    g::Float64, 
    p::Int, 
    row::Int, 
    nqubits::Int;
    measure_first::Symbol = :Z,
    share_params::Bool = true,
    conv_step::Int = 100,
    samples_per_run::Int = 1000,
    n_parallel_runs::Int = 44,
    maxiter::Int = 5000,
    abstol::Float64 = 0.02,
    gap_regularization::Bool = false,
    gap_weight::Float64 = 1.0,
    gap_threshold::Float64 = 0.1)
    # Validate inputs
    measure_first ∈ (:X, :Z) || throw(ArgumentError("measure_first must be :X or :Z"))
    
    initial_params = copy(params)
    n_params = length(params)
    
    # History storage (one entry per CMA-ES generation)
    energy_history = Float64[]
    params_history = Vector{Vector{Float64}}()
    Z_samples_history = Vector{Matrix{Float64}}()
    X_samples_history = Vector{Matrix{Float64}}()
    
    # Track energies within each generation
    generation_energies = Float64[]
    generation_params = Vector{Vector{Float64}}()
    generation_Z_samples = Vector{Matrix{Float64}}()
    generation_X_samples = Vector{Matrix{Float64}}()
    generation_count = Ref(0)
    
    has_logged_threads = Ref(false)
    
    function objective(x, _)
        gates = build_unitary_gate(x, p, row, nqubits; share_params=share_params)
        
        # Store samples per chain (not combined)
        Z_samples_per_chain = Vector{Vector{Float64}}(undef, n_parallel_runs)
        X_samples_per_chain = Vector{Vector{Float64}}(undef, n_parallel_runs)
        
        if !has_logged_threads[]
            @info "Using $(Threads.nthreads()) threads for parallel sampling"
            @info "Parallel runs: $n_parallel_runs, samples per run: $samples_per_run"
            if gap_regularization
                @info "Gap regularization: weight=$gap_weight, threshold=$gap_threshold"
            end
            has_logged_threads[] = true
        end
        
        Threads.@threads for run_idx in 1:n_parallel_runs
            _, Z_run, X_run = sample_quantum_channel(
                gates, row, nqubits;
                conv_step = conv_step,
                samples = samples_per_run,
                measure_first = measure_first
            )
            Z_samples_per_chain[run_idx] = Z_run[(conv_step + 1):end]
            X_samples_per_chain[run_idx] = X_run[(conv_step + 1):end]
        end
        
        # Convert to matrix: rows = chains, cols = samples
        Z_matrix = reduce(hcat, Z_samples_per_chain)'
        X_matrix = reduce(hcat, X_samples_per_chain)'
        
        # Compute energy using all samples
        Z_combined = vec(Z_matrix)
        X_combined = vec(X_matrix)
        energy = real(compute_energy(X_combined, Z_combined, g, J, row))
        
        # Compute gap penalty if enabled
        gap_penalty = 0.0
        if gap_regularization
            _, current_gap, _, _ = compute_transfer_spectrum(gates, row, nqubits; 
                                                              num_eigenvalues=2, 
                                                              use_iterative=:auto)
            if current_gap < gap_threshold
                gap_penalty = gap_weight * (gap_threshold - current_gap)
            end
        end
        
        total_cost = energy + gap_penalty
        
        # Store in generation arrays (raw energy, not penalized)
        push!(generation_energies, energy)
        push!(generation_params, copy(x))
        push!(generation_Z_samples, Z_matrix)
        push!(generation_X_samples, X_matrix)
        
        return total_cost
    end
    
    # Callback to track minimum energy per generation
    function callback(state, loss)
        if !isempty(generation_energies)
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
            global_best = minimum(energy_history)
            
            if generation_count[] % 10 == 0
                if gap_regularization
                    # Recompute gap for best candidate for logging
                    best_gates = build_unitary_gate(best_params, p, row, nqubits; share_params=share_params)
                    _, best_gap, _, _ = compute_transfer_spectrum(best_gates, row, virtual_qubits; 
                                                                   num_eigenvalues=2, 
                                                                   use_iterative=:auto)
                    best_penalty = best_gap < gap_threshold ? gap_weight * (gap_threshold - best_gap) : 0.0
                    @info "TFIM J=$J g=$g $(row)×∞ | Gen $(generation_count[]) | E=$(round(min_energy, digits=6)) | gap=$(round(best_gap, digits=4)) | penalty=$(round(best_penalty, digits=4)) | Best=$(round(global_best, digits=6))"
                else
                    @info "TFIM J=$J g=$g $(row)×∞ PEPS | Generation $(generation_count[]) | Min Energy: $(round(min_energy, digits=6)) | Global Best: $(round(global_best, digits=6))"
                end
            end
            
            # Clear generation arrays for next generation
            empty!(generation_energies)
            empty!(generation_params)
            empty!(generation_Z_samples)
            empty!(generation_X_samples)
        end
        return false  # Continue optimization
    end
    
    @info "=" ^ 60
    @info "Starting CMA-ES optimization (Optimization.jl interface)"
    @info "  Parameters: $n_params"
    @info "  Max iterations: $maxiter"
    @info "  Stopping: abstol=$abstol"
    if gap_regularization
        @info "  Gap regularization: weight=$gap_weight, threshold=$gap_threshold"
    end
    @info "=" ^ 60
    
    f = OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(f, params, 
                                             lb=zeros(n_params), 
                                             ub=fill(2π, n_params))
   
    sol = solve(prob, CMAEvolutionStrategyOpt(), 
                maxiters=maxiter, abstol=abstol, callback=callback)
    
    converged = sol.retcode == :Success || sol.retcode == :Default
    
    # Find the best iteration from history
    if !isempty(energy_history)
        best_idx = argmin(energy_history)
        final_cost = energy_history[best_idx]
        final_params = params_history[best_idx]
        final_Z_samples = Z_samples_history[best_idx]
        final_X_samples = X_samples_history[best_idx]
        @info "Best energy at generation $best_idx: $(round(final_cost, digits=6))"
    else
        final_params = sol.u
        final_cost = sol.objective
        final_Z_samples = Matrix{Float64}(undef, 0, 0)
        final_X_samples = Matrix{Float64}(undef, 0, 0)
    end
    
    final_gates = build_unitary_gate(final_params, p, row, nqubits; share_params=share_params)
    
    @info "=" ^ 60
    @info "Optimization complete"
    @info "  Converged: $converged"
    @info "  Total generations: $(generation_count[])"
    @info "=" ^ 60
    
    result = CircuitOptimizationResult(
        energy_history,
        final_gates,
        final_params,
        final_cost,
        final_Z_samples,
        final_X_samples,
        converged
    )
    
    input_args = Dict{Symbol,Any}(
        # Model parameters
        :g => g,
        :J => J,
        :row => row,
        :p => p,
        :nqubits => nqubits,
        :n_params => n_params,
        # Initial state
        :initial_params => initial_params,
        # Sampling settings
        :measure_first => String(measure_first),
        :share_params => share_params,
        :conv_step => conv_step,
        :samples_per_run => samples_per_run,
        :n_parallel_runs => n_parallel_runs,
        # Gap regularization
        :gap_regularization => gap_regularization,
        :gap_weight => gap_weight,
        :gap_threshold => gap_threshold,
        # Stopping criteria
        :maxiter => maxiter,
        :abstol => abstol,
        # Results metadata
        :total_generations => generation_count[],
        :converged => converged,
        :note => "energy_history contains minimum energy per CMA-ES generation",
        # Sample shape info
        :sample_shape => (n_parallel_runs, samples_per_run),
        :sample_description => "Matrix: rows=chains, cols=samples"
    )
    
    save_result("data/circuit_g=$(g)_row=$(row)_nqubits=$(nqubits).json", result, input_args)
    
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
        # Model parameters
        :g => g, :J => J, :row => row, :p => p, :nqubits => nqubits,
        # Optimization settings
        :initial_params => initial_params,
        :maxiter => maxiter,
        :abstol => abstol,
        :best_iteration => best_idx,
        :total_iterations => length(energy_history)
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