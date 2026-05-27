module ManifoldsManoptExt

using IsoPEPS
using Manifolds
using Manopt

function IsoPEPS.optimize_manifold(gate, row::Int, nqubits::Int, manifold,
                           J::Float64, g::Float64; maxiter=3000, swarm_size=20)
    initial_gate = copy(gate)
    virtual_qubits = (nqubits - 1) ÷ 2
    @info "Starting manifold optimization with $(length(vec(gate))) gate entries (row=$row, nqubits=$nqubits)"

    energy_history = Float64[]
    gap_history = Float64[]
    eigenvalues_history = Vector{Float64}[]
    X_history = Float64[]
    ZZ_vert_history = Float64[]
    ZZ_horiz_history = Float64[]

    function f(M, gate)
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = IsoPEPS.compute_transfer_spectrum(gates, row, nqubits)

        X_cost = real(IsoPEPS.compute_X_expectation(rho, gates, row, virtual_qubits))
        ZZ_vert, ZZ_horiz = IsoPEPS.compute_ZZ_expectation(rho, gates, row, virtual_qubits)
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
        swarm_size = swarm_size,
        stopping_criterion = StopAfterIteration(maxiter) | StopWhenSwarmVelocityLess(1e-6),
        record = [:Iteration, :Cost],
        return_state = true
    )

    final_gate = get_solver_result(result)
    final_energy = f(manifold, final_gate)

    converged = true

    opt_result = IsoPEPS.ManifoldOptimizationResult(
        energy_history,
        final_gate,
        final_energy,
        gap_history,
        converged
    )

    input_args = Dict{Symbol, Any}(
        :g => g, :J => J, :row => row, :nqubits => nqubits,
        :initial_gate => initial_gate,
        :maxiter => maxiter,
        :swarm_size => swarm_size,
        :gate_entry_count => length(vec(gate)),
        :manifold_type => string(typeof(manifold))
    )
    IsoPEPS.save_result("data/manifold_g=$(g)_row=$(row).json", opt_result, input_args)

    return opt_result
end

end # module ManifoldsManoptExt
