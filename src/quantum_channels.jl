# =============================================================================
# Quantum Channel Sampling (Yao-based, optimized)
# =============================================================================

"""
    sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=10000,
                           measure_first=:Z, measure_y=false)

Sample observables from an iterative quantum channel defined by gates.

Optimizations over the naive version:
- Gate blocks (`put`, `matblock`) are built **once** before the loop.
- The Hadamard / S†H blocks are cached.
- `join` and `measure!` still allocate per step (inherent to Yao's register model).

# Arguments
- `measure_first`: First basis to sample (`:X` or `:Z`)
- `measure_y`: If `true`, run a third sampling phase for Y measurements.
  The measurement order is `[measure_first, other_XZ, :Y]`.

# Returns (measure_y=false, default)
- `(rho, Z_samples, X_samples)`

# Returns (measure_y=true)
- `(rho, Z_samples, X_samples, Y_samples)`
"""
function sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=10000,
                                measure_first=:Z, measure_y=false)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end

    remaining_qubits = (nqubits - 1) ÷ 2
    fixed_qubits     = (nqubits + 1) ÷ 2
    n_env            = remaining_qubits * (row + 1)
    total_qubits     = n_env + 1

    # --- Pre-build Yao gate blocks ONCE (not per iteration) ---
    gate_blocks = Vector{Any}(undef, row)
    for j in 1:row
        qpos = tuple((1:fixed_qubits)...,
                     (fixed_qubits + (j-1)*remaining_qubits + 1 :
                      fixed_qubits + j*remaining_qubits)...)
        gate_blocks[j] = put(total_qubits, qpos => matblock(gates[j]))
    end
    # X basis: apply H then measure Z
    H_block = put(total_qubits, 1 => H)
    # Y basis: apply S†H then measure Z  (S† = shift(-π/2))
    SdagH_block = put(total_qubits, 1 => chain(Yao.shift(-π/2), H))

    rho = zero_state(n_env)
    X_samples = Float64[]
    Z_samples = Float64[]
    Y_samples = Float64[]

    n_phases = measure_y ? 3 : 2
    sizehint!(Z_samples, conv_step + samples)
    sizehint!(X_samples, conv_step + samples)
    if measure_y
        sizehint!(Y_samples, samples)
    end

    niters = cld(conv_step + n_phases * samples, row)
    phase2_start = (conv_step + samples) / row
    phase3_start = measure_y ? (conv_step + 2 * samples) / row : Inf

    # Determine the measurement order: [phase1, phase2, phase3]
    # phase1 = measure_first, phase2 = the other of X/Z, phase3 = Y (if enabled)
    second_basis = measure_first == :Z ? :X : :Z

    for i in 1:niters
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, gate_blocks[j])

            # Determine current basis
            current_basis = if i > phase3_start
                :Y
            elseif i > phase2_start
                second_basis
            else
                measure_first
            end

            if current_basis == :Z
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(Z_samples, val.buf)
            elseif current_basis == :X
                Yao.apply!(rho, H_block)
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(X_samples, val.buf)
            else  # :Y
                Yao.apply!(rho, SdagH_block)
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(Y_samples, val.buf)
            end
        end
    end

    if measure_y
        return rho, Z_samples, X_samples, Y_samples
    else
        return rho, Z_samples, X_samples
    end
end

"""
    sample_quantum_channel(gates_odd, gates_even, row, nqubits; ...)

Sample observables from a quantum channel with alternating gate sets (2×2 unit cell).

Odd columns (1, 3, 5, ...) use `gates_odd`, even columns (2, 4, 6, ...) use `gates_even`.
Otherwise identical to the single-gate-set version.

# Returns
Same format as `sample_quantum_channel(gates, ...)`.
"""
function sample_quantum_channel(gates_odd::Vector{<:AbstractMatrix},
                                 gates_even::Vector{<:AbstractMatrix},
                                 row, nqubits;
                                 conv_step=1000, samples=10000,
                                 measure_first=:Z, measure_y=false)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end

    remaining_qubits = (nqubits - 1) ÷ 2
    fixed_qubits     = (nqubits + 1) ÷ 2
    n_env            = remaining_qubits * (row + 1)
    total_qubits     = n_env + 1

    # Pre-build gate blocks for both gate sets
    function _make_blocks(gates)
        blocks = Vector{Any}(undef, row)
        for j in 1:row
            qpos = tuple((1:fixed_qubits)...,
                         (fixed_qubits + (j-1)*remaining_qubits + 1 :
                          fixed_qubits + j*remaining_qubits)...)
            blocks[j] = put(total_qubits, qpos => matblock(gates[j]))
        end
        return blocks
    end
    gate_blocks_odd  = _make_blocks(gates_odd)
    gate_blocks_even = _make_blocks(gates_even)

    H_block = put(total_qubits, 1 => H)
    SdagH_block = put(total_qubits, 1 => chain(Yao.shift(-π/2), H))

    rho = zero_state(n_env)
    X_samples = Float64[]
    Z_samples = Float64[]
    Y_samples = Float64[]

    n_phases = measure_y ? 3 : 2
    sizehint!(Z_samples, conv_step + samples)
    sizehint!(X_samples, conv_step + samples)
    if measure_y
        sizehint!(Y_samples, samples)
    end

    niters = cld(conv_step + n_phases * samples, row)
    phase2_start = (conv_step + samples) / row
    phase3_start = measure_y ? (conv_step + 2 * samples) / row : Inf

    second_basis = measure_first == :Z ? :X : :Z

    for i in 1:niters
        # Alternate gate set per column
        blocks = isodd(i) ? gate_blocks_odd : gate_blocks_even
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, blocks[j])

            current_basis = if i > phase3_start
                :Y
            elseif i > phase2_start
                second_basis
            else
                measure_first
            end

            if current_basis == :Z
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(Z_samples, val.buf)
            elseif current_basis == :X
                Yao.apply!(rho, H_block)
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(X_samples, val.buf)
            else
                Yao.apply!(rho, SdagH_block)
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(Y_samples, val.buf)
            end
        end
    end

    if measure_y
        return rho, Z_samples, X_samples, Y_samples
    else
        return rho, Z_samples, X_samples
    end
end
