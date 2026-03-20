# =============================================================================
# Quantum Channel Sampling (Yao-based, optimized)
# =============================================================================

"""
    sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=10000,
                           model::AbstractModel=TFIM())

Sample observables from an iterative quantum channel defined by gates.

Each basis (Z, X, and optionally Y) is sampled in a separate sequential phase,
each receiving exactly `conv_step + samples` raw measurements.

The model determines which bases are sampled:
- `TFIM()` → Z, X
- `HeisenbergJ1J2()` → Z, X, Y

# Returns
- `(rho, Z_samples, X_samples)` when model needs no Y measurement
- `(rho, Z_samples, X_samples, Y_samples)` when model needs Y measurement
"""
function sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=10000,
                                model::AbstractModel=TFIM())
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
    H_block = put(total_qubits, 1 => H)
    SdagH_block = put(total_qubits, 1 => chain(Yao.shift(-π/2), H))

    rho = zero_state(n_env)
    need_y = needs_y_measurement(model)

    Z_samples = Float64[]
    X_samples = Float64[]
    Y_samples = Float64[]
    sizehint!(Z_samples, conv_step + samples)
    sizehint!(X_samples, conv_step + samples)
    need_y && sizehint!(Y_samples, conv_step + samples)

    iters_per_phase = cld(conv_step + samples, row)

    # Phase 1: Z basis
    for i in 1:iters_per_phase
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, gate_blocks[j])
            val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
            push!(Z_samples, val.buf)
        end
    end

    # Phase 2: X basis
    for i in 1:iters_per_phase
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, gate_blocks[j])
            Yao.apply!(rho, H_block)
            val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
            push!(X_samples, val.buf)
        end
    end

    # Phase 3: Y basis (if needed)
    if need_y
        for i in 1:iters_per_phase
            for j in 1:row
                rho = join(rho, zero_state(1))
                Yao.apply!(rho, gate_blocks[j])
                Yao.apply!(rho, SdagH_block)
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(Y_samples, val.buf)
            end
        end
    end

    need_y ? (rho, Z_samples, X_samples, Y_samples) : (rho, Z_samples, X_samples)
end

"""
    sample_quantum_channel(gates_odd, gates_even, row, nqubits; ...)

Sample observables from a quantum channel with alternating gate sets (2×2 unit cell).

Odd iterations (1, 3, 5, ...) use `gates_odd`, even iterations (2, 4, 6, ...) use `gates_even`.
Each basis is sampled in a separate sequential phase with `conv_step + samples` raw measurements.

# Returns
Same format as `sample_quantum_channel(gates, ...)`.
"""
function sample_quantum_channel(gates_odd::Vector{<:AbstractMatrix},
                                 gates_even::Vector{<:AbstractMatrix},
                                 row, nqubits;
                                 conv_step=100, samples=10000,
                                 model::AbstractModel=TFIM())
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
    need_y = needs_y_measurement(model)

    Z_samples = Float64[]
    X_samples = Float64[]
    Y_samples = Float64[]
    sizehint!(Z_samples, conv_step + samples)
    sizehint!(X_samples, conv_step + samples)
    need_y && sizehint!(Y_samples, conv_step + samples)

    iters_per_phase = cld(conv_step + samples, row)

    # Phase 1: Z basis
    for i in 1:iters_per_phase
        blocks = isodd(i) ? gate_blocks_odd : gate_blocks_even
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, blocks[j])
            val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
            push!(Z_samples, val.buf)
        end
    end

    # Phase 2: X basis
    for i in 1:iters_per_phase
        blocks = isodd(i) ? gate_blocks_odd : gate_blocks_even
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, blocks[j])
            Yao.apply!(rho, H_block)
            val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
            push!(X_samples, val.buf)
        end
    end

    # Phase 3: Y basis (if needed)
    if need_y
        for i in 1:iters_per_phase
            blocks = isodd(i) ? gate_blocks_odd : gate_blocks_even
            for j in 1:row
                rho = join(rho, zero_state(1))
                Yao.apply!(rho, blocks[j])
                Yao.apply!(rho, SdagH_block)
                val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                push!(Y_samples, val.buf)
            end
        end
    end

    need_y ? (rho, Z_samples, X_samples, Y_samples) : (rho, Z_samples, X_samples)
end
