# =============================================================================
# Quantum Channel Sampling (Yao-based, optimized)
# =============================================================================

"""
    sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=10000, measure_first=:Z)

Sample observables from an iterative quantum channel defined by gates.

Optimizations over the naive version:
- Gate blocks (`put`, `matblock`) are built **once** before the loop.
- The Hadamard block is cached.
- `join` and `measure!` still allocate per step (inherent to Yao's register model).

# Returns
- `rho`: Final quantum state (Yao register)
- `Z_samples`: Vector of Z measurement outcomes
- `X_samples`: Vector of X measurement outcomes
"""
function sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=10000, measure_first=:Z)
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
    H_block = put(total_qubits, 1 => H)

    rho = zero_state(n_env)
    X_samples = Float64[]
    Z_samples = Float64[]
    sizehint!(Z_samples, conv_step + samples)
    sizehint!(X_samples, samples + conv_step)

    niters = cld(conv_step + 2*samples, row)
    second_phase_start = (conv_step + samples) / row

    for i in 1:niters
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, gate_blocks[j])

            if i > second_phase_start
                if measure_first == :X
                    Z = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                    push!(Z_samples, Z.buf)
                else
                    Yao.apply!(rho, H_block)
                    X = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                    push!(X_samples, X.buf)
                end
            else
                if measure_first == :X
                    Yao.apply!(rho, H_block)
                    X = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                    push!(X_samples, X.buf)
                else
                    Z = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                    push!(Z_samples, Z.buf)
                end
            end
        end
    end

    return rho, Z_samples, X_samples
end
