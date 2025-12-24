"""
    sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=100000, measure_first=:Z)

Sample observables from an iterative quantum channel defined by gates.

# Arguments
- `gates`: Vector of gate matrices, one per row
- `row`: Number of rows in the PEPS structure  
- `nqubits`: Number of qubits per gate
- `conv_step`: Convergence steps before sampling (default: 1000)
- `samples`: Number of samples to collect (default: 100000)
- `measure_first`: Which observable to measure first, `:X` or `:Z` (default: `:Z`)

# Returns
- `rho`: Final quantum state
- `Z_samples`: Vector of Z measurement outcomes
- `X_samples`: Vector of X measurement outcomes

# Description
Simulates the quantum channel by iteratively applying gates and measuring.
The `measure_first` parameter determines which observable is sampled during
the convergence phase vs the sampling phase.
"""
function sample_quantum_channel(gates, row, nqubits; conv_step=1000, samples=100000, measure_first=:Z)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end
    
    rho = zero_state(Int((nqubits-1)/2)*(row+1))
    total_qubits = Int((nqubits-1)/2)*(row+1)+1
    fixed_qubits = (nqubits+1)÷2
    remaining_qubits = (nqubits-1)÷2
    X_samples = Float64[]
    Z_samples = Float64[]
    
    niters = ceil(Int, (conv_step + 2*samples) / row)
    for i in 1:niters
        for j in 1:row
            rho_p = zero_state(1)
            rho = join(rho, rho_p)
            qubit_positions = tuple((1:fixed_qubits)..., (fixed_qubits + (j-1)*remaining_qubits + 1:fixed_qubits + j*remaining_qubits)...)
            rho = Yao.apply!(rho, put(total_qubits, qubit_positions=>matblock(gates[j]))) 
            
            if i > (conv_step + samples) / row
                # Second phase: measure the other observable
                if measure_first == :X
                    Z = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                    push!(Z_samples, Z.buf)
                else
                    Yao.apply!(rho, put(total_qubits, 1=>H))
                    X = 1 - 2*measure!(RemoveMeasured(), rho, 1)
                    push!(X_samples, X.buf)
                end
            else
                # First phase: measure the primary observable
                if measure_first == :X
                    Yao.apply!(rho, put(total_qubits, 1=>H))
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