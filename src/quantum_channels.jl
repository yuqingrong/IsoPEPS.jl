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
function sample_quantum_channel(gates, row, nqubits; conv_step=100, samples=100000, measure_first=:Z)
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

"""
    track_convergence_to_steady_state(gates, row, nqubits; n_steps=30, n_trajectories=50)

Track how ⟨Z⟩ converges to steady state from random initial states.

The deviation from equilibrium decays as λ₂^k, where λ₂ is the second largest
eigenvalue of the transfer matrix. This verifies the transfer matrix gap.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows in PEPS
- `nqubits`: Number of qubits per gate
- `n_steps`: Number of steps to track (default: 30)
- `n_trajectories`: Number of random initial states to average over (default: 50)

# Returns
- `Z_deviations`: RMS deviation from equilibrium at each step (decays as λ₂^k)
- `Z_equilibrium`: Equilibrium value of ⟨Z⟩
"""
function track_convergence_to_steady_state(gates, row, nqubits; n_steps=30, n_trajectories=50)
    total_qubits = Int((nqubits-1)/2)*(row+1) + 1
    boundary_qubits = total_qubits - 1
    fixed_qubits = (nqubits+1)÷2
    remaining_qubits = (nqubits-1)÷2
    d = 2^boundary_qubits
    
    Z_op = put(total_qubits, 1 => Z)
    
    # Helper: apply one channel step (partial trace + gate)
    function apply_step(rho, j)
        rho_mat = Matrix(rho)
        rho_boundary = zeros(ComplexF64, d, d)
        for ii in 1:d, jj in 1:d
            rho_boundary[ii, jj] = rho_mat[2ii-1, 2jj-1] + rho_mat[2ii, 2jj]
        end
        rho_new = zeros(ComplexF64, 2^total_qubits, 2^total_qubits)
        for ii in 1:d, jj in 1:d
            rho_new[2ii-1, 2jj-1] = rho_boundary[ii, jj]
        end
        rho = DensityMatrix(rho_new)
        qubit_positions = tuple((1:fixed_qubits)..., (fixed_qubits + (j-1)*remaining_qubits + 1:fixed_qubits + j*remaining_qubits)...)
        return Yao.apply(rho, put(total_qubits, qubit_positions => matblock(gates[j])))
    end
    
    # Find equilibrium ⟨Z⟩ by running until convergence
    rho_eq = density_matrix(zero_state(total_qubits))
    for _ in 1:100
        for j in 1:row
            rho_eq = apply_step(rho_eq, j)
        end
    end
    Z_eq = real(tr(Matrix(rho_eq) * Matrix(mat(Z_op))))
    
    # Track deviations from random initial states
    Z_deviations = zeros(n_steps)
    for _ in 1:n_trajectories
        rho = density_matrix(rand_state(total_qubits))
        for step in 1:n_steps
            j = ((step - 1) % row) + 1
            rho = apply_step(rho, j)
            Z_exp = real(tr(Matrix(rho) * Matrix(mat(Z_op))))
            Z_deviations[step] += (Z_exp - Z_eq)^2
        end
    end
    Z_deviations = sqrt.(Z_deviations ./ n_trajectories)
    
    return Z_deviations, Z_eq
end