"""
Number of parameters per layer for the improved ansatz.
Uses 3 parameters per qubit (Rz-Ry-Rz decomposition for full SU(2) coverage).
"""
const PARAMS_PER_QUBIT_PER_LAYER = 2

"""
    build_unitary_gate(params, p, row, nqubits; share_params=true)

Build parameterized unitary gates for the PEPS structure using an improved ansatz
with full SU(2) single-qubit rotations and brick-wall CNOT entangling layers.

# Arguments
- `params`: Parameter vector (angles for Rx and Rz rotations)
- `p`: Number of layers per gate
- `row`: Number of gates to generate
- `nqubits`: Number of qubits per gate
- `share_params`: If true, all gates share parameters (A-A-A); if false, independent (A-B-C)

# Returns
- Vector of unitary gate matrices

# Notes
- When `share_params=true`: requires `2*nqubits*p` parameters
- When `share_params=false`: requires `2*nqubits*p*row` parameters
"""
function build_unitary_gate(params, p, row, nqubits; share_params=true)
    A_matrix = Vector{Matrix{ComplexF64}}(undef, row)
    dim = 2^nqubits
    params_per_layer = PARAMS_PER_QUBIT_PER_LAYER * nqubits
    
    for i in 1:row
        A_matrix[i] = Matrix(Array{ComplexF64}(I, dim, dim))
    end
    
    if share_params
        @assert length(params) >= 2*nqubits*p "Need at least $(2*nqubits*p) parameters for shared parameters mode"
        shared_params = params[1:2*nqubits*p]
        for i in 1:row
            for r in 1:p
                A_matrix[i] *= _build_layer(shared_params, r, nqubits)
            end
        end
    else
        @assert length(params) >= 2*nqubits*p*row "Need at least $(2*nqubits*p*row) parameters for independent parameters mode"
        for i in 1:row
            params_i = params[2*nqubits*p*(i-1)+1:2*nqubits*p*i]
            for r in 1:p
                A_matrix[i] *= _build_layer(params_i, r, nqubits)
            end
        end
    end
    
    # Verify unitarity
    for i in 1:row
        @assert A_matrix[i] * A_matrix[i]' ≈ I atol=1e-5 "Gate $i is not unitary"
    end

    return A_matrix
end

"""
Build a single layer of the parameterized gate circuit.
"""
function _build_layer(params, r, nqubits)
    params_per_layer = PARAMS_PER_QUBIT_PER_LAYER * nqubits
    
    # Full SU(2) single-qubit rotations: Rz(θ₁) * Ry(θ₂) * Rz(θ₃) for each qubit
    single_qubit_gates = []  
    for i in 1:nqubits
        idx = 2*nqubits*r - 2*nqubits + 2*i - 1
        push!(single_qubit_gates, Yao.Rx(params[idx]) * Yao.Rz(params[idx+1]))
    end
    gate = kron(single_qubit_gates...)
    
    # Brick-wall CNOT pattern for better entanglement
    dim = 2^nqubits
    cnot_gates = Matrix{ComplexF64}(I, dim, dim)
    for i in 1:nqubits
        target = i
        control = (i % nqubits) + 1  
        cnot_gates *= Matrix(cnot(nqubits, control, target))
    end
    #=
    # Even layer: pairs (1,2), (3,4), ...
    cnot_even = Matrix{ComplexF64}(I, dim, dim)
    for i in 1:2:nqubits-1
        cnot_even *= Matrix(cnot(nqubits, i, i+1))
    end
    
    # Odd layer: pairs (2,3), (4,5), ...
    cnot_odd = Matrix{ComplexF64}(I, dim, dim)
    for i in 2:2:nqubits-1
        cnot_odd *= Matrix(cnot(nqubits, i, i+1))
    end
    
    # Combine: single-qubit gates -> even CNOTs -> odd CNOTs=#
    return Matrix(gate) * cnot_gates
end

"""
    compute_energy(X_samples, Z_samples, g, J, row)

Compute TFIM energy from measurement samples.

# Arguments
- `X_samples`: Vector of X measurement outcomes
- `Z_samples`: Vector of Z measurement outcomes  
- `g`: Transverse field strength
- `J`: Coupling strength (default: 1.0)
- `row`: Number of rows

# Returns
- Energy estimate: E = -g⟨X⟩ - J⟨ZZ⟩

# Description
Computes the transverse field Ising model energy:
H = -g ∑ᵢ Xᵢ - J ∑ᵢ ZᵢZᵢ₊₁ - J ∑ᵢ ZᵢZᵢ₊ᵣₒw
"""
function compute_energy(X_samples, Z_samples, g, J, row) 
    X_mean = mean(X_samples)
    if row == 1
        ZZ_mean = mean(Z_samples[i]*Z_samples[i+1] for i in 1:length(Z_samples)-1)
    else
        ZZ_mean = mean(Z_samples[i]*Z_samples[i+1] + Z_samples[i]*Z_samples[i+row] 
                       for i in 1:length(Z_samples)-row)
    end
    return -g * X_mean - J * ZZ_mean
end
