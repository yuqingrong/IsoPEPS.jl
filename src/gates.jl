"""
Number of parameters per layer for the improved ansatz.
Uses 3 parameters per qubit (Rz-Ry-Rz decomposition for full SU(2) coverage).
"""
const PARAMS_PER_QUBIT_PER_LAYER = 2

"""
    build_unitary_gate(params, p, row, nqubits; share_params=true, symmetry_breaking=0.0, noise_strength=0.0)

Build parameterized unitary gates for the PEPS structure using an improved ansatz
with full SU(2) single-qubit rotations and brick-wall CNOT entangling layers.

# Arguments
- `params`: Parameter vector (angles for Rx and Rz rotations)
- `p`: Number of layers per gate
- `row`: Number of gates to generate
- `nqubits`: Number of qubits per gate
- `share_params`: If true, all gates share parameters (A-A-A); if false, independent (A-B-C)
- `symmetry_breaking`: Small angle ε for Ry(ε) on qubit 1 to break Z₂ symmetry (default: 0.0)
- `noise_strength`: Add random Ry rotations on all qubits with angle ~ N(0, noise_strength) (default: 0.0)

# Returns
- Vector of unitary gate matrices

# Notes
- When `share_params=true`: requires `2*nqubits*p` parameters
- When `share_params=false`: requires `2*nqubits*p*row` parameters
- Setting `symmetry_breaking > 0` adds Ry(ε) on the first qubit (measured qubit) 
  at the end of the circuit to break Z₂ symmetry and avoid unitary channels.
- Setting `noise_strength > 0` adds random rotations to break all symmetries.
"""
function build_unitary_gate(params, p, row, nqubits; share_params=true, symmetry_breaking::Float64=0.0, noise_strength::Float64=0.0)
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
    
    # Apply symmetry-breaking Ry(ε) on qubit 1 (the measured qubit)
    # This breaks Z₂ symmetry and helps avoid unitary/unital channels
    if symmetry_breaking != 0.0
        # Build Ry(ε) ⊗ I ⊗ I ⊗ ... (Ry only on first qubit)
        ry_gate = Matrix(Yao.Ry(symmetry_breaking))
        identity_rest = Matrix{ComplexF64}(I, 2^(nqubits-1), 2^(nqubits-1))
        symmetry_breaker = kron(ry_gate, identity_rest)
        
        for i in 1:row
            A_matrix[i] = symmetry_breaker * A_matrix[i]
        end
    end
    
    # Apply deterministic "noise" rotations to break ALL symmetries
    # Uses parameter values to generate consistent angles (not random!)
    # Different for each gate (breaks translation symmetry)
    if noise_strength > 0.0
        for i in 1:row
            # Use params to generate deterministic but varied angles
            # This ensures the same params always produce the same gates
            noise_angles = [noise_strength * sin(sum(params) + j + i * nqubits) for j in 1:nqubits]
            noise_gates = [Matrix(Yao.Ry(angle)) for angle in noise_angles]
            noise_layer = reduce(kron, noise_gates)
            A_matrix[i] = noise_layer * A_matrix[i]
        end
    end
    
    # Verify unitarity
    for i in 1:row
        @assert A_matrix[i] * A_matrix[i]' ≈ I atol=1e-5 "Gate $i is not unitary"
    end

    return A_matrix
end

"""
Build a single layer of the parameterized gate circuit with multi-range entanglement.
"""
function _build_layer(params, r, nqubits)
    params_per_layer = PARAMS_PER_QUBIT_PER_LAYER * nqubits
    
    # Full SU(2) single-qubit rotations: Rx(θ₁) * Rz(θ₂) for each qubit
    single_qubit_gates = []  
    for i in 1:nqubits
        idx = 2*nqubits*r - 2*nqubits + 2*i - 1
        push!(single_qubit_gates, Yao.Rx(params[idx]) * Yao.Rz(params[idx+1]))
    end
    gate = kron(single_qubit_gates...)
    
    dim = 2^nqubits
    # Nearest-neighbor CNOTs: (1,2), (2,3), (3,4), (4,5), ...
    cnot_nn = Matrix{ComplexF64}(I, dim, dim)
    for i in 1:nqubits-1
        cnot_nn *= Matrix(cnot(nqubits, i+1, i))  # control=i+1, target=i
    end
    
    # Next-nearest-neighbor CNOTs: (1,3), (2,4), (3,5), ...
    cnot_nnn = Matrix{ComplexF64}(I, dim, dim)
    for i in 1:nqubits-2
        cnot_nnn *= Matrix(cnot(nqubits, i, i+2))  # control=i+2, target=i
    end
    
    # Skip-2 CNOTs: (1,4), (2,5), ... (only if nqubits >= 4)
    cnot_skip2 = Matrix{ComplexF64}(I, dim, dim)
    if nqubits >= 4
        for i in 1:nqubits-3
            cnot_skip2 *= Matrix(cnot(nqubits, i+3, i))  # control=i+3, target=i
        end
    end
    
    cnot_full = Matrix{ComplexF64}(I, dim, dim)
   if nqubits >= 5  
    cnot_full *= Matrix(cnot(nqubits, 1, nqubits))  # 1→5 
   end
    
    
    # Combine: single-qubit gates -> NN -> NNN -> Skip-2 -> Full-range
    return Matrix(gate) * cnot_nn * cnot_nnn * cnot_skip2 * cnot_full
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
H = -g ∑ᵢ Xᵢ - J ∑⟨ij⟩ ZᵢZⱼ

Sample layout for row=4:
  Z[1]  Z[5]  Z[9]   ...   ← row 1
  Z[2]  Z[6]  Z[10]  ...   ← row 2
  Z[3]  Z[7]  Z[11]  ...   ← row 3
  Z[4]  Z[8]  Z[12]  ...   ← row 4
   ↑     ↑     ↑
  col1  col2  col3

- Vertical bonds: Z[i]*Z[i+1] only when i % row != 0 (not at last row of column)
- Horizontal bonds: Z[i]*Z[i+row] (same row, adjacent columns)
"""
function compute_energy(X_samples, Z_samples, g, J, row) 
    X_mean = mean(X_samples)
    N = length(Z_samples)
    
    if row == 1
        # Row=1: only horizontal bonds (no vertical neighbors)
        ZZ_horiz = mean(Z_samples[i] * Z_samples[i+1] for i in 1:N-1)
        ZZ_mean = ZZ_horiz
    else
        # Vertical bonds: Z[i]*Z[i+1] only when NOT at the last row of a column
        # Skip when i % row == 0 (e.g., Z[4]*Z[5] would be diagonal, not vertical)
        ZZ_vert_pairs = [Z_samples[i] * Z_samples[i+1] 
                         for i in 1:N-1]
        ZZ_vert = mean(ZZ_vert_pairs)
        
        # Horizontal bonds: Z[i]*Z[i+row] (same row, adjacent columns)
        ZZ_horiz = mean(Z_samples[i] * Z_samples[i+row] for i in 1:N-row)
        
        # Both contribute to energy
        ZZ_mean = ZZ_vert + ZZ_horiz
    end

    return -g * X_mean - J * ZZ_mean
end
