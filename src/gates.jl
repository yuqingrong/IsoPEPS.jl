"""
Number of parameters per layer for the improved ansatz.
Uses 3 parameters per qubit (Rx-Ry-Rz) for full SU(2) coverage.
"""
const PARAMS_PER_QUBIT_PER_LAYER = 3

# Cache for the parameter-independent CNOT entangling product (keyed by nqubits)
const _CNOT_PRODUCT_CACHE = Dict{Int, Matrix{ComplexF64}}()

"""Return (and cache) the combined CNOT entangling matrix for `nqubits`."""
function _get_cnot_product(nqubits::Int)
    get!(_CNOT_PRODUCT_CACHE, nqubits) do
        dim = 1 << nqubits
        # Nearest-neighbor
        cnot_nn = Matrix{ComplexF64}(I, dim, dim)
        for i in 1:nqubits-1
            cnot_nn *= Matrix(cnot(nqubits, i+1, i))
        end
        # Next-nearest-neighbor
        cnot_nnn = Matrix{ComplexF64}(I, dim, dim)
        for i in 1:nqubits-2
            cnot_nnn *= Matrix(cnot(nqubits, i, i+2))
        end
        # Skip-2
        cnot_skip2 = Matrix{ComplexF64}(I, dim, dim)
        if nqubits >= 4
            for i in 1:nqubits-3
                cnot_skip2 *= Matrix(cnot(nqubits, i+3, i))
            end
        end
        # Full-range
        cnot_full = Matrix{ComplexF64}(I, dim, dim)
        if nqubits >= 5
            cnot_full *= Matrix(cnot(nqubits, nqubits, 1))
        end
        cnot_nn * cnot_nnn * cnot_skip2 * cnot_full
    end
end

"""
    build_unitary_gate(params, p, row, nqubits; share_params=true, symmetry_breaking=0.0, noise_strength=0.0)

Build parameterized unitary gates for the PEPS structure using an improved ansatz
with full SU(2) single-qubit rotations and brick-wall CNOT entangling layers.

# Arguments
- `params`: Parameter vector (angles for Rx, Ry, and Rz rotations)
- `p`: Number of layers per gate
- `row`: Number of gates to generate
- `nqubits`: Number of qubits per gate
- `share_params`: If true, all gates share parameters (A-A-A); if false, independent (A-B-C)
- `symmetry_breaking`: Small angle ε for Ry(ε) on qubit 1 to break Z₂ symmetry (default: 0.0)
- `noise_strength`: Add random Ry rotations on all qubits with angle ~ N(0, noise_strength) (default: 0.0)

# Returns
- Vector of unitary gate matrices

# Notes
- When `share_params=true`: requires `3*nqubits*p` parameters
- When `share_params=false`: requires `3*nqubits*p*row` parameters
- Setting `symmetry_breaking > 0` adds Ry(ε) on the first qubit (measured qubit) 
  at the end of the circuit to break Z₂ symmetry and avoid unitary channels.
- Setting `noise_strength > 0` adds random rotations to break all symmetries.
"""
function build_unitary_gate(params, p, row, nqubits; share_params=true)
    A_matrix = Vector{Matrix{ComplexF64}}(undef, row)
    dim = 2^nqubits
    params_per_layer = PARAMS_PER_QUBIT_PER_LAYER * nqubits
    
    for i in 1:row
        A_matrix[i] = Matrix(Array{ComplexF64}(I, dim, dim))
    end
    
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 3
    if share_params
        @assert length(params) >= ppq*nqubits*p "Need at least $(ppq*nqubits*p) parameters for shared parameters mode"
        shared_params = params[1:ppq*nqubits*p]
        for i in 1:row
            for r in 1:p
                A_matrix[i] *= _build_layer(shared_params, r, nqubits)
            end
        end
    else
        @assert length(params) >= ppq*nqubits*p*row "Need at least $(ppq*nqubits*p*row) parameters for independent parameters mode"
        for i in 1:row
            params_i = params[ppq*nqubits*p*(i-1)+1:ppq*nqubits*p*i]
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
Build a single layer: raw Rx·Ry·Rz rotations ⊗ cached CNOT product.
No Yao objects created — pure matrix arithmetic.
"""
function _build_layer(params, r, nqubits)
    # Compute Rx(θx)·Ry(θy)·Rz(θz) for each qubit as a raw 2×2 matrix
    # Full SU(2) coverage: 3 parameters per qubit per layer
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 3
    gate = Matrix{ComplexF64}(undef, 1, 1)
    gate[1,1] = one(ComplexF64)

    for i in 1:nqubits
        idx = ppq*nqubits*(r-1) + ppq*(i-1) + 1
        θx = params[idx]; θy = params[idx+1]; θz = params[idx+2]
        # Rx(θx)
        cx = cos(θx/2); sx = sin(θx/2)
        Rx = ComplexF64[cx  -im*sx;
                        -im*sx  cx]
        # Ry(θy)
        cy = cos(θy/2); sy = sin(θy/2)
        Ry = ComplexF64[cy  -sy;
                        sy   cy]
        # Rz(θz)
        em = exp(-im * θz/2); ep = exp(im * θz/2)
        Rz = ComplexF64[em  0;
                        0   ep]
        sq = Rx * Ry * Rz
        # kron(sq, gate) puts sq on the higher qubit — Yao convention
        gate = kron(sq, gate)
    end

    return gate * _get_cnot_product(nqubits)
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
                         for i in 1:N-1 if i % row != 0]
        ZZ_vert = mean(ZZ_vert_pairs)
        
        # Horizontal bonds: Z[i]*Z[i+row] (same row, adjacent columns)
        ZZ_horiz = mean(Z_samples[i] * Z_samples[i+row] for i in 1:N-row)
        # Both contribute to energy
        ZZ_mean = ZZ_vert + ZZ_horiz
    end

    return -g * X_mean - J * ZZ_mean
end

"""
    compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)

Compute Heisenberg J1-J2 energy from X, Y, Z measurement samples.

    S_i · S_j = (X_i X_j + Y_i Y_j + Z_i Z_j) / 4

# Arguments
- `X_samples`: Vector of X measurement outcomes
- `Z_samples`: Vector of Z measurement outcomes
- `Y_samples`: Vector of Y measurement outcomes
- `J1`: Nearest-neighbor coupling
- `J2`: Next-nearest-neighbor (diagonal) coupling
- `row`: Number of rows

# Returns
- Energy estimate per column
"""
function compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)
    all_samples = (Z_samples, X_samples, Y_samples)

    # Helper: compute vertical and horizontal correlations for one set of samples
    function _correlations(S, row)
        N = length(S)
        if row == 1
            vert = 0.0
            horiz = mean(S[i] * S[i+1] for i in 1:N-1)
        else
            vert_pairs = [S[i] * S[i+1] for i in 1:N-1 if i % row != 0]
            vert = mean(vert_pairs)
            horiz = mean(S[i] * S[i+row] for i in 1:N-row)
        end
        return vert, horiz
    end

    # Sum XX + YY + ZZ for NN bonds
    SS_vert = 0.0
    SS_horiz = 0.0
    for S in all_samples
        v, h = _correlations(S, row)
        @show v, h
        SS_vert += v
        SS_horiz += h
    end

    energy = J1 * (row == 1 ? SS_horiz : SS_vert + SS_horiz) / 4.0

    # J2: diagonal NNN bonds
    if J2 != 0.0 && row > 1
        SS_diag = 0.0
        for S in all_samples
            N = length(S)
            # Diagonal: (pos,col)->(pos+1,col+1)
            diag1 = [S[i] * S[i+row+1] for i in 1:N-row-1 if i % row != 0]
            # Anti-diagonal: (pos,col)->(pos-1,col+1)
            diag2 = [S[i] * S[i+row-1] for i in 1:N-row+1 if (i-1) % row != 0]
            SS_diag += mean(diag1) + mean(diag2)
        end
        energy += J2 * SS_diag / 4.0
    end
    return energy
end
