"""
Number of parameters per layer for the improved ansatz.
Uses 2 parameters per qubit (Rx-Rz).
"""
const PARAMS_PER_QUBIT_PER_LAYER = 2

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
- When `share_params=true`: requires `$(PARAMS_PER_QUBIT_PER_LAYER)*nqubits*p` parameters
- When `share_params=false`: requires `$(PARAMS_PER_QUBIT_PER_LAYER)*nqubits*p*row` parameters
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
    
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 2
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
    build_unitary_gate_2x2(params, p, row, nqubits)

Build a 2×2 unit cell of unitary gates (A, B, C, D) and tile vertically for `row` rows.

The 4 gates tile the lattice as:
    odd columns:  [A, B, A, B, ...]  (rows 1, 2, 3, 4, ...)
    even columns: [C, D, C, D, ...]

# Arguments
- `params`: Parameter vector of length `4 * PARAMS_PER_QUBIT_PER_LAYER * nqubits * p`
- `p`: Number of layers per gate
- `row`: Number of rows
- `nqubits`: Number of qubits per gate

# Returns
- `(gates_odd, gates_even)`: Tuple of two `Vector{Matrix{ComplexF64}}`, each of length `row`
"""
function build_unitary_gate_2x2(params, p, row, nqubits)
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 2
    chunk = ppq * nqubits * p
    @assert length(params) >= 4 * chunk "Need at least $(4*chunk) parameters for 2×2 unit cell"

    dim = 2^nqubits
    function _build_gate(par)
        G = Matrix{ComplexF64}(I, dim, dim)
        for r in 1:p
            G *= _build_layer(par, r, nqubits)
        end
        @assert G * G' ≈ I atol=1e-5 "Gate is not unitary"
        return G
    end

    # Build the 4 unit-cell gates: A, B, C, D
    A = _build_gate(params[1:chunk])
    B = _build_gate(params[chunk+1:2*chunk])
    C = _build_gate(params[2*chunk+1:3*chunk])
    D = _build_gate(params[3*chunk+1:4*chunk])

    # Tile vertically: odd cols = [A,B,A,B,...], even cols = [C,D,C,D,...]
    ab = [A, B]
    cd = [C, D]
    gates_odd  = Matrix{ComplexF64}[ab[mod1(i, 2)] for i in 1:row]
    gates_even = Matrix{ComplexF64}[cd[mod1(i, 2)] for i in 1:row]
    return gates_odd, gates_even
end

"""
Build a single layer: raw Rx·Rz rotations ⊗ cached CNOT product.
No Yao objects created — pure matrix arithmetic.
"""
function _build_layer(params, r, nqubits)
    # Compute Rx(θx)·Rz(θz) for each qubit as a raw 2×2 matrix
    # 2 parameters per qubit per layer
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 2
    gate = Matrix{ComplexF64}(undef, 1, 1)
    gate[1,1] = one(ComplexF64)

    for i in 1:nqubits
        idx = ppq*nqubits*(r-1) + ppq*(i-1) + 1
        θx = params[idx]; θz = params[idx+1]
        # Rx(θx)
        cx = cos(θx/2); sx = sin(θx/2)
        Rx = ComplexF64[cx  -im*sx;
                        -im*sx  cx]
        # Rz(θz)
        em = exp(-im * θz/2); ep = exp(im * θz/2)
        Rz = ComplexF64[em  0;
                        0   ep]
        sq = Rx * Rz
        # kron(sq, gate) puts sq on the higher qubit — Yao convention
        gate = kron(sq, gate)
    end

    return gate * _get_cnot_product(nqubits)
end
