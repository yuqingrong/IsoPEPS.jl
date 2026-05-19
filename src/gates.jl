"""
Number of parameters per layer for the improved ansatz.
Uses 2 parameters per qubit (Rx-Rz).
"""
const PARAMS_PER_QUBIT_PER_LAYER = 2

# Cache for the parameter-independent CNOT entangling product.
const _CNOT_PRODUCT_CACHE = Dict{Tuple{Int,Int,Int}, Matrix{ComplexF64}}()

"""
    LocalCircuitOp

Symbolic operation in one local circuit block.

Fields:
- `kind`: `:rx`, `:rz`, or `:cnot`
- `qubits`: acted-on qubits; one entry for rotations, `(control, target)` for CNOTs
- `layer`: circuit layer index
- `param_index`: index into the parameter vector for rotations, or `nothing` for CNOTs
"""
struct LocalCircuitOp
    kind::Symbol
    qubits::Tuple{Vararg{Int}}
    layer::Int
    param_index::Union{Int,Nothing}
end

function _cnot_pattern_layers(nqubits::Int; max_stride::Int=nqubits-1, active_nqubits::Int=nqubits)
    1 <= active_nqubits <= nqubits || throw(ArgumentError("active_nqubits must be between 1 and nqubits"))
    active_nqubits == 1 && return Vector{Tuple{Int,Int}}[]
    max_stride = clamp(max_stride, 1, active_nqubits-1)

    if nqubits == 5 && active_nqubits == 5 && max_stride == 4
        return [[(2, 1), (3, 2), (1, 3), (4, 1), (5, 3), (5, 4)]]
    end

    layers = Vector{Vector{Tuple{Int,Int}}}()
    for s in 1:max_stride
        layer = Tuple{Int,Int}[]
        if s == active_nqubits - 1
            push!(layer, (1,active_nqubits))
        else
            for i in 1:active_nqubits-s
                push!(layer, (i+s, i))
            end
        end
        push!(layers, layer)
    end
    return layers
end

"""
    cnot_pattern(nqubits; max_stride=nqubits-1, active_nqubits=nqubits)

Return the ordered CNOT `(control, target)` pairs used by the local circuit
block. This is the same entangling pattern used internally by
`build_unitary_gate`.
"""
function cnot_pattern(nqubits::Int; max_stride::Int=nqubits-1, active_nqubits::Int=nqubits)
    return reduce(vcat, _cnot_pattern_layers(nqubits;
                                             max_stride=max_stride,
                                             active_nqubits=active_nqubits);
                  init=Tuple{Int,Int}[])
end

"""Return (and cache) the combined CNOT entangler for the first `active_nqubits` qubits."""
function _get_cnot_product(nqubits::Int; max_stride::Int=nqubits-1, active_nqubits::Int=nqubits)
    1 <= active_nqubits <= nqubits || throw(ArgumentError("active_nqubits must be between 1 and nqubits"))
    max_stride = active_nqubits == 1 ? 0 : clamp(max_stride, 1, active_nqubits-1)
    get!(_CNOT_PRODUCT_CACHE, (nqubits, max_stride, active_nqubits)) do
        dim = 1 << nqubits
        result = Matrix{ComplexF64}(I, dim, dim)
        for cnot_layer in _cnot_pattern_layers(nqubits;
                                               max_stride=max_stride,
                                               active_nqubits=active_nqubits)
            layer = Matrix{ComplexF64}(I, dim, dim)
            for (control, target) in cnot_layer
                layer *= Matrix(cnot(nqubits, control, target))
            end
            result *= layer
        end
        result
    end
end

"""
    local_circuit_ops(p, nqubits; max_stride=nqubits-1, active_nqubits=nqubits)

Return a symbolic trace of the local circuit block built by
`build_unitary_gate`: per-layer `Rx`, `Rz`, then the ordered CNOT entangler.
The trace is intended for drawing and documentation; numerical optimization
continues to use the dense matrix builder.
"""
function local_circuit_ops(p::Int, nqubits::Int; max_stride::Int=nqubits-1,
                           active_nqubits::Int=nqubits)
    1 <= active_nqubits <= nqubits || throw(ArgumentError("active_nqubits must be between 1 and nqubits"))
    ppq = PARAMS_PER_QUBIT_PER_LAYER
    ops = LocalCircuitOp[]

    for r in 1:p
        for q in 1:active_nqubits
            idx = ppq*nqubits*(r-1) + ppq*(q-1) + 1
            push!(ops, LocalCircuitOp(:rx, (q,), r, idx))
            push!(ops, LocalCircuitOp(:rz, (q,), r, idx + 1))
        end
        for (control, target) in cnot_pattern(nqubits;
                                              max_stride=max_stride,
                                              active_nqubits=active_nqubits)
            push!(ops, LocalCircuitOp(:cnot, (control, target), r, nothing))
        end
    end

    return ops
end

function _quantikz_gate(op::LocalCircuitOp)
    if op.kind == :rx
        return "\\gate{R_x(\\theta_{$(op.param_index)})}"
    elseif op.kind == :rz
        return "\\gate{R_z(\\theta_{$(op.param_index)})}"
    end
    return "\\qw"
end

"""
    circuit_quantikz(p, nqubits; max_stride=nqubits-1, active_nqubits=nqubits)

Generate a Quantikz diagram for the local circuit block. The diagram is built
from `local_circuit_ops`, so it follows the same gate ordering as
`build_unitary_gate`.
"""
function circuit_quantikz(p::Int, nqubits::Int; max_stride::Int=nqubits-1,
                          active_nqubits::Int=nqubits)
    ops = local_circuit_ops(p, nqubits;
                            max_stride=max_stride,
                            active_nqubits=active_nqubits)
    columns = Vector{Vector{String}}()
    for r in 1:p
        rx_column = fill("\\qw", nqubits)
        rz_column = fill("\\qw", nqubits)
        for op in ops
            op.layer == r || continue
            if op.kind == :rx
                rx_column[op.qubits[1]] = _quantikz_gate(op)
            elseif op.kind == :rz
                rz_column[op.qubits[1]] = _quantikz_gate(op)
            end
        end
        push!(columns, rx_column)
        push!(columns, rz_column)

        for op in ops
            op.layer == r && op.kind == :cnot || continue
            control, target = op.qubits
            cnot_column = fill("\\qw", nqubits)
            cnot_column[control] = "\\ctrl{$(target - control)}"
            cnot_column[target] = "\\targ{}"
            push!(columns, cnot_column)
        end
    end

    lines = String[]
    for q in 1:nqubits
        cells = [column[q] for column in columns]
        push!(lines, "\\lstick{q_$q} & " * join(cells, " & ") * " & \\qw")
    end
    return "\\begin{quantikz}\n" * join(lines, " \\\\\n") * "\n\\end{quantikz}"
end

"""
    build_unitary_gate(params, p, row, nqubits; share_params=true, active_nqubits=nqubits)

Build parameterized unitary gates for the PEPS structure using an improved ansatz
with full SU(2) single-qubit rotations and brick-wall CNOT entangling layers.

# Arguments
- `params`: Parameter vector (angles for Rx and Rz rotations)
- `p`: Number of layers per gate
- `row`: Number of gates to generate
- `nqubits`: Number of qubits per gate
- `share_params`: If true, all gates share parameters (A-A-A); if false, independent (A-B-C)
- `active_nqubits`: Number of leading qubits that receive rotations and CNOTs.
  Use `active_nqubits < nqubits` with `embed_params` to keep extra qubits idle.

# Returns
- Vector of unitary gate matrices

# Notes
- When `share_params=true`: requires `$(PARAMS_PER_QUBIT_PER_LAYER)*nqubits*p` parameters
- When `share_params=false`: requires `$(PARAMS_PER_QUBIT_PER_LAYER)*nqubits*p*row` parameters
"""
function build_unitary_gate(params, p, row, nqubits; share_params=true, max_stride::Int=nqubits-1,
                            active_nqubits::Int=nqubits)
    1 <= active_nqubits <= nqubits || throw(ArgumentError("active_nqubits must be between 1 and nqubits"))
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
                A_matrix[i] *= _build_layer(shared_params, r, nqubits;
                                            max_stride=max_stride, active_nqubits=active_nqubits)
            end
        end
    else
        @assert length(params) >= ppq*nqubits*p*row "Need at least $(ppq*nqubits*p*row) parameters for independent parameters mode"
        for i in 1:row
            params_i = params[ppq*nqubits*p*(i-1)+1:ppq*nqubits*p*i]
            for r in 1:p
                A_matrix[i] *= _build_layer(params_i, r, nqubits;
                                            max_stride=max_stride, active_nqubits=active_nqubits)
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
    build_unitary_gate_2x2(params, p, row, nqubits; active_nqubits=nqubits)

Build a 2×2 unit cell of unitary gates (A, B, C, D) and tile vertically for `row` rows.

The 4 gates tile the lattice as:
    odd columns:  [A, B, A, B, ...]  (rows 1, 2, 3, 4, ...)
    even columns: [C, D, C, D, ...]

# Arguments
- `params`: Parameter vector of length `4 * PARAMS_PER_QUBIT_PER_LAYER * nqubits * p`
- `p`: Number of layers per gate
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
- `active_nqubits`: Number of leading qubits that receive rotations and CNOTs.
  Use `active_nqubits < nqubits` with `embed_params` to keep extra qubits idle.

# Returns
- `(gates_odd, gates_even)`: Tuple of two `Vector{Matrix{ComplexF64}}`, each of length `row`
"""
function build_unitary_gate_2x2(params, p, row, nqubits; max_stride::Int=nqubits-1,
                                active_nqubits::Int=nqubits)
    1 <= active_nqubits <= nqubits || throw(ArgumentError("active_nqubits must be between 1 and nqubits"))
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 2
    chunk = ppq * nqubits * p
    @assert length(params) >= 4 * chunk "Need at least $(4*chunk) parameters for 2×2 unit cell"

    dim = 2^nqubits
    function _build_gate(par)
        G = Matrix{ComplexF64}(I, dim, dim)
        for r in 1:p
            G *= _build_layer(par, r, nqubits;
                              max_stride=max_stride, active_nqubits=active_nqubits)
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
function _build_layer(params, r, nqubits; max_stride::Int=nqubits-1, active_nqubits::Int=nqubits)
    # Compute Rx(θx)·Rz(θz) for each qubit as a raw 2×2 matrix
    # 2 parameters per qubit per layer
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 2
    gate = Matrix{ComplexF64}(undef, 1, 1)
    gate[1,1] = one(ComplexF64)

    for i in 1:nqubits
        idx = ppq*nqubits*(r-1) + ppq*(i-1) + 1
        if i <= active_nqubits
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
        else
            sq = Matrix{ComplexF64}(I, 2, 2)
        end
        # kron(sq, gate) puts sq on the higher qubit — Yao convention
        gate = kron(sq, gate)
    end

    return gate * _get_cnot_product(nqubits; max_stride=max_stride, active_nqubits=active_nqubits)
end

"""
    embed_params(params, p, nqubits_from, nqubits_to; unit_cell=:single)

Embed parameters from a smaller nqubits into a larger nqubits parameter space.
Copies existing qubit rotations and sets new qubits to identity (θ=0).

Works for both `:single` (shared params) and `:two_by_two` (4 gate chunks) unit cells.

# Example
```julia
# Optimize at nqubits=3, then warm-start at nqubits=5
params_3 = result.final_params
params_5 = embed_params(params_3, p, 3, 5; unit_cell=:two_by_two)
```
"""
function embed_params(params::Vector{Float64}, p::Int, nqubits_from::Int, nqubits_to::Int;
                      unit_cell::Symbol=:single)
    nqubits_to > nqubits_from || error("nqubits_to ($nqubits_to) must be > nqubits_from ($nqubits_from)")
    ppq = PARAMS_PER_QUBIT_PER_LAYER  # 2

    chunk_from = ppq * nqubits_from * p
    chunk_to   = ppq * nqubits_to * p

    n_chunks = unit_cell == :two_by_two ? 4 : 1
    @assert length(params) >= n_chunks * chunk_from "Expected $(n_chunks * chunk_from) params, got $(length(params))"

    new_params = zeros(Float64, n_chunks * chunk_to)

    for c in 0:(n_chunks-1)
        old_chunk = params[c*chunk_from+1 : (c+1)*chunk_from]
        for r in 1:p
            for i in 1:nqubits_from
                old_idx = ppq * nqubits_from * (r-1) + ppq * (i-1) + 1
                new_idx = c * chunk_to + ppq * nqubits_to * (r-1) + ppq * (i-1) + 1
                new_params[new_idx]   = old_chunk[old_idx]
                new_params[new_idx+1] = old_chunk[old_idx+1]
            end
            # Qubits nqubits_from+1 : nqubits_to stay at 0 (identity rotation)
        end
    end

    return new_params
end
