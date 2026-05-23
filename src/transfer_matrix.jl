# =============================================================================
# Transfer Matrix Core Operations
# =============================================================================
# Functions for building and computing transfer matrix properties

# =============================================================================
# TransferOperator — Unified API for N-column unit cells
# =============================================================================

"""
    TransferOperator{G<:AbstractVector}

Unified transfer operator for arbitrary unit-cell widths.
Wraps one gate vector per column (1 for 1×1, 2 for 2×2, …) and exposes
`apply_transfer(op, v)` (matrix-free T·v), `Matrix(op)` (explicit matrix),
and `matrix_size(op)`.
"""
struct TransferOperator{G<:AbstractVector}
    columns::Vector{G}
    row::Int
    virtual_qubits::Int
end

"""1×1 unit-cell constructor: `TransferOperator(gates, row, nqubits)`"""
TransferOperator(gates, row, nqubits) =
    TransferOperator([gates], row, (nqubits - 1) ÷ 2)

"""2×2 unit-cell constructor: `TransferOperator(gates_odd, gates_even, row, nqubits)`"""
TransferOperator(gates_odd, gates_even, row, nqubits) =
    TransferOperator([gates_odd, gates_even], row, (nqubits - 1) ÷ 2)

"""
    matrix_size(op::TransferOperator)

Transfer matrix dimension: `bond_dim^(2*(row+1))`.
"""
function matrix_size(op::TransferOperator)
    bond_dim = 2^op.virtual_qubits
    return bond_dim^(2 * (op.row + 1))
end

"""Apply one column's transfer matrix to vector `v` (matrix-free)."""
function _apply_single_column(gates, row, virtual_qubits, v)
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    code, total_legs = _build_transfer_contraction_code(A_tensors, row, virtual_qubits)
    tensor_ket = [A_tensors[i] for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    bond_dim = 2^virtual_qubits
    v_tensor = reshape(v, ntuple(_ -> bond_dim, 2 * total_legs)...)
    return vec(apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs))
end

"""
    apply_transfer(op::TransferOperator, v)

Matrix-free T·v.  Columns are applied right-to-left so that for 2×2:
`T_col1 * (T_col2 * v)`, consistent with `Matrix(op) = T_col1 * T_col2`.
"""
function apply_transfer(op::TransferOperator, v)
    for gates in reverse(op.columns)
        v = _apply_single_column(gates, op.row, op.virtual_qubits, v)
    end
    return v
end

function _operator_inserted_tensors(A_tensors, row, operators::Dict{Int,<:AbstractMatrix})
    for pos in keys(operators)
        1 <= pos <= row || error("position must be between 1 and row=$row, got $pos")
    end

    return map(1:row) do i
        if haskey(operators, i)
            O = Matrix{ComplexF64}(operators[i])
            ein"iabcd,ji -> jabcd"(A_tensors[i], O)
        else
            A_tensors[i]
        end
    end
end

function _column_matvec_data(gates, row, virtual_qubits,
                             operators::Dict{Int,<:AbstractMatrix};
                             optimizer=GreedyMethod())
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    tensor_ket = isempty(operators) ?
        [A_tensors[i] for i in 1:row] :
        _operator_inserted_tensors(A_tensors, row, operators)
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    code, total_legs = build_transfer_code(tensor_ket, tensor_bra, row;
                                           for_matvec=true,
                                           optimizer=optimizer)
    bond_dim = 2^virtual_qubits
    return (code=code, total_legs=total_legs, tensor_ket=tensor_ket,
            tensor_bra=tensor_bra, bond_dim=bond_dim)
end

function _apply_column_matvec_data(cd, v)
    v_tensor = reshape(v, ntuple(_ -> cd.bond_dim, 2 * cd.total_legs)...)
    return vec(apply_transfer_matvec(cd.code, cd.tensor_ket, cd.tensor_bra,
                                     v_tensor, cd.total_legs))
end

function _operator_matvec_data(op::TransferOperator,
                               operators::Dict{Tuple{Int,Int},<:AbstractMatrix};
                               optimizer=GreedyMethod())
    N = length(op.columns)
    for (col, pos) in keys(operators)
        1 <= col <= N || error("column must be between 1 and $N, got $col")
        1 <= pos <= op.row || error("position must be between 1 and row=$(op.row), got $pos")
    end

    return map(enumerate(op.columns)) do (c, gates)
        col_ops = Dict(pos => O for ((col, pos), O) in operators if col == c)
        _column_matvec_data(gates, op.row, op.virtual_qubits, col_ops;
                            optimizer=optimizer)
    end
end

"""
    apply_transfer_with_operator(op, operators, v; optimizer=GreedyMethod())

Apply one full period of `op` to `v`, with local physical operators inserted
at `(column, row_position)` sites. This is the matrix-free counterpart of
`get_transfer_matrix_with_operator(op, operators) * v` and never builds the
dense transfer matrix.
"""
function apply_transfer_with_operator(op::TransferOperator,
                                      operators::Dict{Tuple{Int,Int},<:AbstractMatrix},
                                      v;
                                      optimizer=GreedyMethod())
    col_data = _operator_matvec_data(op, operators; optimizer=optimizer)
    for cd in reverse(col_data)
        v = _apply_column_matvec_data(cd, v)
    end
    return v
end

"""
    Matrix(op::TransferOperator; max_size=4096)

Build the explicit combined transfer matrix (product of per-column matrices).
Uses the raw contraction convention (no transpose) consistent with `apply_transfer`.
For large systems, prefer `compute_transfer_spectrum(...; matrix_free=:always)`.
"""
function Base.Matrix(op::TransferOperator; max_size=4096)
    bond_dim = 2^op.virtual_qubits
    total_legs = op.row + 1
    ms = bond_dim^(2 * total_legs)
    if ms > max_size
        error("""
            Refusing to build explicit transfer matrix of size $(ms)×$(ms).
            This would allocate about $(round(ms * ms * sizeof(ComplexF64) / 2.0^40; digits=3)) TiB.
            Use compute_transfer_spectrum(op; matrix_free=:always) or raise max_size explicitly.
            """)
    end
    mats = map(op.columns) do gates
        A_tensors = gates_to_tensors(gates, op.row, op.virtual_qubits)
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:op.row],
                                         [conj(A_tensors[i]) for i in 1:op.row], op.row)
        reshape(T, ms, ms)
    end
    return reduce(*, mats)
end

# =============================================================================
# Unified compute_transfer_spectrum (primary method)
# =============================================================================

"""
    compute_transfer_spectrum(op::TransferOperator; num_eigenvalues=2,
                              use_iterative=:auto, matrix_free=:auto,
                              check_normalization=false)

Compute the transfer matrix spectrum and fixed point for any unit-cell width.

Three solver paths are selected automatically based on `matrix_size(op)`:
  • **matrix-free** (`sz > 1024` or `matrix_free=:always`): KrylovKit with `apply_transfer`
  • **iterative**  (`sz > 256` or `use_iterative=:always`): KrylovKit on explicit matrix
  • **full-eigen** (otherwise): `LinearAlgebra.eigen`

# Returns
- `rho`:  Fixed-point density matrix (dominant eigenvector, trace-normalized)
- `gap`:  Spectral gap = −log|λ₂/λ₁|
- `eigenvalues`:     Top `num_eigenvalues` eigenvalue magnitudes
- `eigenvalues_raw`: Corresponding raw complex eigenvalues
"""
function compute_transfer_spectrum(op::TransferOperator;
                                   num_eigenvalues=2,
                                   use_iterative=:auto,
                                   matrix_free=:auto,
                                   check_normalization=false,
                                   krylovdim=max(30, 2 * num_eigenvalues),
                                   tol=1e-10,
                                   maxiter=300,
                                   eager=true,
                                   explicit_matrix_max_size=4096)
    sz = matrix_size(op)
    should_use_matrix_free = matrix_free == :always || (matrix_free == :auto && sz > 1024)
    should_use_iterative   = use_iterative == :always || (use_iterative == :auto && sz > 256)

    if should_use_matrix_free
        # Precompute tensors and contraction code once per column (not per matvec)
        col_data = map(op.columns) do gates
            A_tensors = gates_to_tensors(gates, op.row, op.virtual_qubits)
            code, total_legs = _build_transfer_contraction_code(A_tensors, op.row, op.virtual_qubits)
            tensor_ket = [A_tensors[i] for i in 1:op.row]
            tensor_bra = [conj(A_tensors[i]) for i in 1:op.row]
            bond_dim   = 2^op.virtual_qubits
            (code=code, total_legs=total_legs, tensor_ket=tensor_ket, tensor_bra=tensor_bra, bond_dim=bond_dim)
        end
        function _matvec_precomputed(v)
            for cd in reverse(col_data)
                v_t = reshape(v, ntuple(_ -> cd.bond_dim, 2 * cd.total_legs)...)
                v   = vec(apply_transfer_matvec(cd.code, cd.tensor_ket, cd.tensor_bra, v_t, cd.total_legs))
            end
            return v
        end
        v0 = randn(ComplexF64, sz); v0 ./= norm(v0)
        vals, vecs, _ = KrylovKit.eigsolve(_matvec_precomputed, v0,
                                           num_eigenvalues, :LM;
                                           ishermitian=false,
                                           krylovdim=krylovdim,
                                           tol=tol,
                                           maxiter=maxiter,
                                           eager=eager)
        sorted          = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted]
        eigenvalues     = abs.(eigenvalues_raw)
        fixed_point     = reshape(vecs[sorted[1]], isqrt(sz), isqrt(sz))

    elseif should_use_iterative
        vals, vecs, _ = KrylovKit.eigsolve(Matrix(op; max_size=explicit_matrix_max_size), num_eigenvalues, :LM;
                                           ishermitian=false,
                                           krylovdim=krylovdim,
                                           tol=tol,
                                           maxiter=maxiter,
                                           eager=eager)
        sorted          = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted]
        eigenvalues     = abs.(eigenvalues_raw)
        fixed_point     = reshape(vecs[sorted[1]], isqrt(sz), isqrt(sz))

    else
        eig             = LinearAlgebra.eigen(Matrix(op; max_size=explicit_matrix_max_size))
        sorted          = sortperm(abs.(eig.values), rev=true)
        eigenvalues_raw = eig.values[sorted]
        eigenvalues     = abs.(eigenvalues_raw)
        fixed_point     = reshape(eig.vectors[:, sorted[1]], isqrt(sz), isqrt(sz))
    end

    if check_normalization
        @assert isapprox(eigenvalues[1], 1.0, atol=1e-6) """
            Transfer matrix dominant eigenvalue |λ₁| = $(eigenvalues[1]) ≠ 1.
            Gates are not properly normalized/isometric (U†U = I required).
            """
    end

    rho = fixed_point ./ tr(fixed_point)
    gap = length(eigenvalues) > 1 ? -log(eigenvalues[2] / eigenvalues[1]) : Inf
    n   = min(num_eigenvalues, length(eigenvalues))
    return rho, gap, eigenvalues[1:n], eigenvalues_raw[1:n]
end

# =============================================================================
# Backward-Compatibility Shims (compute_transfer_spectrum)
# =============================================================================

"""
    compute_transfer_spectrum(gates, row, nqubits; ...)

Legacy 1×1 API — delegates to `TransferOperator` path.
"""
function compute_transfer_spectrum(gates, row, nqubits;
                                   num_eigenvalues=2, use_iterative=:auto,
                                   matrix_free=:auto, channel_type=:virtual,
                                   check_normalization=false,
                                   krylovdim=max(30, 2 * num_eigenvalues),
                                   tol=1e-10,
                                   maxiter=300,
                                   eager=true,
                                   explicit_matrix_max_size=4096)
    channel_type == :virtual ||
        error("channel_type=$channel_type is no longer supported; use TransferOperator API")
    return compute_transfer_spectrum(TransferOperator(gates, row, nqubits);
                                    num_eigenvalues, use_iterative, matrix_free,
                                    check_normalization, krylovdim, tol, maxiter,
                                    eager, explicit_matrix_max_size)
end


"""
    get_transfer_matrix(gates, row, virtual_qubits)

Build and return the full transfer matrix.
"""
function get_transfer_matrix(gates, row, virtual_qubits)
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    matrix_size = bond_dim^(2*total_legs)
    _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
    T = reshape(T, matrix_size, matrix_size)
    return transpose(T)
end

"""
    build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=GreedyMethod())

Build contraction code for transfer matrix operations.

Each tensor should have 5 legs: [physical, down, right, up, left].
Dimensions are automatically inferred from tensor shapes.

# Arguments
- `tensor_ket`: Vector of ket tensors (length = row)
- `tensor_bra`: Vector of bra tensors, typically conj.(tensor_ket)
- `row`: Number of rows in the tensor network
- `for_matvec`: If true, build code for T*v (matrix-vector product); if false, build full transfer matrix
- `optimizer`: Contraction order optimizer (default: GreedyMethod())

# Returns
When `for_matvec=false` (default):
- `code`: Optimized contraction code
- `result`: Contracted transfer matrix with open left/right boundary indices

When `for_matvec=true`:
- `code`: Optimized contraction code for T*v operation
- `total_legs`: Number of legs on each side (for reshaping)

# Modes
- **Full matrix mode** (`for_matvec=false`): Both left and right boundaries are open indices.
  Use this to build the explicit transfer matrix T.
  
- **Matrix-vector mode** (`for_matvec=true`): Left boundary connects to input vector,
  only right boundary is open. Use this for matrix-free eigensolvers where you only
  need T*v without forming T explicitly.
"""
function build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=GreedyMethod())
    store = IndexStore()
    index_ket = Vector{Int}[]
    index_bra = Vector{Int}[]
    output_indices = Int[]
    
    # For matvec mode, create input indices for the left boundary
    # Infer total_legs from tensor dimensions: left leg count = 1 (periodic) + row
    total_legs = row + 1  # periodic boundary index + row left indices
    input_indices = for_matvec ? [newindex!(store) for _ in 1:2*total_legs] : Int[]
    
    # Initialize boundary indices for periodic vertical boundary
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    prev_down_ket = first_down_ket
    prev_down_bra = first_down_bra
    
    # Build index structure for each tensor
    # Tensor leg ordering: [physical, down, right, up, left]
    for i in 1:row
        phyidx = newindex!(store)  # Shared between ket and bra (contracted)
        left_ket, right_ket = newindex!(store), newindex!(store)
        left_bra, right_bra = newindex!(store), newindex!(store)
        
        # Handle periodic boundary: row 1's up connects to row N's down
        up_ket = (i == 1) ? first_up_ket : prev_down_ket
        up_bra = (i == 1) ? first_up_bra : prev_down_bra
        down_ket = (i == 1) ? first_down_ket : newindex!(store)
        down_bra = (i == 1) ? first_down_bra : newindex!(store)
        
        push!(index_ket, [phyidx, down_ket, right_ket, up_ket, left_ket])
        push!(index_bra, [phyidx, down_bra, right_bra, up_bra, left_bra])
        
        prev_down_ket = down_ket
        prev_down_bra = down_bra
    end

    # Right boundary indices (always in output)
    push!(output_indices, index_ket[row][2])  # Periodic boundary index
    append!(output_indices, [index_ket[i][3] for i in 1:row])
    push!(output_indices, index_bra[row][2])
    append!(output_indices, [index_bra[i][3] for i in 1:row])
    
    # Left boundary handling depends on mode
    left_indices_ket = [index_ket[1][4], [index_ket[i][5] for i in 1:row]...]
    left_indices_bra = [index_bra[1][4], [index_bra[i][5] for i in 1:row]...]
    
    if for_matvec
        # Matrix-vector mode: connect left boundary to input vector indices
        # Replace left indices in tensor index lists with input_indices
        for (j, old_idx) in enumerate(left_indices_ket)
            for k in 1:row, l in 1:5
                if index_ket[k][l] == old_idx
                    index_ket[k][l] = input_indices[j]
                end
            end
        end
        for (j, old_idx) in enumerate(left_indices_bra)
            for k in 1:row, l in 1:5
                if index_bra[k][l] == old_idx
                    index_bra[k][l] = input_indices[total_legs + j]
                end
            end
        end
        
        # Build tensor index list with input vector.  Seed the input boundary
        # dimensions directly so planning does not allocate a large dummy vector.
        all_indices = [index_ket..., index_bra..., collect(input_indices)]
        # Infer bond dimension from tensors (index 2 is 'down' which has bond_dim size)
        bond_dim = size(tensor_ket[1], 2)
        all_tensors = [tensor_ket..., tensor_bra...]
        
        size_dict = Dict(idx => bond_dim for idx in input_indices)
        OMEinsum.get_size_dict!([index_ket..., index_bra...], all_tensors, size_dict)
        code = optimize_code(DynamicEinCode(all_indices, output_indices), size_dict, optimizer)
        
        return code, total_legs
    else
        # Full matrix mode: left boundary indices are also in output
        append!(output_indices, left_indices_ket)
        append!(output_indices, left_indices_bra)
        
        all_indices = [index_ket..., index_bra...]
        all_tensors = [tensor_ket..., tensor_bra...]
        
        size_dict = OMEinsum.get_size_dict(all_indices, all_tensors)
        code = optimize_code(DynamicEinCode(all_indices, output_indices), size_dict, optimizer)
        
        return code, code(all_tensors...)
    end
end


"""
    gates_to_tensors(gates, row, virtual_qubits)

Convert gate matrices to tensor form for contraction.
Tensor leg ordering: [physical, down, right, up, left]
"""
function gates_to_tensors(gates, row, virtual_qubits)
    bond_dim = 2^virtual_qubits
    A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
    indices = (ntuple(_ -> Colon(), 3)..., 1,ntuple(_ -> Colon(), 2)...)
    return [reshape(gates[i], A_size)[indices...] for i in 1:row]
end

"""
    apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs)

Apply transfer matrix to a vector using precomputed contraction code.
"""
function apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs)
    result = code(tensor_ket..., tensor_bra..., v_tensor)
    # Infer bond dimension from the input tensor
    bond_dim = size(v_tensor, 1)
    return reshape(result, ntuple(_ -> bond_dim, 2*total_legs)...)
end

# ==============================================================================
# reshape to MPS format and use reference
# ==============================================================================

"""
    reshape_to_mps(gates, row, virtual_qubits)

Reshape one column of the multiline MPS into a single MPS tensor.

Takes `row` tensors from a column, contracts the vertical (up/down) bonds between them,
and fuses the physical legs into a single index.

# Tensor Structure
Each input tensor has legs: [physical, down, right, up, left]
- Vertical bonds: down[i] contracts with up[i+1] for i = 1:row-1
- Open boundaries: up[1] and down[row] (vertical), left[1:row] and right[1:row] (horizontal)

# Output
MPS tensor with shape: (physical_dim, left_bond_dim, right_bond_dim)
- physical_dim = 2^row (fused physical indices)
- left_bond_dim = bond_dim^(row+1) (up[1] + all left indices)
- right_bond_dim = bond_dim^(row+1) (down[row] + all right indices)

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows (tensors in the column)
- `virtual_qubits`: Number of virtual qubits per bond

# Returns
- `mps_tensor`: Array of shape (2^row, bond_dim^(row+1), bond_dim^(row+1))
"""
function reshape_to_mps(gates, row, virtual_qubits)
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    bond_dim = 2^virtual_qubits
    
    # Tensor leg ordering: [physical, down, right, up, left]
    # Each tensor has shape: (2, bond_dim, bond_dim, bond_dim, bond_dim)
    
    # Build einsum contraction for vertical bond contractions
    store = IndexStore()
    index_tensors = Vector{Vector{Int}}()
    
    # Allocate indices for each tensor
    phys_indices = [newindex!(store) for _ in 1:row]
    down_indices = [newindex!(store) for _ in 1:row]
    right_indices = [newindex!(store) for _ in 1:row]
    up_indices = [newindex!(store) for _ in 1:row]
    left_indices = [newindex!(store) for _ in 1:row]
    
    # Create index structure: [physical, down, right, up, left]
    for i in 1:row
        push!(index_tensors, [phys_indices[i], down_indices[i], right_indices[i], 
                              up_indices[i], left_indices[i]])
    end
    
    # Contract vertical bonds: down[i] = up[i+1] for i = 1 to row-1
    # This means up[i+1] should use down[i]'s index
    for i in 1:(row-1)
        # Replace up[i+1] with down[i] in the index structure
        index_tensors[i+1][4] = down_indices[i]
    end
    
    # Output indices ordering:
    # 1. Physical indices (all row of them) - will fuse to dim 2^row
    # 2. Left boundary: up[1], left[1], left[2], ..., left[row] - will fuse to bond_dim^(row+1)  
    # 3. Right boundary: down[row], right[1], right[2], ..., right[row] - will fuse to bond_dim^(row+1)
    output_indices = Int[]
    append!(output_indices, phys_indices)                    # Physical: p1, p2, ..., p_row
    push!(output_indices, up_indices[1])                     # Vertical boundary (top)
    append!(output_indices, left_indices)                    # Horizontal left: l1, l2, ..., l_row
    push!(output_indices, down_indices[row])                 # Vertical boundary (bottom)
    append!(output_indices, right_indices)                   # Horizontal right: r1, r2, ..., r_row
    
    # Build and execute contraction
    size_dict = OMEinsum.get_size_dict(index_tensors, A_tensors)
    code = optimize_code(DynamicEinCode(index_tensors, output_indices), size_dict, GreedyMethod())
    result = code(A_tensors...)
    
    # Result shape: (2, 2, ..., 2, bond_dim, bond_dim, ..., bond_dim, bond_dim, bond_dim, ..., bond_dim)
    #               |___row___|  |______row+1________|  |_______row+1________|
    # Reshape to MPS tensor: (physical_dim, left_bond, right_bond)
    physical_dim = 2^row
    left_bond_dim = bond_dim^(row + 1)   # up[1] + row left indices
    right_bond_dim = bond_dim^(row + 1)  # down[row] + row right indices
    
    mps_tensor = reshape(result, physical_dim, left_bond_dim, right_bond_dim)
    
    return mps_tensor
end





# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

"""Backward compatibility alias for build_transfer_code with for_matvec=false"""
contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=GreedyMethod()) = 
    build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=optimizer)

"""Internal helper to build transfer contraction code for matrix-vector product"""
function _build_transfer_contraction_code(A_tensors, row, total_qubits; optimizer=GreedyMethod())
    tensor_ket = [A_tensors[i] for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    return build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=true, optimizer=optimizer)
end

"""Internal helper to apply transfer matrix to vector"""
_apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_qubits) =
    apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_qubits)

# =============================================================================
# 2×2 Unit Cell: Combined Transfer Matrix
# =============================================================================

"""
    get_combined_transfer_matrix(gates_odd, gates_even, row, virtual_qubits)

Build the combined transfer matrix T_combined = T_odd * T_even for a 2-column unit cell.

# Returns
- `(T_combined, T_odd, T_even)`: The product and individual transfer matrices
"""
function get_combined_transfer_matrix(gates_odd, gates_even, row, virtual_qubits)
    T_odd  = get_transfer_matrix(gates_odd, row, virtual_qubits)
    T_even = get_transfer_matrix(gates_even, row, virtual_qubits)
    return T_odd * T_even, T_odd, T_even
end

"""
    compute_transfer_spectrum_2x2(gates_odd, gates_even, row, nqubits; num_eigenvalues=2)

Legacy 2×2 API — delegates to `TransferOperator` path.
Now supports all three solver paths (matrix-free, iterative, full-eigen).
"""
function compute_transfer_spectrum_2x2(gates_odd, gates_even, row, nqubits;
                                        num_eigenvalues=2)
    return compute_transfer_spectrum(TransferOperator(gates_odd, gates_even, row, nqubits);
                                    num_eigenvalues)
end


# =============================================================================
# Transfer Matrix with Operator Insertion (for Correlation Analysis)
# =============================================================================

"""
    get_transfer_matrix_with_operator(gates, row, virtual_qubits, O::AbstractMatrix; 
                                       position::Int=1, optimizer=GreedyMethod())
    get_transfer_matrix_with_operator(gates, row, virtual_qubits, operators::Dict{Int,<:AbstractMatrix}; 
                                       optimizer=GreedyMethod())

Build transfer matrix E_O with operator(s) inserted at specific row position(s).

This is used for computing correlation functions via eigenmode decomposition.
The transfer matrix E_O contracts physical indices after applying operator(s) at the 
specified position(s).

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `O`: 2×2 operator matrix (single operator case)
- `operators`: Dict mapping position (1 to row) => operator matrix (multiple operators case)
- `position`: Row position (1 to row) where to insert the operator (default: 1, single operator case)
- `optimizer`: Contraction optimizer

# Returns
- `E_O`: Transfer matrix with operator(s) inserted, shape (matrix_size, matrix_size)

# Example
```julia
gates = build_unitary_gate(params, p, row, nqubits)
# Single operator at position 1
E_Z = get_transfer_matrix_with_operator(gates, row, virtual_qubits, Matrix(Z); position=1)
# Multiple operators at different positions
E_ZZ = get_transfer_matrix_with_operator(gates, row, virtual_qubits, Dict(1 => Matrix(Z), 2 => Matrix(Z)))
# Operators at all positions (1 to row)
E_all = get_transfer_matrix_with_operator(gates, row, virtual_qubits, Dict(i => Matrix(Z) for i in 1:row))
```
"""
function get_transfer_matrix_with_operator(gates, row, virtual_qubits, O::AbstractMatrix; 
                                            position::Int=1, optimizer=GreedyMethod())
    # Single operator case - convert to Dict and call the multi-operator version
    operators = Dict(position => O)
    return get_transfer_matrix_with_operator(gates, row, virtual_qubits, operators; optimizer=optimizer)
end

function get_transfer_matrix_with_operator(gates, row, virtual_qubits, operators::Dict{Int,<:AbstractMatrix}; 
                                            optimizer=GreedyMethod())
    # Validate positions
    for pos in keys(operators)
        if pos < 1 || pos > row
            error("position must be between 1 and row=$row, got $pos")
        end
    end
    
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    
    # Insert operators at specified positions: AO = O * A (acting on physical index)
    # Tensor leg ordering: [physical, down, right, up, left]
    # ein"iabcd,ji -> jabcd" applies O to the physical index
    tensor_ket = map(1:row) do i
        if haskey(operators, i)
            ein"iabcd,ji -> jabcd"(A_tensors[i], operators[i])
        else
            A_tensors[i]
        end
    end
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    
    # Contract to get E_O
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    matrix_size = bond_dim^(2*total_legs)
    
    _, T_O = contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=optimizer)
    E_O = reshape(T_O, matrix_size, matrix_size)
    
    return transpose(E_O)
end

# =============================================================================
# TransferOperator: get_transfer_matrix_with_operator
# =============================================================================

"""
    get_transfer_matrix_with_operator(op::TransferOperator, operators::Dict{Tuple{Int,Int},<:AbstractMatrix};
                                      optimizer=GreedyMethod())

Build transfer matrix with operators inserted at `(column, row_position)` sites.

# Example
```julia
op = TransferOperator(gates_odd, gates_even, row, nqubits)
E_ZZ = get_transfer_matrix_with_operator(op, Dict((1,3)=>Z, (2,4)=>Z))
```
"""
function get_transfer_matrix_with_operator(
        op::TransferOperator,
        operators::Dict{Tuple{Int,Int},M};
        optimizer=GreedyMethod()) where {M<:AbstractMatrix}
    mats = map(enumerate(op.columns)) do (c, gates)
        col_ops = Dict(r => O for ((col, r), O) in operators if col == c)
        if isempty(col_ops)
            get_transfer_matrix(gates, op.row, op.virtual_qubits)
        else
            get_transfer_matrix_with_operator(gates, op.row, op.virtual_qubits,
                                              col_ops; optimizer=optimizer)
        end
    end
    return reduce(*, mats)
end

"""
    get_transfer_matrix_with_operator(op::TransferOperator, O::AbstractMatrix;
                                      column=1, position=1, optimizer=GreedyMethod())

Single-operator convenience: insert `O` at `(column, position)`.
"""
function get_transfer_matrix_with_operator(
        op::TransferOperator, O::AbstractMatrix;
        column::Int=1, position::Int=1, optimizer=GreedyMethod())
    return get_transfer_matrix_with_operator(
        op, Dict((column, position) => O); optimizer=optimizer)
end

"""
    compute_correlation_coefficients(gates, row, virtual_qubits, O::AbstractMatrix;
                                      num_modes::Int=10, optimizer=GreedyMethod())

Compute the correlation coefficients c_α for eigenmode decomposition of connected correlations.

For the transfer matrix eigendecomposition E = Σ_α λ_α |r_α⟩⟨l_α|, the connected correlation 
function is:
    ⟨O_i O_{i+r}⟩_c = Σ_{α≥2} c_α λ_α^{r-1}

where c_α = ⟨l₁|E_O|r_α⟩ ⟨l_α|E_O|r₁⟩

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `O`: 2×2 operator matrix (e.g., Pauli Z for ⟨ZZ⟩ correlations)
- `num_modes`: Number of eigenmode coefficients to compute (default: 10)
- `optimizer`: Contraction optimizer

# Returns
- `eigenvalues`: Complex eigenvalues λ_α (sorted by magnitude, descending)
- `coefficients`: Complex coefficients c_α for each mode
- `correlation_length`: ξ = -1/log|λ₂| (from second largest eigenvalue)

# Description
The dominant term in the correlation decay is c_α λ_α^{r-1} for the largest |λ_α| < 1.
If λ is complex (λ = |λ|e^{iθ}), the correlation oscillates: |λ|^{r-1} cos(θr + φ).

# Example
```julia
gates = build_unitary_gate(params, p, row, nqubits)
eigenvalues, coefficients, ξ = compute_correlation_coefficients(gates, row, virtual_qubits, Matrix(Z))
# Dominant decay: coefficients[2] * eigenvalues[2]^(r-1)
```
"""
function compute_correlation_coefficients(gates, row, virtual_qubits, O::AbstractMatrix;
                                           num_modes::Int=64, optimizer=GreedyMethod())
    # Get the transfer matrix
    E = get_transfer_matrix(gates, row, virtual_qubits)
    
    # Get E_O (transfer matrix with operator inserted)
    E_O = get_transfer_matrix_with_operator(gates, row, virtual_qubits, O; position=1, optimizer=optimizer)
    
    # Compute eigendecomposition
    # E[output, input] convention: eigen(E) gives vectors in INPUT space
    # 
    # Mathematical right eigenvector r: E * r = λ * r (lives in input space = physical LEFT)
    # Mathematical left eigenvector l: l† * E = λ * l† (lives in output space = physical RIGHT)
    #
    # eigen(E) → right eigenvectors r_α (physical: left fixed point)
    # eigen(E') → left eigenvectors l_α (since E'*l = conj(λ)*l ⟺ l†*E = λ*l†)
    eig_E = eigen(E')
    sorted_idx = sortperm(abs.(eig_E.values), rev=true)
    eigenvalues = eig_E.values[sorted_idx]
    R = eig_E.vectors[:, sorted_idx]  # Right eigenvectors (physical: left fixed point)
    
    eig_E_adj = eigen(E)
    λ_adj = eig_E_adj.values  # eigenvalues of E' = conj(eigenvalues of E)
    V_adj = eig_E_adj.vectors
    
    # Match left eigenvectors to right eigenvectors by eigenvalue
    # For each right eigenvector with eigenvalue λ, find the left eigenvector
    # with eigenvalue conj(λ) (since E'l = conj(λ)l implies l†E = λl†)
    n = length(eigenvalues)
    L = similar(R)
    for α in 1:n
        λ_target = conj(eigenvalues[α])
        # Find the closest eigenvalue in E'
        distances = abs.(λ_adj .- λ_target)
        best_idx = argmin(distances)
        L[:, α] = V_adj[:, best_idx]
    end
    
    # Compute coefficients c_α = ⟨l₁|E_O|r_α⟩ * ⟨l_α|E_O|r₁⟩ / (⟨l₁|r₁⟩ * ⟨l_α|r_α⟩)
    num_modes = min(num_modes, length(eigenvalues))
    coefficients = zeros(ComplexF64, num_modes)
    
    r_1 = R[:, 1]  # Dominant right eigenvector (physical: left fixed point)
    l_1 = L[:, 1]  # Dominant left eigenvector (physical: right fixed point)
    norm_1 = dot(l_1, r_1)  # Biorthogonal normalization
    
    E_O_r1 = E_O * r_1
    
    for α in 1:num_modes
        r_α = R[:, α]
        l_α = L[:, α]
        norm_α = dot(l_α, r_α)
        
        # c_α = ⟨l₁|E_O|r_α⟩ * ⟨l_α|E_O|r₁⟩ / (⟨l₁|r₁⟩ * ⟨l_α|r_α⟩)
        term1 = dot(l_1, E_O * r_α)
        term2 = dot(l_α, E_O_r1)
        coefficients[α] = term1 * term2 / (norm_1 * norm_α)
    end
    
    # Correlation length from second eigenvalue
    correlation_length = length(eigenvalues) > 1 ? -1.0 / log(abs(eigenvalues[2])) : Inf
    
    return eigenvalues[1:num_modes], coefficients, correlation_length
end

"""
    compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag::Int)

Compute theoretical connected correlation decay from eigenmode decomposition.

# Arguments
- `eigenvalues`: Complex eigenvalues λ_α from `compute_correlation_coefficients`
- `coefficients`: Complex coefficients c_α from `compute_correlation_coefficients`
- `max_lag`: Maximum lag/distance r to compute

# Returns
- `lags`: Vector 1:max_lag
- `correlation`: Theoretical ⟨O_i O_{i+r}⟩_c = Σ_{α≥2} c_α λ_α^{r-1}

# Description
The connected correlation sums over all sub-leading eigenmodes (α ≥ 2, excluding 
the fixed point mode). For complex eigenvalues, this produces oscillatory behavior.
"""
function compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag::Int)
    lags = 1:max_lag
    correlation = zeros(ComplexF64, max_lag)
    
    # Sum over sub-leading modes (α ≥ 2)
    for r in lags
        for α in 2:length(eigenvalues)
            correlation[r] += coefficients[α] * eigenvalues[α]^(r-1)
        end
    end
    
    return collect(lags), correlation
end

"""
    compute_theoretical_lambda_eff(eigenvalues, coefficients, max_lag::Int)

Compute the theoretical effective eigenvalue λ_eff(r) for a sum of exponential modes.

For a correlator C(r) = Σ_{α≥2} c_α λ_α^{r-1}, the effective eigenvalue is:
    λ_eff(r) = C(r+1)/C(r) = Σ_α w_α(r) λ_α / Σ_α w_α(r)
where w_α(r) = c_α λ_α^{r-1} is the weight of mode α at distance r.

This is a weighted average of eigenvalues, where faster-decaying modes contribute
less at larger distances. At large r, λ_eff(r) → |λ₂| (the dominant eigenvalue).

# Arguments
- `eigenvalues`: Complex eigenvalues λ_α from `compute_correlation_coefficients`
- `coefficients`: Complex coefficients c_α from `compute_correlation_coefficients`
- `max_lag`: Maximum lag/distance r to compute

# Returns
- `lags`: Vector 1:(max_lag-1)
- `lambda_eff`: Theoretical λ_eff(r) = C(r+1)/C(r)

# Example
```julia
eigenvalues, coefficients, ξ = compute_correlation_coefficients(gates, row, virtual_qubits, Matrix(Z))
lags, lambda_eff = compute_theoretical_lambda_eff(eigenvalues, coefficients, 200)
```
"""
function compute_theoretical_lambda_eff(eigenvalues, coefficients, max_lag::Int)
    # First compute the correlation C(r) for r = 1 to max_lag
    _, correlation = compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag)
    
    # λ_eff(r) = C(r+1) / C(r)
    lags = 1:(max_lag-1)
    lambda_eff = zeros(ComplexF64, max_lag-1)
    
    for r in lags
        if abs(correlation[r]) > 1e-15
            lambda_eff[r] = correlation[r+1] / correlation[r]
        else
            lambda_eff[r] = NaN
        end
    end
    return collect(lags), lambda_eff
end
