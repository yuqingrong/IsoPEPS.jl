# =============================================================================
# Transfer Matrix Core Operations
# =============================================================================
# Functions for building and computing transfer matrix properties

"""
    compute_transfer_spectrum(gates, row, virtual_qubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)

Compute the transfer matrix spectrum and fixed point.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `num_eigenvalues`: Number of eigenvalues to compute (default: 2 for gap calculation)
- `use_iterative`: `:auto` (default), `:always`, or `:never` for iterative solver
- `matrix_free`: `:auto` (default), `:always`, or `:never` for matrix-free approach
  - When enabled, never forms the full transfer matrix (essential for large row)

# Returns
- `rho`: Fixed point density matrix (normalized)
- `gap`: Spectral gap = -log|λ₂| where λ₂ is second largest eigenvalue
- `eigenvalues`: Sorted eigenvalue magnitudes (top `num_eigenvalues` if iterative)
- `eigenvalues_raw`: Raw complex eigenvalues (sorted by magnitude, descending)

# Description
Constructs the transfer matrix from gates and computes its spectral properties.
For large systems (row >= 5), uses a matrix-free approach that never forms the full
transfer matrix, instead computing T*v on-the-fly via tensor contraction.
The spectral gap quantifies how quickly the channel converges to its fixed point.
"""
function compute_transfer_spectrum(gates, row, nqubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)
    virtual_qubits = (nqubits - 1) ÷ 2
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)

    # Matrix size: bond_dim^(2*total_legs) where total_legs = row + 1 (periodic + row bonds)
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    matrix_size = bond_dim^(2*total_legs)
    # Decide whether to use matrix-free approach (essential for large row)
    should_use_matrix_free = if matrix_free == :auto
        matrix_size > 1024  # Use matrix-free for matrices larger than 1024x1024
    elseif matrix_free == :always
        true
    else
        false
    end
    
    # Decide whether to use iterative solver
    should_use_iterative = if use_iterative == :auto
        matrix_size > 256
    elseif use_iterative == :always
        true
    else
        false
    end
    
    if should_use_matrix_free
        # Matrix-free approach: define T*v as a function
        # This avoids O(n²) memory for large matrices
        
        # Precompute the contraction code for applying transfer matrix
        bond_dim = 2^virtual_qubits
        code, total_legs = _build_transfer_contraction_code(A_tensors, row, virtual_qubits)
        tensor_ket = [A_tensors[i] for i in 1:row]
        tensor_bra = [conj(A_tensors[i]) for i in 1:row]
        
        # Define the linear map T*v
        function apply_transfer(v)
            # v is a vector of length bond_dim^(2*total_legs) = matrix_size
            # Reshape to tensor form for left indices (2*total_legs indices, each dim bond_dim)
            v_tensor = reshape(v, ntuple(_ -> bond_dim, 2*total_legs)...)
            # Apply transfer matrix via contraction
            result = _apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_legs)
            return vec(result)
        end
        
        # Use KrylovKit with the linear map
        v0 = randn(ComplexF64, matrix_size)
        v0 = v0 / norm(v0)
        vals, vecs, info = KrylovKit.eigsolve(apply_transfer, v0, num_eigenvalues, :LM; 
                                              ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
        
        # Sort by magnitude (descending)
        sorted_indices = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
        
        gap = -log(eigenvalues[2]/eigenvalues[1])
        
        fixed_point = reshape(vecs[sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    elseif should_use_iterative
        # Build full matrix but use iterative eigensolver
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        
        vals, vecs, info = KrylovKit.eigsolve(T, num_eigenvalues, :LM; 
                                              ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
        
        sorted_indices = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
        
        # Handle case with only 1 eigenvalue
        gap = length(eigenvalues) > 1 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
        
        fixed_point = reshape(vecs[sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    else
        # Full eigendecomposition for small matrices
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        eig_result = LinearAlgebra.eigen(T)
        sorted_indices = sortperm(abs.(eig_result.values), rev=true)
        eigenvalues_raw = eig_result.values[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
        # Handle case with only 1 eigenvalue
        gap = length(eigenvalues) > 1 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
        
        fixed_point = reshape(eig_result.vectors[:, sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    end
    rho = fixed_point ./ tr(fixed_point)
    
    return rho, gap, eigenvalues, eigenvalues_raw
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
    return T
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
        
        # Build tensor index list with input vector
        all_indices = [index_ket..., index_bra..., collect(input_indices)]
        # Infer bond dimension from tensors (index 2 is 'down' which has bond_dim size)
        bond_dim = size(tensor_ket[1], 2)
        dummy_input = zeros(ComplexF64, ntuple(_ -> bond_dim, 2*total_legs)...)
        all_tensors = [tensor_ket..., tensor_bra..., dummy_input]
        
        size_dict = OMEinsum.get_size_dict(all_indices, all_tensors)
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

