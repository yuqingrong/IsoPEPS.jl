# =============================================================================
# Transfer Matrix Core Operations
# =============================================================================
# Functions for building and computing transfer matrix properties

"""
    compute_transfer_spectrum(gates, row, nqubits; channel_type=:virtual, num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)

Compute the transfer matrix spectrum and fixed point.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `nqubits`: Number of qubits per gate (e.g., 3 for 8×8 gates with 1 virtual qubit)
- `channel_type`: `:virtual` (default) or `:physical`
  - `:virtual`: Virtual transfer matrix (contracts physical indices, virtual boundaries open)
    - Matrix dimension: `bond_dim^(2*(row+1))`
  - `:physical`: Physical channel (contracts virtual indices with environment)
    - Matrix dimension: `2^row × 2^row`
    - Requires computing virtual fixed point first
- `num_eigenvalues`: Number of eigenvalues to compute (default: 2 for gap calculation)
- `use_iterative`: `:auto` (default), `:always`, or `:never` for iterative solver
- `matrix_free`: `:auto` (default), `:always`, or `:never` for matrix-free approach
  - When enabled, never forms the full transfer matrix (essential for large row)
  - Only applies to `:virtual` channel_type

# Returns
- `rho_or_E`: For `:virtual`: fixed point density matrix ρ; For `:physical`: physical channel matrix E
- `gap`: Spectral gap = -log|λ₂/λ₁|
- `eigenvalues`: Sorted eigenvalue magnitudes (top `num_eigenvalues` if iterative)
- `eigenvalues_raw`: Raw complex eigenvalues (sorted by magnitude, descending)

# Description
Constructs the transfer matrix from gates and computes its spectral properties.
- For `:virtual`: The virtual transfer matrix propagates in the virtual boundary space.
  The spectral gap quantifies correlation length in the horizontal direction.
- For `:physical`: The physical channel maps physical ket states to physical bra states,
  with virtual indices contracted using the fixed point ρ (left) and identity R (right).
"""
function compute_transfer_spectrum(gates, row, nqubits; channel_type=:virtual, num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)
    virtual_qubits = (nqubits - 1) ÷ 2
    
    # Handle physical channel type
    if channel_type == :physical
        return _compute_physical_channel_spectrum(gates, row, nqubits, virtual_qubits; num_eigenvalues=num_eigenvalues)
    elseif channel_type != :virtual
        error("Unknown channel_type: $channel_type. Use :virtual or :physical")
    end
    
    # Virtual channel computation (existing code)
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
    
    # Assert that the dominant eigenvalue magnitude is approximately 1
    # For isometric PEPS, the transfer matrix should satisfy |λ₁| = 1
    @assert isapprox(eigenvalues[1], 1.0, atol=1e-6) """
        Transfer matrix dominant eigenvalue |λ₁| = $(eigenvalues[1]) ≠ 1.
        This indicates the gates are not properly normalized/isometric.
        For isometric PEPS, we require |λ₁| = 1.
        Check that gates satisfy U†U = I (isometry condition).
        """
    
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

"""
    _to_ITensor(mps_tensor, row, virtual_qubits)

Convert MPS tensor from reshape_to_mps to ITensor format for ITensorInfiniteMPS.

# Arguments
- `mps_tensor`: Array of shape (physical_dim, left_bond, right_bond) from reshape_to_mps
- `row`: Number of rows (for metadata)
- `virtual_qubits`: Number of virtual qubits per bond

# Returns
- `A_itensor`: ITensor object with indices (left, site, right)
- `indices`: Named tuple (site=s, left=l, right=r) of Index objects
"""
function _to_ITensor(mps_tensor, row, virtual_qubits)
    physical_dim, left_bond, right_bond = size(mps_tensor)
    
    # Create ITensor indices
    s = Index(physical_dim, "Site,n=1")
    l = Index(left_bond, "Link,l=0")
    r = Index(right_bond, "Link,l=1")
    
    # Permute from (phys, left, right) to ITensor MPS convention: (left, phys, right)
    A_permuted = permutedims(mps_tensor, (2, 1, 3))
    A_itensor = ITensor(A_permuted, l, s, r)
    
    return A_itensor, (site=s, left=l, right=r)
end

"""
    _to_MPSKit(mps_tensor, row, virtual_qubits)

Convert MPS tensor from reshape_to_mps to TensorKit TensorMap format.

# Arguments
- `mps_tensor`: Array of shape (physical_dim, left_bond, right_bond) from reshape_to_mps
- `row`: Number of rows (for metadata)
- `virtual_qubits`: Number of virtual qubits per bond

# Returns
- `A_tensormap`: TensorMap with structure left_virtual ← physical ⊗ right_virtual
"""
function _to_MPSKit(mps_tensor, row, virtual_qubits)
    physical_dim, left_bond, right_bond = size(mps_tensor)
    
    # Permute from (phys, left, right) to MPSKit convention: [left, phys, right]
    A_permuted = permutedims(mps_tensor, (2, 1, 3))
    
    # Create TensorMap: left_virtual ← physical ⊗ right_virtual
    A_tensormap = TensorMap(A_permuted, ComplexSpace(left_bond), 
                            ComplexSpace(physical_dim) ⊗ ComplexSpace(right_bond))
    
    return A_tensormap
end

"""
    transfer_matrix_ITensor(gates, row, virtual_qubits; num_eigenvalues=10)

Compute transfer matrix spectrum using ITensorInfiniteMPS.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `num_eigenvalues`: Number of eigenvalues to compute (default: 10)

# Returns
- `eigenvalues`: Transfer matrix eigenvalues sorted by magnitude
- `correlation_length`: ξ = -1/log|λ₂|
"""
function transfer_matrix_ITensor(gates, row, virtual_qubits; num_eigenvalues=64)
    # Get MPS tensor from gates
    mps_tensor = reshape_to_mps(gates, row, virtual_qubits)
    A, indices = _to_ITensor(mps_tensor, row, virtual_qubits)
    
    # Build transfer matrix: T = A * dag(A')
    # Contract physical index, keep virtual indices open
    s, l, r = indices.site, indices.left, indices.right
    
    # Create primed copies for bra layer
    l_prime = prime(l)
    r_prime = prime(r)
    A_dag = dag(A)
    A_dag = replaceinds(A_dag, [l, r], [l_prime, r_prime])
    
    # Contract physical index to form transfer matrix
    # T has indices: (l, r, l', r') 
    T = A * A_dag
    
    # Combine indices for matrix form
    # Left combined index: (l, l'), Right combined index: (r, r')
    left_comb = combiner(l, l_prime)
    right_comb = combiner(r, r_prime)
    T_matrix = T * left_comb * right_comb
    
    # Extract as dense matrix and compute eigenvalues
    T_array = Array(T_matrix, combinedind(left_comb), combinedind(right_comb))
    
    # Compute eigenvalues using KrylovKit for large matrices
    matrix_size = size(T_array, 1)
    if matrix_size > 256
        vals, _, _ = KrylovKit.eigsolve(T_array, num_eigenvalues, :LM; 
                                        ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
    else
        eig_result = LinearAlgebra.eigen(T_array)
        vals = eig_result.values
    end
    
    # Sort by magnitude
    sorted_idx = sortperm(abs.(vals), rev=true)
    eigenvalues = vals[sorted_idx][1:min(num_eigenvalues, length(vals))]
    
    # Correlation length from second eigenvalue
    correlation_length = length(eigenvalues) > 1 ? -1.0 / log(abs(eigenvalues[2] / eigenvalues[1])) : Inf
    
    return T_matrix, eigenvalues, correlation_length
end

"""
    spectrum_MPSKit(gates, row, virtual_qubits; num_eigenvalues=64)

Compute transfer matrix spectrum using TensorKit operations (MPSKit-style).

Uses TensorKit's @tensor macro to build the transfer matrix and compute eigenvalues,
which is equivalent to what MPSKit does internally for InfiniteMPS.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `num_eigenvalues`: Number of eigenvalues to compute (default: 64)

# Returns
- `spectrum`: Transfer matrix eigenvalue spectrum
- `corr_length`: Correlation length ξ = -1/log|λ₂|
"""
function spectrum_MPSKit(gates, row, virtual_qubits; num_eigenvalues=64)
    # Get MPS tensor from gates
    mps_tensor = reshape_to_mps(gates, row, virtual_qubits)
    
    # Convert to TensorMap
    A = _to_MPSKit(mps_tensor, row, virtual_qubits)
    
    # Build transfer matrix using TensorKit: T = Σ_s A[s] ⊗ conj(A[s])
    # A has structure: left ← physical ⊗ right
    # Transfer matrix: (left ⊗ left') → (right ⊗ right')
    
    # Contract over physical index to form transfer matrix
    @tensor T[-1 -2; -3 -4] := A[-1 1 -3] * conj(A[-2 1 -4])
    
    # Reshape to matrix form and compute eigenvalues
    left_dim = TensorKit.dim(codomain(T))
    right_dim = TensorKit.dim(domain(T))
    T_array = reshape(convert(Array, T), left_dim, right_dim)
    
    # Compute eigenvalues
    if left_dim > 256
        vals, _, _ = KrylovKit.eigsolve(T_array, num_eigenvalues, :LM;
                                        ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
    else
        eig_result = LinearAlgebra.eigen(T_array)
        vals = eig_result.values
    end
    
    # Sort by magnitude
    sorted_idx = sortperm(abs.(vals), rev=true)
    spectrum = vals[sorted_idx][1:min(num_eigenvalues, length(vals))]
    
    # Correlation length from second eigenvalue
    corr_length = length(spectrum) > 1 ? -1.0 / log(abs(spectrum[2] / spectrum[1])) : Inf
    
    return spectrum, corr_length
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
# Transfer Matrix with Operator Insertion (for Correlation Analysis)
# =============================================================================

"""
    get_transfer_matrix_with_operator(gates, row, virtual_qubits, O::AbstractMatrix; 
                                       position::Int=1, optimizer=GreedyMethod())

Build transfer matrix E_O with operator O inserted at a specific row position.

This is used for computing connected correlation functions via eigenmode decomposition.
The transfer matrix E_O contracts physical indices after applying operator O at the 
specified position: E_O[i,j] = ⟨i|E_O|j⟩ where the physical leg at `position` has O inserted.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `O`: 2×2 operator matrix (e.g., Pauli X, Z)
- `position`: Row position (1 to row) where to insert the operator (default: 1)
- `optimizer`: Contraction optimizer

# Returns
- `E_O`: Transfer matrix with operator inserted, shape (matrix_size, matrix_size)

# Description
For connected correlation ⟨O_i O_{i+r}⟩_c, we need E_O defined as:
    E_O = Σ_s ⟨s|A† ⊗ O ⊗ A|s⟩
where the sum is over the physical index at the operator position.

# Example
```julia
gates = build_unitary_gate(params, p, row, nqubits)
E_Z = get_transfer_matrix_with_operator(gates, row, virtual_qubits, Matrix(Z); position=1)
```
"""
function get_transfer_matrix_with_operator(gates, row, virtual_qubits, O::AbstractMatrix; 
                                            position::Int=1, optimizer=GreedyMethod())
    if position < 1 || position > row
        error("position must be between 1 and row=$row, got $position")
    end
    
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    
    # Insert operator at specified position: AO = O * A (acting on physical index)
    # Tensor leg ordering: [physical, down, right, up, left]
    # ein"iabcd,ji -> jabcd" applies O to the physical index
    AO_tensor = ein"iabcd,ji -> jabcd"(A_tensors[position], O)
    
    # Create tensor lists with operator inserted
    tensor_ket = [i == position ? AO_tensor : A_tensors[i] for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    
    # Contract to get E_O
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    matrix_size = bond_dim^(2*total_legs)
    
    _, T_O = contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=optimizer)
    E_O = reshape(T_O, matrix_size, matrix_size)
    
    return transpose(E_O)
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
                                           num_modes::Int=10, optimizer=GreedyMethod())
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
    @show mean(lambda_eff)
    return collect(lags), lambda_eff
end
