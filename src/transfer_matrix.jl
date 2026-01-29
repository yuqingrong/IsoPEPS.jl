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
    _compute_physical_channel_spectrum(gates, row, nqubits, virtual_qubits; num_eigenvalues=2)

Internal helper to compute the physical channel spectrum.

First computes the virtual transfer matrix fixed points (left ρ and right R), then builds 
the physical channel E using both fixed points, and computes the spectrum of E.

For a proper trace-preserving quantum channel, we need both left and right fixed points
and proper normalization so that the dominant eigenvalue is exactly 1.
"""
function _compute_physical_channel_spectrum(gates, row, nqubits, virtual_qubits; num_eigenvalues=2)
    # Step 1: Build full transfer matrix and compute both fixed points
    T = get_transfer_matrix(gates, row, virtual_qubits)
    matrix_size = size(T, 1)
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    
    # Compute left fixed point (dominant right eigenvector): T * ρ = λ₁ * ρ
    vals_left, vecs_left, _ = KrylovKit.eigsolve(T, randn(ComplexF64, matrix_size), 2, :LM;
                                                  ishermitian=false, krylovdim=max(30, 4))
    sorted_idx_left = sortperm(abs.(vals_left), rev=true)
    λ₁ = vals_left[sorted_idx_left[1]]  # Dominant eigenvalue of transfer matrix
    rho_vec = vecs_left[sorted_idx_left[1]]
    rho_virtual = reshape(rho_vec, Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    rho_virtual = rho_virtual ./ tr(rho_virtual)  # Normalize trace to 1
    
    # Compute right fixed point (dominant left eigenvector): R * T = λ₁ * R, or T' * R = λ₁ * R
    vals_right, vecs_right, _ = KrylovKit.eigsolve(T', randn(ComplexF64, matrix_size), 2, :LM;
                                                    ishermitian=false, krylovdim=max(30, 4))
    sorted_idx_right = sortperm(abs.(vals_right), rev=true)
    R_vec = vecs_right[sorted_idx_right[1]]
    R_virtual = reshape(R_vec, Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    
    # Normalize so that tr(ρ · R) = 1
    norm_factor = tr(rho_virtual * R_virtual)
    R_virtual = R_virtual ./ norm_factor
    
    # Step 2: Build physical channel using both fixed points
    E = get_physical_channel(gates, row, virtual_qubits, rho_virtual; R=R_virtual)
    
    # Step 3: Compute spectrum of physical channel
    # Note: The physical channel eigenvalue is bounded by |λ₁| of the transfer matrix.
    # If |λ₁| < 1, the physical channel dominant eigenvalue will also be < 1.
    phys_dim = 2^row
    
    if phys_dim <= 64
        # Full eigendecomposition for small matrices
        eig_result = LinearAlgebra.eigen(E)
        sorted_indices = sortperm(abs.(eig_result.values), rev=true)
        eigenvalues_raw = eig_result.values[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
    else
        # Iterative solver for larger matrices
        vals, vecs, info = KrylovKit.eigsolve(E, num_eigenvalues, :LM; 
                                              ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
        sorted_indices = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
    end
    
    # Compute spectral gap
    gap = length(eigenvalues) > 1 && eigenvalues[1] > 0 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
    
    return E, gap, eigenvalues, eigenvalues_raw
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




"""
    build_physical_channel_code(tensor_ket, tensor_bra, row, rho, R; optimizer=GreedyMethod())

Build contraction code for physical channel (physical space → physical space).

Unlike `build_transfer_code` which contracts physical indices and leaves virtual boundaries open,
this function keeps physical indices open and contracts all virtual indices with environment tensors.

Each tensor should have 5 legs: [physical, down, right, up, left].

# Arguments
- `tensor_ket`: Vector of ket tensors (length = row)
- `tensor_bra`: Vector of bra tensors, typically conj.(tensor_ket)
- `row`: Number of rows in the tensor network
- `rho`: Left boundary environment tensor (fixed point density matrix)
- `R`: Right boundary environment tensor (typically identity)
- `optimizer`: Contraction order optimizer (default: GreedyMethod())

# Returns
- `code`: Optimized contraction code
- `result`: Physical channel matrix of shape (2^row, 2^row)

# Description
The physical channel E maps physical ket states to physical bra states:
    E[p_ket, p_bra] = Tr_virtual[ρ_left ⊗ A_ket ⊗ A_bra^† ⊗ R_right]

This is useful for studying the quantum channel properties of the PEPS in physical space.
"""
function build_physical_channel_code(tensor_ket, tensor_bra, row, rho, R; optimizer=GreedyMethod())
    store = IndexStore()
    index_ket = Vector{Int}[]
    index_bra = Vector{Int}[]
    output_indices = Int[]
    
    bond_dim = size(tensor_ket[1], 2)  # down leg dimension
    total_legs = row + 1  # periodic boundary index + row bonds
       
    index_rho = [newindex!(store) for _ in 1:2*total_legs]
    index_R = [newindex!(store) for _ in 1:2*total_legs]
    
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    prev_down_ket = first_down_ket
    prev_down_bra = first_down_bra
    
    for i in 1:row
       
        phyidx_ket = newindex!(store)
        phyidx_bra = newindex!(store)
          
        right_ket = index_R[1 + i]  # indices 2 to row+1 for ket right bonds
        right_bra = index_R[total_legs + 1 + i]  # indices total_legs+2 to 2*total_legs for bra right bonds
        
        
        left_ket = index_rho[1 + i]  # indices 2 to row+1 for ket left bonds
        left_bra = index_rho[total_legs + 1 + i]  # indices total_legs+2 to 2*total_legs for bra left bonds
      
        if i == 1
            up_ket = first_up_ket
            up_bra = first_up_bra
            down_ket = first_down_ket
            down_bra = first_down_bra
        else
            up_ket = prev_down_ket
            up_bra = prev_down_bra
            down_ket = newindex!(store)
            down_bra = newindex!(store)
        end
        
        push!(index_ket, [phyidx_ket, down_ket, right_ket, up_ket, left_ket])
        push!(index_bra, [phyidx_bra, down_bra, right_bra, up_bra, left_bra])
        push!(output_indices, phyidx_ket)
        
        prev_down_ket = down_ket
        prev_down_bra = down_bra
    end
    for i in 1:row
        push!(output_indices, index_bra[i][1])
    end
    for k in 1:row, l in 1:5
        if index_ket[k][l] == first_up_ket
            index_ket[k][l] = index_rho[1]
        end
        if index_bra[k][l] == first_up_bra
            index_bra[k][l] = index_rho[total_legs + 1]
        end
    end
    last_down_ket = index_ket[row][2]
    last_down_bra = index_bra[row][2]
    
    for k in 1:row, l in 1:5
        if index_ket[k][l] == last_down_ket
            index_ket[k][l] = index_R[1]
        end
        if index_bra[k][l] == last_down_bra
            index_bra[k][l] = index_R[total_legs + 1]
        end
    end

    all_indices = [index_ket..., index_bra..., collect(index_rho), collect(index_R)]
    all_tensors = [tensor_ket..., tensor_bra..., rho, R]
    
    size_dict = OMEinsum.get_size_dict(all_indices, all_tensors)
    code = optimize_code(DynamicEinCode(all_indices, output_indices), size_dict, optimizer)
    
    result = code(all_tensors...)
    phys_dim = 2^row
    result_matrix = reshape(result, phys_dim, phys_dim)
    
    return code, result_matrix
end

"""
    get_physical_channel(gates, row, virtual_qubits, rho; R=nothing, optimizer=GreedyMethod())

Build and return the physical channel matrix from gates.

This is a convenience wrapper around `build_physical_channel_code` that handles
tensor conversion and default environment construction.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `rho`: Fixed point density matrix (from compute_transfer_spectrum)
- `R`: Right boundary environment (default: identity matrix)
- `optimizer`: Contraction optimizer

# Returns
- `E`: Physical channel matrix of shape (2^row, 2^row)

# Example
```julia
gates = build_unitary_gate(params, p, row, nqubits)
rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
E = get_physical_channel(gates, row, virtual_qubits, rho)
```
"""
function get_physical_channel(gates, row, virtual_qubits, rho; R=nothing, optimizer=GreedyMethod())
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    total_qubits = virtual_qubits * total_legs
    
    # Prepare environment tensors
    env_size = ntuple(_ -> bond_dim, 2*total_legs)
    rho_tensor = reshape(rho, env_size...)
    
    # Default R is identity
    if R === nothing
        R_matrix = Matrix{ComplexF64}(I, bond_dim^total_legs, bond_dim^total_legs)
        R_tensor = reshape(R_matrix, env_size...)
    else
        R_tensor = reshape(R, env_size...)
    end
    
    # Convert gates to tensors
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    tensor_ket = A_tensors
    tensor_bra = [conj(A) for A in A_tensors]
    
    # Build and return physical channel
    _, E = build_physical_channel_code(tensor_ket, tensor_bra, row, rho_tensor, R_tensor; optimizer=optimizer)
    return E
end
