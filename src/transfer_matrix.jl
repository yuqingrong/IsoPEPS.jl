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
    
    return E_O
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
    
    # Compute full eigendecomposition
    # For right eigenvectors: E * r_α = λ_α * r_α
    eig_right = eigen(E)
    
    # Sort by magnitude (descending)
    sorted_idx = sortperm(abs.(eig_right.values), rev=true)
    eigenvalues = eig_right.values[sorted_idx]
    R = eig_right.vectors[:, sorted_idx]  # Right eigenvectors as columns
    
    # For left eigenvectors: l_α' * E = λ_α * l_α', equivalently E' * l_α = conj(λ_α) * l_α
    # Since E is generally not Hermitian, we need left eigenvectors separately
    eig_left = eigen(E')
    
    # Sort left eigenvectors to match right eigenvectors by eigenvalue
    sorted_idx_left = sortperm(abs.(eig_left.values), rev=true)
    L = eig_left.vectors[:, sorted_idx_left]  # Left eigenvectors as columns
    
    # Normalize so that ⟨l_α|r_β⟩ = δ_{αβ}
    # The biorthogonal normalization: L' * R should be diagonal
    # Rescale L columns so that diag(L' * R) = 1
    overlap = L' * R
    for α in 1:min(num_modes, size(L, 2))
        if abs(overlap[α, α]) > 1e-12
            L[:, α] ./= overlap[α, α]
        end
    end
    
    # Compute coefficients c_α = ⟨l₁|E_O|r_α⟩ * ⟨l_α|E_O|r₁⟩
    num_modes = min(num_modes, length(eigenvalues))
    coefficients = zeros(ComplexF64, num_modes)
    
    r_1 = R[:, 1]  # Fixed point right eigenvector
    l_1 = L[:, 1]  # Fixed point left eigenvector
    
    # E_O * r_α and l_α' * E_O * r_1
    E_O_r1 = E_O * r_1
    
    for α in 1:num_modes
        r_α = R[:, α]
        l_α = L[:, α]
        
        # c_α = ⟨l₁|E_O|r_α⟩ * ⟨l_α|E_O|r₁⟩
        term1 = dot(l_1, E_O * r_α)
        term2 = dot(l_α, E_O_r1)
        coefficients[α] = term1 * term2
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
