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

# Description
Constructs the transfer matrix from gates and computes its spectral properties.
For large systems (row >= 5), uses a matrix-free approach that never forms the full
transfer matrix, instead computing T*v on-the-fly via tensor contraction.
The spectral gap quantifies how quickly the channel converges to its fixed point.
"""
function compute_transfer_spectrum(gates, row, virtual_qubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)

    total_qubits = 1 + virtual_qubits + virtual_qubits * row
    matrix_size = 4^(total_qubits-1)
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
        boundary_qubits = total_qubits - 1  # qubits on each boundary side
        code, _ = _build_transfer_contraction_code(A_tensors, row, boundary_qubits)
        tensor_ket = [A_tensors[i] for i in 1:row]
        tensor_bra = [conj(A_tensors[i]) for i in 1:row]
        
        # Define the linear map T*v
        function apply_transfer(v)
            # v is a vector of length 4^boundary_qubits = matrix_size
            # Reshape to tensor form for left indices (2*boundary_qubits indices, each dim 2)
            v_tensor = reshape(v, ntuple(_ -> 2, 2*boundary_qubits)...)
            # Apply transfer matrix via contraction
            result = _apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, boundary_qubits)
            return vec(result)
        end
        
        # Use KrylovKit with the linear map
        v0 = randn(ComplexF64, matrix_size)
        v0 = v0 / norm(v0)
        vals, vecs, info = KrylovKit.eigsolve(apply_transfer, v0, num_eigenvalues, :LM; 
                                              ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
        
        # Sort by magnitude (descending)
        sorted_indices = sortperm(abs.(vals), rev=true)
        eigenvalues = abs.(vals[sorted_indices])
        
        gap = -log(eigenvalues[2])
        @assert isapprox(eigenvalues[1], 1.0, atol=1e-6) "Largest eigenvalue should be 1, got $(eigenvalues[1])"
        
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
        eigenvalues = abs.(vals[sorted_indices])
        
        gap = -log(eigenvalues[2])
        @assert isapprox(eigenvalues[1], 1.0, atol=1e-6) "Largest eigenvalue should be 1, got $(eigenvalues[1])"
        
        fixed_point = reshape(vecs[sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    else
        # Full eigendecomposition for small matrices
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        
        eig_result = LinearAlgebra.eigen(T)
        eigenvalues = sort(abs.(eig_result.values))
        gap = -log(eigenvalues[end-1])
        @assert eigenvalues[end] ≈ 1.0 "Largest eigenvalue should be 1"
        
        fixed_point = reshape(eig_result.vectors[:, end], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    end
    rho = fixed_point ./ tr(fixed_point)
    
    return rho, gap, eigenvalues
end

function get_transfer_matrix(gates, row, virtual_qubits)
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    total_qubits = 1 + virtual_qubits + virtual_qubits * row
    matrix_size = 4^(total_qubits-1)
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
        dummy_input = zeros(ComplexF64, ntuple(_ -> 2, 2*total_legs)...)
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
Convert gate matrices to tensor form for contraction.
"""
function gates_to_tensors(gates, row, virtual_qubits)
    bond_dim = 2^virtual_qubits
    A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
    indices = (ntuple(_ -> Colon(), 3)..., 1, ntuple(_ -> Colon(), 2)...)
    return [reshape(gates[i], A_size)[indices...] for i in 1:row]
end

"""
    apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs)

Apply transfer matrix to a vector using precomputed contraction code.
"""
function apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs)
    result = code(tensor_ket..., tensor_bra..., v_tensor)
    return reshape(result, ntuple(_ -> 2, 2*total_legs)...)
end

# Backward compatibility aliases
contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=GreedyMethod()) = 
    build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=optimizer)

function _build_transfer_contraction_code(A_tensors, row, total_qubits; optimizer=GreedyMethod())
    tensor_ket = [A_tensors[i] for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    return build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=true, optimizer=optimizer)
end

_apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_qubits) = 
    apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_qubits)

"""
    compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())

Compute ⟨X⟩ expectation value from fixed point density matrix.

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `optimizer`: Contraction optimizer

# Returns
Average X expectation value across all sites.
"""
function compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    total_qubits = 1 + virtual_qubits + virtual_qubits * row
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = _gates_to_tensors(gates, row, virtual_qubits)
    AX_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], Matrix(X)) for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    
    results = map(1:row) do pos
        tensor_ket = [i == pos ? AX_tensors[i] : A_tensors[i] for i in 1:row]
        _, list = contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=optimizer)
        
        store = IndexStore()
        index_list = [newindex!(store) for _ in 1:4*total_qubits]
        index_rho = index_list[2*total_qubits+1:4*total_qubits] 
        index_R = index_list[1:2*total_qubits]
        index = [index_list, index_rho, index_R]
        
        size_dict = OMEinsum.get_size_dict(index, [list, rho, R])
        code = optimize_code(DynamicEinCode(index, Int[]), size_dict, optimizer)
        code(list, rho, R)[]
    end
    
    return sum(results) / row
end

"""
    compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())

Compute ⟨ZZ⟩ expectation values (vertical and horizontal bonds).

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `optimizer`: Contraction optimizer

# Returns
- `ZZ_vertical`: Vertical bond ⟨ZᵢZᵢ₊₁⟩
- `ZZ_horizontal`: Horizontal bond ⟨ZᵢZᵢ₊ᵣₒw⟩
"""
function compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    total_qubits = 1 + virtual_qubits + virtual_qubits * row
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = _gates_to_tensors(gates, row, virtual_qubits)
    AZ_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], Matrix(Z)) for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]

    # Vertical: Z on sites 1 and 2
    tensor_ket_vert = [i == 1 || i == 2 ? AZ_tensors[i] : A_tensors[i] for i in 1:row]
    # Horizontal: Z on site 1 only (correlates across transfer matrix)
    tensor_ket_horiz = [i == 1 ? AZ_tensors[i] : A_tensors[i] for i in 1:row]

    ZZ_vert = _contract_ZZ(tensor_ket_vert, A_tensors, tensor_bra, rho, R, row, total_qubits, optimizer)
    ZZ_horiz = _contract_ZZ(tensor_ket_horiz, tensor_ket_horiz, tensor_bra, rho, R, row, total_qubits, optimizer)
    
    return ZZ_vert, ZZ_horiz
end

"""
Helper function to contract ZZ expectation value.
"""
function _contract_ZZ(tensor_ket_a, tensor_ket_b, tensor_bra, rho, R, row, total_qubits, optimizer)
    _, list1 = contract_transfer_matrix(tensor_ket_a, tensor_bra, row; optimizer=optimizer)
    _, list2 = contract_transfer_matrix(tensor_ket_b, tensor_bra, row; optimizer=optimizer)
    
    store = IndexStore()
    index_list1 = [newindex!(store) for _ in 1:4*total_qubits]
    index_list2 = [[newindex!(store) for _ in 1:2*total_qubits]..., index_list1[1:2*total_qubits]...]
    
    index_rho = index_list1[2*total_qubits+1:4*total_qubits]
    index_R = index_list2[1:2*total_qubits]
    index = [index_list1, index_list2, index_rho, index_R]
    
    size_dict = OMEinsum.get_size_dict(index, [list1, list2, rho, R])
    code = optimize_code(DynamicEinCode(index, Int[]), size_dict, optimizer)
    return code(list1, list2, rho, R)[]
end



"""
    compute_exact_energy(params, g, J, p, row, virtual_qubits; optimizer=GreedyMethod())

Compute exact energy from parameters using tensor contraction.

# Arguments
- `params`: Parameter vector
- `g`: Transverse field strength
- `J`: Coupling strength
- `p`: Number of circuit layers
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `optimizer`: Contraction optimizer

# Returns
- `gap`: Spectral gap
- `energy`: Ground state energy estimate
"""
function compute_exact_energy(params::Vector{Float64}, g::Float64, J::Float64, 
                               p::Int, row::Int, virtual_qubits::Int; optimizer=GreedyMethod())
    gates = build_unitary_gate(params, p, row, virtual_qubits; share_params=true)
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, virtual_qubits)
    
    X_cost = real(compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=optimizer))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=optimizer)
    ZZ_vert = real(ZZ_vert)
    ZZ_horiz = real(ZZ_horiz)
    
    energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz) 
    return gap, energy
end
