"""
    compute_transfer_spectrum(gates, row, nqubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)

Compute the transfer matrix spectrum and fixed point.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
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
function compute_transfer_spectrum(gates, row, nqubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)
    A_tensors = _gates_to_tensors(gates, row, nqubits)
    total_qubits = Int((row+1)*(nqubits-1)/2)
    matrix_size = 4^total_qubits
    
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
        code, _ = _build_transfer_contraction_code(A_tensors, row, total_qubits)
        tensor_ket = [A_tensors[i] for i in 1:row]
        tensor_bra = [conj(A_tensors[i]) for i in 1:row]
        
        # Define the linear map T*v
        function apply_transfer(v)
            # v is a vector of length 4^total_qubits
            # Reshape to tensor form for left indices
            v_tensor = reshape(v, ntuple(_ -> 2, 2*total_qubits)...)
            # Apply transfer matrix via contraction
            result = _apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_qubits)
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

"""
Build the contraction code for transfer matrix application (without computing the full matrix).
"""
function _build_transfer_contraction_code(A_tensors, row, total_qubits; optimizer=GreedyMethod())
    store = IndexStore()
    index_bra = Vector{Int}[]
    index_ket = Vector{Int}[]
    
    # Input vector indices (left side of transfer matrix)
    input_indices = [newindex!(store) for _ in 1:2*total_qubits]
    # Output vector indices (right side of transfer matrix)  
    output_indices = Int[]
    
    # Initialize boundary indices
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    previdx_down_ket = first_down_ket
    previdx_down_bra = first_down_bra
    
    # Build index structure for each row
    for i in 1:row
        phyidx = newindex!(store)
        left_ket = newindex!(store)
        right_ket = newindex!(store)
        left_bra = newindex!(store)
        right_bra = newindex!(store)
        
        next_up_ket = i == 1 ? first_up_ket : previdx_down_ket
        next_up_bra = i == 1 ? first_up_bra : previdx_down_bra
        next_down_ket = i == 1 ? first_down_ket : newindex!(store)
        next_down_bra = i == 1 ? first_down_bra : newindex!(store)
        
        push!(index_ket, [phyidx, next_down_ket, right_ket, next_up_ket, left_ket])
        push!(index_bra, [phyidx, next_down_bra, right_bra, next_up_bra, left_bra])
        
        previdx_down_ket = next_down_ket
        previdx_down_bra = next_down_bra
    end

    # Output indices (right side)
    append!(output_indices, index_ket[row][2])
    append!(output_indices, [index_ket[i][3] for i in 1:row])
    append!(output_indices, index_bra[row][2])
    append!(output_indices, [index_bra[i][3] for i in 1:row])
    
    # Connect input indices to left side
    # The input vector connects to the left indices of the transfer matrix
    left_indices_ket = [index_ket[1][4], [index_ket[i][end] for i in 1:row]...]
    left_indices_bra = [index_bra[1][4], [index_bra[i][end] for i in 1:row]...]
    
    # Map input_indices to the left side
    for (j, idx) in enumerate(left_indices_ket)
        # Replace with input indices
        for k in 1:row
            for l in 1:5
                if index_ket[k][l] == idx
                    index_ket[k][l] = input_indices[j]
                end
            end
        end
    end
    for (j, idx) in enumerate(left_indices_bra)
        for k in 1:row
            for l in 1:5
                if index_bra[k][l] == idx
                    index_bra[k][l] = input_indices[total_qubits + j]
                end
            end
        end
    end
    
    # Build tensor index list
    index_tensors = [index_ket..., index_bra...]
    # Add input vector
    push!(index_tensors, collect(input_indices))
    
    size_dict = OMEinsum.get_size_dict(index_tensors, [A_tensors..., [conj(A_tensors[i]) for i in 1:row]..., 
                                                        zeros(ComplexF64, ntuple(_ -> 2, 2*total_qubits)...)])
    code = optimize_code(DynamicEinCode(index_tensors, output_indices), size_dict, optimizer)
    
    return code, output_indices
end

"""
Apply transfer matrix to a vector using precomputed contraction code.
"""
function _apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_qubits)
    result = code(tensor_ket..., tensor_bra..., v_tensor)
    return reshape(result, ntuple(_ -> 2, 2*total_qubits)...)
end

"""
    compute_single_transfer(gates, nqubits)

Compute transfer matrix for a single gate (row=1 case).
"""
function compute_single_transfer(gates, nqubits)
    A_size = ntuple(i -> 2, 2*nqubits)
    indices = (ntuple(_ -> Colon(), nqubits)..., 1, ntuple(_ -> Colon(), nqubits-1)...)
    A_single = reshape(gates[1], A_size)[indices...]
    
    T = ein"iabcd,iefgh-> abefcdgh"(A_single, conj(A_single))
    T = reshape(T, 4^(nqubits-1), 4^(nqubits-1))
    
    eigenvalues = sort(abs.(LinearAlgebra.eigen(T).values))
    gap = -log(eigenvalues[end-1])
    @assert eigenvalues[end] ≈ 1.0 "Largest eigenvalue should be 1"
    
    fixed_point = reshape(LinearAlgebra.eigen(T).vectors[:, end], 
                          Int(sqrt(4^(nqubits-1))), Int(sqrt(4^(nqubits-1))))
    rho = fixed_point ./ tr(fixed_point)
    
    return rho, gap, eigenvalues
end

"""
    contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=GreedyMethod())

Contract tensors to form the transfer matrix.

# Arguments
- `tensor_ket`: Ket tensors (row of them)
- `tensor_bra`: Bra tensors (conjugates)
- `row`: Number of rows
- `optimizer`: Contraction order optimizer

# Returns
- `code`: Optimized contraction code
- `result`: Contracted transfer matrix
"""
function contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=GreedyMethod())
    store = IndexStore()
    index_bra = Vector{Int}[]
    index_ket = Vector{Int}[]
    index_output = Int[]
    
    # Initialize boundary indices
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    previdx_down_ket = first_down_ket
    previdx_down_bra = first_down_bra
    
    # Build index structure for each row
    for i in 1:row
        phyidx = newindex!(store)
        left_ket = newindex!(store)
        right_ket = newindex!(store)
        left_bra = newindex!(store)
        right_bra = newindex!(store)
        
        next_up_ket = i == 1 ? first_up_ket : previdx_down_ket
        next_up_bra = i == 1 ? first_up_bra : previdx_down_bra
        next_down_ket = i == 1 ? first_down_ket : newindex!(store)
        next_down_bra = i == 1 ? first_down_bra : newindex!(store)
        
        push!(index_ket, [phyidx, next_down_ket, right_ket, next_up_ket, left_ket])
        push!(index_bra, [phyidx, next_down_bra, right_bra, next_up_bra, left_bra])
        
        previdx_down_ket = next_down_ket
        previdx_down_bra = next_down_bra
    end

    # Collect output indices
    append!(index_output, index_ket[row][2])
    append!(index_output, [index_ket[i][3] for i in 1:row])
    append!(index_output, index_bra[row][2])
    append!(index_output, [index_bra[i][3] for i in 1:row])

    append!(index_output, index_ket[1][4])
    append!(index_output, [index_ket[i][end] for i in 1:row])
    append!(index_output, index_bra[1][4])
    append!(index_output, [index_bra[i][end] for i in 1:row])
    
    index = [index_ket..., index_bra...]

    size_dict = OMEinsum.get_size_dict(index, [tensor_ket..., tensor_bra...])
    code = optimize_code(DynamicEinCode(index, index_output), size_dict, optimizer)
    
    return code, code(tensor_ket..., tensor_bra...)
end

"""
    compute_X_expectation(rho, gates, row, nqubits; optimizer=GreedyMethod())

Compute ⟨X⟩ expectation value from fixed point density matrix.

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
- `optimizer`: Contraction optimizer

# Returns
Average X expectation value across all sites.
"""
function compute_X_expectation(rho, gates, row, nqubits; optimizer=GreedyMethod())
    total_qubits = Int((row+1)*(nqubits-1)/2)
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = _gates_to_tensors(gates, row, nqubits)
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
    compute_ZZ_expectation(rho, gates, row, nqubits; optimizer=GreedyMethod())

Compute ⟨ZZ⟩ expectation values (vertical and horizontal bonds).

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
- `optimizer`: Contraction optimizer

# Returns
- `ZZ_vertical`: Vertical bond ⟨ZᵢZᵢ₊₁⟩
- `ZZ_horizontal`: Horizontal bond ⟨ZᵢZᵢ₊ᵣₒw⟩
"""
function compute_ZZ_expectation(rho, gates, row, nqubits; optimizer=GreedyMethod())
    total_qubits = Int((row+1)*(nqubits-1)/2)
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = _gates_to_tensors(gates, row, nqubits)
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
Convert gate matrices to tensor form for contraction.
"""
function _gates_to_tensors(gates, row, nqubits)
    A_size = ntuple(i -> 2, 2 * nqubits)
    indices = (ntuple(_ -> Colon(), nqubits)..., 1, ntuple(_ -> Colon(), nqubits-1)...)
    return [reshape(gates[i], A_size)[indices...] for i in 1:row]
end

"""
    compute_exact_energy(params, g, J, p, row, nqubits; optimizer=GreedyMethod())

Compute exact energy from parameters using tensor contraction.

# Arguments
- `params`: Parameter vector
- `g`: Transverse field strength
- `J`: Coupling strength
- `p`: Number of circuit layers
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
- `optimizer`: Contraction optimizer

# Returns
- `gap`: Spectral gap
- `energy`: Ground state energy estimate
"""
function compute_exact_energy(params::Vector{Float64}, g::Float64, J::Float64, 
                               p::Int, row::Int, nqubits::Int; optimizer=GreedyMethod())
    gates = build_unitary_gate(params, p, row, nqubits; share_params=true)
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    X_cost = real(compute_X_expectation(rho, gates, row, nqubits; optimizer=optimizer))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, nqubits; optimizer=optimizer)
    ZZ_vert = real(ZZ_vert)
    ZZ_horiz = real(ZZ_horiz)
    
    energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz) 
    return gap, energy
end
