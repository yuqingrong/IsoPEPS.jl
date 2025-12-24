"""
    compute_transfer_spectrum(gates, row, nqubits)

Compute the transfer matrix spectrum and fixed point.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `nqubits`: Number of qubits per gate

# Returns
- `rho`: Fixed point density matrix (normalized)
- `gap`: Spectral gap = -log|λ₂| where λ₂ is second largest eigenvalue
- `eigenvalues`: Sorted eigenvalue magnitudes

# Description
Constructs the transfer matrix from gates and computes its spectral properties.
The spectral gap quantifies how quickly the channel converges to its fixed point.
"""
function compute_transfer_spectrum(gates, row, nqubits)
    A_tensors = _gates_to_tensors(gates, row, nqubits)
    total_qubits = Int((row+1)*(nqubits-1)/2)
    
    _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                     [conj(A_tensors[i]) for i in 1:row], row)
    T = reshape(T, 4^total_qubits, 4^total_qubits)
    
    eigenvalues = sort(abs.(LinearAlgebra.eigen(T).values))
    gap = -log(eigenvalues[end-1])
    @assert eigenvalues[end] ≈ 1.0 "Largest eigenvalue should be 1"
    
    fixed_point = reshape(LinearAlgebra.eigen(T).vectors[:, end], 
                          Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits)))
    rho = fixed_point ./ tr(fixed_point)
    
    return rho, gap, eigenvalues
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
