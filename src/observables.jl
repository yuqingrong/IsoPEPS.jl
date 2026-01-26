# =============================================================================
# Observable Expectation Values
# =============================================================================
# Functions for computing expectation values from the transfer matrix fixed point

"""
    compute_single_expectation(rho, gates, row, virtual_qubits, observable; optimizer=GreedyMethod())

Compute single-site expectation value ⟨O⟩ from fixed point density matrix.

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `observable`: Either `:X` or `:Z` (or a 2x2 matrix)
- `optimizer`: Contraction optimizer

# Returns
Average expectation value across all sites.
"""
function compute_single_expectation(rho, gates, row, virtual_qubits, observable::Union{Symbol,AbstractMatrix}; optimizer=GreedyMethod())
    # Get the operator matrix
    O = if observable isa Symbol
        observable == :X ? Matrix(X) : (observable == :Z ? Matrix(Z) : error("Unknown observable: $observable"))
    else
        observable
    end
    
    # total_qubits must match compute_transfer_spectrum: bond_dim^(2*total_legs) = 2^(2*v*(row+1))
    total_qubits = virtual_qubits * (row + 1)
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    AO_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], O) for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    
    results = map(1:row) do pos
        tensor_ket = [i == pos ? AO_tensors[i] : A_tensors[i] for i in 1:row]
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
    compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())

Compute ⟨X⟩ expectation value from fixed point density matrix.
Wrapper around compute_single_expectation for backward compatibility.
"""
function compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    compute_single_expectation(rho, gates, row, virtual_qubits, :X; optimizer=optimizer)
end

"""
    compute_Z_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())

Compute ⟨Z⟩ expectation value from fixed point density matrix.
"""
function compute_Z_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    compute_single_expectation(rho, gates, row, virtual_qubits, :Z; optimizer=optimizer)
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
    # total_qubits must match compute_transfer_spectrum: bond_dim^(2*total_legs) = 2^(2*v*(row+1))
    total_qubits = virtual_qubits * (row + 1)
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
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
    _contract_ZZ(tensor_ket_a, tensor_ket_b, tensor_bra, rho, R, row, total_qubits, optimizer)

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
    compute_exact_energy(params, g, J, p, row, nqubits; optimizer=GreedyMethod())

Compute exact energy from parameters using tensor contraction.

# Arguments
- `params`: Parameter vector
- `g`: Transverse field strength
- `J`: Coupling strength
- `p`: Number of circuit layers
- `row`: Number of rows
- `nqubits`: Number of qubits per gate (e.g., 3 for 8x8 gates)
- `optimizer`: Contraction optimizer

# Returns
- `gap`: Spectral gap
- `energy`: Ground state energy estimate
"""
function compute_exact_energy(params::Vector{Float64}, g::Float64, J::Float64, 
                               p::Int, row::Int, nqubits::Int; optimizer=GreedyMethod())
    virtual_qubits = (nqubits - 1) ÷ 2
    gates = build_unitary_gate(params, p, row, nqubits; share_params=true)
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    # Note: compute_X/ZZ_expectation expect virtual_qubits, not nqubits
    X_cost = real(compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=optimizer))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=optimizer)
    ZZ_vert = real(ZZ_vert)
    ZZ_horiz = real(ZZ_horiz)
    
    energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz) 
    return gap, energy
end

