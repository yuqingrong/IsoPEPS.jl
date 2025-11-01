"""
    exact_left_eigen(gate, nsites)

Compute the left eigenvector and spectral gap of the transfer matrix.

# Arguments
- `gate`: Quantum gate defining the channel
- `nsites`: Number of sites

# Returns
- `rho`: Normalized density matrix (fixed point)
- `gap`: Spectral gap (-log|λ₂|)

# Description
Constructs the transfer matrix from the gate and computes its spectral properties.
The spectral gap quantifies how quickly the channel approaches its fixed point.
"""
function exact_left_eigen(gate, nsites)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    
    _, T = contract_Elist([A for _ in 1:nsites], [conj(A) for _ in 1:nsites], nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    
    # Get second largest eigenvalue for gap
    λ₁ = partialsort(abs.(LinearAlgebra.eigen(T).values), 2; rev=true)
    gap = -log(abs(λ₁))
    
    # Verify normalization
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.0
    
    # Extract fixed point density matrix
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(nsites+1))), Int(sqrt(4^(nsites+1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    
    return rho, gap
end

"""
Transfer matrix contraction for tensor networks.

Provides efficient tensor network contraction routines for computing
transfer matrices in infinite PEPS.
"""

"""
    contract_Elist(tensor_ket, tensor_bra, row; optimizer=GreedyMethod())

Contract a list of tensors to form the transfer matrix.

# Arguments
- `tensor_ket`: List of ket tensors
- `tensor_bra`: List of bra tensors
- `row`: Number of rows
- `optimizer`: Contraction order optimizer (default: GreedyMethod())

# Returns
- `code`: Optimized einsum contraction code
- Result of the contraction

# Description
Constructs and contracts the tensor network representing the transfer matrix
by pairing ket and bra tensors. Automatically generates optimal contraction
orders for efficiency.
"""
function contract_Elist(tensor_ket, tensor_bra, row; optimizer=GreedyMethod())
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

    # Optimize contraction order
    size_dict = OMEinsum.get_size_dict(index, [tensor_ket..., tensor_bra...])
    code = optimize_code(DynamicEinCode(index, index_output), size_dict, optimizer)
    
    return code, code(tensor_ket..., tensor_bra...)
end

"""
    cost_X(rho, row, gate)

Compute X observable expectation by direct tensor contraction.

# Arguments
- `rho`: Density matrix
- `row`: Number of rows
- `gate`: Quantum gate

# Returns
Expectation value ⟨X⟩

# Description
More accurate than measurement-based approach, uses exact tensor contraction.
"""
function cost_X(rho, row, gate)
    nqubits = Int(log2(size(rho, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho, shape...)
    
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AX = ein"iabcd,ij -> jabcd"(A, Matrix(X))
    
    tensor_ket = [AX, A, A]
    tensor_bra = [conj(A), conj(A), conj(A)]
    _, list = contract_Elist(tensor_ket, tensor_bra, row)
    result = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list)

    return result[]
end

"""
    cost_ZZ(rho, row, gate)

Compute ZZ correlation by direct tensor contraction.

# Arguments
- `rho`: Density matrix
- `row`: Number of rows
- `gate`: Quantum gate

# Returns
ZZ correlation value ⟨Z₁Z₂⟩

# Description
Computes the expectation value by contracting the tensor network with
Z operators applied at different positions. Averages over all possible
nearest-neighbor configurations.
"""
function cost_ZZ(rho, row, gate)
    nqubits = Int(log2(size(rho, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho, shape...)
    
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AZ = ein"iabcd,ij -> jabcd"(A, Matrix(Z))
    
    # Compute three different configurations
    tensor_bra = [conj(A), conj(A), conj(A)]
    
    _, list1 = contract_Elist([AZ, AZ, A], tensor_bra, row)
    result1 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list1)
    
    _, list2 = contract_Elist([A, AZ, AZ], tensor_bra, row)
    result2 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list2)
    
    _, list3 = contract_Elist([AZ, A, AZ], tensor_bra, row)
    result3 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list3)
    
    return 2*(result1[] + result2[] + result3[]) / 3
end
