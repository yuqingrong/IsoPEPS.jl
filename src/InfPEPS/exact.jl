"""
    exact_left_eigen(gate, nsites)

Compute the left eigenvector and spectral gap of the transfer matrix.

# Arguments
- `gate`: Quantum gate defining the channel
- `nsites`: Number of sites

# Returns
- `rho`: Normalized density matrix (fixed point)
- `gap`: Spectral gap (-log|λ₂|) where λ₂ is the second largest eigenvalue

# Description
Constructs the transfer matrix from the gate and computes its spectral properties.
The spectral gap quantifies how quickly the channel approaches its fixed point.
"""

#index of single A:
#          4  5
#           \ |
#         6 -|A|- 3
#             | \
#             2  1

function exact_left_eigen(A_matrix, row)
    A_tensors = Vector{Array{ComplexF64, 5}}(undef, row)
    for i in 1:row
        A_tensors[i] = reshape(A_matrix[i], (2, 2, 2, 2, 2, 2))[:, :, :, 1, :, :]   # TODO:not suits for arbitrary nqubits
    end
    
    _, T = contract_Elist([A_tensors[i] for i in 1:row], [conj(A_tensors[i]) for i in 1:row], row)
    T = reshape(T, 4^(row+1), 4^(row+1))    # D=2
    eigenvalues = sort(abs.(LinearAlgebra.eigen(T).values))
    gap = -log(eigenvalues[end-1])  # Second largest eigenvalue
    @assert eigenvalues[end] ≈ 1.0
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(row+1))), Int(sqrt(4^(row+1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    return rho, gap, eigenvalues
end

function single_transfer(A_matrix, nqubits)
    A_single = reshape(A_matrix[1], (2, 2, 2, 2))[:, :, 1, :]   # TODO:not suits for arbitrary nqubits
    T = ein"iab,icd-> acbd"(A_single, conj(A_single))
    T = reshape(T, 4, 4)    # D=2
    eigenvalues = sort(abs.(LinearAlgebra.eigen(T).values))
    gap = -log(eigenvalues[end-1])  # Second largest eigenvalue
    @assert eigenvalues[end] ≈ 1.0
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4)), Int(sqrt(4)))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    return rho, gap, eigenvalues
end

function cost_X_single(rho, A_matrix)
    A = reshape(A_matrix[1], (2, 2, 2, 2))[:, :, 1, :]
    X_exp = ein"bd, iab, ij, jad ->"(rho, A, Matrix(X), conj(A))
    return X_exp[]
end

function cost_ZZ_single(rho, A_matrix)
    A = reshape(A_matrix[1], (2, 2, 2, 2))[:, :, 1, :]
    mid_res1 = ein"bd, iab, ij, jcd -> ac"(rho, A, Matrix(Z), conj(A))
    ZZ_exp1 = ein"bd, iab, ij, jad ->"(mid_res1, A, Matrix(Z), conj(A))

    mid_res2 = ein"bd, iab, icd -> ac"(mid_res1, A, conj(A))
    mid_res3 = ein"bd, iab, icd -> ac"(mid_res2, A, conj(A))
    mid_res4 = ein"bd, iab, icd -> ac"(mid_res3, A, conj(A))    #TODO: relation with row
    ZZ_exp2 = ein"bd, iab, ij, jad ->"(mid_res4, A, Matrix(Z), conj(A))

    return ZZ_exp1[], ZZ_exp2[]
end

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

    tensor_bra = [conj(A), conj(A), conj(A)]
    tensor_ket1 = [AX, A, A]
    _, list1 = contract_Elist(tensor_ket1, tensor_bra, row)
    result1 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list1)

    tensor_ket2 = [A, AX, A]
    _, list2 = contract_Elist(tensor_ket2, tensor_bra, row)
    result2 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list2)

    tensor_ket3 = [A, A, AX]
    _, list3 = contract_Elist(tensor_ket3, tensor_bra, row)
    result3 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list3)
    return (result1[] + result2[] + result3[]) / 3
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
    
    _, list = contract_Elist([AZ, AZ, A], tensor_bra, row)
    Z_value = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list)

    _, list1 = contract_Elist([AZ, AZ, A], tensor_bra, row)
    result1 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list1)
    
    _, list2 = contract_Elist([A, AZ, AZ], tensor_bra, row)
    result2 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list2)
    
    _, list3 = contract_Elist([AZ, A, AZ], tensor_bra, row)
    result3 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list3)
    
    _, list4 = contract_Elist([AZ, A, A], tensor_bra, row)
    rho2 = ein"abcdefgh,ijklmnopabcdefgh ->ijklmnop"(rho, list4)
    _, list5 = contract_Elist([AZ, A, A], tensor_bra, row)
    result4 = ein"abcdefgh,ijklijklabcdefgh ->"(rho2, list5)

    _, list6 = contract_Elist([A, AZ, A], tensor_bra, row)
    rho2 = ein"abcdefgh,ijklmnopabcdefgh ->ijklmnop"(rho, list6)
    _, list7 = contract_Elist([A, AZ, A], tensor_bra, row)
    result5 = ein"abcdefgh,ijklijklabcdefgh ->"(rho2, list7)

    _, list8 = contract_Elist([A, A, AZ], tensor_bra, row)
    rho2 = ein"abcdefgh,ijklmnopabcdefgh ->ijklmnop"(rho, list8)
    _, list9 = contract_Elist([A, A, AZ], tensor_bra, row)
    result6 = ein"abcdefgh,ijklijklabcdefgh ->"(rho2, list9)

    return real(Z_value[]), (result1[] + result2[] + result3[] + result4[] + result5[] + result6[]) / 3
end
