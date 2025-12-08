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

function exact_left_eigen(A_matrix, row, nqubits)
    A_tensors = _A_matrix2A_tensors(A_matrix, row, nqubits)
    total_qubits = Int((row+1)*(nqubits-1)/2)
    _, T = contract_Elist([A_tensors[i] for i in 1:row], [conj(A_tensors[i]) for i in 1:row], row)
    T = reshape(T, 4^total_qubits, 4^(total_qubits))    # D=2
    eigenvalues = sort(abs.(LinearAlgebra.eigen(T).values))
    gap = -log(eigenvalues[end-1])  # Second largest eigenvalue
    @assert eigenvalues[end] ≈ 1.0
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits)))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    return rho, gap, eigenvalues
end

function single_transfer(A_matrix, nqubits)
    A_size = ntuple(i -> 2, 2*nqubits)
    indices = (ntuple(_ -> Colon(), nqubits)..., 1, ntuple(_ -> Colon(), nqubits-1)...)
    A_single = reshape(A_matrix[1], A_size)[indices...]
    T = ein"iabcd,iefgh-> abefcdgh"(A_single, conj(A_single))
    T = reshape(T, 4^(nqubits-1), 4^(nqubits-1))    # D=2
    eigenvalues = sort(abs.(LinearAlgebra.eigen(T).values))
    gap = -log(eigenvalues[end-1])  # Second largest eigenvalue
    @assert eigenvalues[end] ≈ 1.0
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(nqubits-1))), Int(sqrt(4^(nqubits-1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    return rho, gap, eigenvalues
end

function cost_singleop(rho, A_matrix, nqubits, op)
    A_size = ntuple(i -> 2, 2*nqubits)
    env_size = ntuple(i -> 2, 2*(nqubits-1))
    indices = (ntuple(_ -> Colon(), nqubits)..., 1, ntuple(_ -> Colon(), nqubits-1)...)
    A = reshape(A_matrix[1], A_size)[indices...]
    rho = reshape(rho, env_size)
    R = reshape(Matrix(I,4^(nqubits-1),4^(nqubits-1)), env_size)

    store = IndexStore()
    index_A = [newindex!(store) for _ in 1:2*nqubits-1]
    index_conjA = [newindex!(store) for i in 1:2*nqubits-1]
    index_op = [index_A[1], index_conjA[1]]
    index_rho = [index_A[Int(end+1)/2+1:end]..., index_conjA[Int(end+1)/2+1:end]...]
    index_R = [index_A[2:Int(end+1)/2]..., index_conjA[2:Int(end+1)/2]...]

    index = [index_A..., index_conjA..., index_op..., index_rho..., index_R...]

    # Optimize contraction order
    size_dict = OMEinsum.get_size_dict(index, [A, conj(A), op, rho, R])
    code = optimize_code(DynamicEinCode(index, Int[]), size_dict, optimizer)
    
    return code, code(A, conj(A), op, rho, R)[]
end

function cost_doubleop_ver(rho, A_matrix)
    A = reshape(A_matrix[1], (2, 2, 2, 2))[:, :, 1, :]
    mid_res1 = ein"bd, iab, ij, jcd -> ac"(rho, A, Matrix(Z), conj(A))
    ZZ_exp1 = ein"bd, iab, ij, jad ->"(mid_res1, A, Matrix(Z), conj(A))

    mid_res2 = ein"bd, iab, icd -> ac"(mid_res1, A, conj(A))
    mid_res3 = ein"bd, iab, icd -> ac"(mid_res2, A, conj(A))
    mid_res4 = ein"bd, iab, icd -> ac"(mid_res3, A, conj(A))    #TODO: relation with row
    ZZ_exp2 = ein"bd, iab, ij, jad ->"(mid_res4, A, Matrix(Z), conj(A))

    return ZZ_exp1[], ZZ_exp2[]
end

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

function cost_X(rho, A_matrix, row, nqubits; optimizer=GreedyMethod())
    total_qubits = Int((row+1)*(nqubits-1)/2)
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I,Int(sqrt(4^total_qubits)),Int(sqrt(4^total_qubits))), env_size)
    A_tensors = _A_matrix2A_tensors(A_matrix, row, nqubits)
    # Prepare AX tensors for each position (apply X operator)
    AX_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], Matrix(X)) for i in 1:row]   #TODO: still 3 qubits per gate
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    
    # Compute expectation value at each position and average
    results = map(1:row) do pos
        tensor_ket = [i == pos ? AX_tensors[i] : A_tensors[i] for i in 1:row]
        _, list = contract_Elist(tensor_ket, tensor_bra, row)
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

function _compute_ZZ_contraction(tensor_ket_a, tensor_ket_b, tensor_bra, rho, R, row, total_qubits, optimizer)
    # Helper function to compute ZZ contraction for a pair of tensor kets.
    _, list1 = contract_Elist(tensor_ket_a, tensor_bra, row)
    _, list2 = contract_Elist(tensor_ket_b, tensor_bra, row)
    
    store = IndexStore()
    index_list1 = [newindex!(store) for _ in 1:4*total_qubits]
    index_list2 = [[newindex!(store) for _ in 1:2*total_qubits]..., index_list1[1:2*total_qubits]...]
    
    index_rho = index_list1[2*total_qubits+1:4*total_qubits]  # Last half of list1
    index_R = index_list2[1:2*total_qubits]  # First half of list2
    index = [index_list1, index_list2, index_rho, index_R]
    
    size_dict = OMEinsum.get_size_dict(index, [list1, list2, rho, R])
    code = optimize_code(DynamicEinCode(index, Int[]), size_dict, optimizer)
    return code(list1, list2, rho, R)[]
end

function cost_ZZ(rho, A_matrix, row, nqubits; optimizer=GreedyMethod())
    total_qubits = Int((row+1)*(nqubits-1)/2)
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I,Int(sqrt(4^total_qubits)),Int(sqrt(4^total_qubits))), env_size)
    A_tensors = _A_matrix2A_tensors(A_matrix, row, nqubits)
    # Prepare AZ tensors for each position (apply Z operator)
    AZ_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], Matrix(Z)) for i in 1:row]
    tensor_ket = A_tensors
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]

    tensor_ket1 = [i == 1 || i == 2 ? AZ_tensors[i] : A_tensors[i] for i in 1:row]
    tensor_ket2 = [i == 1 ? AZ_tensors[i] : A_tensors[i] for i in 1:row]

    # Compute ZZ expectation values
    ZZ_ver = _compute_ZZ_contraction(tensor_ket1, tensor_ket, tensor_bra, rho, R, row, total_qubits, optimizer)
    ZZ_hor = _compute_ZZ_contraction(tensor_ket2, tensor_ket2, tensor_bra, rho, R, row, total_qubits, optimizer)
    
    return ZZ_ver, ZZ_hor
end

function _A_matrix2A_tensors(A_matrix, row, nqubits)
    total_qubits = Int((row+1)*(nqubits-1)/2)
    A_size = ntuple(i -> 2, 2 * nqubits)
    indices = (ntuple(_ -> Colon(), nqubits)..., 1, ntuple(_ -> Colon(), nqubits-1)...)
    A_tensors = [reshape(A_matrix[i], A_size)[indices...] for i in 1:row]
    return A_tensors
end

function exact_E_from_params(g::Float64, J::Float64, p::Int, row::Int, nqubits::Int; data_dir="data", optimizer=GreedyMethod())
    params_file = joinpath(data_dir, "compile_params_history_row=$(row)_g=$(g).dat")
    params = parse.(Float64, split(readlines(params_file)[end]))
    params = [0.21306935871527355, 0.934507440458287, 0.05229259440633531, 0.9113544925693889, 0.19728530675603592, 1.4359355474671014, 0.016551505630408005, 0.8767865530009604, 0.2266873583865921, 2.451715699073036, 0.0005442109650379405, 0.7591169490501166, 0.19447280296550473, 0.014583435860538011, 0.0526894648388774, 0.8146217029061544, 0.2813917600942367, 0.5794862606743785]
    A_matrix = build_gate_from_params(params, p, row, nqubits; share_params=true)
    rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
    X_cost = real(cost_X(rho, A_matrix, row, nqubits))
    ZZ_exp1, ZZ_exp2 = cost_ZZ(rho, A_matrix, row, nqubits)
    ZZ_exp1 = real(ZZ_exp1)
    ZZ_exp2 = real(ZZ_exp2)
    energy = -g*X_cost - J*(row == 1 ? ZZ_exp2 : ZZ_exp1 + ZZ_exp2) 
    @show X_cost, ZZ_exp1, ZZ_exp2
    return gap, energy
end