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

