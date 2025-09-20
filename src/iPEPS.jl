function iterate_channel_PEPS(gate, niters, row)
    rho = density_matrix(zero_state(row+1))
    for i in 1:niters
        for j in 1:row
            rho_p = density_matrix(zero_state(1))
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate))       
            rho = partial_tr(rho, 1) |> normalize!
        end
        @info "eigenvalues of ρ = " eigen(Hermitian(rho.state)).values
        @show isapprox(rho.state, rho.state') 
    end
    return rho.state
end

function exact_left_eigen(gate, nsites)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    E = reshape(ein"iabcd,iefgh -> abefcdgh"(A, conj(A)), 4,4,4,4)
    _, T = contract_Elist(E, nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.
    @show LinearAlgebra.eigen(T).values[end], LinearAlgebra.eigen(T).values[end-1]
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(nsites+1))), Int(sqrt(4^(nsites+1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    @show isapprox(rho, rho') 
    return rho
end

# contract the horizontal indices. output: all top*bottom arrays
function contract_Elist(E, row; optimizer=GreedyMethod())
    store = IndexStore()
    index = Vector{Int}[]
    first_right = newindex!(store)
    first_left = newindex!(store)
    previdx_right = first_right
    for i in 1:row
        topidx = newindex!(store)
        bottomidx = newindex!(store)
        next_right = i == 1 ? first_right : newindex!(store)
        next_left = i == 1 ? first_left : previdx_right
        push!(index, [next_right, topidx, next_left, bottomidx])
        previdx_right = next_right
    end
    output_indices = Int[]
    push!(output_indices, index[row][1])
    for i in 1:row
        topidx = index[i][2]    
        push!(output_indices, topidx)
    end
    push!(output_indices, first_left)
    for i in 1: row
        bottomidx = index[i][4] 
        push!(output_indices,bottomidx)
    end
    
    tensors = [E for _ in 1:row]
    size_dict = OMEinsum.get_size_dict(index, tensors)
    code = optimize_code(DynamicEinCode(index, output_indices), size_dict, optimizer)
    return code, code(tensors...)
end

