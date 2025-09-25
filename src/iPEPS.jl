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
    #E = reshape(ein"iabcd,iefgh -> abefcdgh"(A, conj(A)), 4,4,4,4)
    _, T = contract_Elist(A, nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.
    fixed_point_rho = reshape(LinearAlgebra.eigen(T).vectors[:, end], Int(sqrt(4^(nsites+1))), Int(sqrt(4^(nsites+1))))
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    @show isapprox(rho, rho') 
    return rho
end

# contract list of A and A^dagger to form transfer matrix  
function contract_Elist(A, row; optimizer=GreedyMethod())
    store = IndexStore()
    index_bra = Vector{Int}[]
    index_ket = Vector{Int}[]
    index_output = Int[]
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    previdx_down_ket = first_down_ket
    previdx_down_bra = first_down_bra
    for i in 1:row
        phyidx = newindex!(store)
        left_ket = newindex!(store)
        right_ket = newindex!(store)
        left_bra = newindex!(store)
        right_bra = newindex!(store)
        next_up_ket = i ==1 ? first_up_ket : previdx_down_ket
        next_up_bra = i ==1 ? first_up_bra : previdx_down_bra
        next_down_ket = i == 1 ? first_down_ket : newindex!(store)
        next_down_bra = i == 1 ? first_down_bra : newindex!(store)
        push!(index_ket, [phyidx, next_down_ket, right_ket, next_up_ket, left_ket])
        push!(index_bra, [phyidx, next_down_bra, right_bra, next_up_bra, left_bra])
        previdx_down_ket = next_down_ket
        previdx_down_bra = next_down_bra
    end

    append!(index_output, index_ket[row][2])
    append!(index_output, [index_ket[i][3] for i in 1:row])
    append!(index_output, index_bra[row][2])
    append!(index_output, [index_bra[i][3] for i in 1:row])

    append!(index_output, index_ket[1][4])
    append!(index_output, [index_ket[i][end] for i in 1:row])
    append!(index_output, index_bra[1][4])
    append!(index_output, [index_bra[i][end] for i in 1:row])
    index=[index_ket...,index_bra...]
    tensors_ket = [A for _ in 1:row]
    tensors_bra = [conj(A) for _ in 1:row]
    size_dict = OMEinsum.get_size_dict(index, [tensors_ket...,tensors_bra...])
    code = optimize_code(DynamicEinCode(index, index_output), size_dict, optimizer)
    return code, code(tensors_ket...,tensors_bra...)
end

function exact_energy_PEPS(d::Int,D::Int,g::Float64,row::Int)
    # Create MPS with 3-site unit cell to match InfiniteStrip(3)
    mps = InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H0 = transverse_field_ising(InfiniteStrip(row); g=g)
    psi,_= find_groundstate(mps, H0, VUMPS())
    E = real(expectation_value(psi,H0))/row
    return E
end