struct IndexStore{LT}
    indices::Vector{LT}
end
IndexStore() = IndexStore(Int[])
Base.length(store::IndexStore) = length(store.indices)
function newindex!(store::IndexStore{LT}) where LT <: Integer
    index = length(store) == 0 ? 1 : store.indices[end] + 1
    push!(store.indices, index)
    return index
end

function truncated_svd(tmat, dmax::Int, atol::Real)
    u, s, v = svd(tmat)
    dmax = min(searchsortedfirst(s, atol, rev=true), dmax, length(s))
    return u[:, 1:dmax], s[1:dmax], v'[1:dmax, :], sum(s[dmax+1:end])
end

function mps_dot_mpo(mps::MPS,mpo::MPO)
    @assert length(mps.tensors)==length(mpo.tensors)
    new_mps=deepcopy(mps)
    for i in 1:length(mps.tensors)
        mps_mpo=ein"mpjn,ijk->mipnk"(mpo.tensors[i],mps.tensors[i])     
        new_shape=(size(mps_mpo,1)*size(mps_mpo,2),
                   size(mps_mpo,3),
                   size(mps_mpo,4)*size(mps_mpo,5))
        new_mps.tensors[i]=reshape(mps_mpo,new_shape)
    end
    return new_mps  
end
  
function code_sandwich(bra::MPS, op::MPO, ket::MPS; optimizer=GreedyMethod())
    store = IndexStore()
    ixs_bra = Vector{Int}[]
    ixs_op = Vector{Int}[]
    ixs_ket = Vector{Int}[]
    firstidx_bra = newindex!(store)
    previdx_bra = firstidx_bra
    firstidx_op = newindex!(store)
    previdx_op = firstidx_op
    firstidx_ket = newindex!(store)
    previdx_ket = firstidx_ket
    nsite=length(bra.tensors)
    for k = 1:nsite
        physical_bra = newindex!(store)
        physical_ket = newindex!(store)
        nextidx_bra = k == nsite ? firstidx_bra : newindex!(store)
        nextidx_op = k == nsite ? firstidx_op : newindex!(store)
        nextidx_ket = k == nsite ? firstidx_ket : newindex!(store)
        push!(ixs_bra, [previdx_bra, physical_bra, nextidx_bra])
        push!(ixs_op, [previdx_op, physical_bra, physical_ket, nextidx_op])
        push!(ixs_ket, [previdx_ket, physical_ket, nextidx_ket])
        previdx_bra = nextidx_bra
        previdx_op = nextidx_op
        previdx_ket = nextidx_ket
    end
    ixs = [ixs_bra..., ixs_op..., ixs_ket...]
    size_dict = OMEinsum.get_size_dict(ixs, [bra.tensors..., op.tensors..., ket.tensors...])
    code=optimize_code(DynamicEinCode(ixs, Int[]), size_dict, optimizer)
    return code,code(conj.(bra.tensors)..., op.tensors..., ket.tensors...)[]
end


