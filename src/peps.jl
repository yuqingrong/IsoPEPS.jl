const OrderedEinCode{LT} = Union{NestedEinsum{LT}, SlicedEinsum{LT,<:NestedEinsum{LT}}}
abstract type PEPS{T,NF,LT} <:AbstractRegister{NF} end

struct GeneralPEPS{T,NF,LT<:Union{Int,Char},Ein<:OrderedEinCode} <: PEPS{T,NF,LT}
    physical_labels::Vector{LT}
    virtual_labels::Vector{LT}

    vertex_labels::Vector{Vector{LT}}
    vertex_tensors::Vector{<:AbstractArray{T}}
    max_index::LT

    # optimized contraction codes
    code_statetensor::Ein
    code_inner_product::Ein
    
    D::Int
end

function GeneralPEPS{NF}(
    physical_labels::Vector{LT},   
    virtual_labels::Vector{LT},

    vertex_labels::Vector{Vector{LT}},
    vertex_tensors::Vector{<:AbstractArray{T}},
    max_index::LT,

    code_statetensor::Ein,
    code_inner_product::Ein,

    D::Int
    ) where {LT,T,NF,Ein}
    GeneralPEPS{T,NF,LT,Ein}(physical_labels, virtual_labels, vertex_labels, vertex_tensors, max_index, code_statetensor, code_inner_product, D)
end

function GeneralPEPS{NF}(vertex_labels::AbstractVector{<:AbstractVector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}}, virtual_labels::AbstractVector{LT}, D::Int, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {LT,T,NF}
    physical_labels = [vl[findall(∉(virtual_labels), vl)[]] for vl in vertex_labels]
                                                                            
    max_ind = max(maximum(physical_labels), maximum(virtual_labels))

    # optimal contraction orders
    optcode_statetensor, optcode_inner_product = _optimized_code(vertex_labels, physical_labels, virtual_labels, max_ind, NF, D, optimizer, simplifier )
    GeneralPEPS{NF}(physical_labels, virtual_labels, vertex_labels, vertex_tensors, max_ind, optcode_statetensor, optcode_inner_product, D)
end

function convert_type(::Type{T}, peps::GeneralPEPS{T2, NF}) where {T, T2, NF}
    p = GeneralPEPS{NF}(peps.physical_labels,
            peps.virtual_labels,
            peps.vertex_labels,
            map(t -> T.(t), peps.vertex_tensors),
            peps.max_index,
            peps.code_statetensor,
            peps.code_inner_product,
            peps.D)
    @show typeof(p)
    return p
end

function _optimized_code(alllabels, physical_labels::AbstractVector{LT}, virtual_labels, max_ind, nflavor, D, optimizer, simplifier) where LT
    code_statetensor = EinCode(alllabels, physical_labels)
    size_dict = Dict([[l=>nflavor for l in physical_labels]..., [l=>D for l in virtual_labels]...])
    optcode_statetensor = optimize_code(code_statetensor, size_dict, optimizer, simplifier)  # generate a good contraction order according to the label and dimension.
    rep = [l=>max_ind+i for (i, l) in enumerate(virtual_labels)]  # max_ind+i: for virtual_labels of the other peps
    merge!(size_dict, Dict([l.second=>D for l in rep]))
    code_inner_product = EinCode([alllabels..., [replace(l, rep...) for l in alllabels]...], LT[])
    
    optcode_inner_product = optimize_code(code_inner_product, size_dict, optimizer, simplifier)

    return optcode_statetensor, optcode_inner_product
end

function inner_product(p1::PEPS, p2::PEPS)
    p1c = conj(p1)
    p1.code_inner_product(alltensors(p1c)..., alltensors(p2)...) 
end

function Base.conj(peps::PEPS)
    replace_tensors(peps, conj.(alltensors(peps)))
end

function replace_tensors(peps::GeneralPEPS{T,NF}, tensors) where {T,NF}
    GeneralPEPS{NF}(peps.physical_labels, peps.virtual_labels,
        peps.vertex_labels, tensors, peps.max_index,
        peps.code_statetensor, peps.code_inner_product, peps.D
    )
end

function zero_peps(::Type{T}, g, D::Int, nflavor::Int, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where T
    virtual_labels = collect(nv(g)+1:nv(g)+ne(g))  # nv(g): number of vertices; ne(g): number of edges
    vertex_labels = Vector{Int}[] 
    vertex_tensors = Array{T}[]
    edge_map = Dict(zip(edges(g), virtual_labels))  # edges(g) returns edge between 2 vertices, [(1,2),(2,3)]
    for i=1:nv(g)
        push!(vertex_labels, [i,[get(edge_map, SimpleEdge(i,nb), get(edge_map,SimpleEdge(nb,i),0)) for nb in neighbors(g, i)]...])  # write physical_labels and the corresponding virtual_labels together.
    
        t = zeros(T, nflavor, fill(D, degree(g, i))...)  
        t[1] = 1  # normalization?
        push!(vertex_tensors, t)
    end
    GeneralPEPS{nflavor}(vertex_labels, vertex_tensors, virtual_labels, D, optimizer, simplifier)
end

function rand_peps(::Type{T}, g, D::Int, nflavor::Int, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where T
    randn!(zero_peps(T, g, D, nflavor, optimizer, simplifier))
end

function Random.randn!(peps::GeneralPEPS)    
    for t in alltensors(peps)
        randn!(t)
    end
    return peps
end

Base.vec(peps::PEPS) = vec(statetensor(peps))
function statetensor(peps::PEPS)
    peps.code_statetensor(alltensors(peps)...)  
end
Yao.statevec(peps::PEPS) = vec(peps)

alllabels(s::GeneralPEPS) = s.vertex_labels
alltensors(s::GeneralPEPS) = s.vertex_tensors


function apply_onsite!(peps::GeneralPEPS{T,NF,LT}, i, mat::AbstractMatrix) where {T,NF,LT}
    @assert size(mat, 1) == size(mat, 2)
    ti = peps.vertex_tensors[i]
    old = getvlabel(peps, i)
    mlabel = [newlabel(peps, 1), getphysicallabel(peps, i)] 
    peps.vertex_tensors[i] = EinCode([old, mlabel], replace(old, mlabel[2]=>mlabel[1]))(ti, mat) 
    return peps                                     # if mlabel[2] in old, replace it with mlabel[1]
end

function single_sandwich_code(peps::GeneralPEPS{T,NF,LT}, i, mat::AbstractMatrix, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {T,NF,LT}
    @assert size(mat, 1) == size(mat, 2)
    nflavor, D = NF, peps.D # TODO: Need to be modified
    size_dict = Dict([[l=>nflavor for l in peps.physical_labels]..., [l=>D for l in peps.virtual_labels]...])
    virtual = [l=>peps.max_index+1+i for (i, l) in enumerate(peps.virtual_labels)]
    rep = [peps.physical_labels[i]=>peps.max_index+1, virtual...] 
    mlabel = [getphysicallabel(peps, i), rep[1].second]
    merge!(size_dict, Dict([l=>nflavor for l in mlabel]), Dict(rep[1].second => nflavor), Dict([l.second => D for l in virtual]))
    single_sandwich_code=EinCode([peps.vertex_labels..., mlabel, [replace(l, rep...) for l in peps.vertex_labels]...], LT[])
    optcode_single_sandwich = optimize_code(single_sandwich_code, size_dict, optimizer, simplifier)
end
function single_sandwich(p1::GeneralPEPS{T}, p2::GeneralPEPS{T}, i, mat::AbstractMatrix{T}, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where T
    p1c = conj(p1)
    code = single_sandwich_code(p1, i, mat, optimizer,simplifier) 
    cost, gradient = OMEinsum.cost_and_gradient(code, (p1c.vertex_tensors..., mat, p2.vertex_tensors...))
    return cost[], gradient
end 


function two_sandwich_code(peps::GeneralPEPS{T,NF,LT}, i, j, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {T,NF,LT}
    @assert size(mat, 1) == size(mat, 2)
    nflavor, D = NF, peps.D
    size_dict = Dict([[l=>nflavor for l in peps.physical_labels]..., [l=>D for l in peps.virtual_labels]...])
    virtual = [l=>peps.max_index+2+i for (i, l) in enumerate(peps.virtual_labels)]
    rep = [peps.physical_labels[i]=>peps.max_index+1, peps.physical_labels[j]=>peps.max_index+2, virtual...] 
    mlabel = [getphysicallabel(peps, i), getphysicallabel(peps, j), rep[1].second, rep[2].second]
    merge!(size_dict, Dict([l=>nflavor for l in mlabel]), Dict(rep[1].second => nflavor, rep[2].second => nflavor), Dict([l.second => D for l in virtual])) 
    two_sandwich_code = EinCode([peps.vertex_labels..., mlabel, [replace(l, rep...) for l in peps.vertex_labels]...], LT[])
    optcode_two_sandwich = optimize_code(two_sandwich_code, size_dict, optimizer,simplifier)
end
function two_sandwich(p1::PEPS, p2::PEPS, i, j, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    p1c = conj(p1)
    code = two_sandwich_code(p1, i, j,  mat, optimizer,simplifier) 
    cost, gradient = OMEinsum.cost_and_gradient(code, (p1c.vertex_tensors..., mat, p2.vertex_tensors...))
    return cost[], gradient
end

nsite(peps::PEPS) = length(peps.physical_labels)
nflavor(::PEPS{T,NF}) where NF = NF
D(peps::PEPS) = peps.D
getvlabel(peps::PEPS, i::Int) = peps.vertex_labels[i]  # vertex tensor labels
getphysicallabel(peps::PEPS, i::Int) = peps.physical_labels[i]  # physical label
newlabel(peps::PEPS, offset) = peps.max_index + offset  # create a new label



# variational optimization begin

variables(peps::GeneralPEPS) = vcat(vec.(alltensors(peps))...) # peps->vec
function load_variables!(peps::GeneralPEPS{T}, variables::AbstractVector{T}) where T  # vec->peps
    k = 0
    for t in alltensors(peps)
        # t .*= zero(eltype(variables))
        t .= reshape(variables[k+1:k+length(t)], size(t))
        k += length(t)
    end
    return peps
end

function f1(peps::PEPS{T}, variables::AbstractVector{T}, i, mat::AbstractArray{T}, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where T
    load_variables!(peps, variables)
    energy, _ = single_sandwich(peps, peps, i, mat, optimizer, simplifier)
    norm = inner_product(peps, peps)
    # return real(energy / norm[]) 
    return (energy + energy') / 2 / norm[]
end

function g1!(G1, peps::PEPS, variables, i, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    load_variables!(peps, variables)
    energy, gradientA = single_sandwich(peps, peps, i, mat, optimizer, simplifier)
    flatten_gradientA = 2*vcat(reshape.(gradientA, :)...)[Int(end/2+3):end]  
    norm, gradientB = OMEinsum.cost_and_gradient(peps.code_inner_product, (peps.vertex_tensors..., peps.vertex_tensors...))
    flatten_gradientB = 2*vcat(reshape.(gradientB, :)...)[Int(end/2+1):end]
    G1 .=(flatten_gradientA*norm[]-flatten_gradientB*energy)/norm[]^2
    return G1
end


function peps_optimize1(peps::PEPS, i, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    params = variables(peps)
    f_closure1(params) =  f1(peps, params, i, mat, optimizer, simplifier)
    g_closure1!(G1, params) = g1!(G1, peps, params, i, mat, optimizer, simplifier)
    result = IsoPEPS.optimize(f_closure1, g_closure1!, params, IsoPEPS.LBFGS(), Optim.Options(f_tol = 1e-6))
    return result.minimum
end

function f2(peps::PEPS, variables, i, j, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    load_variables!(peps, variables)
    energy, _ = two_sandwich(peps, peps, i, j, mat, optimizer, simplifier)
    norm = inner_product(peps, peps)
    #return real(energy / norm[]) 
    return (energy + energy') / 2 / norm[]
end

function g2!(G2, peps::PEPS, variables, i, j, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    load_variables!(peps, variables)
    energy, gradientA = two_sandwich(peps, peps, i, j, mat, optimizer, simplifier)
    flatten_gradientA = 2*vcat(reshape.(gradientA, :)...)[Int(end/2+9):end]  
    norm, gradientB = OMEinsum.cost_and_gradient(peps.code_inner_product, (peps.vertex_tensors..., peps.vertex_tensors...))
    flatten_gradientB = 2*vcat(reshape.(gradientB, :)...)[Int(end/2+1):end]
    G2 .=(flatten_gradientA*norm[]-flatten_gradientB*energy)/norm[]^2
    return G2
end

function peps_optimize2(peps::PEPS, i, j, mat::AbstractArray, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    params = variables(peps)
    f_closure2(params) =  f2(peps, params, i, j, mat, optimizer, simplifier)
    g_closure2!(G2, params) = g2!(G2, peps, params, i, j, mat, optimizer, simplifier)
    result = IsoPEPS.optimize(f_closure2, g_closure2!, params, IsoPEPS.LBFGS(), Optim.Options(f_tol = 1e-6))
    @show result
    return result.minimum
end

function f_ising(peps::PEPS{T},  variables::AbstractVector{T}, g, J::Float64, h::Float64, optimizer::CodeOptimizer, simplifier::CodeSimplifier ) where {T}
    energy = 0  
    for i in peps.physical_labels
        energy+=f1(peps, variables, i, -h*Matrix{T}(X), optimizer, simplifier)
        
        for nb in neighbors(g, i)
            energy+=0.5*f2(peps, variables, i, nb, -J*reshape(kron(Matrix{T}(Z),Matrix{T}(Z)),2,2,2,2), optimizer, simplifier)
        end
    end
    return energy
end

function g_ising!(G, peps::PEPS{T}, variables, g, J::Float64, h::Float64, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    G1 = zeros(eltype(variables),size(variables))
    G2 = zeros(eltype(variables),size(variables))
    fill!(G, 0)
    for i in peps.physical_labels
        G .+= g1!(G1, peps, variables, i, -h*Matrix{T}(X), optimizer::CodeOptimizer, simplifier::CodeSimplifier)
  
        for nb in neighbors(g, i)
            G .+= 0.5*g2!(G2, peps, variables, i, nb, -J*reshape(kron(Matrix{T}(Z),Matrix{T}(Z)),2,2,2,2), optimizer::CodeOptimizer, simplifier::CodeSimplifier)
        end
    end
    return G
end

function peps_optimize_ising(peps::PEPS, g, J::Float64, h::Float64, optimizer::CodeOptimizer, simplifier::CodeSimplifier)
    params = variables(peps)
   
    f_closure_ising(params) =  f_ising(peps, params,g,J,h, optimizer, simplifier)
    g_closure_ising!(G, params) = g_ising!(G, peps, params,g,J,h, optimizer, simplifier)
    result = IsoPEPS.optimize(f_closure_ising, g_closure_ising!, params, IsoPEPS.LBFGS())
    @show result
    return result.minimum
end


# use AD
#=
function ChainRulesCore.rrule(::typeof(einsum), code::EinCode, @nospecialize(xs), size_dict)
    y = einsum(code, xs, size_dict)
    function einsum_pullback(dy)
        dy = convert(typeof(y), dy)  # for filled array/cuarray et al.
        dxs = ChainRulesCore.@thunk ntuple(i -> einsum_grad(getixs(code), xs, getiy(code), size_dict, conj(dy), i), length(xs))
        return (NoTangent(), NoTangent(), dxs, NoTangent())
    end
    einsum_pullback(::NoTangent) = (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    return y, einsum_pullback
end


Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(f_closure_ising), Float64}
=#

















function truncated_bmps(bmps,dmax::Int)
    for i in 1:(length(bmps)-1)        
        u,s,v=svd(reshape(bmps[i],:,size(bmps[i])[end])*reshape(bmps[i+1],size(bmps[i+1],1),:))
        dmax1 = min(searchsortedfirst(s, 1e-4, rev=true),dmax, length(s))
        u=u[:, 1:dmax1]; s=s[1:dmax1]; v=v[:,1:dmax1]
        bmps[i]=reshape(u,size(bmps[i])[1:end-1]...,size(u, 2))
        bmps[i+1]=reshape(Diagonal(s) *v',size(v', 1),size(bmps[i+1])[2:end]...)
    end
    return bmps
end

function contract_2peps(bra::PEPS,ket::PEPS)
    Ly=length(bra.tensors)
    Lx=length(bra.tensors[1])
    bra_ket = Vector{Vector{AbstractArray{Float64, 4}}}(undef, Ly)
    for y in 1:Ly
        bra_ket[y] = Vector{AbstractArray{Float64, 4}}(undef, Lx)
    end
    for y in 1:Ly
        bra_row=bra.tensors[y]
        ket_row=ket.tensors[y]
        for x in 1:Lx
            T=bra_row[x]
            T_dagger = Array(conj(ket_row[x]))
            T_contracted = ein"ijklc,mnpqc->ijklmnpq"(T, T_dagger)

            new_shape = (size(T_contracted, 1) * size(T_contracted, 5), 
                         size(T_contracted, 2) * size(T_contracted, 6),  
                         size(T_contracted, 3) * size(T_contracted, 7),  
                         size(T_contracted, 4) * size(T_contracted, 8))  
            bra_ket[y][x] = reshape(T_contracted, new_shape)
        end
    end
    return bra_ket
end

function overlap_peps(bra_ket,dmax::Int)
    Ly=length(bra_ket)
    Lx=length(bra_ket[1])
    bmps=truncated_bmps(bra_ket[1],dmax)
    for y in 2:(Ly-1)
        bra_ket_row=bra_ket[y]
        for x in 1:Lx
            bmps_contracted=ein"ijkl,mknp->imjnlp"(bmps[x],bra_ket_row[x])
            new_shape=(size(bmps_contracted,1)*size(bmps_contracted,2),
                       size(bmps_contracted,3),size(bmps_contracted,4),
                       size(bmps_contracted,5)*size(bmps_contracted,6))
            bmps[x]=reshape(bmps_contracted,new_shape)
        end
        bmps=truncated_bmps(bmps,dmax)
    end
    store=IndexStore()
    index_bra=Vector{Int}[]
    index_ket=Vector{Int}[]
    leftidx_bra=newindex!(store)
    firstleftbra=leftidx_bra
    leftidx_ket=newindex!(store)
    firstleftket=leftidx_ket
    for k=1:length(bmps)
        upidx_bra=newindex!(store)
        downidx_ket=upidx_bra
        downidx_bra=newindex!(store)
        upidx_ket=downidx_bra
        rightidix_bra=k==length(bmps) ? firstleftbra : newindex!(store)
        rightidix_ket=k==length(bmps) ? firstleftket : newindex!(store)
        push!(index_bra,[leftidx_bra,upidx_bra,downidx_bra,rightidix_bra])
        push!(index_ket,[leftidx_ket,upidx_ket,downidx_ket,rightidix_ket])
        leftidx_bra=rightidix_bra
        leftidx_ket=rightidix_ket
    end
    ixs=[index_bra...,index_ket...]
    eincode = OMEinsum.DynamicEinCode(ixs, Int[])
    nested_ein = optimize_code(eincode, uniformsize(eincode, 2), TreeSA())
    return nested_ein(bmps..., bra_ket[end]...)[1]
end

function get_tensors(psi::PEPS)
    psi_tensors=Vector{Any}()
    Ly=length(psi.tensors)
    Lx=length(psi.tensors[1])
    for i in 1:Ly
        for j in 1:Lx
            push!(psi_tensors,psi.tensors[i][j])
        end
    end
    return psi_tensors
end

function local_sandwich(bra::PEPS,ket::PEPS,op,y::Vector,x::Vector;optimizer=GreedyMethod())
    store = IndexStore()
    ixs_bra = Vector{Int}[]
    ixs_op = Vector{Int}[]
    ixs_ket = Vector{Int}[]
    Ly=length(bra.tensors)
    Lx=length(bra.tensors[1])
    for i in 1:Ly
        leftidx_bra = newindex!(store)
        firstleftbra=leftidx_bra
        leftidx_ket = newindex!(store)
        firstleftket=leftidx_ket
        for j in 1:Lx
            rightidx_bra = j==Lx ? firstleftbra : newindex!(store)    
            rightidx_ket = j==Lx ? firstleftket : newindex!(store)   
            push!(ixs_bra,[leftidx_bra,rightidx_bra])
            push!(ixs_ket,[leftidx_ket,rightidx_ket])
            #@show size(ixs_bra[j+3*(i-1)])
            leftidx_bra=rightidx_bra
            leftidx_ket=rightidx_ket
        end
    end

    for k in 1:Lx
        upidx_bra=newindex!(store)
        firstupbra=upidx_bra
        upidx_ket=newindex!(store)
        firstupket=upidx_ket
        for l in 1:Ly
            downidx_bra = l==Ly ? firstupbra : newindex!(store)    
            downidx_ket = l==Ly ? firstupket : newindex!(store) 
            ixs_bra[k+Lx*(l-1)]=vcat(ixs_bra[k+Lx*(l-1)][1],[upidx_bra,downidx_bra],ixs_bra[k+Lx*(l-1)][2])
            ixs_ket[k+Lx*(l-1)]=vcat(ixs_ket[k+Lx*(l-1)][1],[upidx_ket,downidx_ket],ixs_ket[k+Lx*(l-1)][2])
            upidx_bra=downidx_bra
            upidx_ket=downidx_ket
        end
    end 

    for m in 1:Ly
        for n in 1:Lx
            physidx1=newindex!(store)
            push!(ixs_bra[n+Lx*(m-1)],physidx1)
            physidx2 = physidx1
            for i in 1:length(y)
                if m==y[i] && n==x[i]
                    physidx2=newindex!(store)
                    push!(ixs_op,[physidx1,physidx2])
                end
            end
            push!(ixs_ket[n+Lx*(m-1)],physidx2)
        end
    end
    ixs_op=vcat(ixs_op...)
    ixs=[ixs_bra...,ixs_op,ixs_ket...]
    size_dict=OMEinsum.get_size_dict(ixs,[get_tensors(bra)...,op,get_tensors(ket)...])
    code=optimize_code(DynamicEinCode(ixs,Int[]),size_dict,optimizer)
    return code,code(conj.(get_tensors(bra))...,op, get_tensors(ket)...)[]
end

function local_h(Lx::Int,Ly::Int,bra::PEPS,ket::PEPS)
    nsites=Ly*Lx
    mg1_sum=get_tensors(bra)
    mg1_sum = [zeros(size(mg1_sum[i])) for i in 1:nsites]

    energy=0
    for y in 1:Ly
        y=[y]
        for x in 1:Lx
            x=[x]
            h=Float64.(-0.2*Matrix(X))
            code1,energy1=local_sandwich(bra,ket,h,y,x)
            energy+=energy1
            
            cost1, mg1 = IsoPEPS.OMEinsum.cost_and_gradient(code1, (conj.(get_tensors(bra))..., h, get_tensors(ket)...))
           
            mg1_sum+=mg1[nsites+2:end]
        end
    end
   

    for y in 1:Ly
        y=[y,y]
        for x in 1:(Lx-1)
            x=[x,x+1]
            h=Float64.(-reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2))
            code1,energy1=local_sandwich(bra,ket,h,y,x)
            energy+=energy1
            cost1, mg1 = IsoPEPS.OMEinsum.cost_and_gradient(code1, (conj.(get_tensors(bra))..., h, get_tensors(ket)...))
            @show size(mg1)
            mg1_sum+=mg1[nsites+2:end]
        end
    end

    for x in 1:Lx
        x=[x,x]
        for y in 1:(Ly-1)
            y=[y,y+1]
            h=Float64.(-reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2))
            code1,energy1=local_sandwich(bra,ket,h,y,x)
            energy+=energy1
            cost1, mg1 = IsoPEPS.OMEinsum.cost_and_gradient(code1, (conj.(get_tensors(bra))..., h, get_tensors(ket)...))
            mg1_sum+=mg1[nsites+2:end]
        end
    end
    
    mg1_sum = vcat(map(vec, mg1_sum)...) 
    code2,norm=local_sandwich(bra,ket,Float64.(Matrix(I2)),[1],[1])
    cost2,mg2=IsoPEPS.OMEinsum.cost_and_gradient(code2, (conj.(get_tensors(bra))..., Float64.(Matrix(I2)), get_tensors(ket)...))
    
    mg2=vcat(map(vec,mg2[nsites+2:end])...)
    gradient_E=(mg1_sum*cost2[]-mg2*energy)/cost2[]^2
    energy=energy/cost2[]
    @show energy,norm,cost2[]

    return energy,gradient_E
end

#=
function update_peps_from_params!(psi, params)
    idx = 1
    for i in 1:length(psi.tensors)
        for j in 1:length(psi.tensors[i])
            size_tensor = size(psi.tensors[i][j])  
            n_elements = prod(size_tensor)         
            psi.tensors[i][j] .= reshape(params[idx:idx + n_elements-1], size_tensor)
            idx += n_elements  
        end
    end
end

function f(params::Vector{Float64}, psi::PEPS, Ly::Int, Lx::Int)
    update_peps_from_params!(psi, params)
    energy, _ = local_h(Ly, Lx, psi, psi)
    return real(energy)
end

function g!(G, params::Vector{Float64}, psi::PEPS, Ly::Int, Lx::Int)
    update_peps_from_params!(psi, params)
    _, gradient_E = local_h(Ly, Lx, psi, psi)
    G .= gradient_E
end

function peps_variation(Ly::Int, Lx::Int, bond_dim::Int)
    psi = generate_peps(bond_dim, Ly, Lx)
    @show psi
  
    params = Float64[]
    for i in 1:Ly 
        append!(params, map(vec, psi.tensors[i])...)
    end  
    @show params

    f_closure(params) =  f(params, psi, Ly, Lx)
    g_closure!(G, params) = g!(G, params, psi, Ly, Lx)

    result = IsoPEPS.optimize(f_closure, g_closure!, params, IsoPEPS.LBFGS())
    @show result
    return result.minimum
end 
=#