struct PEPS{T,AT<:AbstractArray{T,5}}
    tensors::Vector{Vector{AT}}
    function PEPS(tensors::Vector{Vector{AT}}) where {T, AT<:AbstractArray{T,5}}
        Ly=length(tensors)
        Lx=length(tensors[1])
        physical_dim=size(tensors[1][1],5)
        @assert all(size(tensors[y][x],5)==physical_dim for y in 1:Ly, x in 1:Lx)
        @assert all(size(tensors[y][x],4)==size(tensors[y][mod1(x+1,Lx)],1) for y in 1:Ly, x in 1:Lx)
        @assert all(size(tensors[y][x],3)==size(tensors[mod1(y+1,Ly)][x],2) for y in 1:Ly, x in 1:Lx)
        new{T,AT}(tensors)
    end
end

function generate_peps(::Type{T},bond_dim::Int,Ly::Int,Lx::Int;d::Int=2) where T
    tensors=Vector{Vector{AbstractArray{T, 5}}}(undef,Ly)
    for y in 1: Ly
        row = Vector{AbstractArray{T, 5}}(undef, Lx)
        for x in 1: Lx
            left_dim=x==1 ? 1 : bond_dim
            right_dim=x==Lx ? 1 : bond_dim
            up_dim=y==1 ? 1 : bond_dim
            down_dim=y==Ly ? 1 : bond_dim
            row[x]=randn(T,left_dim,up_dim,down_dim,right_dim,d)
        end
        tensors[y]=row
    end

    return PEPS(tensors)
end

generate_peps(bond_dim::Int,Ly::Int,Lx::Int; d::Int=2) = generate_peps(Float64, bond_dim,Ly,Lx;d)

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





function local_sandwich(bra::PEPS,ket::PEPS,op,y::Vector,x::Vector)
    nsites=Ly*Lx
    store = IndexStore()
    ixs_bra = Vector{Int}[]
    ixs_op = Vector{Int}[]
    ixs_ket = Vector{Int}[]

    for i in 1:Ly
        leftidx_bra = newindex!(store)
        firstleftbra=leftidx_bra
        leftidx_ket = newindex!(store)
        firstleftket=leftidx_ket
        for j in 1:Lx
            rightidix_bra = j==Lx ? firstleftbra : newindex!(store)    
            rightidix_ket = j==Lx ? firstleftket : newindex!(store)   
            push!(ixs_bra,[leftidx_bra,rightidix_bra])
            push!(ixs_ket,[leftidx_ket,rightidix_ket])
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
            downidix_bra = l==Ly ? firstupbra : newindex!(store)    
            downidix_ket = l==Ly ? firstupket : newindex!(store) 
            splice!(ixs_bra[Lx+3*(Ly-1)],2,[upidx_bra,downidix_bra])
            splice!(ixs_ket[Lx+3*(Ly-1)],2,[upidx_ket,downidix_ket])
            upidx_bra=downidx_bra
            upidx_ket=downidx_ket
        end
    end 

    for m in 1:Ly
        for n in 1:Lx
            physidx1=newindex!(store)
            push!(ixs_bra[n+3*(m-1)],physidx1)
            for i in length(y)
                physidx2=(m==y[i] && n==x[i]) ? newindex!(store) : physidx1
                push!(ixs_ket[n+3*(m-1)],physidx2)
                push!(ixs_op,[physidx1,physidx2])
        end
    end
    ixs=[ixs_bra...,ixs_op...,ixs_ket...]
    size_dict=OMEinsum.get_size_dict(ixs,[bra.tensors...,op,ket.tensors...])
    code=optimize_code(DynamicEinCode(ixs,Int[]),size_dict,optimizer)
    return code,code(conj.(bra.tensors)...,op, ket.tensors...)[]
end

function local_h(Lx,Ly)
    mg1_sum=zeros(1,nsites) 
    energy=0
    for y in 1:Ly
        for x in 1:Lx
            y=[y]
            x=[x]
            h=Matrix(X)
            code1,energy1=local_sandwich(psi,psi,h,y,x)
            energy+=energy1
            cost1, mg1 = IsoPEPS.OMEinsum.cost_and_gradient(code1, (conj.(psi.tensors)..., h, psi.tensors...))
            mg1_sum+=mg1[nsites+2:end]
            
        end
    end

    for y in 1:Ly
        for x in 1:(Lx-1)
            y=[y,y]
            x=[x,x+1]
            h=reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)
            code1,energy1=local_sandwich(psi,psi,h,y,x)
            energy+=energy1
            cost1, mg1 = IsoPEPS.OMEinsum.cost_and_gradient(code2, (conj.(psi.tensors)..., h, psi.tensors...))
            mg1_sum+=mg1[nsites+2:end]
        end
    end

    for x in 1:Lx
        for y in 1:(Ly-1)
            x=[x,x]
            y=[y,y+1]
            h=reshape(kron(Matrix(Z),Matrix(Z)),2,2,2,2)
            code1,energy1=local_sandwich(psi,psi,h,y,x)
            energy+=energy1
            cost1, mg1 = IsoPEPS.OMEinsum.cost_and_gradient(code2, (conj.(psi.tensors)..., h, psi.tensors...))
            mg1_sum+=mg1[nsites+2:end]
        end
    end
    
    mg1_sum = vcat(map(vec, mg1_sum)...) 
    code2,norm=local_sandwich(psi,psi,Matrix(I),y,x)
    cost2,mg2=IsoPEPS.OMEinsum.cost_and_gradient(code1, (conj.(psi.tensors)..., Matrix(I), psi.tensors...))
    mg2=vcat(map(vec,mg2[nsites+1:end]))
    gradient_E=(mg1*cost2[]-mg2*energy)/cost2[]^2 

    energy/=norm
    return energy,gradient_E
end


function peps_variation(Ly::Int,Lx::Int,bond_dim::Int,h::Float64)
    psi=generate_peps(bond_dim,Ly,Lx)
    params=Float64[]
    for i in 1:Ly 
        append!(params,map(vec, psi.tensors[i])...)
    end
    @show size(params)
    function update_peps_from_params!(psi, params)
        idx = 1
        for i in 1:length(psi.tensors)
            for j in 1:length(psi.tensors[i])
                size_tensor = size(psi.tensors[i][j])  
                n_elements = prod(size_tensor)         
                psi.tensors[i][j] .= reshape(params[idx:idx + n_elements - 1], size_tensor)
                idx += n_elements  
            end
        end
    end

    update_peps_from_params!(psi, params)
    energy,gradient_E=local_h(Lx,Ly)

    function f(params)
        return energy
    end
    
    function g!(G, params)
        G .= gradient_E
    end
    
    result = IsoPEPS.optimize(f,g!,params,IsoPEPS.LBFGS())
    return result
end