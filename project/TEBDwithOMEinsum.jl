using OMEinsum
using LinearAlgebra
using TensorOperations
using IsoPEPS
using Yao

J = 1.0  
h = 0.2
nsites = 10
tau = 0.01  
iter = 1000
 
σz = [1.0 0.0; 0.0 -1.0] 
σx = [0.0 1.0; 1.0 0.0]
σi = [1.0 0.0; 0.0 1.0]
U_X = exp(-tau * h * σx)
U_Z = exp(-tau * J * kron(σz, σz))
U_ZZ=reshape(U_Z,2,2,2,2)
H_X = -h*σx
H_Z = -J*kron(σz, σz)
H_ZZ = reshape(H_Z,2,2,2,2)

function generate_random_mps(nsites::Int, d::Int, bond_dim::Int)
    tensors = Vector{AbstractArray}(undef, nsites)
    tensors[1] = svd(rand( d, bond_dim)).U
    for i in 2:nsites-1
        tensors[i] = reshape(svd(rand( bond_dim* d, bond_dim)).U,bond_dim, d, bond_dim)
    end 
    tensors[nsites] = svd(rand( bond_dim, d)).U
    return MPS(tensors)
end
mps=generate_random_mps(nsites,2,2)

@show inner_product(mps,mps)
"""
mps=Array{Any,1}(undef,nsites)1·
mps[1]=randn(2,2);mps[nsites]=randn(2,2)
for k = 2:(nsites-1)
   mps[k] = randn(2,2,2)
end

function normalize!(mps)
    for i in 1:nsites
        mps.tensors[i] ./= norm(mps.tensors[i])
    end
end
"""

function normalize!(mps)
    global_norm = sqrt(inner_product(mps,mps))
    mps.tensors[1] ./= global_norm
end


normalize!(mps)

@show inner_product(mps,mps)

function compute_energy(mps, nsites, J, h)
    energy=0.0
    epsilon = 1e-12
    for i in 1:nsites
        mps_H=deepcopy(mps)
        mps_H0=deepcopy(mps)
        if i==1  
            D=ein"abcd,ai,bij->cdj"(H_ZZ,mps_H.tensors[i],mps_H.tensors[i+1])
            D=D .+ epsilon
            F=svd(reshape(D,2,4))
            S_truncated = Diagonal(F.S)
            mps_H.tensors[i]=reshape(F.U,2,2);
            mps_H.tensors[i+1]=reshape((S_truncated*F.Vt),2,2,2)
            energy+=inner_product(mps,mps_H)
            G=ein"ab,ac->bc"(U_X,mps_H0.tensors[i])
            mps_H0.tensors[i]=G
            energy+=inner_product(mps,mps_H0)
            
    
        elseif i==(nsites-1)
            D=ein"abcd,aij,bj->cdi"(H_ZZ,mps_H.tensors[i],mps_H.tensors[i+1])
            D=D .+ epsilon
            F=svd(reshape(D,4,2))
            S_truncated = Diagonal(F.S)
            mps_H.tensors[i]=reshape(F.U,2,2,2);
            mps_H.tensors[i+1]=reshape((S_truncated*F.Vt),2,2)
            energy+=inner_product(mps,mps_H)
            G=ein"ab,acd->bcd"(H_X,mps_H0.tensors[i])
            mps_H.tensors[i]=G
            energy+=inner_product(mps,mps_H0)
           
        elseif i==nsites
            G=ein"ab,ac->bc"(H_X,mps_H0.tensors[i])
            mps_H.tensors[i]=G
            energy+=inner_product(mps,mps_H0)
            
        else
            D=ein"abcd,aij,bjk->cdik"(H_ZZ,mps_H.tensors[i],mps_H.tensors[i+1])
            D=D .+ epsilon
            F=svd(reshape(D,4,4))
            S_truncated = Diagonal(F.S)
            mps_H.tensors[i]=reshape(F.U[:,1:2],2,2,2);
            mps_H.tensors[i+1]=reshape((S_truncated[1:2,1:2]*F.Vt[1:2,:]),2,2,2)
            energy+=inner_product(mps,mps_H)
            G=ein"ab,aij->bij"(H_X,mps_H0.tensors[i])
            mps_H.tensors[i]=G
            energy+=inner_product(mps,mps_H0)
        end
    end
    return energy
end
"""
function inner_product(mps1, mps2)
    # Initialize with an identity for the bond dimension contraction
    overlap = 1.0
    nsites = length(mps1)

    for i in 1:nsites-1
        # Assume each mps tensor has the form: (left_bond, physical, right_bond)
        # Contract overlap with the current tensors from mps1 and mps2
        if i==1
            overlap = @tensor overlap * conj(mps1[i])[a, b] * mps2[i][a, b]
        elseif i==nsites
            overlap = @tensor overlap * conj(mps1[i])[a, b] * mps2[i][a, b]
        else 
            overlap = @tensor overlap * conj(mps1[i])[a, b, c] * mps2[i][a, b, c]
        end
    end

    return overlap
end
"""


function time_evolve(iter::Int)
    epsilon = 1e-12
    for p in 1:iter
        for i in 1:2:(nsites - 1)
            if i==1  
                D=ein"abcd,ai,bij->cdj"(U_ZZ,mps.tensors[i],mps.tensors[i+1])
                D=D .+ epsilon
                F=svd(reshape(D,2,4))
                S_truncated = Diagonal(F.S)
                mps.tensors[i]=reshape(F.U,2,2);
                mps.tensors[i+1]=reshape((S_truncated*F.Vt),2,2,2)
         
            elseif i==nsites-1
                D=ein"abcd,aij,bj->cdi"(U_ZZ,mps.tensors[i],mps.tensors[i+1])
                D=D .+ epsilon
                F=svd(reshape(D,4,2))
                S_truncated = Diagonal(F.S)
                mps.tensors[i]=reshape(F.U,2,2,2);
                mps.tensors[i+1]=reshape((S_truncated*F.Vt),2,2)
            
            else
                D=ein"abcd,aij,bjk->cdik"(U_ZZ,mps.tensors[i],mps.tensors[i+1])
                D=D .+ epsilon
                F=svd(reshape(D,4,4))
                S_truncated = Diagonal(F.S)
                mps.tensors[i]=reshape(F.U[:,1:2],2,2,2);
                mps.tensors[i+1]=reshape((S_truncated[1:2,1:2]*F.Vt[1:2,:]),2,2,2)
             
            end 
            #normalize!(mps)
        end
        for i in 2:2:(nsites - 1) 
            D=ein"abcd,aij,bjk->cdik"(U_ZZ,mps.tensors[i],mps.tensors[i+1])
                D=D .+ epsilon
                F=svd(reshape(D,4,4))
                S_truncated = Diagonal(F.S)
                mps.tensors[i]=reshape(F.U[:,1:2],2,2,2);
                mps.tensors[i+1]=reshape((S_truncated[1:2,1:2]*F.Vt[1:2,:]),2,2,2)
            #normalize!(mps)
        end
        for i in 1:nsites
            if i==1 || i==nsites
                G=ein"ab,ac->bc"(U_X,mps.tensors[i])
                mps.tensors[i]=G
            else
                G=ein"ab,aij->bij"(U_X,mps.tensors[i])
                mps.tensors[i]=G
            end
            #normalize!(mps)
        end
        normalize!(mps)
        if p%100==0
            println(compute_energy(mps, nsites, J, h))
        end
        
    end
    
end
    


# Compute the energy of the system
time_evolve(iter)
product1=inner_product(mps, mps)

EG=compute_energy(mps, nsites, J, h)/product1
@show EG,product1
#@show mps
#EG=eincode(mps..., mps...)[1]

