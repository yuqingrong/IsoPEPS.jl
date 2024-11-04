using OMEinsum
using LinearAlgebra


J = 1.0  
h = 0.2  
nsites = 10  
tau = 0.1  
iter = 20  

σz = [1.0 0.0; 0.0 -1.0] 
σx = [0.0 1.0; 1.0 0.0]
U_X = exp(-tau * h * σx)
U_X=reshape(U_X,2,2)
U_ZZ = exp(-tau * J * kron(σz, σz))
U_ZZ=reshape(U_ZZ,2,2,2,2)

mps0=mps=Array{Any,1}(undef,nsites)
    for k = 1:nsites
        mps[k] = rand(2,2,2)
    end
normalize!(mps)

for p in iter
    for i in 1:(nsites-1)
        D=ein"abcd,aij,bjk->cdik"(U_ZZ,mps[i],mps[i+1])
        U,S,V=svd(reshape(D,4,4))
        S_truncated = Diagonal(S)
        mps[i]=reshape(U[:,1:2],2,2,2);
        mps[i+1]=reshape((S_truncated*V)[:,1:2],2,2,2)
        G=ein"ab,aij->bij"(U_X,mps[i])
        normalize!(mps)
    end
end
EG=ein""(mps,mps0)
println("基态虚时间演化完成。")