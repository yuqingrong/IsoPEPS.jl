"""1D,Ising model,MPO version"""

using Printf
using LinearAlgebra
using Plots

include("ncon.jl");
include("doDMRG_MPO.jl");

chi = 16; # maximum bond dimension
Nsites = 50; # number of lattice sites

OPTS_numsweeps = 4; # number of DMRG sweeps
OPTS_dispon = 2; # level of output display
OPTS_updateon = true; # update MPS tensors
OPTS_maxit = 2; # iterations of Lanczos method
OPTS_krydim = 4; # dimension of Krylov subspace

#MPO initialize
chid = 2;
sX = [0 1; 1 0]; sY = [0 -im; im 0];
sZ = [1 0; 0 -1]; sI = [1 0; 0 1];
M = zeros(4,4,chid,chid);
M[1,1,:,:] = sI; M[4,4,:,:] = sI;
M[1,2,:,:] = sZ; M[2,4,:,:] = sZ;
M[1,3,:,:] = sX; M[3,4,:,:] = sX;
ML = reshape([1;0;0;0],4,1,1); #left MPO boundary
MR = reshape([0;0;0;1],4,1,1); #right MPO boundary

#MPS Initialize
A = Array{Any,1}(undef,Nsites); A[1] = rand(1,chid,min(chi,chid));
for k = 2:Nsites
    A[k] = rand(size(A[k-1],3),chid,min(min(chi,size(A[k-1],3)*chid),chid^(Nsites-k)));
end

#DMRG sweeps
En, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi; numsweeps = OPTS_numsweeps, dispon = OPTS_dispon, updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim);

#exact energy (fermions)
H = diagm(1 => ones(Nsites-1)) + diagm(-1 => ones(Nsites-1));
D = eigen(0.5*(H+H'));
EnExact = 2*sum(D.values[D.values .< 0]);

#error
EnErr = En[end] - EnExact; # should equal to zero
@printf "NumSites: %d, Energy: %e, EnErr: %e \n" Nsites  En[end] EnErr

#plot(1:length(En1),log10.(En1 .-EnExact),xlabel = "Update Step",
    #ylabel = "-log10(Ground Energy Error)", label = "chi = 16", title = "DMRG for XX model");
#display(plot!(1:length(En2),log10.(En2 .-EnExact), label = "chi = 32"));

#Compute 2-site reduced density matrices, local energy profile
rhotwo = Array{Any,1}(undef,Nsites-1);
hamloc = reshape(real(-kron(sX,sX)+0.5*kron(sI,sZ)+0.5*kron(sZ,sI)),2,2,2,2);
Enloc = zeros(Nsites-1);
for k = 1:Nsites-1
    rhotwo[k] = ncon(Any[A[k],conj(A[k]),A[k+1],conj(A[k+1]),sWeight[k+2],sWeight[k+2]],Any[[1,-3,2],[1,-1,3],[2,-4,4],[3,-2,5],[4,6],[5,6]]);
    Enloc[k] = ncon(Any[hamloc,rhotwo[k]],Any[[1,2,3,4],[1,2,3,4]]);
end