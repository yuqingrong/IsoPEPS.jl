# TODO: isoPEPS is not a valid module name, capitalize it
module isoPEPS

using Yao, Yao.EasyBuild
using KrylovKit: eigsolve
using LinearAlgebra
using CairoMakie

export ising_hamiltonian, ed_groundstate

include("LanczosAlgorithm.jl")
include("KrylovkitYao.jl")

end