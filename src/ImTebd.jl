export itime_groundstate!
using Yao
using Yao.EasyBuild
using LinearAlgebra
using Test

function transverse_ising(nbit::Int; J, h, periodic::Bool=true)
    ising_term = map(1:(periodic ? nbit : nbit - 1)) do i #ZZ
        j = (i % nbit) + 1  
        J * repeat(nbit, Z, (i, j))  
    end |> sum
    transverse_field = sum(map(i -> h * put(nbit, i => X), 1:nbit))#X
    return (ising_term + transverse_field)
end

itime_groundstate!(H::AbstractBlock; t::Int64=t, p::Float64=p) = reg -> itime_groundstate!(reg, H; t=t,p=p)
function itime_groundstate!(reg, H; t,p)
    te = time_evolve(H, -im*p)
    for i = 1:t÷p
        reg |> te |> normalize!
    end
    if t%p != 0
        reg |> time_evolve(H, t%p) |> normalize!
    end
    reg
end


#@test ishermitian(h)


# using imaginary time Evolution

#@test isapprox(EG, -0.4564, atol=1e-4)
#include("Lanczos.jl")
#### Compare with exact results (computed from free fermions)


