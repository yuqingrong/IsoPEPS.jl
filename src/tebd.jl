export itime_groundstate!
using Yao
using Yao.EasyBuild
using LinearAlgebra
using Test

function transverse_ising(nbit::Int; J=1.0, h=0.2, periodic::Bool=true)
    # ZZ interaction terms between neighboring qubits
    ising_term = map(1:(periodic ? nbit : nbit - 1)) do i
        j = (i % nbit) + 1  # Neighboring qubit, wraps around if periodic
        J * repeat(nbit, Z, (i, j))  # Apply ZZ interaction with coupling constant J
    end |> sum

    # Transverse field terms on each qubit (X direction)
    transverse_field = sum(map(i -> h * put(nbit, i => X), 1:nbit))

    # Combine the interaction and transverse field terms to form the full Hamiltonian
    ising_term + transverse_field
end

itime_groundstate!(h::AbstractBlock; τ::Real=20, tol=1e-4) = reg -> itime_groundstate!(reg, h; τ=τ, tol=tol)
function itime_groundstate!(reg::AbstractRegister, h::AbstractBlock; τ::Int=20, tol=1e-4)
    span = 1
    te = time_evolve(h, -im*span)
    for i = 1:τ÷span
        reg |> te |> normalize!
    end
    if τ%span != 0
        reg |> time_evolve(h, τ%span) |> normalize!
    end
    reg
end

nsites = 10
h = transverse_ising(nsites;J=1.0, h=0.2) |> cache
#@test ishermitian(h)

# using imaginary time Evolution
reg = rand_state(nqubits(h))
reg |> itime_groundstate!(h, τ=20)
EG = expect(h, reg)/nsites
println(EG)
#@test isapprox(EG, -0.4564, atol=1e-4)

#### Compare with exact results (computed from free fermions)
H = diagm(1 => ones(nsites-1)) + diagm(-1 => ones(nsites-1));
D = eigen(0.5*(H+H'));
EnExact = 2*sum(D.values[D.values .< 0]);
EnExact = EnExact/nsites
println(EnExact)

