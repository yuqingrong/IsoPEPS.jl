function transverse_ising(nbit::Int; J, h, periodic::Bool=true)
    #=
    ising_term = map(1:(periodic ? nbit : nbit - 1)) do i #ZZ
        j = (i % nbit) + 1  
        J/4 * repeat(nbit, Z, (i, j))  
    end |> sum
    transverse_field = sum(map(i -> h/2 * put(nbit, i => X), 1:nbit))#X
    ising_term + transverse_field
    =#
    sum(-J/4*kron(nbit,i=>Z,mod1(i+1,nbit)=>Z) for i in 1:(nbit- (periodic ? 0 : 1))) + sum(-h/2*put(nbit,i=>X) for i in 1:nbit)
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