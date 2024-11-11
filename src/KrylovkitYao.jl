function ising_hamiltonian(nbit::Int, J::Float64, h::Float64)
    sz = i->put(nbit, i=>Z)
    sx = i->put(nbit, i=>X)
    sum(1:(nbit-1)) do i
        sz(i)*sz(i+1)
    end * (-J) + sum(1:nbit) do i
        sx(i)
    end * (-h)
end

# TODO: study why the default initial state is sparse vector.
function ed_groundstate(h::AbstractBlock)
    x0 = statevec(rand_state(nqubits(h)))
    E, V = eigsolve(h |> mat, x0, 1, :SR, ishermitian=true)
    E[1], V[1]
end