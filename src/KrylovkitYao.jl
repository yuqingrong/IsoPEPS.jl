function ising_hamiltonian(nbit::Int, J::Float64, h::Float64)
    -J * sum([kron(nbit, i=>Z, i+1=>Z) for i=1:nbit-1]) -
        h * sum([kron(nbit, i=>X) for i=1:nbit])
end

function ising_hamiltonian_2d(m::Int, n::Int, J::Float64, h::Float64)
    lis = LinearIndices((m, n))  # CartesianIndices
    -J * sum([[kron(m * n, lis[i, j]=>Z, lis[i+1,j]=>Z) for i=1:m-1, j=1:n]..., 
              [kron(m * n, lis[i, j]=>Z, lis[i, j+1]=>Z) for i=1:m, j=1:n-1]...]) -
        h * sum(vec([kron(m * n, lis[i, j]=>X) for i=1:m, j=1:n]))
end


function ed_groundstate(h::AbstractBlock)
    x0 = statevec(rand_state(nqubits(h)))
    E, V = eigsolve(h |> mat, x0, 1, :SR, ishermitian=true)
    E[1], V[1]
end


