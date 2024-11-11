using LinearAlgebra
using SparseArrays
using Arpack

function Lanczos(nsites::Int64, J::Float64, h::Float64)
    H = spzeros(2^nsites, 2^nsites)
    
    for state in 0:(2^nsites - 1)
        σ = digits(state, base=2, pad=nsites)

        for i in 1:(nsites - 1)
            if σ[i] == σ[i + 1]
                H[state + 1, state + 1] -= J
            else
                H[state + 1, state + 1] += J
            end
        end
        
        for i in 1:nsites
            new_state = state ⊻ (1 << (i - 1))
            H[state + 1, new_state + 1] -= h
        end
    end
    
    return H
end



# Print the ground state vector
#println("Ground state vector: ", ground_state_vector)