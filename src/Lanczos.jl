using LinearAlgebra
using SparseArrays
using Arpack

function ising_hamiltonian(nsites, J, h)
    # Initialize a sparse matrix with the correct dimensions
    H = spzeros(2^nsites, 2^nsites)
    
    # Loop over all possible states
    for state in 0:(2^nsites - 1)
        # Convert state to binary array
        σ = digits(state, base=2, pad=nsites)
        
        # Diagonal elements (σ_i^z σ_{i+1}^z term)
        for i in 1:(nsites - 1)
            if σ[i] == σ[i + 1]
                H[state + 1, state + 1] -= J
            else
                H[state + 1, state + 1] += J
            end
        end
        
        # Off-diagonal elements (σ_i^x term)
        for i in 1:nsites
            new_state = state ⊻ (1 << (i - 1))
            H[state + 1, new_state + 1] -= h
        end
    end
    
    return H
end



# Print the ground state vector
#println("Ground state vector: ", ground_state_vector)