"""
Quantum channel operations for iPEPS.

This module provides functions for iterating quantum channels and computing
spectral properties of transfer matrices.
"""

"""
    iterate_channel_PEPS(gate, row; niters=10000)

Iterate a quantum channel defined by `gate` on a system with `row` rows.

# Arguments
- `gate`: Quantum gate defining the channel
- `row`: Number of rows in the system
- `niters`: Number of iterations (default: 10000)

# Returns
- `rho`: Final density matrix
- `Z1_list`: List of Z measurements
- `X1_list`: List of X measurements

# Description
Applies the gate repeatedly while measuring observables. 
- First 3/4 of iterations: Measure X observable
- Last 1/4 of iterations: Measure Z observable
"""
function iterate_channel_PEPS(gate, row; niters=10000)
    rho = zero_state(row+1)
    Z1_list = Float64[]
    X1_list = Float64[]
    
    for i in 1:niters
        for j in 1:row
            rho_p = zero_state(1)
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate)) 

            if i > niters*3 ÷ 4  
                Z = 1-2*measure!(RemoveMeasured(), rho, 1)
                push!(Z1_list, Z.buf)
            else
                Yao.apply!(rho, put(2+row, 1=>H)) 
                X = 1-2*measure!(RemoveMeasured(), rho, 1)
                push!(X1_list, X.buf)
            end
        end
    end
    
    return rho, Z1_list, X1_list
end

"""
    exact_left_eigen(gate, nsites)

Compute the left eigenvector and spectral gap of the transfer matrix.

# Arguments
- `gate`: Quantum gate defining the channel
- `nsites`: Number of sites

# Returns
- `rho`: Normalized density matrix (fixed point)
- `gap`: Spectral gap (-log|λ₂|)

# Description
Constructs the transfer matrix from the gate and computes its spectral properties.
The spectral gap quantifies how quickly the channel approaches its fixed point.
"""
function exact_left_eigen(gate, nsites)
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    
    _, T = contract_Elist([A for _ in 1:nsites], [conj(A) for _ in 1:nsites], nsites)
    T = reshape(T, 4^(nsites+1), 4^(nsites+1))
    
    # Get second largest eigenvalue for gap
    λ₁ = partialsort(abs.(LinearAlgebra.eigen(T).values), 2; rev=true)
    gap = -log(abs(λ₁))
    
    # Verify normalization
    @assert LinearAlgebra.eigen(T).values[end] ≈ 1.0
    
    # Extract fixed point density matrix
    fixed_point_rho = reshape(
        LinearAlgebra.eigen(T).vectors[:, end], 
        Int(sqrt(4^(nsites+1))), 
        Int(sqrt(4^(nsites+1)))
    )
    rho = fixed_point_rho ./ tr(fixed_point_rho)
    
    return rho, gap
end

