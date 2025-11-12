"""
Quantum circuit operations for iPEPS.
"""

"""
    iterate_channel_PEPS(gate, row; niters=10000, measure_first=:X)

Iterate a quantum channel defined by `gate` on a system with `row` rows.

# Arguments
- `gate`: Quantum gate defining the channel
- `row`: Number of rows in the system
- `niters`: Number of iterations (default: 10000)
- `measure_first`: Which observable to measure first, either `:X` or `:Z` (default: `:X`)

# Returns
- `rho`: Final density matrix
- `X_list`: List of X measurements (3/4 of iterations if X first, 1/4 if Z first)
- `Z_list`: List of Z measurements (1/4 of iterations if X first, 3/4 if Z first)

# Description
Applies the gate repeatedly while measuring observables.
- If `measure_first=:X`: First 3/4 measure X, last 1/4 measure Z
- If `measure_first=:Z`: First 3/4 measure Z, last 1/4 measure X
Always returns (rho, X_list, Z_list) in that order regardless of measure_first.
"""
function iterate_channel_PEPS(gate, row; niters=10000, measure_first=:X)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end
    
    rho = zero_state(row+1)
    X_list = Float64[]
    Z_list = Float64[]
    
    for i in 1:niters
        for j in 1:row
            rho_p = zero_state(1)
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate)) 

            if i > niters*3 ÷ 4
                # Last 1/4 of iterations: measure the second observable
                if measure_first == :X
                    Z = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(Z_list, Z.buf)
                else
                    Yao.apply!(rho, put(2+row, 1=>H))
                    X = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(X_list, X.buf)
                end
            else
                # First 3/4 of iterations: measure the first observable
                if measure_first == :X
                    Yao.apply!(rho, put(2+row, 1=>H))
                    X = 1-2*measure!(RemoveMeasured(), rho, 1)                  
                    push!(X_list, X.buf)
                else
                    Z = 1-2*measure!(RemoveMeasured(), rho, 1)               
                    push!(Z_list, Z.buf)
                end
            end
        end
    end
    
    return rho, Z_list, X_list
end


function iterate_dm(gate, row; niters=10000, measure_first=:X)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end
    
    rho = density_matrix(zero_state(row+1))
    X_list = Float64[]
    Z_list = Float64[]
    
    for i in 1:niters
        for j in 1:row
            rho_p = density_matrix(zero_state(1))   
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate)) 

            if i > niters*3 ÷ 4
                # Last 1/4 of iterations: compute expectation value of second observable
                if measure_first == :X
                    # Compute ⟨Z⟩ = Tr(ρ σ_z) on qubit 1
                    Z_expect = real(expect(put(nqubits(rho), 1=>Z), rho))
                    push!(Z_list, Z_expect)
                else
                    # Compute ⟨X⟩ = Tr(ρ σ_x) on qubit 1
                    X_expect = real(expect(put(nqubits(rho), 1=>X), rho))
                    push!(X_list, X_expect)
                end
            else
                # First 3/4 of iterations: compute expectation value of first observable
                if measure_first == :X
                    # Compute ⟨X⟩ = Tr(ρ σ_x) on qubit 1
                    X_expect = real(expect(put(nqubits(rho), 1=>X), rho))
                    push!(X_list, X_expect)
                else
                    # Compute ⟨Z⟩ = Tr(ρ σ_z) on qubit 1
                    Z_expect = real(expect(put(nqubits(rho), 1=>Z), rho))
                    push!(Z_list, Z_expect)
                end    
            end
            rho = Yao.partial_tr(rho, 1)
        end
    end
    
    return rho, Z_list, X_list
end
