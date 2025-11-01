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
- `first_list`: List of measurements for the first observable (3/4 of iterations)
- `second_list`: List of measurements for the second observable (last 1/4 of iterations)

# Description
Applies the gate repeatedly while measuring observables.
- If `measure_first=:X`: First 3/4 measure X, last 1/4 measure Z
- If `measure_first=:Z`: First 3/4 measure Z, last 1/4 measure X
"""
function iterate_channel_PEPS(gate, row; niters=10000, measure_first=:X)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end
    
    rho = zero_state(row+1)
    first_list = Float64[]
    second_list = Float64[]
    
    for i in 1:niters
        for j in 1:row
            rho_p = zero_state(1)
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>gate)) 

            if i > niters*3 ÷ 4
                if measure_first == :X
                    Z = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(second_list, Z.buf)
                else
                    Yao.apply!(rho, put(2+row, 1=>H))
                    X = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(second_list, X.buf)
                end
            else
                if measure_first == :X
                    Yao.apply!(rho, put(2+row, 1=>H))
                    X = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(first_list, X.buf)
                else
                    Z = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(first_list, Z.buf)
                end
            end
        end
    end
    
    return rho, first_list, second_list
end
