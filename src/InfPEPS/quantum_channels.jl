
function iterate_channel_PEPS(A_matrix, row; conv_step=1000, samples=100000,measure_first=:X)
    if measure_first ∉ (:X, :Z)
        throw(ArgumentError("measure_first must be either :X or :Z, got $measure_first"))
    end
    
    rho = zero_state(row+1)
    X_list = Float64[]
    Z_list = Float64[]
    
    niters = ceil(Int, (conv_step + 2*samples)/ row)
    for i in 1:niters
        for j in 1:row
            rho_p = zero_state(1)
            rho = join(rho, rho_p)
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>matblock(A_matrix[j]))) 

            if i > (conv_step + samples)/ row
                if measure_first == :X
                    Z = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(Z_list, Z.buf)
                else
                    Yao.apply!(rho, put(2+row, 1=>H))
                    X = 1-2*measure!(RemoveMeasured(), rho, 1)
                    push!(X_list, X.buf)
                end
            else
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
            rho = Yao.apply!(rho, put(2+row,(1, 2, j+2)=>matblock(gate[j]))) 

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
