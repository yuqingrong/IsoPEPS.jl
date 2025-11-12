
function _build_parameterized_gate(params, r)
    gate = kron(
        Yao.Rx(params[6*r-5]) * Yao.Rz(params[6*r-4]), 
        Yao.Rx(params[6*r-3]) * Yao.Rz(params[6*r-2]), 
        Yao.Rx(params[6*r-1]) * Yao.Rz(params[6*r])
    )

    cnot_12 = cnot(3, 2,1)
    cnot_23 = cnot(3, 3,2)
    cnot_31 = cnot(3, 1,3)
    return Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
end
function build_gate_from_params(params, p)
    A_matrix = Vector{}(undef, 3)
    for i in 1:3
        A_matrix[i] = Matrix(Array{ComplexF64}(I, 8, 8))
    end
    # A_matrix[1] uses params 1-12 (indices 1:6*p)
    params_1 = params[1:6*p]
    for r in 1:p
        A_matrix[1] *= _build_parameterized_gate(params_1, r)
    end
    
    # A_matrix[2] uses params 13-24 (indices 6*p+1:12*p)
    params_2 = params[6*p+1:12*p]
    for r in 1:p
        A_matrix[2] *= _build_parameterized_gate(params_2, r)
    end
    
    # A_matrix[3] uses params 25-36 (indices 12*p+1:18*p)
    params_3 = params[12*p+1:18*p]
    for r in 1:p
        A_matrix[3] *= _build_parameterized_gate(params_3, r)
    end
    
    for i in 1:3
        @assert A_matrix[i] * A_matrix[i]' ≈ I atol=1e-5
        @assert A_matrix[i]' * A_matrix[i] ≈ I atol=1e-5
    end

    return A_matrix
end



"""
    extract_Z_configurations(Z_list, row): seperate the Z_list to row sublists, here row=3

    --| A_1 |--| A_4 |--
         |        |
    --| A_2 |--| A_5 |--
         |        |
    --| A_3 |--| A_6 |--
    
    energy_measure(X_list, separ_Z_lists, g, J, row): ⟨H⟩ = -g ∑ᵢ⟨Xᵢ⟩ - J∑ᵢⱼ⟨ZᵢZⱼ⟩

    ⟨ZᵢZⱼ⟩ includes vertical and horizontal interactions, mean(⟨Z₁Z₂⟩ + ⟨Z₂Z₃⟩ + ⟨Z₃Z₁⟩ + ⟨Z₁Z₄⟩ + ⟨Z₂Z₅⟩ + ⟨Z₃Z₆⟩)
"""
function extract_Z_configurations(Z_list, row)
    @assert length(Z_list) % row == 0 
    Z_configs = ntuple(i -> Z_list[i:row:end], row)
    return Z_configs
end

function energy_measure(X_list, separ_Z_lists, g, J, row; niters=niters) 
    interaction_pairs = Vector{Tuple{Int,Int}}(undef, row)
    @inbounds for i in 1:row
        j = (i % row) + 1
        interaction_pairs[i] = (i, j)
    
    end

    X_mean = mean(@view X_list[Int(1+end-3/4*niters):end])
    
    # interaction within row
    ZZ_wr_mean = mean(separ_Z_lists[i][k] * separ_Z_lists[j][k] for (i, j) in interaction_pairs for k in 1:length(separ_Z_lists[1]))
    
    # interaction between rows
    ZZ_br_mean = mean(separ_Z_lists[i][j] * separ_Z_lists[i][j+1] for i in 1:row for j in 1:(length(separ_Z_lists[i])-1))
    
    energy = -g * X_mean - J * (ZZ_wr_mean + ZZ_br_mean)
    
    return energy
end




