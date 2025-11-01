"""
Gate construction utilities for parameterized quantum circuits.

Provides functions for building quantum gates from variational parameters.
"""

"""
    build_parameterized_gate(params, r, p)

Build a single layer gate from parameters.

# Arguments
- `params`: Parameter vector
- `r`: Layer index
- `p`: Total number of layers

# Returns
Quantum gate for the specified layer

# Description
Constructs a gate using RX and RZ rotations followed by CNOTs.
Each layer uses 6 parameters (2 per qubit for a 3-qubit gate).
"""
function _build_parameterized_gate(params, r)
    gate = kron(
        Yao.Rx(params[6*r-5]) * Yao.Rz(params[6*r-4]), 
        Yao.Rx(params[6*r-3]) * Yao.Rz(params[6*r-2]), 
        Yao.Rx(params[6*r-1]) * Yao.Rz(params[6*r])
    )
    cnot_12 = cnot(3, 2, 1)
    cnot_23 = cnot(3, 3, 2)
    cnot_31 = cnot(3, 1, 3)
    return Matrix(gate) * Matrix(cnot_12) * Matrix(cnot_23) * Matrix(cnot_31)
end

"""
    build_gate_from_params(params, p)

Build complete unitary gate from all parameters.

# Arguments
- `params`: Full parameter vector
- `p`: Number of layers

# Returns
Complete unitary matrix

# Description
Constructs the full gate by composing all layers. The resulting gate
should be unitary within numerical precision.
"""
function build_gate_from_params(params, p)
    A_matrix = Matrix(I, 8, 8)
    for r in 1:p
        A_matrix *= _build_parameterized_gate(params, r)
    end
    
    # Verify unitarity
    @assert A_matrix * A_matrix' ≈ I atol=1e-5
    @assert A_matrix' * A_matrix ≈ I atol=1e-5

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

function energy_measure(X_list, separ_Z_lists, g, J, row) 
    interaction_pairs = Vector{Tuple{Int,Int}}(undef, row)
    @inbounds for i in 1:row
        j = (i % row) + 1
        interaction_pairs[i] = (i, j)
    
    end

    X_mean = mean(@view X_list[1+end-3/4*niters:end])
    
    # interaction within row
    ZZ_wr_mean = mean(separ_Z_lists[i][k] * separ_Z_lists[j][k] for (i, j) in interaction_pairs for k in 1:length(separ_Z_lists[1]))
    
    # interaction between rows
    ZZ_br_mean = mean(separ_Z_lists[i][j] * separ_Z_lists[i][j+1] for i in 1:row for j in 1:(length(separ_Z_lists[i])-1))
    
    energy = -g * X_mean - J * (ZZ_wr_mean + ZZ_br_mean)
    
    return energy
end


