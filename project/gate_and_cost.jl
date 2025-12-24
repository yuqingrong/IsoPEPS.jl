
function _build_parameterized_gate(params, r, nqubits)
    single_qubit_gates = []  
    for i in 1:nqubits
        idx = 2*nqubits*r - 2*nqubits + 2*i - 1   # 2 means 2 parameters for Rx and Rz
        push!(single_qubit_gates, Yao.Rx(params[idx]) * Yao.Rz(params[idx+1]))
    end
    gate = kron(single_qubit_gates...)
    
    # Build CNOT gates in a ring: 2->1, 3->2, ..., row->(row-1), 1->row TODO: check if no 1 -> row
    
    cnot_gates = Matrix{ComplexF64}(I, 2^nqubits, 2^nqubits)
    for i in 1:nqubits
        target = i
        control= (i % nqubits) + 1  
        cnot_gates *= Matrix(cnot(nqubits, control, target))
    end
  
    return Matrix(gate) * Matrix(cnot_gates)
end

"""
    build_gate_from_params(params, p, row; share_params=true)

Build multiple parameterized gates from parameters.

# Arguments
- `params`: Parameter vector
- `p`: Number of layers per gate
- `row`: Number of gates to generate
- `share_params`: If true, all gates use the same parameters; if false, each gate uses different parameters (default: true)

# Returns
- Vector of gate matrices, each of size 2^row × 2^row

# Notes
- When `share_params=true`: requires `2*row*p` parameters total (all gates share the same parameters), A-A-A structure
- When `share_params=false`: requires `2*row*p*row` parameters total (each gate has independent parameters), A-B-C structure
"""
function build_gate_from_params(params, p, row, nqubits; share_params=true)
    A_matrix = Vector{Matrix{ComplexF64}}(undef, row)
    dim = 2^nqubits
    for i in 1:row
        A_matrix[i] = Matrix(Array{ComplexF64}(I, dim, dim))
    end
    
    if share_params
        @assert length(params) >= 2*nqubits*p "Need at least $(2*nqubits*p) parameters for shar ed parameters mode"
        shared_params = params[1:2*nqubits*p]
        for i in 1:row
            for r in 1:p
                A_matrix[i] *= _build_parameterized_gate(shared_params, r, nqubits)
            end
        end
    else
        @assert length(params) >= 2*nqubits*p*row "Need at least $(2*nqubits*p*row) parameters for independent parameters mode"
        for i in 1:row
            # Gate i uses params from indices (2*row*p*(i-1)+1):(2*row*p*i)
            params_i = params[2*nqubits*p*(i-1)+1:2*nqubits*p*i]
            for r in 1:p
                A_matrix[i] *= _build_parameterized_gate(params_i, r, nqubits)
            end
        end
    end
    
    for i in 1:row
        @assert A_matrix[i] * A_matrix[i]' ≈ I atol=1e-5
        @assert A_matrix[i]' * A_matrix[i] ≈ I atol=1e-5
    end

    return A_matrix
end


"""
    --| A_1 |--| A_1 |--
         |        |
    --| A_2 |--| A_2 |--
         |        |
    --| A_3 |--| A_3 |--
                                                              __________________________
                                           __________________|________                  |
                                          |                  |        |                 |
MPS string + long range interaction: --| A_1 |--| A_2 |--| A_3 |--| A_1 |--| A_2 |--| A_3 |--
                                                   |                           |
                                                    ___________________________

H = -g ∑ᵢ Xᵢ - J ∑ᵢ ZᵢZᵢ₊₁ - J ∑ᵢ ZᵢZᵢ₊ᵣₒᵥ
"""

function energy_measure(X_list, Z_list, g, J, row) 
    X_mean = mean(X_list)
    ZZ_mean = row == 1 ? mean(Z_list[i]*Z_list[i+1] for i in 1:length(Z_list)-1) : mean(Z_list[i]*Z_list[i+1] + Z_list[i]*Z_list[i+row] for i in 1:length(Z_list)-row)
    energy = -g * X_mean - J * ZZ_mean
    return energy
end


function energy_recal(g::Float64, J::Float64, p::Int, row::Int, nqubits::Int; data_dir="data", optimizer=GreedyMethod())
    X_file = joinpath(data_dir, "compile_X_list_list_row=$(row)_g=$(g).dat")
    X_list = parse.(Float64, split(readlines(X_file)[end]))
    Z_file = joinpath(data_dir, "compile_Z_list_list_row=$(row)_g=$(g).dat")
    Z_list = parse.(Float64, split(readlines(Z_file)[end]))
    energy = energy_measure(X_list, Z_list, g, J, row)
    return energy
end

