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
function build_parameterized_gate(params, r, p)
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
        A_matrix *= build_parameterized_gate(params, r, p)
    end
    
    # Verify unitarity
    @assert A_matrix * A_matrix' ≈ I atol=1e-5
    @assert A_matrix' * A_matrix ≈ I atol=1e-5
    
    return A_matrix
end

"""
    compute_energy_from_measurements(X_list, Z_lists, g, J, row)

Compute energy from measurement statistics.

# Arguments
- `X_list`: X measurement results
- `Z_lists`: Tuple of Z measurement results for different configurations
- `g`: Transverse field strength
- `J`: Coupling strength
- `row`: Number of rows

# Returns
Energy estimate

# Description
Computes energy from the last portion of measurements to ensure
convergence. Uses a hardcoded window of 7500 measurements.

# TODO
Make the measurement window size configurable.
"""
function compute_energy_from_measurements(X_list, Z_lists, g, J, row; measurement_window=7500)
    Z1_list, Z2_list, Z3_list, Z4_list, Z5_list, Z6_list = Z_lists
    
    energy = -g * mean(X_list[end-measurement_window:end]) - 
             J * mean(
                 Z1_list .* Z2_list + Z2_list .* Z3_list + 
                 Z1_list .* Z3_list + Z1_list .* Z4_list + 
                 Z2_list .* Z5_list + Z3_list .* Z6_list
             ) / row
    
    return energy
end

"""
    extract_Z_configurations(Z_list, row)

Extract Z measurement configurations for different sublattices.

# Arguments
- `Z_list`: Full Z measurement list
- `row`: Number of rows (determines sublattice spacing)

# Returns
Tuple of 6 Z lists for different sublattices
"""
function extract_Z_configurations(Z_list, row)
    Z1_list = Z_list[1:2*row:end]
    Z2_list = Z_list[2:2*row:end]  
    Z3_list = Z_list[3:2*row:end]
    Z4_list = Z_list[4:2*row:end]
    Z5_list = Z_list[5:2*row:end]
    Z6_list = Z_list[6:2*row:end]
    
    return (Z1_list, Z2_list, Z3_list, Z4_list, Z5_list, Z6_list)
end

