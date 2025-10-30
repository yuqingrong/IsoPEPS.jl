"""
Cost functions for variational optimization of iPEPS.

Provides various cost function implementations for computing expectation
values of observables in infinite PEPS states.
"""

"""
    cost_X_circ(rho, row, gate; niters=50)

Compute X observable expectation using circuit measurement.

# Arguments
- `rho`: Density matrix
- `row`: Number of rows
- `gate`: Quantum gate
- `niters`: Number of measurement shots (unused, kept for compatibility)

# Returns
Expectation value of X operator
"""
function cost_X_circ(rho, row, gate; niters=50)
    rho2 = join(rho, density_matrix(zero_state(1)))
    rho2 = Yao.apply!(rho2, put(2+row, (1, 2, 3)=>gate))   
    Yao.apply!(rho2, put(2+row, 1=>H)) 
    return 1-2*mean(measure(rho2, 1; nshots=1000000))
end

"""
    cost_ZZ_circ(rho, row, gate; niters=50)

Compute ZZ correlation using circuit measurement.

# Arguments
- `rho`: Density matrix
- `row`: Number of rows
- `gate`: Quantum gate
- `niters`: Number of measurement shots (unused, kept for compatibility)

# Returns
ZZ correlation value ⟨Z₁Z₂⟩
"""
function cost_ZZ_circ(rho, row, gate; niters=50)
    rho2 = join(rho, density_matrix(zero_state(1)))
    rho2 = Yao.apply!(rho2, put(2+row, (1, 2, 3)=>gate))   
    Z1 = measure(rho2, 1; nshots=1000000)
    Z1 = 1-2*mean(Z1)
    
    # Trace out the measured qubit to reduce register size
    rho2 = partial_tr(rho2, 1)
    rho2 = join(rho2, density_matrix(zero_state(1)))
    Yao.apply!(rho2, put(2+row, (1,2,4)=>gate))
    Z2 = measure(rho2, 1; nshots=1000000)
    Z2 = 1-2*mean(Z2)
    
    return Z1*Z2
end

"""
    cost_X(rho, row, gate)

Compute X observable expectation by direct tensor contraction.

# Arguments
- `rho`: Density matrix
- `row`: Number of rows
- `gate`: Quantum gate

# Returns
Expectation value ⟨X⟩

# Description
More accurate than measurement-based approach, uses exact tensor contraction.
"""
function cost_X(rho, row, gate)
    nqubits = Int(log2(size(rho, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho, shape...)
    
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AX = ein"iabcd,ij -> jabcd"(A, Matrix(X))
    
    tensor_ket = [AX, A, A]
    tensor_bra = [conj(A), conj(A), conj(A)]
    _, list = contract_Elist(tensor_ket, tensor_bra, row)
    result = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list)

    return result[]
end

"""
    cost_ZZ(rho, row, gate)

Compute ZZ correlation by direct tensor contraction.

# Arguments
- `rho`: Density matrix
- `row`: Number of rows
- `gate`: Quantum gate

# Returns
ZZ correlation value ⟨Z₁Z₂⟩

# Description
Computes the expectation value by contracting the tensor network with
Z operators applied at different positions. Averages over all possible
nearest-neighbor configurations.
"""
function cost_ZZ(rho, row, gate)
    nqubits = Int(log2(size(rho, 1)))
    shape = ntuple(_ -> 2, 2 * nqubits)
    rho = reshape(rho, shape...)
    
    A = reshape(Matrix(gate), 2, 2, 2, 2, 2, 2)[:, :, :, 1, :, :]
    AZ = ein"iabcd,ij -> jabcd"(A, Matrix(Z))
    
    # Compute three different configurations
    tensor_bra = [conj(A), conj(A), conj(A)]
    
    _, list1 = contract_Elist([AZ, AZ, A], tensor_bra, row)
    result1 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list1)
    
    _, list2 = contract_Elist([A, AZ, AZ], tensor_bra, row)
    result2 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list2)
    
    _, list3 = contract_Elist([AZ, A, AZ], tensor_bra, row)
    result3 = ein"abcdefgh,ijklijklabcdefgh ->"(rho, list3)
    
    return 2*(result1[] + result2[] + result3[]) / 3
end

"""
    exact_energy_PEPS(d::Int, D::Int, g::Float64, row::Int)

Compute exact ground state energy using MPSKit.

# Arguments
- `d`: Physical dimension
- `D`: Bond dimension
- `g`: Transverse field strength
- `row`: Number of rows (for cylinder)

# Returns
Ground state energy density

# Description
Uses VUMPS algorithm from MPSKit to find the ground state of the
transverse field Ising model on an infinite cylinder.
"""
function exact_energy_PEPS(d::Int, D::Int, g::Float64, row::Int)
    mps = InfiniteMPS([ComplexSpace(d) for _ in 1:row], [ComplexSpace(D) for _ in 1:row])
    H0 = transverse_field_ising(InfiniteCylinder(row); g=g)
    psi, _ = find_groundstate(mps, H0, VUMPS())
    E = real(expectation_value(psi, H0)) / row
    return E
end

