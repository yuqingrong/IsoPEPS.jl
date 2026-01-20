"""
    compute_transfer_spectrum(gates, row, virtual_qubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)

Compute the transfer matrix spectrum and fixed point.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `num_eigenvalues`: Number of eigenvalues to compute (default: 2 for gap calculation)
- `use_iterative`: `:auto` (default), `:always`, or `:never` for iterative solver
- `matrix_free`: `:auto` (default), `:always`, or `:never` for matrix-free approach
  - When enabled, never forms the full transfer matrix (essential for large row)

# Returns
- `rho`: Fixed point density matrix (normalized)
- `gap`: Spectral gap = -log|λ₂| where λ₂ is second largest eigenvalue
- `eigenvalues`: Sorted eigenvalue magnitudes (top `num_eigenvalues` if iterative)
- `eigenvalues_raw`: Raw complex eigenvalues (sorted by magnitude, descending)

# Description
Constructs the transfer matrix from gates and computes its spectral properties.
For large systems (row >= 5), uses a matrix-free approach that never forms the full
transfer matrix, instead computing T*v on-the-fly via tensor contraction.
The spectral gap quantifies how quickly the channel converges to its fixed point.
"""
function compute_transfer_spectrum(gates, row, nqubits; num_eigenvalues=2, use_iterative=:auto, matrix_free=:auto)
    virtual_qubits = (nqubits - 1) ÷ 2
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)

    # Matrix size: bond_dim^(2*total_legs) where total_legs = row + 1 (periodic + row bonds)
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    matrix_size = bond_dim^(2*total_legs)
    # Decide whether to use matrix-free approach (essential for large row)
    should_use_matrix_free = if matrix_free == :auto
        matrix_size > 1024  # Use matrix-free for matrices larger than 1024x1024
    elseif matrix_free == :always
        true
    else
        false
    end
    
    # Decide whether to use iterative solver
    should_use_iterative = if use_iterative == :auto
        matrix_size > 256
    elseif use_iterative == :always
        true
    else
        false
    end
    
    if should_use_matrix_free
        # Matrix-free approach: define T*v as a function
        # This avoids O(n²) memory for large matrices
        
        # Precompute the contraction code for applying transfer matrix
        bond_dim = 2^virtual_qubits
        code, total_legs = _build_transfer_contraction_code(A_tensors, row, virtual_qubits)
        tensor_ket = [A_tensors[i] for i in 1:row]
        tensor_bra = [conj(A_tensors[i]) for i in 1:row]
        
        # Define the linear map T*v
        function apply_transfer(v)
            # v is a vector of length bond_dim^(2*total_legs) = matrix_size
            # Reshape to tensor form for left indices (2*total_legs indices, each dim bond_dim)
            v_tensor = reshape(v, ntuple(_ -> bond_dim, 2*total_legs)...)
            # Apply transfer matrix via contraction
            result = _apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_legs)
            return vec(result)
        end
        
        # Use KrylovKit with the linear map
        v0 = randn(ComplexF64, matrix_size)
        v0 = v0 / norm(v0)
        vals, vecs, info = KrylovKit.eigsolve(apply_transfer, v0, num_eigenvalues, :LM; 
                                              ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
        
        # Sort by magnitude (descending)
        sorted_indices = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
        
        gap = -log(eigenvalues[2])
        @assert isapprox(eigenvalues[1], 1.0, atol=1e-6) "Largest eigenvalue should be 1, got $(eigenvalues[1])"
        
        fixed_point = reshape(vecs[sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    elseif should_use_iterative
        # Build full matrix but use iterative eigensolver
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        
        vals, vecs, info = KrylovKit.eigsolve(T, num_eigenvalues, :LM; 
                                              ishermitian=false, krylovdim=max(30, 2*num_eigenvalues))
        
        sorted_indices = sortperm(abs.(vals), rev=true)
        eigenvalues_raw = vals[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
        
        gap = -log(eigenvalues[2])
        @assert isapprox(eigenvalues[1], 1.0, atol=1e-6) "Largest eigenvalue should be 1, got $(eigenvalues[1])"
        
        fixed_point = reshape(vecs[sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    else
        # Full eigendecomposition for small matrices
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        eig_result = LinearAlgebra.eigen(T)
        sorted_indices = sortperm(abs.(eig_result.values), rev=true)
        eigenvalues_raw = eig_result.values[sorted_indices]
        eigenvalues = abs.(eigenvalues_raw)
        gap = -log(eigenvalues[2])
        @assert eigenvalues[1] ≈ 1.0 "Largest eigenvalue should be 1"
        
        fixed_point = reshape(eig_result.vectors[:, sorted_indices[1]], 
                              Int(sqrt(matrix_size)), Int(sqrt(matrix_size)))
    end
    rho = fixed_point ./ tr(fixed_point)
    
    return rho, gap, eigenvalues, eigenvalues_raw
end

function get_transfer_matrix(gates, row, virtual_qubits)
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    bond_dim = 2^virtual_qubits
    total_legs = row + 1
    matrix_size = bond_dim^(2*total_legs)
    _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                         [conj(A_tensors[i]) for i in 1:row], row)
    T = reshape(T, matrix_size, matrix_size)
    return T
end

"""
    build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=GreedyMethod())

Build contraction code for transfer matrix operations.

Each tensor should have 5 legs: [physical, down, right, up, left].
Dimensions are automatically inferred from tensor shapes.

# Arguments
- `tensor_ket`: Vector of ket tensors (length = row)
- `tensor_bra`: Vector of bra tensors, typically conj.(tensor_ket)
- `row`: Number of rows in the tensor network
- `for_matvec`: If true, build code for T*v (matrix-vector product); if false, build full transfer matrix
- `optimizer`: Contraction order optimizer (default: GreedyMethod())

# Returns
When `for_matvec=false` (default):
- `code`: Optimized contraction code
- `result`: Contracted transfer matrix with open left/right boundary indices

When `for_matvec=true`:
- `code`: Optimized contraction code for T*v operation
- `total_legs`: Number of legs on each side (for reshaping)

# Modes
- **Full matrix mode** (`for_matvec=false`): Both left and right boundaries are open indices.
  Use this to build the explicit transfer matrix T.
  
- **Matrix-vector mode** (`for_matvec=true`): Left boundary connects to input vector,
  only right boundary is open. Use this for matrix-free eigensolvers where you only
  need T*v without forming T explicitly.
"""
function build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=GreedyMethod())
    store = IndexStore()
    index_ket = Vector{Int}[]
    index_bra = Vector{Int}[]
    output_indices = Int[]
    
    # For matvec mode, create input indices for the left boundary
    # Infer total_legs from tensor dimensions: left leg count = 1 (periodic) + row
    total_legs = row + 1  # periodic boundary index + row left indices
    input_indices = for_matvec ? [newindex!(store) for _ in 1:2*total_legs] : Int[]
    
    # Initialize boundary indices for periodic vertical boundary
    first_down_ket = newindex!(store)
    first_up_ket = newindex!(store)
    first_down_bra = newindex!(store)
    first_up_bra = newindex!(store)
    prev_down_ket = first_down_ket
    prev_down_bra = first_down_bra
    
    # Build index structure for each tensor
    # Tensor leg ordering: [physical, down, right, up, left]
    for i in 1:row
        phyidx = newindex!(store)  # Shared between ket and bra (contracted)
        left_ket, right_ket = newindex!(store), newindex!(store)
        left_bra, right_bra = newindex!(store), newindex!(store)
        
        # Handle periodic boundary: row 1's up connects to row N's down
        up_ket = (i == 1) ? first_up_ket : prev_down_ket
        up_bra = (i == 1) ? first_up_bra : prev_down_bra
        down_ket = (i == 1) ? first_down_ket : newindex!(store)
        down_bra = (i == 1) ? first_down_bra : newindex!(store)
        
        push!(index_ket, [phyidx, down_ket, right_ket, up_ket, left_ket])
        push!(index_bra, [phyidx, down_bra, right_bra, up_bra, left_bra])
        
        prev_down_ket = down_ket
        prev_down_bra = down_bra
    end

    # Right boundary indices (always in output)
    push!(output_indices, index_ket[row][2])  # Periodic boundary index
    append!(output_indices, [index_ket[i][3] for i in 1:row])
    push!(output_indices, index_bra[row][2])
    append!(output_indices, [index_bra[i][3] for i in 1:row])
    
    # Left boundary handling depends on mode
    left_indices_ket = [index_ket[1][4], [index_ket[i][5] for i in 1:row]...]
    left_indices_bra = [index_bra[1][4], [index_bra[i][5] for i in 1:row]...]
    
    if for_matvec
        # Matrix-vector mode: connect left boundary to input vector indices
        # Replace left indices in tensor index lists with input_indices
        for (j, old_idx) in enumerate(left_indices_ket)
            for k in 1:row, l in 1:5
                if index_ket[k][l] == old_idx
                    index_ket[k][l] = input_indices[j]
                end
            end
        end
        for (j, old_idx) in enumerate(left_indices_bra)
            for k in 1:row, l in 1:5
                if index_bra[k][l] == old_idx
                    index_bra[k][l] = input_indices[total_legs + j]
                end
            end
        end
        
        # Build tensor index list with input vector
        all_indices = [index_ket..., index_bra..., collect(input_indices)]
        # Infer bond dimension from tensors (index 2 is 'down' which has bond_dim size)
        bond_dim = size(tensor_ket[1], 2)
        dummy_input = zeros(ComplexF64, ntuple(_ -> bond_dim, 2*total_legs)...)
        all_tensors = [tensor_ket..., tensor_bra..., dummy_input]
        
        size_dict = OMEinsum.get_size_dict(all_indices, all_tensors)
        code = optimize_code(DynamicEinCode(all_indices, output_indices), size_dict, optimizer)
        
        return code, total_legs
    else
        # Full matrix mode: left boundary indices are also in output
        append!(output_indices, left_indices_ket)
        append!(output_indices, left_indices_bra)
        
        all_indices = [index_ket..., index_bra...]
        all_tensors = [tensor_ket..., tensor_bra...]
        
        size_dict = OMEinsum.get_size_dict(all_indices, all_tensors)
        code = optimize_code(DynamicEinCode(all_indices, output_indices), size_dict, optimizer)
        
        return code, code(all_tensors...)
    end
end

"""
Convert gate matrices to tensor form for contraction.
"""
function gates_to_tensors(gates, row, virtual_qubits)
    bond_dim = 2^virtual_qubits
    A_size = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
    indices = (ntuple(_ -> Colon(), 3)..., 1,ntuple(_ -> Colon(), 2)...)
    return [reshape(gates[i], A_size)[indices...] for i in 1:row]
end

"""
    apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs)

Apply transfer matrix to a vector using precomputed contraction code.
"""
function apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_legs)
    result = code(tensor_ket..., tensor_bra..., v_tensor)
    # Infer bond dimension from the input tensor
    bond_dim = size(v_tensor, 1)
    return reshape(result, ntuple(_ -> bond_dim, 2*total_legs)...)
end

# Backward compatibility aliases
contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=GreedyMethod()) = 
    build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=false, optimizer=optimizer)

function _build_transfer_contraction_code(A_tensors, row, total_qubits; optimizer=GreedyMethod())
    tensor_ket = [A_tensors[i] for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    return build_transfer_code(tensor_ket, tensor_bra, row; for_matvec=true, optimizer=optimizer)
end

_apply_transfer_to_vector(code, tensor_ket, tensor_bra, v_tensor, total_qubits) = 
    apply_transfer_matvec(code, tensor_ket, tensor_bra, v_tensor, total_qubits)

"""
    compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())

Compute ⟨X⟩ expectation value from fixed point density matrix.

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `optimizer`: Contraction optimizer

# Returns
Average X expectation value across all sites.
"""
function compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    total_qubits = 1 + virtual_qubits + virtual_qubits * row
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = _gates_to_tensors(gates, row, virtual_qubits)
    AX_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], Matrix(X)) for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]
    
    results = map(1:row) do pos
        tensor_ket = [i == pos ? AX_tensors[i] : A_tensors[i] for i in 1:row]
        _, list = contract_transfer_matrix(tensor_ket, tensor_bra, row; optimizer=optimizer)
        
        store = IndexStore()
        index_list = [newindex!(store) for _ in 1:4*total_qubits]
        index_rho = index_list[2*total_qubits+1:4*total_qubits] 
        index_R = index_list[1:2*total_qubits]
        index = [index_list, index_rho, index_R]
        
        size_dict = OMEinsum.get_size_dict(index, [list, rho, R])
        code = optimize_code(DynamicEinCode(index, Int[]), size_dict, optimizer)
        code(list, rho, R)[]
    end
    
    return sum(results) / row
end

"""
    compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())

Compute ⟨ZZ⟩ expectation values (vertical and horizontal bonds).

# Arguments
- `rho`: Fixed point density matrix
- `gates`: Gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `optimizer`: Contraction optimizer

# Returns
- `ZZ_vertical`: Vertical bond ⟨ZᵢZᵢ₊₁⟩
- `ZZ_horizontal`: Horizontal bond ⟨ZᵢZᵢ₊ᵣₒw⟩
"""
function compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    total_qubits = 1 + virtual_qubits + virtual_qubits * row
    env_size = ntuple(i -> 2, 2*total_qubits)
    rho = reshape(rho, env_size...)
    R = reshape(Matrix(I, Int(sqrt(4^total_qubits)), Int(sqrt(4^total_qubits))), env_size)
    
    A_tensors = _gates_to_tensors(gates, row, virtual_qubits)
    AZ_tensors = [ein"iabcd,ij -> jabcd"(A_tensors[i], Matrix(Z)) for i in 1:row]
    tensor_bra = [conj(A_tensors[i]) for i in 1:row]

    # Vertical: Z on sites 1 and 2
    tensor_ket_vert = [i == 1 || i == 2 ? AZ_tensors[i] : A_tensors[i] for i in 1:row]
    # Horizontal: Z on site 1 only (correlates across transfer matrix)
    tensor_ket_horiz = [i == 1 ? AZ_tensors[i] : A_tensors[i] for i in 1:row]

    ZZ_vert = _contract_ZZ(tensor_ket_vert, A_tensors, tensor_bra, rho, R, row, total_qubits, optimizer)
    ZZ_horiz = _contract_ZZ(tensor_ket_horiz, tensor_ket_horiz, tensor_bra, rho, R, row, total_qubits, optimizer)
    
    return ZZ_vert, ZZ_horiz
end

"""
Helper function to contract ZZ expectation value.
"""
function _contract_ZZ(tensor_ket_a, tensor_ket_b, tensor_bra, rho, R, row, total_qubits, optimizer)
    _, list1 = contract_transfer_matrix(tensor_ket_a, tensor_bra, row; optimizer=optimizer)
    _, list2 = contract_transfer_matrix(tensor_ket_b, tensor_bra, row; optimizer=optimizer)
    
    store = IndexStore()
    index_list1 = [newindex!(store) for _ in 1:4*total_qubits]
    index_list2 = [[newindex!(store) for _ in 1:2*total_qubits]..., index_list1[1:2*total_qubits]...]
    
    index_rho = index_list1[2*total_qubits+1:4*total_qubits]
    index_R = index_list2[1:2*total_qubits]
    index = [index_list1, index_list2, index_rho, index_R]
    
    size_dict = OMEinsum.get_size_dict(index, [list1, list2, rho, R])
    code = optimize_code(DynamicEinCode(index, Int[]), size_dict, optimizer)
    return code(list1, list2, rho, R)[]
end



"""
    compute_exact_energy(params, g, J, p, row, virtual_qubits; optimizer=GreedyMethod())

Compute exact energy from parameters using tensor contraction.

# Arguments
- `params`: Parameter vector
- `g`: Transverse field strength
- `J`: Coupling strength
- `p`: Number of circuit layers
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits
- `optimizer`: Contraction optimizer

# Returns
- `gap`: Spectral gap
- `energy`: Ground state energy estimate
"""
function compute_exact_energy(params::Vector{Float64}, g::Float64, J::Float64, 
                               p::Int, row::Int, virtual_qubits::Int; optimizer=GreedyMethod())
    gates = build_unitary_gate(params, p, row, virtual_qubits; share_params=true)
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, virtual_qubits)
    
    X_cost = real(compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=optimizer))
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=optimizer)
    ZZ_vert = real(ZZ_vert)
    ZZ_horiz = real(ZZ_horiz)
    
    energy = -g*X_cost - J*(row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz) 
    return gap, energy
end

"""
    diagnose_transfer_channel(gates, row, virtual_qubits; verbose=true)

Diagnose why the transfer matrix behaves like a unitary (reversible) channel.

# Arguments
- `gates`: Vector of gate matrices (one per row)
- `row`: Number of rows in the PEPS
- `virtual_qubits`: Number of boundary/virtual qubits = (nqubits-1)÷2
  - For nqubits=3 gates (8×8), use virtual_qubits=1
  - For nqubits=5 gates (32×32), use virtual_qubits=2
- `verbose`: Print detailed diagnostics (default: true)

# Returns
Named tuple with:
- `eigenvalues`: All transfer matrix eigenvalues (sorted by magnitude)
- `gap`: Spectral gap = -log|λ₂|
- `unitality`: Deviation from unital channel (0 = perfectly unital/reversible)
- `kraus_structure`: Kraus operator analysis
- `gate_structure`: Gate structure analysis
- `diagnosis`: String explaining the likely cause

# Description
A quantum channel T[ρ] = Σᵢ KᵢρKᵢ† is:
- **Trace-preserving**: Σᵢ Kᵢ†Kᵢ = I (always satisfied)
- **Unital (reversible)**: Σᵢ KᵢKᵢ† = I (preserves identity)

When the channel is nearly unital, eigenvalues cluster near 1, giving small spectral gap.
This happens when the measurement outcome barely affects the boundary state.

# Example
```julia
gates, _, _, _ = reconstruct_gates("results/circuit.json")
virtual_qubits = 1  # for nqubits=3 gates
diag = diagnose_transfer_channel(gates, row, virtual_qubits)
```
"""
function diagnose_transfer_channel(gates, row, virtual_qubits; verbose=true)
    # Get full transfer matrix
    T = get_transfer_matrix(gates, row, virtual_qubits)
    
    # Compute all eigenvalues
    λs_all = eigvals(T)
    λs_mags = sort(abs.(λs_all), rev=true)
    
    # Spectral gap
    gap = length(λs_mags) > 1 ? -log(λs_mags[2]) : Inf
    
    # Count eigenvalues close to 1
    n_close_to_1 = count(x -> x > 0.99, λs_mags)
    n_close_to_095 = count(x -> x > 0.95, λs_mags)
    n_close_to_09 = count(x -> x > 0.9, λs_mags)
    
    # Check distance to identity
    I_mat = Matrix{ComplexF64}(I, size(T)...)
    dist_to_identity = norm(T - I_mat) / norm(I_mat)
    
    # Analyze Kraus operator structure (for row=1 case, generalizable)
    kraus_analysis = Dict{String, Any}()
    unitality = NaN
    is_unital = false
    is_unitary_channel = false
    
    if row == 1
        U = gates[1]
        # Infer nqubits from gate dimension (more robust)
        nqubits_gate = Int(log2(size(U, 1)))  # e.g., 8×8 gate → nqubits=3
        d_boundary = 2^(nqubits_gate - 1)  # Boundary dimension (excluding physical qubit)
        
        # Reshape gate to extract Kraus operators: K_m = ⟨m|U|0⟩
        U_tensor = reshape(U, ntuple(_ -> 2, 2 * nqubits_gate)...)
        
        # Fix phys_in = 0 (index value 1 in Julia)
        A = U_tensor[:, ntuple(_ -> Colon(), nqubits_gate - 1)..., 1, ntuple(_ -> Colon(), nqubits_gate - 1)...]
        
        K0 = reshape(A[1, ntuple(_ -> Colon(), 2*(nqubits_gate-1))...], d_boundary, d_boundary)
        K1 = reshape(A[2, ntuple(_ -> Colon(), 2*(nqubits_gate-1))...], d_boundary, d_boundary)
        
        # Check trace-preserving: Σᵢ Kᵢ†Kᵢ = I
        TP_check = K0' * K0 + K1' * K1
        TP_deviation = norm(TP_check - I(d_boundary))
        
        # Check unital: Σᵢ KᵢKᵢ† = I 
        unital_check = K0 * K0' + K1 * K1'
        unitality = norm(unital_check - I(d_boundary))
        is_unital = unitality < 0.01
        
        # Check unitarity of individual Kraus operators
        # A unitary Kraus operator K satisfies: K†K = KK† = I
        K0_dagger_K0 = K0' * K0
        K0_K0_dagger = K0 * K0'
        K1_dagger_K1 = K1' * K1
        K1_K1_dagger = K1 * K1'
        
        K0_isometry_dev = norm(K0_dagger_K0 - I(d_boundary))  # K†K = I (isometry)
        K0_coisometry_dev = norm(K0_K0_dagger - I(d_boundary))  # KK† = I (co-isometry)
        K1_isometry_dev = norm(K1_dagger_K1 - I(d_boundary))
        K1_coisometry_dev = norm(K1_K1_dagger - I(d_boundary))
        
        K0_is_unitary = (K0_isometry_dev < 0.01) && (K0_coisometry_dev < 0.01)
        K1_is_unitary = (K1_isometry_dev < 0.01) && (K1_coisometry_dev < 0.01)
        
        # Channel is unitary if one Kraus operator is unitary (and dominates)
        is_unitary_channel = K0_is_unitary || K1_is_unitary
        
        # Kraus operator properties
        K0_norm = norm(K0)
        K1_norm = norm(K1)
        K0_rank = rank(K0, rtol=1e-6)
        K1_rank = rank(K1, rtol=1e-6)
        
        # Check if Kraus operators are proportional (degenerate channel)
        if K1_norm > 1e-10 && K0_norm > 1e-10
            K0_normalized = K0 / K0_norm
            K1_normalized = K1 / K1_norm
            kraus_overlap = abs(tr(K0_normalized' * K1_normalized)) / d_boundary
        else
            kraus_overlap = NaN
        end
        
        kraus_analysis = Dict(
            "K0_norm" => K0_norm,
            "K1_norm" => K1_norm,
            "K0_rank" => K0_rank,
            "K1_rank" => K1_rank,
            "trace_preserving_deviation" => TP_deviation,
            "unitality_deviation" => unitality,
            "is_unital" => is_unital,
            "K0_isometry_deviation" => K0_isometry_dev,
            "K0_coisometry_deviation" => K0_coisometry_dev,
            "K0_is_unitary" => K0_is_unitary,
            "K1_isometry_deviation" => K1_isometry_dev,
            "K1_coisometry_deviation" => K1_coisometry_dev,
            "K1_is_unitary" => K1_is_unitary,
            "is_unitary_channel" => is_unitary_channel,
            "kraus_overlap" => kraus_overlap,
            "K0" => K0,
            "K1" => K1
        )
    else
        # For multi-row, extract Kraus operators from transfer matrix via Choi matrix
        d = Int(sqrt(size(T, 1)))
        
        # Convert transfer matrix T to Choi matrix C
        # T acts as vec(T[ρ]) = T · vec(ρ), where T = Σᵢ Kᵢ ⊗ K̄ᵢ
        # Choi matrix: C[i,j,k,l] = T[(i-1)*d+k, (j-1)*d+l] (reshuffle indices)
        C = zeros(ComplexF64, d^2, d^2)
        for i in 1:d, j in 1:d, k in 1:d, l in 1:d
            # T index: row = (i-1)*d + k, col = (j-1)*d + l
            # C index: row = (i-1)*d + j, col = (k-1)*d + l
            T_row = (i-1)*d + k
            T_col = (j-1)*d + l
            C_row = (i-1)*d + j
            C_col = (k-1)*d + l
            C[C_row, C_col] = T[T_row, T_col]
        end
        
        # Eigendecompose Choi matrix to get Kraus operators
        # C = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|, so Kᵢ = √λᵢ · reshape(vᵢ, d, d)
        choi_eig = eigen(Hermitian(C))
        choi_vals = real.(choi_eig.values)
        choi_vecs = choi_eig.vectors
        
        # Extract significant Kraus operators (those with λ > threshold)
        threshold = 1e-10
        significant_indices = findall(λ -> λ > threshold, choi_vals)
        n_kraus = length(significant_indices)
        
        kraus_ops = Vector{Matrix{ComplexF64}}(undef, n_kraus)
        for (idx, i) in enumerate(significant_indices)
            kraus_ops[idx] = sqrt(choi_vals[i]) * reshape(choi_vecs[:, i], d, d)
        end
        
        # Check trace-preserving: Σᵢ Kᵢ†Kᵢ = I
        TP_check = sum(K' * K for K in kraus_ops)
        TP_deviation = norm(TP_check - I(d))
        
        # Check unital: Σᵢ KᵢKᵢ† = I
        unital_check = sum(K * K' for K in kraus_ops)
        unitality = norm(unital_check - I(d))
        is_unital = unitality < 0.01
        
        # Check unitarity: ∃ Kᵢ such that Kᵢ†Kᵢ = KᵢKᵢ† = I
        kraus_unitarity_info = []
        is_unitary_channel = false
        for (idx, K) in enumerate(kraus_ops)
            K_dagger_K = K' * K
            K_K_dagger = K * K'
            isometry_dev = norm(K_dagger_K - I(d))
            coisometry_dev = norm(K_K_dagger - I(d))
            K_is_unitary = (isometry_dev < 0.01) && (coisometry_dev < 0.01)
            if K_is_unitary
                is_unitary_channel = true
            end
            push!(kraus_unitarity_info, Dict(
                "index" => idx,
                "norm" => norm(K),
                "isometry_deviation" => isometry_dev,
                "coisometry_deviation" => coisometry_dev,
                "is_unitary" => K_is_unitary
            ))
        end
        
        kraus_analysis = Dict(
            "n_kraus_operators" => n_kraus,
            "trace_preserving_deviation" => TP_deviation,
            "unitality_deviation" => unitality,
            "is_unital" => is_unital,
            "is_unitary_channel" => is_unitary_channel,
            "kraus_unitarity_info" => kraus_unitarity_info,
            "kraus_operators" => kraus_ops
        )
    end
    
    # Analyze gate structure
    gate_analysis = Dict{String, Any}()
    for (i, U) in enumerate(gates)
        U_mat = Matrix(U)
        
        # Check unitarity
        is_unitary = norm(U_mat * U_mat' - I) < 1e-10
        
        # Gate eigenvalues (phases)
        gate_eigs = eigvals(U_mat)
        phases = angle.(gate_eigs)
        phase_spread = maximum(phases) - minimum(phases)
        
        # Distance to identity
        gate_dist_to_I = norm(U_mat - I) / norm(U_mat)
        
        # Check for block structure (indicative of symmetry)
        U_abs = abs.(U_mat)
        off_diag_density = norm(U_abs - Diagonal(diag(U_abs))) / norm(U_abs)
        
        gate_analysis["gate_$i"] = Dict(
            "is_unitary" => is_unitary,
            "phase_spread" => phase_spread,
            "dist_to_identity" => gate_dist_to_I,
            "off_diagonal_density" => off_diag_density
        )
    end
    
    # Generate diagnosis
    diagnosis = _generate_diagnosis(λs_mags, unitality, is_unital, is_unitary_channel, gap, n_close_to_1, dist_to_identity)
    
    if verbose
        println("=" ^ 70)
        println("TRANSFER CHANNEL DIAGNOSTIC REPORT")
        println("=" ^ 70)
        
        println("\n📊 EIGENVALUE SPECTRUM")
        println("   Top 10 eigenvalue magnitudes:")
        for i in 1:min(10, length(λs_mags))
            marker = λs_mags[i] > 0.99 ? " ⚠️" : (λs_mags[i] > 0.95 ? " ⚡" : "")
            println("     λ_$i = $(round(λs_mags[i], digits=6))$marker")
        end
        println("   Spectral gap: $(round(gap, digits=4))")
        println("   Eigenvalues > 0.99: $n_close_to_1 / $(length(λs_mags))")
        println("   Eigenvalues > 0.95: $n_close_to_095 / $(length(λs_mags))")
        println("   Eigenvalues > 0.90: $n_close_to_09 / $(length(λs_mags))")
        
        println("\n🔄 CHANNEL PROPERTIES")
        println("   Distance to identity: $(round(dist_to_identity, digits=6))")
        
        # Unitality check: Σᵢ KᵢKᵢ† = I
        println("\n   📐 UNITALITY CHECK (Σᵢ KᵢKᵢ† = I):")
        println("      Deviation from identity: $(round(unitality, digits=6))")
        if is_unital
            println("      ⚠️  Channel IS UNITAL - preserves maximally mixed state")
        elseif unitality < 0.1
            println("      ⚠️  Channel is nearly unital")
        elseif unitality < 0.3
            println("      ⚡ Channel has moderate unitality deviation")
        else
            println("      ✓  Channel is properly dissipative (non-unital)")
        end
        
        # Unitarity check for multi-row (based on Kraus operators)
        if row > 1 && haskey(kraus_analysis, "kraus_unitarity_info")
            println("\n   🔲 UNITARITY CHECK (∃ K: K†K = KK† = I):")
            println("      Number of Kraus operators: $(kraus_analysis["n_kraus_operators"])")
            # Show all Kraus operators
            kraus_info = kraus_analysis["kraus_unitarity_info"]
            for info in kraus_info
                status = info["is_unitary"] ? " ⚠️ UNITARY" : ""
                println("      K_$(info["index"]): ||K||=$(round(info["norm"], digits=4)), K†K-I=$(round(info["isometry_deviation"], digits=6)), KK†-I=$(round(info["coisometry_deviation"], digits=6))$status")
            end
            if is_unitary_channel
                println("      ⚠️  Channel IS UNITARY - one Kraus operator K satisfies K†K = KK† = I")
            else
                println("      ✓  Channel is NOT unitary - proper quantum channel")
            end
        end
        
        if row == 1 && haskey(kraus_analysis, "K0_norm")
            # Unitarity check: ∃ Kᵢ such that Kᵢ†Kᵢ = KᵢKᵢ† = I
            println("\n   🔲 UNITARITY CHECK (∃ K: K†K = KK† = I):")
            println("      K₀: K₀†K₀-I = $(round(kraus_analysis["K0_isometry_deviation"], digits=6)), K₀K₀†-I = $(round(kraus_analysis["K0_coisometry_deviation"], digits=6))")
            println("      K₁: K₁†K₁-I = $(round(kraus_analysis["K1_isometry_deviation"], digits=6)), K₁K₁†-I = $(round(kraus_analysis["K1_coisometry_deviation"], digits=6))")
            if kraus_analysis["K0_is_unitary"]
                println("      ⚠️  K₀ IS UNITARY")
            end
            if kraus_analysis["K1_is_unitary"]
                println("      ⚠️  K₁ IS UNITARY")
            end
            if kraus_analysis["is_unitary_channel"]
                println("      ⚠️  Channel IS UNITARY - one Kraus operator dominates as a unitary")
            else
                println("      ✓  Channel is NOT unitary - proper quantum channel")
            end
            
            println("\n🔧 KRAUS OPERATOR ANALYSIS (row=1)")
            println("   K₀ norm: $(round(kraus_analysis["K0_norm"], digits=4)), rank: $(kraus_analysis["K0_rank"])")
            println("   K₁ norm: $(round(kraus_analysis["K1_norm"], digits=4)), rank: $(kraus_analysis["K1_rank"])")
            println("   Trace-preserving (Σᵢ Kᵢ†Kᵢ = I) deviation: $(round(kraus_analysis["trace_preserving_deviation"], digits=8))")
            println("   Kraus overlap |tr(K₀†K₁)|/d: $(round(kraus_analysis["kraus_overlap"], digits=4))")
            
            # Physical interpretation
            K0_prob = kraus_analysis["K0_norm"]^2
            K1_prob = kraus_analysis["K1_norm"]^2
            total = K0_prob + K1_prob
            println("   Measurement probabilities: P(0)≈$(round(K0_prob/total, digits=3)), P(1)≈$(round(K1_prob/total, digits=3))")
        end
        
        # Kraus analysis for multi-row (extracted from Choi matrix)
        if row > 1 && haskey(kraus_analysis, "n_kraus_operators")
            println("\n🔧 KRAUS OPERATOR ANALYSIS (row=$row, from Choi matrix)")
            println("   Number of Kraus operators: $(kraus_analysis["n_kraus_operators"])")
            println("   Trace-preserving (Σᵢ Kᵢ†Kᵢ = I) deviation: $(round(kraus_analysis["trace_preserving_deviation"], digits=8))")
        end
        
        println("\n🎯 GATE STRUCTURE")
        for (key, val) in gate_analysis
            println("   $key:")
            println("      Phase spread: $(round(val["phase_spread"], digits=4)) rad")
            println("      Distance to I: $(round(val["dist_to_identity"], digits=4))")
            println("      Off-diagonal density: $(round(val["off_diagonal_density"], digits=4))")
        end
        
        println("\n" * "=" ^ 70)
        println("📋 DIAGNOSIS")
        println("=" ^ 70)
        println(diagnosis)
        println("=" ^ 70)
    end
    
    return (
        eigenvalues = λs_mags,
        eigenvalues_complex = λs_all,
        gap = gap,
        unitality = unitality,
        is_unital = is_unital,
        is_unitary_channel = is_unitary_channel,
        kraus_structure = kraus_analysis,
        gate_structure = gate_analysis,
        dist_to_identity = dist_to_identity,
        diagnosis = diagnosis,
        transfer_matrix = T
    )
end

"""
Generate a diagnosis string based on the analysis.
"""
function _generate_diagnosis(λs_mags, unitality, is_unital, is_unitary_channel, gap, n_close_to_1, dist_to_identity)
    issues = String[]
    suggestions = String[]
    
    # Check for unitary channel (∃ K: K†K = KK† = I)
    if is_unitary_channel
        push!(issues, "• The channel is UNITARY (one Kraus operator K satisfies K†K = KK† = I)")
        push!(issues, "  This means the channel acts as a unitary transformation, not a true quantum channel.")
        push!(issues, "  The transfer matrix will have all eigenvalues on the unit circle.")
        push!(suggestions, "→ This is problematic: unitary channels have no unique fixed point")
        push!(suggestions, "→ The gate is not performing proper measurement/decoherence")
        push!(suggestions, "→ Try different initialization or add noise to break unitarity")
    end
    
    # Check for unital channel (Σᵢ KᵢKᵢ† = I)
    if is_unital
        push!(issues, "• The channel is UNITAL (Σᵢ KᵢKᵢ† = I, deviation = $(round(unitality, digits=6)))")
        push!(issues, "  This means the channel preserves the maximally mixed state.")
        push!(issues, "  Physically: measurement outcomes don't bias the boundary state.")
        push!(suggestions, "→ Unital channels can still have good spectral gaps")
        push!(suggestions, "→ But if gap is small, add gap regularization to the objective")
    elseif !isnan(unitality) && unitality < 0.1
        push!(issues, "• The channel is nearly UNITAL (unitality deviation = $(round(unitality, digits=4)))")
        push!(issues, "  This means Σᵢ KᵢKᵢ† ≈ I, so the channel nearly preserves the maximally mixed state.")
        push!(suggestions, "→ Add gap regularization to the objective function")
        push!(suggestions, "→ Use a different ansatz that prevents product-state gates")
    end
    
    # Check for multiple eigenvalues near 1
    if n_close_to_1 > 1
        push!(issues, "• Multiple eigenvalues ($(n_close_to_1)) are close to 1")
        push!(issues, "  This indicates (near-)degenerate fixed points or symmetry.")
        push!(suggestions, "→ Check for discrete symmetries (Z₂, etc.) in the gate")
        push!(suggestions, "→ Try symmetry-breaking initialization")
    end
    
    # Check for small gap
    if gap < 0.05
        push!(issues, "• Spectral gap is very small ($(round(gap, digits=4)))")
        push!(issues, "  Correlation length ξ = 1/gap = $(round(1/gap, digits=1)) is very large.")
        push!(suggestions, "→ If near critical point (g ≈ J), this may be physical")
        push!(suggestions, "→ Otherwise, add gap penalty: objective += α * max(0, gap_min - gap)")
    end
    
    # Check if close to identity
    if dist_to_identity < 0.3
        push!(issues, "• Transfer matrix is close to identity (dist = $(round(dist_to_identity, digits=4)))")
        push!(issues, "  The channel is barely transforming the state.")
        push!(suggestions, "→ Check if gate parameters have converged to trivial values")
        push!(suggestions, "→ Try different random initialization")
    end
    
    # Compile diagnosis
    if isempty(issues)
        return "✓ No obvious issues detected. The channel appears healthy."
    end
    
    diagnosis = "ISSUES DETECTED:\n" * join(issues, "\n") * "\n\n"
    diagnosis *= "SUGGESTIONS:\n" * join(suggestions, "\n")
    
    return diagnosis
end

"""
    diagnose_from_params(params, p, row, nqubits; share_params=true, verbose=true)

Convenience function to diagnose transfer channel from circuit parameters.

# Example
```julia
params = result.final_params  # from CircuitOptimizationResult
diag = diagnose_from_params(params, 4, 3, 3)
```
"""
function diagnose_from_params(params, p, row, nqubits; share_params=true, verbose=true)
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    virtual_qubits = (nqubits - 1) ÷ 2  # Convert nqubits to virtual_qubits
    return diagnose_transfer_channel(gates, row, virtual_qubits; verbose=verbose)
end
