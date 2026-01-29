# =============================================================================
# Entanglement Entropy Calculations
# =============================================================================
# Functions for computing entanglement entropy of MPS states
# Uses KrylovKit for efficient eigensolvers

# =============================================================================
# Single-Line MPS Entanglement
# =============================================================================

"""
    mps_bond_entanglement(A; tol=1e-12)

Compute the bond entanglement of an MPS tensor via SVD.
This measures entanglement between (physical ⊗ left) and right subsystems.

# Arguments
- `A`: MPS tensor with indices (physical, left, right)
- `tol`: Tolerance for filtering small singular values

# Returns
- `S`: Entanglement entropy S = -Σᵢ λᵢ² log(λᵢ²)
- `schmidt_values`: Normalized Schmidt values
"""
function mps_bond_entanglement(A; tol=1e-12)
    phys_dim, left_dim, right_dim = size(A)
    M = reshape(A, phys_dim * left_dim, right_dim)
    
    svd_result = svd(M)
    σ = filter(s -> s > tol, svd_result.S)
    
    isempty(σ) && return 0.0, Float64[]
    
    schmidt_values = σ ./ norm(σ)
    S = -sum(λ -> λ^2 * log(λ^2), schmidt_values)
    
    return S, schmidt_values
end

"""
    mps_physical_entanglement(A, N; tol=1e-12)

Compute the true physical bipartite entanglement entropy of a uniform MPS.
Constructs the full state vector and computes entanglement across a middle cut.

# Arguments
- `A`: MPS tensor with indices (physical, left, right)
- `N`: Number of sites (must be even, kept small due to exponential scaling)
- `tol`: Tolerance for filtering small singular values

# Returns
- `S`: Physical entanglement entropy across the middle cut
- `schmidt_values`: Schmidt coefficients

# Notes
This function constructs the full 2^N dimensional state vector,
so it only works for small N (≲ 20).
"""
function mps_physical_entanglement(A, N; tol=1e-12)
    phys_dim, bond_dim, _ = size(A)
    total_dim = phys_dim^N
    ψ = zeros(ComplexF64, total_dim)
    
    # Build |ψ⟩ = Σ_{s₁...sₙ} Tr(M^{s₁}...M^{sₙ}) |s₁...sₙ⟩
    for config in 0:(total_dim-1)
        indices = digits(config, base=phys_dim, pad=N) .+ 1
        M_product = Matrix{ComplexF64}(I, bond_dim, bond_dim)
        for s in indices
            M_product = M_product * A[s, :, :]
        end
        ψ[config + 1] = tr(M_product)
    end
    
    ψ = ψ / norm(ψ)
    
    # Bipartite cut in the middle
    left_sites = N ÷ 2
    left_dim = phys_dim^left_sites
    right_dim = phys_dim^(N - left_sites)
    
    ψ_matrix = reshape(ψ, left_dim, right_dim)
    σ = svd(ψ_matrix).S
    σ = filter(s -> s > tol, σ)
    
    isempty(σ) && return 0.0, Float64[]
    
    S = -sum(λ -> λ > tol ? λ^2 * log(λ^2) : 0.0, σ)
    
    return S, σ
end

"""
    mps_physical_entanglement_infinite(A; tol=1e-12)

Compute the physical bipartite entanglement entropy of a single-line uniform MPS 
in the thermodynamic limit using transfer matrix fixed points.

Uses KrylovKit for efficient eigensolver.

# Arguments
- `A`: MPS tensor with indices (physical, left, right)
- `tol`: Tolerance for filtering small singular values

# Returns
- `S`: Entanglement entropy in the thermodynamic limit
- `schmidt_values`: Schmidt coefficients

# Limitations
⚠️ This algorithm assumes the transfer matrix has a **unique dominant eigenvalue**.
It FAILS for states with degenerate transfer matrices (e.g., GHZ, cat states).
"""
function mps_physical_entanglement_infinite(A; tol=1e-12)
    phys_dim, bond_dim, _ = size(A)
    
    # Build transfer matrix using einsum: E = Σ_s A^s ⊗ (A^s)*
    # E_{(α,α'), (β,β')} = Σ_s A[s,α,β] * conj(A[s,α',β'])
    E = zeros(ComplexF64, bond_dim^2, bond_dim^2)
    for s in 1:phys_dim
        As = A[s, :, :]
        E .+= kron(As, conj(As))
    end
    
    # Get right fixed point using KrylovKit
    vals, vecs, _ = KrylovKit.eigsolve(E, randn(ComplexF64, bond_dim^2), 1, :LM;
                                       ishermitian=false, krylovdim=min(30, bond_dim^2))
    r = reshape(vecs[1], bond_dim, bond_dim)
    
    # Get left fixed point  
    vals_l, vecs_l, _ = KrylovKit.eigsolve(E', randn(ComplexF64, bond_dim^2), 1, :LM;
                                           ishermitian=false, krylovdim=min(30, bond_dim^2))
    l = reshape(vecs_l[1], bond_dim, bond_dim)
    
    # Make Hermitian and normalize
    r = (r + r') / 2
    l = (l + l') / 2
    norm_factor = tr(l * r)
    if abs(norm_factor) > tol
        r = r / norm_factor
    end
    
    # Truncate to physical subspace and get Schmidt values
    Λl, Ul = eigen(Hermitian(l))
    Λr, Ur = eigen(Hermitian(r))
    
    # Keep only significant eigenvalues
    keep_l = findall(λ -> λ > tol, real.(Λl))
    keep_r = findall(λ -> λ > tol, real.(Λr))
    
    isempty(keep_l) || isempty(keep_r) && return 0.0, Float64[]
    
    Λl = real.(Λl[keep_l])
    Λr = real.(Λr[keep_r])
    Ul = Ul[:, keep_l]
    Ur = Ur[:, keep_r]
    
    X = Diagonal(sqrt.(Λl)) * Ul'
    Y = Diagonal(sqrt.(Λr)) * Ur'
    
    C = X * Y'
    σ = svd(C).S
    σ = filter(s -> s > tol, σ)
    
    isempty(σ) && return 0.0, Float64[]
    
    σ = σ ./ norm(σ)
    
    S = -sum(λ -> λ > tol ? λ^2 * log(λ^2) : 0.0, σ)
    return S, σ
end

# =============================================================================
# Multiline MPS Entanglement
# =============================================================================

"""
    _is_tensor_input(input)

Check if input is already a tensor (5D array) or a gate matrix (2D array).
"""
function _is_tensor_input(input::Vector)
    isempty(input) && return false
    return ndims(input[1]) == 5
end

"""
    _infer_parameters_from_tensors(A_tensors)

Infer virtual_qubits and bond_dim from tensor dimensions.
"""
function _infer_parameters_from_tensors(A_tensors)
    A = A_tensors[1]
    bond_dim = size(A, 2)
    virtual_qubits = Int(log2(bond_dim))
    return virtual_qubits, bond_dim
end

"""
    multiline_mps_entanglement(input, row; nqubits=nothing, tol=1e-12, use_iterative=:auto, matrix_free=:auto)

Compute the physical bipartite entanglement entropy across the vertical cut 
for a multiline uniform MPS in the thermodynamic limit.

Uses KrylovKit for efficient transfer matrix eigensolvers.

# Arguments
- `input`: Vector of gate matrices or tensors
- `row`: Number of rows
- `nqubits`: Number of qubits per gate (required for gates)
- `tol`: Tolerance for filtering
- `use_iterative`: `:auto`, `:always`, or `:never`
- `matrix_free`: `:auto`, `:always`, or `:never`

# Returns
- `S`: Entanglement entropy
- `spectrum`: Schmidt values squared
- `gap`: Transfer matrix spectral gap
"""
function multiline_mps_entanglement(input, row; nqubits=nothing, tol=1e-12, use_iterative=:auto, matrix_free=:auto)
    # Get tensors
    if _is_tensor_input(input)
        A_tensors = input
        virtual_qubits, bond_dim = _infer_parameters_from_tensors(A_tensors)
    else
        nqubits === nothing && error("nqubits must be provided for gate matrices")
        virtual_qubits = (nqubits - 1) ÷ 2
        bond_dim = 2^virtual_qubits
        A_tensors = gates_to_tensors(input, row, virtual_qubits)
    end
    
    total_legs = row + 1
    env_dim = bond_dim^total_legs
    matrix_size = env_dim^2
    
    # Build transfer matrix
    _, T = contract_transfer_matrix(A_tensors, [conj(A) for A in A_tensors], row)
    T_matrix = reshape(T, matrix_size, matrix_size)
    
    # Decide on solver
    should_use_iterative = use_iterative == :always || (use_iterative == :auto && matrix_size > 256)
    
    if should_use_iterative
        # KrylovKit for large matrices
        vals, vecs, _ = KrylovKit.eigsolve(T_matrix, randn(ComplexF64, matrix_size), 2, :LM;
                                           ishermitian=false, krylovdim=min(30, matrix_size))
        sorted_idx = sortperm(abs.(vals), rev=true)
        eigenvalues = abs.(vals[sorted_idx])
        gap = length(eigenvalues) > 1 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
        
        rho_R = reshape(vecs[sorted_idx[1]], env_dim, env_dim)
        rho_R = rho_R ./ tr(rho_R)
        
        vals_l, vecs_l, _ = KrylovKit.eigsolve(T_matrix', randn(ComplexF64, matrix_size), 1, :LM;
                                               ishermitian=false, krylovdim=min(30, matrix_size))
        rho_L = reshape(vecs_l[argmax(abs.(vals_l))], env_dim, env_dim)
    else
        # Full eigendecomposition
        eig_result = eigen(T_matrix)
        sorted_idx = sortperm(abs.(eig_result.values), rev=true)
        eigenvalues = abs.(eig_result.values[sorted_idx])
        gap = length(eigenvalues) > 1 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
        
        rho_R = reshape(eig_result.vectors[:, sorted_idx[1]], env_dim, env_dim)
        rho_R = rho_R ./ tr(rho_R)
        
        eig_l = eigen(T_matrix')
        rho_L = reshape(eig_l.vectors[:, argmax(abs.(eig_l.values))], env_dim, env_dim)
    end
    
    # Compute entanglement
    S, spectrum = _compute_bipartite_entanglement(A_tensors, rho_L, rho_R, row, bond_dim, tol)
    
    return S, spectrum, gap
end

# Backward compatible
function multiline_mps_entanglement(gates, row, nqubits::Int; tol=1e-12, use_iterative=:auto, matrix_free=:auto)
    return multiline_mps_entanglement(gates, row; nqubits=nqubits, tol=tol, 
                                       use_iterative=use_iterative, matrix_free=matrix_free)
end

"""
    _compute_bipartite_entanglement(A_tensors, rho_L, rho_R, row, bond_dim, tol)

Compute physical bipartite entanglement including the column tensor.
Truncates to physical subspace to handle gauge redundancy.
"""
function _compute_bipartite_entanglement(A_tensors, rho_L, rho_R, row, bond_dim, tol)
    boundary_dim = size(rho_L, 1)
    phys_dim = 2^row
    
    # Make Hermitian
    rho_L = Hermitian((rho_L + rho_L') / 2)
    rho_R = Hermitian((rho_R + rho_R') / 2)
    
    # Normalize
    norm_factor = tr(rho_L * rho_R)
    if abs(norm_factor) > tol
        rho_R = rho_R / norm_factor
    end
    
    # Eigendecompose and truncate
    Λ_L_full, U_L_full = eigen(rho_L)
    Λ_R_full, U_R_full = eigen(rho_R)
    
    keep_L = findall(λ -> real(λ) > tol, Λ_L_full)
    keep_R = findall(λ -> real(λ) > tol, Λ_R_full)
    
    (isempty(keep_L) || isempty(keep_R)) && return 0.0, Float64[]
    
    Λ_L = real.(Λ_L_full[keep_L])
    Λ_R = real.(Λ_R_full[keep_R])
    U_L = U_L_full[:, keep_L]
    U_R = U_R_full[:, keep_R]
    
    rank_L = length(Λ_L)
    rank_R = length(Λ_R)
    
    # Gauge transformations
    X = Diagonal(1.0 ./ sqrt.(Λ_L)) * U_L'
    Y = Diagonal(1.0 ./ sqrt.(Λ_R)) * U_R'
    
    # Build transformed column tensor
    C_transformed = zeros(ComplexF64, phys_dim, rank_L, rank_R)
    
    for s in 0:(phys_dim-1)
        site_values = digits(s, base=2, pad=row) .+ 1
        M_col = _contract_column_to_matrix(A_tensors, site_values, bond_dim, row)
        C_transformed[s+1, :, :] = X * M_col * Y'
    end
    
    # SVD
    C_matrix = reshape(C_transformed, phys_dim * rank_L, rank_R)
    σ = svd(C_matrix).S
    σ = σ ./ norm(σ)
    σ = filter(s -> s > tol, σ)
    
    isempty(σ) && return 0.0, Float64[]
    
    spectrum = σ.^2
    S = -sum(p -> p > tol ? p * log(p) : 0.0, spectrum)
    
    return S, spectrum
end

"""
    _contract_column_to_matrix(A_tensors, phys_indices, bond_dim, row)

Contract a single column with fixed physical indices.
Returns boundary-to-boundary matrix.
"""
function _contract_column_to_matrix(A_tensors, phys_indices, bond_dim, row)
    # A_tensors[r] has shape (physical=2, down, right, up, left)
    tensors_fixed = [A_tensors[r][phys_indices[r], :, :, :, :] for r in 1:row]
    
    boundary_dim = bond_dim^(row+1)
    M = zeros(ComplexF64, boundary_dim, boundary_dim)
    
    # Sum over internal vertical bonds
    n_internal = row - 1
    n_configs = bond_dim^n_internal
    
    for internal_config in 0:(n_configs-1)
        internal_bonds = n_internal > 0 ? digits(internal_config, base=bond_dim, pad=n_internal) .+ 1 : Int[]
        
        for left_config in 0:(boundary_dim-1)
            left_bonds = digits(left_config, base=bond_dim, pad=row+1) .+ 1
            
            for right_config in 0:(boundary_dim-1)
                right_bonds = digits(right_config, base=bond_dim, pad=row+1) .+ 1
                
                col_amplitude = ComplexF64(1.0)
                for r in 1:row
                    A = tensors_fixed[r]  # [down, right, up, left]
                    
                    up_idx = (r == 1) ? left_bonds[1] : internal_bonds[r-1]
                    down_idx = (r == row) ? left_bonds[1] : internal_bonds[r]
                    left_idx = left_bonds[r+1]
                    right_idx = right_bonds[r+1]
                    
                    col_amplitude *= A[down_idx, right_idx, up_idx, left_idx]
                end
                
                M[left_config+1, right_config+1] += col_amplitude
            end
        end
    end
    
    return M
end

"""
    multiline_mps_entanglement_from_params(params, p, row, nqubits; share_params=true, tol=1e-12)

Compute entanglement directly from circuit parameters.
"""
function multiline_mps_entanglement_from_params(params::Vector{Float64}, p::Int, row::Int, nqubits::Int; 
                                                 share_params::Bool=true, tol::Float64=1e-12)
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    return multiline_mps_entanglement(gates, row, nqubits; tol=tol)
end

# =============================================================================
# Finite-Width Physical Entanglement
# =============================================================================

"""
    multiline_mps_physical_entanglement(input, row, width; nqubits=nothing, tol=1e-12)

Compute physical bipartite entanglement for a FINITE-width system.

# Arguments
- `input`: Vector of gate matrices or tensors
- `row`: Number of rows
- `width`: Number of columns (≤ 6 for row=3)
- `nqubits`: Number of qubits per gate
- `tol`: Tolerance

# Returns
- `S`: Entanglement entropy
- `spectrum`: Schmidt values squared
- `gap`: Transfer matrix gap
"""
function multiline_mps_physical_entanglement(input, row, width; nqubits=nothing, tol=1e-12)
    if _is_tensor_input(input)
        A_tensors = input
        virtual_qubits, bond_dim = _infer_parameters_from_tensors(A_tensors)
    else
        nqubits === nothing && error("nqubits must be provided")
        virtual_qubits = (nqubits - 1) ÷ 2
        bond_dim = 2^virtual_qubits
        A_tensors = gates_to_tensors(input, row, virtual_qubits)
    end
    
    total_sites = row * width
    phys_dim = 2
    total_dim = phys_dim^total_sites
    
    total_sites > 20 && error("Too many sites ($total_sites). Use width ≤ $(20 ÷ row)")
    
    boundary_dim = bond_dim^(row+1)
    ψ = zeros(ComplexF64, total_dim)
    
    # Build state vector
    for config in 0:(total_dim-1)
        site_values = digits(config, base=phys_dim, pad=total_sites) .+ 1
        
        M_product = Matrix{ComplexF64}(I, boundary_dim, boundary_dim)
        for col in 1:width
            col_phys = [site_values[(col-1)*row + r] for r in 1:row]
            M_col = _contract_column_to_matrix(A_tensors, col_phys, bond_dim, row)
            M_product = M_product * M_col
        end
        
        ψ[config + 1] = tr(M_product)
    end
    
    ψ = ψ / norm(ψ)
    
    # Bipartite SVD
    left_sites = row * (width ÷ 2)
    left_dim = phys_dim^left_sites
    right_dim = phys_dim^(total_sites - left_sites)
    
    ψ_matrix = reshape(ψ, left_dim, right_dim)
    σ = svd(ψ_matrix).S
    σ = filter(s -> s > tol, σ)
    
    # Compute gap
    _, T = contract_transfer_matrix(A_tensors, [conj(A) for A in A_tensors], row)
    T_matrix = reshape(T, boundary_dim^2, boundary_dim^2)
    eigs = abs.(eigvals(T_matrix))
    eigs_sorted = sort(eigs, rev=true)
    gap = length(eigs_sorted) > 1 ? -log(eigs_sorted[2] / eigs_sorted[1]) : Inf
    
    isempty(σ) && return 0.0, Float64[], gap
    
    S = -sum(λ -> λ > tol ? λ^2 * log(λ^2) : 0.0, σ)
    spectrum = σ.^2
    
    return S, spectrum, gap
end

"""
    multiline_mps_physical_entanglement_from_params(params, p, row, width, nqubits; share_params=true, tol=1e-12)

Compute finite-width entanglement from circuit parameters.
"""
function multiline_mps_physical_entanglement_from_params(params::Vector{Float64}, p::Int, row::Int, 
                                                          width::Int, nqubits::Int; 
                                                          share_params::Bool=true, tol::Float64=1e-12)
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    return multiline_mps_physical_entanglement(gates, row, width; nqubits=nqubits, tol=tol)
end

# =============================================================================
# Infinite-Width Physical Entanglement via Physical Channel
# =============================================================================

"""
    multiline_mps_physical_entanglement_infinite(input, row; nqubits=nothing, tol=1e-12)

Compute physical bipartite entanglement for an INFINITE-width system using the physical channel.

This function computes the correct physical entanglement when the virtual transfer matrix
has degenerate eigenvalues (all ~1). It uses the physical channel fixed point instead of
the virtual transfer matrix fixed point.

# Arguments
- `input`: Vector of gate matrices or tensors
- `row`: Number of rows
- `nqubits`: Number of qubits per gate (required for gate matrices)
- `tol`: Tolerance for filtering small eigenvalues

# Returns
- `S`: Physical entanglement entropy (von Neumann entropy of physical channel fixed point)
- `spectrum`: Eigenvalues of the physical fixed point (probabilities)
- `gap`: Physical channel spectral gap

# Notes
For isometric PEPS:
- When the physical channel has a large gap, the physical state is close to a product state
- The entropy S should be small in this case
- This differs from `multiline_mps_entanglement` which computes virtual bond entanglement
"""
function multiline_mps_physical_entanglement_infinite(input, row; nqubits=nothing, tol=1e-12)
    # Get tensors and parameters
    if _is_tensor_input(input)
        A_tensors = input
        virtual_qubits, bond_dim = _infer_parameters_from_tensors(A_tensors)
        nqubits_computed = 2 * virtual_qubits + 1
    else
        nqubits === nothing && error("nqubits must be provided for gate matrices")
        virtual_qubits = (nqubits - 1) ÷ 2
        bond_dim = 2^virtual_qubits
        nqubits_computed = nqubits
        A_tensors = gates_to_tensors(input, row, virtual_qubits)
    end
    
    total_legs = row + 1
    env_dim = bond_dim^total_legs
    phys_dim = 2^row
    
    # Step 1: Compute virtual transfer matrix fixed point (needed for physical channel)
    _, T = contract_transfer_matrix(A_tensors, [conj(A) for A in A_tensors], row)
    T_matrix = reshape(T, env_dim^2, env_dim^2)
    
    # Get virtual fixed points
    eig_result = eigen(T_matrix)
    sorted_idx = sortperm(abs.(eig_result.values), rev=true)
    rho_virtual = reshape(eig_result.vectors[:, sorted_idx[1]], env_dim, env_dim)
    rho_virtual = rho_virtual ./ tr(rho_virtual)
    
    # Step 2: Build physical channel using virtual fixed point
    E = get_physical_channel(input, row, virtual_qubits, rho_virtual)
    
    # Step 3: Compute physical channel spectrum and fixed point
    E_eig = eigen(E)
    E_sorted_idx = sortperm(abs.(E_eig.values), rev=true)
    E_eigenvalues = abs.(E_eig.values[E_sorted_idx])
    
    # Physical channel gap
    gap = length(E_eigenvalues) > 1 && E_eigenvalues[1] > tol ? 
          -log(E_eigenvalues[2] / E_eigenvalues[1]) : Inf
    
    # Fixed point of physical channel (dominant eigenvector)
    # This is a vector of length phys_dim representing the stationary distribution
    # The physical channel E is phys_dim × phys_dim, so its eigenvector has length phys_dim
    sigma_vec = E_eig.vectors[:, E_sorted_idx[1]]
    
    # The fixed point represents probabilities over physical configurations
    # Take absolute value squared to get probabilities (eigenvector can have complex phases)
    probs = abs.(sigma_vec).^2
    
    # Normalize to get probability distribution
    probs = probs ./ sum(probs)
    
    # Filter small probabilities
    probs_filtered = filter(p -> p > tol, probs)
    
    if isempty(probs_filtered)
        return 0.0, Float64[], gap
    end
    
    # Renormalize after filtering
    probs_filtered = probs_filtered ./ sum(probs_filtered)
    
    # Shannon entropy of the probability distribution
    # This is the physical entanglement entropy in the infinite system limit
    S = -sum(p -> p > tol ? p * log(p) : 0.0, probs_filtered)
    
    return S, probs_filtered, gap
end

"""
    multiline_mps_physical_entanglement_infinite(gates, row, nqubits::Int; tol=1e-12)

Backward compatible version with positional nqubits argument.
"""
function multiline_mps_physical_entanglement_infinite(gates, row, nqubits::Int; tol=1e-12)
    return multiline_mps_physical_entanglement_infinite(gates, row; nqubits=nqubits, tol=tol)
end
