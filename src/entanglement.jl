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

Uses the canonical form procedure:
1. Build transfer matrix E = Σ_s A^s ⊗ (A^s)*
2. Get left fixed point l: E(l) = l (or l·E = l as a row vector)
3. Get right fixed point r: E†(r) = r
4. Normalize so Tr(l·r) = 1
5. Build reduced density matrix: ρ = √l · r · √l
6. Compute entropy: S = -Tr(ρ log ρ)

# Arguments
- `A`: MPS tensor with indices (physical, left, right)
- `tol`: Tolerance for filtering small eigenvalues

# Returns
- `S`: Entanglement entropy in the thermodynamic limit
- `spectrum`: Eigenvalues of ρ (the entanglement spectrum)

# Limitations
⚠️ This algorithm assumes the transfer matrix has a **unique dominant eigenvalue**.
It may give incorrect results for states with degenerate transfer matrices (e.g., GHZ, cat states).
"""
function mps_physical_entanglement_infinite(A; tol=1e-12)
    phys_dim, bond_dim, _ = size(A)
    
    # Step 1: Build transfer matrix E = Σ_s A^s ⊗ (A^s)*
    # E_{(α,α'), (β,β')} = Σ_s A[s,α,β] * conj(A[s,α',β'])
    E = zeros(ComplexF64, bond_dim^2, bond_dim^2)
    for s in 1:phys_dim
        As = A[s, :, :]
        E .+= kron(As, conj(As))
    end
    
    # Step 2: Get right fixed point r: E·r = λ·r (dominant eigenvector)
    vals, vecs, _ = KrylovKit.eigsolve(E, randn(ComplexF64, bond_dim^2), 1, :LM;
                                       ishermitian=false, krylovdim=min(30, bond_dim^2))
    l = reshape(vecs[1], bond_dim, bond_dim)
    
    # Step 3: Get left fixed point l: E†·l = λ·l (or l·E = λ·l)
    vals_r, vecs_r, _ = KrylovKit.eigsolve(E', randn(ComplexF64, bond_dim^2), 1, :LM;
                                           ishermitian=false, krylovdim=min(30, bond_dim^2))
    r = reshape(vecs_r[1], bond_dim, bond_dim)
    
    # Step 4: Use helper to compute entanglement from fixed points
    return _compute_entanglement_from_fixed_points(l, r; tol=tol)
end

"""
    _compute_entanglement_from_fixed_points(l, r; tol=1e-12)

Compute entanglement entropy from left and right fixed points using the canonical form:
    ρ = √l · r · √l
    S = -Tr(ρ log ρ)

# Arguments
- `l`: Left fixed point matrix (bond_dim × bond_dim)
- `r`: Right fixed point matrix (bond_dim × bond_dim)  
- `tol`: Tolerance for filtering small eigenvalues

# Returns
- `S`: Entanglement entropy
- `spectrum`: Eigenvalues of ρ (entanglement spectrum)

# Notes
The matrices l and r should satisfy:
- l, r are positive semi-definite (made Hermitian if not)
- After normalization: Tr(l·r) = 1

For non-isometric MPS, the fixed points may have arbitrary phases. We ensure
positive semi-definiteness by taking absolute values of eigenvalues.
"""
function _compute_entanglement_from_fixed_points(l, r; tol=1e-12)
    # Make Hermitian (fixed points should be positive semi-definite)
    l = (l + l') / 2
    r = (r + r') / 2
    
    # For general MPS, fixed points may have wrong overall sign/phase
    # Project to positive semi-definite by taking absolute eigenvalues
    Λl, Ul = eigen(Hermitian(l))
    Λr, Ur = eigen(Hermitian(r))
    
    # Make eigenvalues positive (for non-isometric MPS, phases can be arbitrary)
    Λl_abs = abs.(real.(Λl))
    Λr_abs = abs.(real.(Λr))
    
    # Reconstruct with positive eigenvalues
    l_psd = Ul * Diagonal(Λl_abs) * Ul'
    r_psd = Ur * Diagonal(Λr_abs) * Ur'
    
    # Normalize so that Tr(l·r) = 1
    norm_factor = tr(l_psd * r_psd)
    if abs(norm_factor) > tol
        r_psd = r_psd / norm_factor
    end
    
    # Keep only significant eigenvalues for sqrt(l)
    keep_l = findall(λ -> λ > tol, Λl_abs)
    isempty(keep_l) && return 0.0, Float64[]
    
    Λl_pos = Λl_abs[keep_l]
    Ul_pos = Ul[:, keep_l]
    
    # Build √l in the truncated subspace
    sqrt_l = Ul_pos * Diagonal(sqrt.(Λl_pos)) * Ul_pos'
    
    # Build ρ = √l · r · √l
    ρ = sqrt_l * r_psd * sqrt_l
    
    # Make ρ Hermitian (should already be, but ensure numerical stability)
    ρ = (ρ + ρ') / 2
    
    # Compute eigenvalues of ρ (the entanglement spectrum)
    spectrum_raw = eigvals(Hermitian(ρ))
    
    # Filter to positive eigenvalues and normalize
    spectrum = filter(p -> real(p) > tol, real.(spectrum_raw))
    isempty(spectrum) && return 0.0, Float64[]
    
    # Normalize spectrum (should sum to 1, but ensure it)
    spectrum = spectrum ./ sum(spectrum)
    
    # Compute entropy: S = -Tr(ρ log ρ) = -Σ p_i log(p_i)
    S = -sum(p -> p > tol ? p * log(p) : 0.0, spectrum)
    
    return S, spectrum
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
    _contract_multiline_to_effective_mps(A_tensors, row, bond_dim)

Contract the vertical legs of multiline MPS tensors to create an effective single-row MPS tensor.

For a multiline MPS with tensors A_1, ..., A_row (each with legs [phys, down, right, up, left]),
this function contracts all vertical bonds (up/down) and stacks the physical indices to create
an effective MPS tensor with:
- Effective physical dimension: 2^row (all physical indices combined)
- Left bond dimension: bond_dim^(row+1) (all left + periodic indices)
- Right bond dimension: bond_dim^(row+1) (all right + periodic indices)

# Arguments
- `A_tensors`: Vector of tensors, each with shape (2, bond_dim, bond_dim, bond_dim, bond_dim)
               Leg ordering: [physical, down, right, up, left]
- `row`: Number of rows
- `bond_dim`: Bond dimension

# Returns
- `A_eff`: Effective MPS tensor with shape (2^row, bond_dim^(row+1), bond_dim^(row+1))
           Leg ordering: [effective_physical, left_boundary, right_boundary]

# Notes
The boundary indices combine: [periodic_bond, row_1_horizontal, row_2_horizontal, ..., row_n_horizontal]
Vertical bonds are contracted in a periodic manner (row 1's up connects to row n's down).
"""
function _contract_multiline_to_effective_mps(A_tensors, row, bond_dim)
    phys_dim = 2^row
    boundary_dim = bond_dim^(row + 1)  # periodic bond + row horizontal bonds
    
    # Initialize effective tensor
    A_eff = zeros(ComplexF64, phys_dim, boundary_dim, boundary_dim)
    
    # Iterate over all physical configurations
    for phys_config in 0:(phys_dim - 1)
        # Get physical index for each row (0-indexed, then +1 for Julia)
        phys_indices = digits(phys_config, base=2, pad=row) .+ 1
        
        # Iterate over all left boundary configurations
        for left_config in 0:(boundary_dim - 1)
            left_indices = digits(left_config, base=bond_dim, pad=row + 1) .+ 1
            # left_indices[1] = periodic bond (connecting row 1's up to row N's down)
            # left_indices[2:end] = horizontal left bonds for each row
            
            # Iterate over all right boundary configurations
            for right_config in 0:(boundary_dim - 1)
                right_indices = digits(right_config, base=bond_dim, pad=row + 1) .+ 1
                
                # Compute amplitude by contracting the column with internal bond summation
                amplitude = _contract_column_amplitude(A_tensors, phys_indices, 
                                                       left_indices, right_indices, 
                                                       row, bond_dim)
                
                A_eff[phys_config + 1, left_config + 1, right_config + 1] = amplitude
            end
        end
    end
    
    return A_eff
end

"""
    _contract_column_amplitude(A_tensors, phys_indices, left_indices, right_indices, row, bond_dim)

Compute the amplitude for a single column with fixed physical and boundary indices.
Sums over all internal vertical bonds.
"""
function _contract_column_amplitude(A_tensors, phys_indices, left_indices, right_indices, row, bond_dim)
    # Number of internal vertical bonds = row - 1
    n_internal = row - 1
    
    if n_internal == 0
        # Single row case: no internal bonds to sum
        A = A_tensors[1]
        # A[phys, down, right, up, left]
        # Periodic: down = up (both use the periodic index)
        # left_indices[1] = right_indices[1] = periodic bond
        p = phys_indices[1]
        periodic_left = left_indices[1]
        periodic_right = right_indices[1]
        left_h = left_indices[2]
        right_h = right_indices[2]
        
        # For single row: up and down both connect to the periodic bond
        # But left and right boundaries can have different periodic indices
        # Actually, for proper contraction: up = left_periodic, down = right_periodic
        return A[p, periodic_right, right_h, periodic_left, left_h]
    end
    
    # Multiple rows: sum over internal vertical bonds
    amplitude = ComplexF64(0.0)
    
    for internal_config in 0:(bond_dim^n_internal - 1)
        internal_bonds = digits(internal_config, base=bond_dim, pad=n_internal) .+ 1
        
        term = ComplexF64(1.0)
        for r in 1:row
            A = A_tensors[r]
            p = phys_indices[r]
            left_h = left_indices[r + 1]
            right_h = right_indices[r + 1]
            
            # Determine vertical indices
            if r == 1
                up_idx = left_indices[1]  # periodic from left
                down_idx = internal_bonds[1]  # first internal bond
            elseif r == row
                up_idx = internal_bonds[n_internal]  # last internal bond
                down_idx = right_indices[1]  # periodic to right
            else
                up_idx = internal_bonds[r - 1]
                down_idx = internal_bonds[r]
            end
            
            # A[phys, down, right, up, left]
            term *= A[p, down_idx, right_h, up_idx, left_h]
        end
        
        amplitude += term
    end
    
    return amplitude
end

"""
    multiline_mps_entanglement(input, row; nqubits=nothing, tol=1e-12, use_iterative=:auto, matrix_free=:auto)

Compute the physical bipartite entanglement entropy across the vertical cut 
for a multiline uniform MPS in the thermodynamic limit.

Uses the canonical form approach:
1. Contract vertical legs to get effective MPS tensor A_eff with shape (2^row, D, D)
2. Build transfer matrix E = Σ_s A_eff^s ⊗ (A_eff^s)* 
3. Get left/right fixed points l, r with Tr(l·r) = 1
4. Compute ρ = √l · r · √l
5. S = -Tr(ρ log ρ)

# Arguments
- `input`: Vector of gate matrices or tensors
- `row`: Number of rows
- `nqubits`: Number of qubits per gate (required for gates)
- `tol`: Tolerance for filtering
- `use_iterative`: `:auto`, `:always`, or `:never`
- `matrix_free`: `:auto`, `:always`, or `:never`

# Returns
- `S`: Entanglement entropy
- `spectrum`: Eigenvalues of ρ (entanglement spectrum)
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
    env_dim = bond_dim^total_legs  # Effective bond dimension for multiline MPS
    matrix_size = env_dim^2
    
    # Step 1: Contract vertical legs to get effective MPS tensor
    # A_eff has shape (2^row, env_dim, env_dim) = (effective_phys, left, right)
    A_eff = _contract_multiline_to_effective_mps(A_tensors, row, bond_dim)
    
    # Step 2: Build transfer matrix for effective MPS: E = Σ_s A_eff^s ⊗ (A_eff^s)*
    phys_dim_eff = 2^row
    E = zeros(ComplexF64, matrix_size, matrix_size)
    for s in 1:phys_dim_eff
        As = A_eff[s, :, :]  # env_dim × env_dim matrix
        E .+= kron(As, conj(As))
    end
    
    # Decide on solver
    should_use_iterative = use_iterative == :always || (use_iterative == :auto && matrix_size > 256)
    
    # Step 3: Get right fixed point r: E·r = λ·r
    if should_use_iterative
        vals, vecs, _ = KrylovKit.eigsolve(E, randn(ComplexF64, matrix_size), 2, :LM;
                                           ishermitian=false, krylovdim=min(30, matrix_size))
        sorted_idx = sortperm(abs.(vals), rev=true)
        eigenvalues = abs.(vals[sorted_idx])
        gap = length(eigenvalues) > 1 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
        
        r = reshape(vecs[sorted_idx[1]], env_dim, env_dim)
        
        # Get left fixed point l: E†·l = λ·l
        vals_l, vecs_l, _ = KrylovKit.eigsolve(E', randn(ComplexF64, matrix_size), 1, :LM;
                                               ishermitian=false, krylovdim=min(30, matrix_size))
        l = reshape(vecs_l[argmax(abs.(vals_l))], env_dim, env_dim)
    else
        # Full eigendecomposition
        eig_result = eigen(E)
        sorted_idx = sortperm(abs.(eig_result.values), rev=true)
        eigenvalues = abs.(eig_result.values[sorted_idx])
        gap = length(eigenvalues) > 1 ? -log(eigenvalues[2]/eigenvalues[1]) : Inf
        
        r = reshape(eig_result.vectors[:, sorted_idx[1]], env_dim, env_dim)
        
        eig_l = eigen(E')
        l = reshape(eig_l.vectors[:, argmax(abs.(eig_l.values))], env_dim, env_dim)
    end
    
    # Step 4 & 5: Use canonical form to compute entanglement: ρ = √l · r · √l
    S, spectrum = _compute_entanglement_from_fixed_points(l, r; tol=tol)
    
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
