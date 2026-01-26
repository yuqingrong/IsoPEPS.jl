# =============================================================================
# Entanglement Entropy Calculations
# =============================================================================
# Functions for computing entanglement entropy of MPS states

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

# Arguments
- `A`: MPS tensor with indices (physical, left, right)
- `tol`: Tolerance for filtering small singular values

# Returns
- `S`: Entanglement entropy in the thermodynamic limit
- `schmidt_values`: Schmidt coefficients
"""
function mps_physical_entanglement_infinite(A; tol=1e-12)
    phys_dim, bond_dim, _ = size(A)
    
    # Build transfer matrix E_{αα',ββ'} = Σ_s A^s_{αβ} (A^s_{α'β'})^*
    E = zeros(ComplexF64, bond_dim^2, bond_dim^2)
    for s in 1:phys_dim
        As = A[s, :, :]
        E .+= kron(As, conj(As))
    end
    
    # Get fixed points (left and right eigenvectors for dominant eigenvalue)
    eig = eigen(E)
    idx = argmax(abs.(eig.values))
    r = reshape(eig.vectors[:, idx], bond_dim, bond_dim)
    
    eig_left = eigen(E')
    idx_left = argmax(abs.(eig_left.values))
    l = reshape(eig_left.vectors[:, idx_left], bond_dim, bond_dim)
    
    # Make Hermitian and normalize
    r = (r + r') / 2
    l = (l + l') / 2
    r /= tr(l * r)
    
    # Gauge transform: X l X† = I, Y r Y† = I
    # Then entanglement spectrum = singular values of X Y†
    Λl, Ul = eigen(Hermitian(l))
    Λr, Ur = eigen(Hermitian(r))
    
    X = Diagonal(sqrt.(max.(Λl, 0))) * Ul'
    Y = Diagonal(sqrt.(max.(Λr, 0))) * Ur'
    
    C = X * Y'
    σ = svd(C).S
    σ = filter(s -> s > tol, σ)
    σ = σ ./ norm(σ)
    
    S = -sum(λ -> λ > tol ? λ^2 * log(λ^2) : 0.0, σ)
    return S, σ
end

# =============================================================================
# Multiline MPS Entanglement
# =============================================================================

"""
    multiline_mps_entanglement(gates, row, nqubits; tol=1e-12, use_iterative=:auto, matrix_free=:auto)

Compute the TRUE physical bipartite entanglement entropy across the vertical cut 
(between columns) for a multiline uniform MPS in the thermodynamic limit.

This requires BOTH left and right transfer matrix fixed points to construct the 
Schmidt decomposition. The Schmidt values σᵢ come from the gauge transformation 
that brings the MPS to mixed canonical form.

# Arguments
- `gates`: Vector of gate matrices (one per row)
- `row`: Number of rows in the multiline MPS
- `nqubits`: Number of qubits per gate (determines bond dimension)
- `tol`: Tolerance for filtering small eigenvalues
- `use_iterative`: `:auto`, `:always`, or `:never` for iterative eigensolver
- `matrix_free`: `:auto`, `:always`, or `:never` for matrix-free approach

# Returns
- `S`: Physical bipartite entanglement entropy across the vertical cut
- `spectrum`: Entanglement spectrum (Schmidt values squared, λᵢ = σᵢ²)
- `gap`: Transfer matrix spectral gap (correlation length ξ = 1/gap)

# Notes
For a uniform MPS with transfer matrix T:
- Right fixed point r: T r = λ r  (environment from contracting right half-chain)
- Left fixed point l:  l T = λ l  (environment from contracting left half-chain)
- Schmidt values = singular values of l^{1/2} r^{1/2} (after proper normalization)

This is NOT the same as eigenvalues of ρ, which only gives bond entanglement!
"""
function multiline_mps_entanglement(gates, row, nqubits; tol=1e-12, use_iterative=:auto, matrix_free=:auto)
    virtual_qubits = (nqubits - 1) ÷ 2
    bond_dim = 2^virtual_qubits
    total_legs = row + 1  # periodic boundary + row bonds
    env_dim = bond_dim^total_legs
    matrix_size = env_dim^2
    
    A_tensors = gates_to_tensors(gates, row, virtual_qubits)
    
    # Get right fixed point and gap from compute_transfer_spectrum
    rho_R, gap, _, _ = compute_transfer_spectrum(gates, row, nqubits; 
                                                  num_eigenvalues=2, 
                                                  use_iterative=use_iterative, 
                                                  matrix_free=matrix_free)
    
    # Now compute the LEFT fixed point (eigenvector of T†)
    should_use_matrix_free = matrix_free == :always || (matrix_free == :auto && matrix_size > 1024)
    should_use_iterative = use_iterative == :always || (use_iterative == :auto && matrix_size > 256)
    
    if should_use_matrix_free
        # Matrix-free approach for left fixed point
        code, total_legs_code = _build_transfer_contraction_code(A_tensors, row, virtual_qubits)
        tensor_ket = [A_tensors[i] for i in 1:row]
        tensor_bra = [conj(A_tensors[i]) for i in 1:row]
        
        # Left fixed point: solve l† T = λ l†, i.e., T† l = λ* l
        # Swapping ket↔bra in the contraction gives T†
        function apply_transfer_adjoint(v)
            v_tensor = reshape(v, ntuple(_ -> bond_dim, 2*total_legs_code)...)
            result = _apply_transfer_to_vector(code, tensor_bra, tensor_ket, v_tensor, total_legs_code)
            return vec(result)
        end
        
        v0 = randn(ComplexF64, matrix_size)
        v0 = v0 / norm(v0)
        vals_l, vecs_l, _ = KrylovKit.eigsolve(apply_transfer_adjoint, v0, 1, :LM; 
                                                ishermitian=false, krylovdim=30)
        l_vec = vecs_l[argmax(abs.(vals_l))]
        
    elseif should_use_iterative
        # Build T and use iterative solver for T†
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                        [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        
        v0 = randn(ComplexF64, matrix_size)
        v0 = v0 / norm(v0)
        vals_l, vecs_l, _ = KrylovKit.eigsolve(T', v0, 1, :LM; ishermitian=false, krylovdim=30)
        l_vec = vecs_l[argmax(abs.(vals_l))]
        
    else
        # Full eigendecomposition
        _, T = contract_transfer_matrix([A_tensors[i] for i in 1:row], 
                                        [conj(A_tensors[i]) for i in 1:row], row)
        T = reshape(T, matrix_size, matrix_size)
        
        eig_l = eigen(T')
        l_vec = eig_l.vectors[:, argmax(abs.(eig_l.values))]
    end
    
    # Reshape to matrix form
    rho_L = reshape(l_vec, env_dim, env_dim)
    
    # Compute true bipartite entanglement from both fixed points
    S, spectrum = _compute_bipartite_entanglement(rho_L, rho_R, tol)
    
    return S, spectrum, gap
end

"""
    _compute_bipartite_entanglement(rho_L, rho_R, tol)

Compute TRUE physical bipartite entanglement from left and right fixed points.

For uniform MPS, the Schmidt decomposition at a bond requires both environments:
- l (left fixed point): encodes the left half-chain  
- r (right fixed point): encodes the right half-chain

The Schmidt values σᵢ are obtained by:
1. Decompose l = X† X and r = Y Y† (via eigendecomposition)
2. σᵢ = singular values of (X Y†), properly normalized

# Arguments
- `rho_L`: Left fixed point (matrix form)
- `rho_R`: Right fixed point (matrix form)  
- `tol`: Tolerance for filtering

# Returns
- `S`: Bipartite entanglement entropy
- `spectrum`: Schmidt values squared (λᵢ = σᵢ²)
"""
function _compute_bipartite_entanglement(rho_L, rho_R, tol)
    # Make Hermitian (fixed points of quantum channels should be Hermitian)
    rho_L = Hermitian((rho_L + rho_L') / 2)
    rho_R = Hermitian((rho_R + rho_R') / 2)
    
    # Normalize: tr(l · r) = 1
    norm_factor = tr(rho_L * rho_R)
    if abs(norm_factor) > tol
        rho_R = rho_R / norm_factor
    end
    
    # Eigendecompose: l = U_L Λ_L U_L†, r = U_R Λ_R U_R†
    Λ_L, U_L = eigen(rho_L)
    Λ_R, U_R = eigen(rho_R)
    
    # Filter small/negative eigenvalues
    Λ_L = real.(Λ_L)
    Λ_R = real.(Λ_R)
    Λ_L = max.(Λ_L, tol)
    Λ_R = max.(Λ_R, tol)
    
    # Gauge transformation to canonical form:
    # X = Λ_L^{1/2} U_L†  such that X l X† = I
    # Y = Λ_R^{1/2} U_R†  such that Y r Y† = I
    # Schmidt values = singular values of C = X Y† = Λ_L^{1/2} U_L† U_R Λ_R^{1/2}
    X = Diagonal(sqrt.(Λ_L)) * U_L'
    Y = Diagonal(sqrt.(Λ_R)) * U_R'
    
    C = X * Y'
    σ = svd(C).S
    
    # Normalize Schmidt values
    σ = σ ./ norm(σ)
    σ = filter(s -> s > tol, σ)
    
    # Entanglement spectrum (λᵢ = σᵢ²) and entropy
    spectrum = σ.^2
    S = -sum(p -> p > tol ? p * log(p) : 0.0, spectrum)
    
    return S, spectrum
end

"""
    multiline_mps_entanglement_from_params(params, p, row, nqubits; share_params=true, tol=1e-12)

Compute multiline MPS entanglement directly from circuit parameters.

# Arguments
- `params`: Parameter vector for the quantum circuit
- `p`: Number of circuit layers
- `row`: Number of rows
- `nqubits`: Number of qubits per gate
- `share_params`: Whether parameters are shared across layers
- `tol`: Tolerance for filtering

# Returns
- `S`: Entanglement entropy
- `spectrum`: Entanglement spectrum
- `gap`: Transfer matrix spectral gap
"""
function multiline_mps_entanglement_from_params(params::Vector{Float64}, p::Int, row::Int, nqubits::Int; 
                                                 share_params::Bool=true, tol::Float64=1e-12)
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    return multiline_mps_entanglement(gates, row, nqubits; tol=tol)
end

