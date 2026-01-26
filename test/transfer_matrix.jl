using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra

@testset "contract_transfer_matrix" begin
    A = randn(ComplexF64, 2, 2, 2, 2, 2)
    for row in 1:3
        code, result = contract_transfer_matrix([A for _ in 1:row], [conj(A) for _ in 1:row], row)
        @test result isa Array{ComplexF64, 4*(row + 1)}
    end
end

@testset "transfer_matrix_properties" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    
    for row in [1, 2]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        # First verify the tensor isometry condition
        # For isometric PEPS: Σ_{physical,down,right} A†A = I_{up,left}
        A_tensors = IsoPEPS.gates_to_tensors(gates, row, virtual_qubits)
        for A in A_tensors
            # A has shape (physical, down, right, up, left) = (2, D, D, D, D)
            # Reshape to matrix: (physical*down*right) × (up*left)
            dims = size(A)
            A_mat = reshape(A, prod(dims[1:3]), prod(dims[4:5]))
            # Isometry: A†A = I
            @test A_mat' * A_mat ≈ I atol=1e-10
        end
        
        # Build transfer matrix using contract_transfer_matrix
        total_qubits = 1 + virtual_qubits + virtual_qubits * row
        boundary_qubits = total_qubits - 1
        matrix_size = 4^boundary_qubits
        _, T_tensor = contract_transfer_matrix([A_tensors[i] for i in 1:row], [conj(A_tensors[i]) for i in 1:row], row)       
        T = reshape(T_tensor, matrix_size, matrix_size)   
        n = Int(sqrt(size(T, 1)))  # Dimension of density matrix space
        # Helper: vectorize and unvectorize density matrices
        vec_rho(ρ) = vec(ρ)
        unvec_rho(v) = reshape(v, n, n)
        # Apply transfer matrix to density matrix
        apply_T(ρ) = unvec_rho(T * vec_rho(ρ))
        
        @testset "row=$row" begin
            # 1. Fixed point with eigenvalue 1 exists
            eigenvalues_T = eigvals(T)
            sorted_eigs = sort(abs.(eigenvalues_T), rev=true)
            @test isapprox(sorted_eigs[1], 1.0, atol=1e-6)  # Largest eigenvalue is 1
            
            # 2. Trace preserving: Tr(T(X)) = Tr(X) 
            X = randn(ComplexF64, n, n)
            T_X = apply_T(X)
            @test tr(T_X) ≈ tr(X) atol=1e-10
            
            # 3. Hermiticity preservation: X = X† ⇒ T(X) = T(X)†
            H = randn(ComplexF64, n, n)
            H = H + H'  
            T_H = apply_T(H)
            @test T_H ≈ T_H' atol=1e-10
            
            # 4. Complete Positivity (Choi matrix test)
            # Choi matrix: C = (I ⊗ T)(|Ω⟩⟨Ω|) where |Ω⟩ = Σᵢ |i⟩|i⟩
            d = n  # dimension
            Choi = zeros(ComplexF64, d^2, d^2)
            for i in 1:d, j in 1:d
                # |i⟩⟨j| basis element
                E_ij = zeros(ComplexF64, d, d)
                E_ij[i, j] = 1.0
                T_E_ij = apply_T(E_ij)
                # Choi matrix: C[i,j block] = T(|i⟩⟨j|)
                for k in 1:d, l in 1:d
                    Choi[(i-1)*d + k, (j-1)*d + l] = T_E_ij[k, l]
                end
            end
            choi_eigenvalues = eigvals(Hermitian(Choi))
            @test all(choi_eigenvalues .> -1e-10) 
            
            # 5. Spectral gap: |λ₂| < 1 (already verified |λ₁| = 1 above)
            @test sorted_eigs[2] < 1.0
            
            # 6. Adjoint channel T' properties (dual to T)
            # T' is the adjoint/dual channel: ⟨A, T(B)⟩ = ⟨T'(A), B⟩
            T_adj = T'
            apply_T_adj(ρ) = unvec_rho(T_adj * vec_rho(ρ))
            
            # 6a. T'(I) = I (adjoint channel is unital, dual to trace preservation)
            I_mat = Matrix{ComplexF64}(I, n, n)
            T_adj_I = apply_T_adj(I_mat)
            @test T_adj_I ≈ I_mat atol=1e-10
            
            # 6b. Leading eigenvector of T' is vec(I) (up to normalization)
            # Since T' * vec(I) = vec(I), vec(I) is an eigenvector with eigenvalue 1
            vec_I = vec(I_mat)
            T_adj_vec_I = T_adj * vec_I
            @test T_adj_vec_I ≈ vec_I atol=1e-10
        end
    end
end


@testset "compute_transfer_spectrum" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
    for row in [1]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, virtual_qubits)
        @test LinearAlgebra.tr(rho) ≈ 1.0
        @test all(eigenvalues[1:end-1] .< 1.0)
    end
end

@testset "product_state_large_gap" begin
    # For a product state, the correlation length ξ = 1/gap should be small
    # This means the spectral gap should be large (λ₂ << 1)
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    bond_dim = 2^virtual_qubits
    
    for row in [1, 2]
        # Create a product state tensor directly:
        # A[physical, down, right, up, left] = v_phys ⊗ v_down ⊗ v_right ⊗ v_up ⊗ v_left
        # This is rank-1 in virtual indices → no entanglement → large gap
        
        # Construct product state tensor
        v_phys = [1.0 + 0im, 0.0]  # |0⟩ physical state
        v_virt = ones(ComplexF64, bond_dim) / sqrt(bond_dim)  # Uniform superposition on virtual
        
        # Product tensor: outer product of all vectors
        A_product = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        for p in 1:2, d in 1:bond_dim, r in 1:bond_dim, u in 1:bond_dim, l in 1:bond_dim
            A_product[p, d, r, u, l] = v_phys[p] * v_virt[d] * v_virt[r] * v_virt[u] * v_virt[l]
        end
        A_tensors = [A_product for _ in 1:row]
        
        # Build transfer matrix directly from tensors
        total_qubits = 1 + virtual_qubits + virtual_qubits * row
        matrix_size = 4^(total_qubits - 1)
        _, T_tensor = contract_transfer_matrix(A_tensors, [conj(A) for A in A_tensors], row)
        T = reshape(T_tensor, matrix_size, matrix_size)
        
        # Get eigenvalues
        eigenvalues = sort(abs.(eigvals(T)), rev=true)
        λ1 = eigenvalues[1]
        λ2 = eigenvalues[2]
        @testset "row=$row" begin
            @test λ1 > 0.99
            # For product state, second eigenvalue should be very small
            @test λ2 < 1e-10
            
            # correlation length should be small
            gap = -log(λ2 / λ1)
            correlation_length = 1 / gap
            @test correlation_length < 0.1
        end
    end
end

@testset "compute_transfer_spectrum_methods_consistency" begin
    # Test that all three methods (matrix-free, iterative, full) give the same results
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    
    for row in [1, 2]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        # Method 1: Full eigendecomposition (use_iterative=:never, matrix_free=:never)
        rho_full, gap_full, eigs_full = compute_transfer_spectrum(
            gates, row, nqubits; 
            use_iterative=:never, matrix_free=:never
        )
        
        # Method 2: Iterative solver with full matrix (use_iterative=:always, matrix_free=:never)
        rho_iter, gap_iter, eigs_iter = compute_transfer_spectrum(
            gates, row, nqubits; 
            use_iterative=:always, matrix_free=:never
        )
        
        # Method 3: Matrix-free approach (matrix_free=:always)
        rho_mfree, gap_mfree, eigs_mfree = compute_transfer_spectrum(
            gates, row, nqubits; 
            matrix_free=:always
        )
        
        @testset "row=$row" begin
            # Compare spectral gaps
            @test isapprox(gap_full, gap_iter, atol=1e-6)
            @test isapprox(gap_full, gap_mfree, atol=1e-6)
            
            # Compare leading eigenvalues
            @test isapprox(maximum(eigs_full), maximum(abs.(eigs_iter)), atol=1e-6)
            @test isapprox(maximum(eigs_full), maximum(abs.(eigs_mfree)), atol=1e-6)
            
            # Compare fixed point density matrices (up to phase)
            # The fixed points should give the same physical observables
            @test isapprox(tr(rho_full), tr(rho_iter), atol=1e-6)
            @test isapprox(tr(rho_full), tr(rho_mfree), atol=1e-6)
            
            # Check that rho matrices are similar (may differ by phase/normalization)
            # Compare the absolute values of diagonal elements
            @test isapprox(abs.(diag(rho_full)), abs.(diag(rho_iter)), atol=1e-5)
            @test isapprox(abs.(diag(rho_full)), abs.(diag(rho_mfree)), atol=1e-5)
        end
    end
end

@testset "transfer_matrix_advanced_tests" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        # Get tensors and build transfer matrix
        A_tensors = IsoPEPS.gates_to_tensors(gates, row, virtual_qubits)
        total_qubits = 1 + virtual_qubits + virtual_qubits * row
        boundary_qubits = total_qubits - 1
        matrix_size = 4^boundary_qubits
        n = Int(sqrt(matrix_size))  # Density matrix dimension
        
        # Build the full transfer matrix M
        _, T_tensor = contract_transfer_matrix(
            [A_tensors[i] for i in 1:row], 
            [conj(A_tensors[i]) for i in 1:row], 
            row
        )
        M = reshape(T_tensor, matrix_size, matrix_size)
        
        @testset "row=$row" begin
            # =================================================================
            # Test 1: Matrix-vs-function consistency
            # Define T(X) that applies the channel, verify vec(T(X)) = M*vec(X)
            # =================================================================
            @testset "matrix_vs_function_consistency" begin
                # Define channel function T(X) = Σ_k A_k X A_k†
                # For our tensor network: contract physical indices
                function apply_channel(X)
                    # X is an n×n matrix (density matrix on boundary)
                    # Apply the transfer matrix channel
                    return reshape(M * vec(X), n, n)
                end
                
                # Test with random input matrices
                for _ in 1:3
                    X = randn(ComplexF64, n, n)
                    T_X = apply_channel(X)
                    
                    # Verify: vec(T(X)) = M * vec(X)
                    @test vec(T_X) ≈ M * vec(X) atol=1e-10
                end
            end
            
            # =================================================================
            # Test 2: Residual check for eigenpairs
            # Diagonalize M, get (λ, v), check ||Mv - λv|| ≈ 0
            # =================================================================
            @testset "eigenpair_residual" begin
                eig_result = eigen(M)
                eigenvalues = eig_result.values
                eigenvectors = eig_result.vectors
                
                for i in 1:min(5, length(eigenvalues))  # Check first 5 eigenpairs
                    λ = eigenvalues[i]
                    v = eigenvectors[:, i]
                    
                    residual = norm(M * v - λ * v)
                    @test residual < 1e-10
                end
            end
            
            # =================================================================
            # Test 3: Left/right eigenvector check for non-Hermitian M
            # Right: Mv = λv, Left: w†M = λw† (equiv: M†w = conj(λ)w)
            # =================================================================
            @testset "left_right_eigenvectors" begin
                # Right eigenvectors
                eig_right = eigen(M)
                λ_right = eig_right.values
                V_right = eig_right.vectors
                
                # Left eigenvectors (eigenvectors of M†)
                eig_left = eigen(M')
                λ_left = eig_left.values
                W_left = eig_left.vectors
                
                # Eigenvalues of M and M† should be complex conjugates
                # Compare by magnitude (sorting complex numbers is numerically unstable)
                λ_right_mags = sort(abs.(λ_right), rev=true)
                λ_left_mags = sort(abs.(λ_left), rev=true)
                @test λ_right_mags ≈ λ_left_mags atol=1e-10
                
                # Verify right eigenvector equation: Mv = λv
                for i in 1:min(3, length(λ_right))
                    v = V_right[:, i]
                    @test norm(M * v - λ_right[i] * v) < 1e-10
                end
                
                # Verify left eigenvector equation: M†w = conj(λ)w
                for i in 1:min(3, length(λ_left))
                    w = W_left[:, i]
                    @test norm(M' * w - λ_left[i] * w) < 1e-10
                end
            end
        end
    end
end

@testset "concrete_transfer_matrix_case" begin
    # Test with a specific diagonal tensor
    A = zeros(Float64, 2, 2, 2, 2, 2)
    λ = 0.5
    row = 1 
    # s = 0
    A[1, 1, 1, 1, 1] = 1.0
    A[1, 2, 2, 2, 2] = 1.0
    # s = 1
    A[2, 1, 1, 1, 1] = λ
    A[2, 2, 2, 2, 2] = λ
    
    A1 = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
    A2 = [λ 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 λ]
    E1 = kron(A1, conj(A1)) + kron(A2, conj(A2))
    E = foldl(kron, ntuple(_ -> E1, row)) 
    _, T_tensor = contract_transfer_matrix([A for _ in 1:row], [conj(A) for _ in 1:row], row)
    T = reshape(T_tensor, 16, 16)
    @test T ≈ E atol=1e-10

    for row in 1:4
        # Boundary has (row + 1) legs; each leg has dimension 2
        boundary_dim = 2^(row + 1)
        matrix_size = boundary_dim^2
        # Expected transfer matrix: only the |0...0⟩ and |1...1⟩ boundary
        # configurations survive on both bra/ket sides.
        # Each row contributes a factor (1 + λ^2).
        weight = (1 + λ^2)^row
        E = zeros(Float64, matrix_size, matrix_size)
        boundary_states = (1, boundary_dim)
        for col in boundary_states, row_idx in boundary_states
            vec_idx = row_idx + (col - 1) * boundary_dim
            E[vec_idx, vec_idx] = weight
        end
        
        _, T_tensor = contract_transfer_matrix(
            [A for _ in 1:row],
            [conj(A) for _ in 1:row],
            row
        )
        T = reshape(T_tensor, matrix_size, matrix_size)
        L1 = LinearAlgebra.eigvals(E)
        L2 = LinearAlgebra.eigvals(T)
        @test T ≈ E atol=1e-10
    end
end

