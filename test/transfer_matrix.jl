using Test
using IsoPEPS
using Yao, YaoBlocks
using LinearAlgebra
using ITensors

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

@testset "get_transfer_matrix_with_operator" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    
    # Define Pauli operators
    Z_op = ComplexF64[1 0; 0 -1]
    X_op = ComplexF64[0 1; 1 0]
    I_op = ComplexF64[1 0; 0 1]
    
    for row in [1, 2]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        @testset "row=$row" begin
            # Test 1: Basic construction - E_Z has correct size
            E_Z = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, Z_op; position=1)
            bond_dim = 2^virtual_qubits
            total_legs = row + 1
            expected_size = bond_dim^(2*total_legs)
            @test size(E_Z) == (expected_size, expected_size)
            @test eltype(E_Z) <: Complex
            
            # Test 2: Identity operator should give same as regular transfer matrix
            E = IsoPEPS.get_transfer_matrix(gates, row, virtual_qubits)
            E_I = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, I_op; position=1)
            @test E_I ≈ E atol=1e-10
            
            # Test 3: Different positions give different matrices
            if row >= 2
                E_Z_pos1 = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, Z_op; position=1)
                E_Z_pos2 = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, Z_op; position=2)
                @test norm(E_Z_pos1 - E_Z_pos2) > 1e-5
            end
            
            # Test 4: Different operators give different matrices
            E_X = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, X_op; position=1)
            @test norm(E_Z - E_X) > 1e-10
            
            # Test 5: Eigenvalue check - E_O should have different spectrum than E
            eigs_E = sort(abs.(eigvals(E)), rev=true)
            eigs_E_Z = sort(abs.(eigvals(E_Z)), rev=true)
            # The dominant eigenvalue of E is 1, E_Z generally different
            @test isapprox(eigs_E[1], 1.0, atol=1e-6)
        end
    end
    
    # Test 6: Error handling for invalid positions
    @testset "error_handling" begin
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:2]
        row = 2
        
        @test_throws ErrorException IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, Z_op; position=0)
        @test_throws ErrorException IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits, Z_op; position=row+1)
    end
    
    # Test 7: Analytical test with diagonal tensor
    # For diagonal tensor: A[s,d,r,u,l] = δ_{d,r,u,l} * amplitude(s)
    # E = Σ_s A_s ⊗ A*_s, E_Z = Σ_s Z[s,s] * A_s ⊗ A*_s
    # With Z = diag(1,-1): E_Z = E_0 - E_1 where E_s is contribution from physical index s
    @testset "analytical_diagonal_tensor" begin
        λ = 0.5
        bond_dim = 2
        
        for row in [1, 2]
            @testset "row=$row" begin
                # Create diagonal PEPS tensor (same as concrete_transfer_matrix_case)
                A = zeros(Float64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
                # physical = 0 (index 1): amplitude 1.0
                A[1, 1, 1, 1, 1] = 1.0
                A[1, 2, 2, 2, 2] = 1.0
                # physical = 1 (index 2): amplitude λ
                A[2, 1, 1, 1, 1] = λ
                A[2, 2, 2, 2, 2] = λ
                
                A_tensors = [A for _ in 1:row]
                
                # Build E and E_Z directly using tensors
                total_legs = row + 1
                matrix_size = bond_dim^(2*total_legs)
                
                # E: standard transfer matrix
                _, T_tensor = contract_transfer_matrix(A_tensors, [conj(A) for A in A_tensors], row)
                E = reshape(T_tensor, matrix_size, matrix_size)
                
                # E_Z: apply Z to first tensor's physical index
                # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                A_Z = copy(A)
                A_Z[2, :, :, :, :] .*= -1  # Z flips sign of s=1 component
                A_Z_tensors = [i == 1 ? A_Z : A for i in 1:row]
                
                _, T_Z_tensor = contract_transfer_matrix(A_Z_tensors, [conj(A) for A in A_tensors], row)
                E_Z_expected = reshape(T_Z_tensor, matrix_size, matrix_size)
                
                # Now use the function we're testing
                # Need to convert tensor to gate format
                # Gate shape: (2^nqubits, 2^nqubits) where nqubits = 1 + 2*virtual_qubits = 3
                # Tensor shape: (physical, down, right, up, left) = (2, 2, 2, 2, 2)
                # Gate indices: physical ⊗ down ⊗ right ⊗ up_fixed=0 ⊗ left
                nqubits = 3
                virtual_qubits_test = 1
                
                # Reshape tensor to gate format: need to match gates_to_tensors inverse
                # gates_to_tensors does: reshape(gate, (2, D, D, 2, D, D))[..., 1, ...] 
                # So gate has shape (2, D, D, 2, D, D) with up_physical=0 slice giving A
                gate_shape = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
                gate_tensor = zeros(Float64, gate_shape...)
                gate_tensor[:, :, :, 1, :, :] = A  # up_physical index = 1 (i.e., 0)
                gate = reshape(gate_tensor, 2^nqubits, 2^nqubits)
                gates = [gate for _ in 1:row]
                
                Z_op = [1.0 0.0; 0.0 -1.0]
                E_Z_computed = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits_test, Z_op; position=1)
                
                # Test: E_Z from function should match analytical E_Z
                @test E_Z_computed ≈ E_Z_expected atol=1e-10
                
                # Test eigenvalue relationship:
                # E has eigenvalues at diagonal boundary configs: (1 + λ²)^row
                # E_Z has Z only at position 1, so: (1 - λ²) × (1 + λ²)^(row-1)
                eigs_E = sort(abs.(eigvals(E)), rev=true)
                eigs_E_Z = sort(abs.(eigvals(E_Z_computed)), rev=true)
                
                expected_E_top = (1 + λ^2)^row
                # Z at position 1 flips sign only for that layer
                expected_E_Z_top = (1 - λ^2) * (1 + λ^2)^(row-1)
                
                @test isapprox(eigs_E[1], expected_E_top, atol=1e-10)
                @test isapprox(eigs_E_Z[1], expected_E_Z_top, atol=1e-10)
                
                # Test: Tr(E_Z) / Tr(E) - Z only affects one layer
                # ratio = (1-λ²)/(1+λ²) (independent of row, since Z is at one position)
                expected_ratio = (1 - λ^2)/(1 + λ^2)
                actual_ratio = tr(E_Z_computed) / tr(E)
                @test isapprox(real(actual_ratio), expected_ratio, atol=1e-10)
            end
        end
    end
    
    # Test 8: Product state - all coefficients should be related to expectation values
    @testset "product_state_EO" begin
        bond_dim = 2
        row = 1
        
        # Product state: A[s, virtual] = |s⟩ ⊗ |uniform⟩
        # Physical state |ψ⟩ = α|0⟩ + β|1⟩
        α, β = 1/sqrt(2), 1/sqrt(2)  # |+⟩ state
        
        A = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        v = ones(ComplexF64, bond_dim) / sqrt(bond_dim)  # uniform virtual
        for d in 1:bond_dim, r in 1:bond_dim, u in 1:bond_dim, l in 1:bond_dim
            A[1, d, r, u, l] = α * v[d] * v[r] * v[u] * v[l]
            A[2, d, r, u, l] = β * v[d] * v[r] * v[u] * v[l]
        end
        
        # For product state, ⟨Z⟩ = |α|² - |β|² = 0 for |+⟩
        expected_Z = abs(α)^2 - abs(β)^2
        
        # Build gate from tensor
        nqubits = 3
        virtual_qubits_test = 1
        gate_shape = (2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_tensor = zeros(ComplexF64, gate_shape...)
        gate_tensor[:, :, :, 1, :, :] = A
        gate = reshape(gate_tensor, 2^nqubits, 2^nqubits)
        gates = [gate]
        
        E = IsoPEPS.get_transfer_matrix(gates, row, virtual_qubits_test)
        Z_op = ComplexF64[1 0; 0 -1]
        E_Z = IsoPEPS.get_transfer_matrix_with_operator(gates, row, virtual_qubits_test, Z_op; position=1)
        
        # For product state: Tr(E_Z)/Tr(E) = ⟨Z⟩
        computed_Z = real(tr(E_Z) / tr(E))
        @test isapprox(computed_Z, expected_Z, atol=1e-10)
        
        # Test with |0⟩ state: ⟨Z⟩ = 1
        A_zero = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        for d in 1:bond_dim, r in 1:bond_dim, u in 1:bond_dim, l in 1:bond_dim
            A_zero[1, d, r, u, l] = v[d] * v[r] * v[u] * v[l]  # Only |0⟩ component
        end
        gate_tensor_zero = zeros(ComplexF64, gate_shape...)
        gate_tensor_zero[:, :, :, 1, :, :] = A_zero
        gate_zero = reshape(gate_tensor_zero, 2^nqubits, 2^nqubits)
        gates_zero = [gate_zero]
        
        E_zero = IsoPEPS.get_transfer_matrix(gates_zero, row, virtual_qubits_test)
        E_Z_zero = IsoPEPS.get_transfer_matrix_with_operator(gates_zero, row, virtual_qubits_test, Z_op; position=1)
        
        computed_Z_zero = real(tr(E_Z_zero) / tr(E_zero))
        @test isapprox(computed_Z_zero, 1.0, atol=1e-10)  # ⟨0|Z|0⟩ = 1
    end
end

@testset "compute_correlation_coefficients" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits
    
    Z_op = ComplexF64[1 0; 0 -1]
    
    for row in [1, 2]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        
        @testset "row=$row" begin
            num_modes = 5
            eigenvalues, coefficients, correlation_length = IsoPEPS.compute_correlation_coefficients(
                gates, row, virtual_qubits, Z_op; num_modes=num_modes
            )
            
            # Test 1: Return structure - correct number of modes
            @test length(eigenvalues) == num_modes
            @test length(coefficients) == num_modes
            
            # Test 2: Dominant eigenvalue should have magnitude ~1
            @test isapprox(abs(eigenvalues[1]), 1.0, atol=1e-6)
            
            # Test 3: Eigenvalues sorted by magnitude (descending)
            mags = abs.(eigenvalues)
            @test issorted(mags, rev=true)
            
            # Test 4: Correlation length should be positive and finite for generic states
            @test correlation_length > 0
            @test isfinite(correlation_length)
            
            # Test 5: Sub-leading eigenvalues should have |λ| < 1
            @test all(abs.(eigenvalues[2:end]) .< 1.0)
            
            # Test 6: Verify coefficients are computed (non-trivial)
            # At least some coefficients should be non-zero
            @test any(abs.(coefficients[2:end]) .> 1e-12)
        end
    end
    
    # Test with more modes
    @testset "num_modes_parameter" begin
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:2]
        row = 2
        
        eigenvalues_5, coefficients_5, ξ_5 = IsoPEPS.compute_correlation_coefficients(
            gates, row, virtual_qubits, Z_op; num_modes=5
        )
        eigenvalues_10, coefficients_10, ξ_10 = IsoPEPS.compute_correlation_coefficients(
            gates, row, virtual_qubits, Z_op; num_modes=10
        )
        
        # First 5 eigenvalues should match (eigenvalues are consistent)
        @test eigenvalues_5 ≈ eigenvalues_10[1:5] atol=1e-10
        
        # Correlation lengths should match (derived from λ₂)
        @test isapprox(ξ_5, ξ_10, atol=1e-10)
    end
    
    # Test: Product state - connected correlations should be zero
    # For product state: E is rank-1, so c_α = 0 for all α ≥ 2
    # The connected correlation ⟨OO⟩_c = ⟨OO⟩ - ⟨O⟩² = 0
    @testset "product_state_zero_correlation" begin
        bond_dim = 2
        row = 1
        nqubits_test = 3
        virtual_qubits_test = 1
        
        v = ones(ComplexF64, bond_dim) / sqrt(bond_dim)
        Z_op = ComplexF64[1 0; 0 -1]
        
        # |0⟩ state: ⟨Z⟩ = 1, c₁ = ⟨Z⟩² = 1, c_α = 0 for α ≥ 2
        A_zero = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        for d in 1:bond_dim, r in 1:bond_dim, u in 1:bond_dim, l in 1:bond_dim
            A_zero[1, d, r, u, l] = v[d] * v[r] * v[u] * v[l]  # Only |0⟩
        end
        gate_tensor = zeros(ComplexF64, 2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_tensor[:, :, :, 1, :, :] = A_zero
        gate_zero = reshape(gate_tensor, 2^nqubits_test, 2^nqubits_test)
        
        eigenvalues_0, coefficients_0, ξ_0 = IsoPEPS.compute_correlation_coefficients(
            [gate_zero], row, virtual_qubits_test, Z_op; num_modes=5
        )
        
        # Transfer matrix is rank-1, so only λ₁ ≈ 1, others ≈ 0
        @test isapprox(abs(eigenvalues_0[1]), 1.0, atol=1e-6)
        @test all(abs.(eigenvalues_0[2:end]) .< 1e-10)
        
        # c₁ = ⟨Z⟩² = 1 for |0⟩ state
        @test isapprox(abs(coefficients_0[1]), 1.0, atol=1e-6)
        
        # All c_α = 0 for α ≥ 2 (product state has no connected correlation)
        @test all(abs.(coefficients_0[2:end]) .< 1e-10)
        
        # |+⟩ state: ⟨Z⟩ = 0, so c₁ = 0, c_α = 0 for all α
        A_plus = zeros(ComplexF64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        for d in 1:bond_dim, r in 1:bond_dim, u in 1:bond_dim, l in 1:bond_dim
            A_plus[1, d, r, u, l] = (1/sqrt(2)) * v[d] * v[r] * v[u] * v[l]
            A_plus[2, d, r, u, l] = (1/sqrt(2)) * v[d] * v[r] * v[u] * v[l]
        end
        gate_tensor_plus = zeros(ComplexF64, 2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_tensor_plus[:, :, :, 1, :, :] = A_plus
        gate_plus = reshape(gate_tensor_plus, 2^nqubits_test, 2^nqubits_test)
        
        eigenvalues_plus, coefficients_plus, _ = IsoPEPS.compute_correlation_coefficients(
            [gate_plus], row, virtual_qubits_test, Z_op; num_modes=5
        )
        
        # c₁ = ⟨Z⟩² = 0 for |+⟩ state
        @test isapprox(abs(coefficients_plus[1]), 0.0, atol=1e-10)
        
        # All c_α = 0 for product state
        @test all(abs.(coefficients_plus[2:end]) .< 1e-10)
    end
    
    # Test: Diagonal tensor - verify eigenvalue structure
    @testset "diagonal_tensor_eigenvalues" begin
        λ = 0.5
        bond_dim = 2
        row = 1
        nqubits_test = 3
        virtual_qubits_test = 1
        
        # Diagonal tensor: only |0000⟩ and |1111⟩ configurations survive
        A = zeros(Float64, 2, bond_dim, bond_dim, bond_dim, bond_dim)
        A[1, 1, 1, 1, 1] = 1.0
        A[1, 2, 2, 2, 2] = 1.0
        A[2, 1, 1, 1, 1] = λ
        A[2, 2, 2, 2, 2] = λ
        
        gate_tensor = zeros(Float64, 2, bond_dim, bond_dim, 2, bond_dim, bond_dim)
        gate_tensor[:, :, :, 1, :, :] = A
        gate = reshape(gate_tensor, 2^nqubits_test, 2^nqubits_test)
        
        Z_op = ComplexF64[1 0; 0 -1]
        eigenvalues_diag, coefficients_diag, ξ_diag = IsoPEPS.compute_correlation_coefficients(
            [gate], row, virtual_qubits_test, Z_op; num_modes=5
        )
        
        # For diagonal tensor, transfer matrix has eigenvalues:
        # λ₁ = 1 + λ² (dominant, normalized to 1 after trace normalization)
        # The matrix is sparse with rank 2
        # Dominant eigenvalue should be ~ (1 + λ²)
        total_legs = row + 1
        matrix_size = bond_dim^(2*total_legs)
        
        # Build E directly to verify eigenvalues
        A_tensors = [A]
        _, T_tensor = contract_transfer_matrix(A_tensors, [conj(A) for _ in 1:row], row)
        E = reshape(T_tensor, matrix_size, matrix_size)
        E_normalized = E / tr(E)  # Normalize like in compute_transfer_spectrum
        
        eigs_E = sort(abs.(eigvals(E)), rev=true)
        
        # The eigenvalues from compute_correlation_coefficients should match E (not normalized)
        @test isapprox(abs(eigenvalues_diag[1]), eigs_E[1], atol=1e-10)
        
        # Second eigenvalue should also match
        @test isapprox(abs(eigenvalues_diag[2]), eigs_E[2], atol=1e-10)
    end
    
end

@testset "compute_theoretical_correlation_decay" begin
    # Test 1: Output structure with synthetic eigenvalues/coefficients
    @testset "output_structure" begin
        # Create simple synthetic data
        eigenvalues = ComplexF64[1.0, 0.5, 0.3, 0.1]
        coefficients = ComplexF64[0.0, 1.0, 0.5, 0.2]  # c_1 = 0 (no fixed point contribution)
        max_lag = 20
        
        lags, correlation = IsoPEPS.compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag)
        
        @test length(lags) == max_lag
        @test length(correlation) == max_lag
        @test lags == 1:max_lag
        @test eltype(correlation) <: Complex
    end
    
    # Test 2: Decay behavior - correlations should decay for |λ| < 1
    @testset "decay_behavior" begin
        # Single mode with |λ| < 1
        eigenvalues = ComplexF64[1.0, 0.5]  # Only second mode contributes
        coefficients = ComplexF64[0.0, 1.0]
        max_lag = 50
        
        lags, correlation = IsoPEPS.compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag)
        
        # Correlation should decay: |C(r+1)| < |C(r)|
        mags = abs.(correlation)
        @test all(mags[2:end] .< mags[1:end-1] .+ 1e-12)  # Allow small numerical tolerance
        
        # Should approach zero at large lags
        @test abs(correlation[end]) < 1e-10
        
        # Verify: C(r) = c_2 * λ_2^(r-1) = 1.0 * 0.5^(r-1)
        for r in 1:10
            expected = 0.5^(r-1)
            @test isapprox(abs(correlation[r]), expected, atol=1e-12)
        end
    end
    
    # Test 3: Complex eigenvalue handling - oscillatory decay
    @testset "oscillatory_decay" begin
        # Complex eigenvalue produces oscillations: λ = |λ|e^{iθ}
        θ = π/4  # 45 degrees
        λ_mag = 0.8
        λ_complex = λ_mag * exp(im * θ)
        
        eigenvalues = ComplexF64[1.0, λ_complex]
        coefficients = ComplexF64[0.0, 1.0]
        max_lag = 20
        
        lags, correlation = IsoPEPS.compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag)
        
        # Magnitude should decay exponentially
        mags = abs.(correlation)
        for r in 1:max_lag
            expected_mag = λ_mag^(r-1)
            @test isapprox(mags[r], expected_mag, atol=1e-12)
        end
        
        # Phase should advance by θ each step
        for r in 2:10
            phase_diff = angle(correlation[r]) - angle(correlation[r-1])
            # Normalize to [-π, π]
            phase_diff = mod(phase_diff + π, 2π) - π
            @test isapprox(phase_diff, θ, atol=1e-10)
        end
    end
    
    # Test 4: Multiple modes sum correctly
    @testset "multiple_modes" begin
        eigenvalues = ComplexF64[1.0, 0.8, 0.5, 0.3]
        coefficients = ComplexF64[0.0, 1.0, 2.0, 0.5]
        max_lag = 10
        
        lags, correlation = IsoPEPS.compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag)
        
        # Verify manual calculation: C(r) = Σ_{α≥2} c_α λ_α^(r-1)
        for r in 1:max_lag
            expected = coefficients[2] * eigenvalues[2]^(r-1) +
                       coefficients[3] * eigenvalues[3]^(r-1) +
                       coefficients[4] * eigenvalues[4]^(r-1)
            @test isapprox(correlation[r], expected, atol=1e-12)
        end
    end
    
    # Test 5: Integration test with actual gates
    @testset "integration_with_gates" begin
        virtual_qubits = 1
        nqubits = 1 + 2*virtual_qubits
        Z_op = ComplexF64[1 0; 0 -1]
        
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:2]
        row = 2
        
        eigenvalues, coefficients, ξ = IsoPEPS.compute_correlation_coefficients(
            gates, row, virtual_qubits, Z_op; num_modes=5
        )
        
        max_lag = 20
        lags, correlation = IsoPEPS.compute_theoretical_correlation_decay(eigenvalues, coefficients, max_lag)
        
        # Verify decay is consistent with correlation length
        # |C(r)| ~ exp(-r/ξ) for large r
        # After ~3ξ, correlation should be significantly decayed
        if isfinite(ξ) && ξ > 0
            decay_at_3xi = abs(correlation[min(Int(ceil(3*ξ)), max_lag)])
            @test decay_at_3xi < 0.1 * abs(correlation[1]) || abs(correlation[1]) < 1e-10
        end
    end
end

@testset "transfer_matrix_consistency" begin
    virtual_qubits = 1
    nqubits = 1 + 2 * virtual_qubits
    
    for row in [1, 2, 3]
        @testset "row=$row" begin
            # Generate random unitary gates
            gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
            gates = [Matrix(gate) for _ in 1:row]
            
            # Get transfer matrix from custom implementation
            T_custom = get_transfer_matrix(gates, row, virtual_qubits)
            
            # Get transfer matrix from ITensor implementation
            T_itensor, eigs_itensor, ξ_itensor = transfer_matrix_ITensor(gates, row, virtual_qubits)
            T_itensor_array = Array(T_itensor, inds(T_itensor)...)  # Convert ITensor to array
            
            # Test 1: Transfer matrices should match (up to index ordering/transpose)
            # Note: ITensor and custom implementation may have different index conventions
            # which results in a transpose. Both T and T' have the same spectrum.
            @test size(T_custom) == size(T_itensor_array)
            @test isapprox(T_custom, T_itensor_array, atol=1e-10) 
            
            # Get spectrum from all three methods
            _, gap_custom, eigenvalues_custom, _ = compute_transfer_spectrum(gates, row, nqubits)
            spectrum_mpskit, ξ_mpskit = spectrum_MPSKit(gates, row, virtual_qubits)
            
            # Test 2: Eigenvalue magnitudes should match
            eigs_custom_sorted = sort(abs.(eigenvalues_custom), rev=true)
            eigs_itensor_sorted = sort(abs.(eigs_itensor), rev=true)
            eigs_mpskit_sorted = sort(abs.(spectrum_mpskit), rev=true)
            
            n_compare = min(length(eigs_custom_sorted), length(eigs_itensor_sorted), length(eigs_mpskit_sorted))
            @test eigs_custom_sorted[1:n_compare] ≈ eigs_itensor_sorted[1:n_compare] atol=1e-6
            @test eigs_custom_sorted[1:n_compare] ≈ eigs_mpskit_sorted[1:n_compare] atol=1e-6
            
            # Test 3: Correlation lengths should match
            @test isapprox(1/gap_custom, ξ_itensor, atol=1e-6)
            @test isapprox(1/gap_custom, ξ_mpskit, atol=1e-6)
        end
    end
end

