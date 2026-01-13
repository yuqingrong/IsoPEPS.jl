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
    virtual_qubits = 2
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
        
        T = get_transfer_matrix(gates, row, virtual_qubits)
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
            
            # 2. Trace preserving: Tr(T(X)) = Tr(X) (follows from isometry A†A = I)
            X = randn(ComplexF64, n, n)
            T_X = apply_T(X)
            @test tr(T_X) ≈ tr(X) atol=1e-10
            
            # 3. Hermiticity preservation: X = X† ⇒ T(X) = T(X)†
            H = randn(ComplexF64, n, n)
            H = H + H'  # Make Hermitian
            T_H = apply_T(H)
            @test T_H ≈ T_H' atol=1e-10
            
            # 4. Complete Positivity (Choi matrix test)
            # Choi matrix: C = (I ⊗ T)(|Ω⟩⟨Ω|) where |Ω⟩ = Σᵢ |i⟩|i⟩
            # For a CP map, C should be positive semidefinite
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
            @test all(choi_eigenvalues .> -1e-10)  # Positive semidefinite
            
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
        @show eigenvalues[end-1]
        @test LinearAlgebra.tr(rho) ≈ 1.0
        @test all(eigenvalues[1:end-1] .< 1.0)
    end
end

@testset "compute_X_expectation" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, virtual_qubits)
        X_exp = compute_X_expectation(rho, gates, row, virtual_qubits)
        @test abs(imag(X_exp)) < 1e-10
        @test -1.0 ≤ real(X_exp) ≤ 1.0
    end
end

@testset "compute_ZZ_expectation" begin
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
    for row in [1, 2, 3]
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        gates = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, virtual_qubits)
        ZZ_vert, ZZ_horiz = compute_ZZ_expectation(rho, gates, row, virtual_qubits)
        @test abs(imag(ZZ_vert)) < 1e-10
        @test -1.0 ≤ real(ZZ_vert) ≤ 1.0
        @test abs(imag(ZZ_horiz)) < 1e-10
        @test -1.0 ≤ real(ZZ_horiz) ≤ 1.0
    end
end

@testset "compute_exact_energy" begin
    g = 2.0
    J = 1.0
    p = 3
    row = 3
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # Gate qubits needed for tensor structure
    # Uses 3 params per qubit per layer (Rz-Ry-Rz decomposition)
    params = rand(3 * nqubits * p)
    _, energy = compute_exact_energy(params, g, J, p, row, virtual_qubits)
    @test energy isa Float64
end
