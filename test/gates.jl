using Test
using IsoPEPS
using LinearAlgebra
using Statistics
using Yao, YaoBlocks
using Random

@testset "build_unitary_gate" begin
    Random.seed!(1234)
    for row in 3:6, p in 2:4, nqubits in [3]
        # Shared parameters: all gates should be equal
        # Uses 3 params per qubit per layer (Rz-Ry-Rz decomposition)
        params = rand(3 * nqubits * p) .* 2π
        gates = build_unitary_gate(params, p, row, nqubits)
        @test all(gates[i] == gates[1] for i in 2:row)

        # Independent parameters: all gates should be different
        params = rand(3 * nqubits * p * row) .* 2π
        gates = build_unitary_gate(params, p, row, nqubits; share_params=false)
        @test all(gates[i] != gates[1] for i in 2:row)
    end
end

@testset "compute_energy" begin
    Random.seed!(2345)
    for row in 3:6
        a = rand()
        b = rand()
        X_samples = a * ones(100)
        Z_samples = b * ones(100)
        g = 1.0
        J = 1.0
        energy = compute_energy(X_samples, Z_samples, g, J, row)
        @test energy ≈ -a - 2 * b^2 atol = 1e-5
    end
end

@testset "build_unitary_gate_transfer_matrix_spectrum" begin
    Random.seed!(3456)
    virtual_qubits = 1
    nqubits = 1 + 2*virtual_qubits  # = 3 qubits
    
    println("\n" * "="^70)
    println("Testing build_unitary_gate transfer matrix eigenvalue spectrum")
    println("="^70)
    
    for row in [1, 2]
        @testset "row=$row" begin
            println("\n--- Row = $row ---")
            
            # Test 1: Random unitary gates (baseline)
            println("\n[1] Random unitary gates (baseline):")
            rand_gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
            rand_gates = [Matrix(rand_gate) for _ in 1:row]
            
            A_tensors_rand = IsoPEPS.gates_to_tensors(rand_gates, row, virtual_qubits)
            total_qubits = 1 + virtual_qubits + virtual_qubits * row
            matrix_size = 4^(total_qubits - 1)
            
            _, T_tensor_rand = contract_transfer_matrix(A_tensors_rand, [conj(A) for A in A_tensors_rand], row)
            T_rand = reshape(T_tensor_rand, matrix_size, matrix_size)
            
            eigs_rand = sort(abs.(eigvals(T_rand)), rev=true)
            gap_rand = -log(eigs_rand[2])
            println("   Eigenvalue magnitudes: ", round.(eigs_rand[1:min(6, end)], digits=6))
            println("   λ₁ = $(round(eigs_rand[1], digits=6)), λ₂ = $(round(eigs_rand[2], digits=6))")
            println("   Spectral gap = $(round(gap_rand, digits=4))")
            
            @test isapprox(eigs_rand[1], 1.0, atol=1e-6)
            @test eigs_rand[2] < 1.0  # Should have a gap
            
            # Test 2: build_unitary_gate with RANDOM parameters
            println("\n[2] build_unitary_gate with RANDOM params:")
            for p in [1, 2, 3, 4, 5]
                n_params = 2 * nqubits * p
                params_rand = rand(n_params)
                gates_built = build_unitary_gate(params_rand, p, row, nqubits)
                
                A_tensors_built = IsoPEPS.gates_to_tensors(gates_built, row, virtual_qubits)
                _, T_tensor_built = contract_transfer_matrix(A_tensors_built, [conj(A) for A in A_tensors_built], row)
                T_built = reshape(T_tensor_built, matrix_size, matrix_size)
                
                eigs_built = sort(abs.(eigvals(T_built)), rev=true)
                gap_built = eigs_built[2] < 1e-10 ? Inf : -log(eigs_built[2])
                
                println("   p=$p layers: λ₁=$(round(eigs_built[1], digits=6)), λ₂=$(round(eigs_built[2], digits=6)), gap=$(round(gap_built, digits=4))")
                println("      Full spectrum: ", round.(eigs_built[1:min(6, end)], digits=6))
                
                @test isapprox(eigs_built[1], 1.0, atol=1e-6)
            end
            
            # Test 3: build_unitary_gate with SMALL parameters (near identity)
            println("\n[3] build_unitary_gate with SMALL params (near identity):")
            for scale in [0.01, 0.1, 0.5, 1.0]
                p = 2
                n_params = 2 * nqubits * p
                params_small = scale * randn(n_params)
                gates_small = build_unitary_gate(params_small, p, row, nqubits)
                
                A_tensors_small = IsoPEPS.gates_to_tensors(gates_small, row, virtual_qubits)
                _, T_tensor_small = contract_transfer_matrix(A_tensors_small, [conj(A) for A in A_tensors_small], row)
                T_small = reshape(T_tensor_small, matrix_size, matrix_size)
                
                eigs_small = sort(abs.(eigvals(T_small)), rev=true)
                
                # Count eigenvalues close to 1
                n_close_to_1 = sum(abs.(eigs_small .- 1.0) .< 0.1)
                
                println("   scale=$scale: λ₂=$(round(eigs_small[2], digits=6)), #(|λ|>0.9)=$n_close_to_1 / $(length(eigs_small))")
                
                @test isapprox(eigs_small[1], 1.0, atol=1e-6)
            end
            
            # Test 4: Check if build_unitary_gate gives systematically different spectrum
            println("\n[4] Statistical comparison (10 trials each):")
            
            # Random unitaries
            gaps_random = Float64[]
            for _ in 1:10
                rand_g = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
                rand_gs = [Matrix(rand_g) for _ in 1:row]
                A_r = IsoPEPS.gates_to_tensors(rand_gs, row, virtual_qubits)
                _, T_r = contract_transfer_matrix(A_r, [conj(A) for A in A_r], row)
                T_r = reshape(T_r, matrix_size, matrix_size)
                eigs_r = sort(abs.(eigvals(T_r)), rev=true)
                push!(gaps_random, -log(eigs_r[2]))
            end
            
            # Built gates with random params
            gaps_built = Float64[]
            for _ in 1:10
                p = 2
                n_params = 2 * nqubits * p
                params_b = rand(n_params) .* 2π
                gates_b = build_unitary_gate(params_b, p, row, nqubits)
                A_b = IsoPEPS.gates_to_tensors(gates_b, row, virtual_qubits)
                _, T_b = contract_transfer_matrix(A_b, [conj(A) for A in A_b], row)
                T_b = reshape(T_b, matrix_size, matrix_size)
                eigs_b = sort(abs.(eigvals(T_b)), rev=true)
                gap_b = eigs_b[2] < 1e-10 ? Inf : -log(eigs_b[2])
                push!(gaps_built, gap_b)
            end
            
            println("   Random unitary gaps: mean=$(round(mean(gaps_random), digits=4)), std=$(round(std(gaps_random), digits=4))")
            println("   build_unitary_gate gaps: mean=$(round(mean(filter(isfinite, gaps_built)), digits=4)), std=$(round(std(filter(isfinite, gaps_built)), digits=4))")
            println("   Random unitary λ₂ range: [$(round(exp(-maximum(gaps_random)), digits=4)), $(round(exp(-minimum(gaps_random)), digits=4))]")
            
            finite_built = filter(isfinite, gaps_built)
            if !isempty(finite_built)
                println("   build_unitary_gate λ₂ range: [$(round(exp(-maximum(finite_built)), digits=4)), $(round(exp(-minimum(finite_built)), digits=4))]")
            end
        end
    end
    
    println("\n" * "="^70)
end

