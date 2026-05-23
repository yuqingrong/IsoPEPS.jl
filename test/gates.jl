using Test
using IsoPEPS
using LinearAlgebra
using Statistics
using Yao, YaoBlocks
using Random

@testset "build_unitary_gate" begin
    Random.seed!(1234)
    ppq = IsoPEPS.PARAMS_PER_QUBIT_PER_LAYER

    for row in 3:6, p in 2:4, nqubits in [3]
        # Shared parameters: all gates should be identical
        params = rand(ppq * nqubits * p) .* 2π
        gates = build_unitary_gate(params, p, row, nqubits)
        @test length(gates) == row
        @test all(gates[i] == gates[1] for i in 2:row)
        @test all(isapprox(g * g', I, atol=1e-6) for g in gates)

        # Independent parameters: all gates should differ
        params = rand(ppq * nqubits * p * row) .* 2π
        gates = build_unitary_gate(params, p, row, nqubits; share_params=false)
        @test length(gates) == row
        @test all(gates[i] != gates[1] for i in 2:row)
        @test all(isapprox(g * g', I, atol=1e-6) for g in gates)
    end

    # Wrong param count should throw
    @test_throws AssertionError build_unitary_gate([0.1], 2, 1, 3)
    @test_throws AssertionError build_unitary_gate([0.1], 2, 2, 3; share_params=false)
    @test gate_parameter_count(2, 5) == ppq * 5 * 2
    @test gate_parameter_count(2, 5; active_nqubits=3) == ppq * 5 * 2

    # max_stride kwarg: stride=1 restricts entangling structure
    Random.seed!(42)
    nqubits = 4; p = 2; row = 2
    params = rand(ppq * nqubits * p) .* 2π
    gates_full  = build_unitary_gate(params, p, row, nqubits; max_stride=nqubits-1)
    gates_short = build_unitary_gate(params, p, row, nqubits; max_stride=1)
    @test gates_full[1] != gates_short[1]
    @test all(isapprox(g * g', I, atol=1e-6) for g in vcat(gates_full, gates_short))

    # active_nqubits keeps a smaller embedded circuit's CNOT pattern intact.
    p = 3; row = 2; nfrom = 3; nto = 5
    params3 = rand(ppq * nfrom * p) .* 2π
    params5 = embed_params(params3, p, nfrom, nto; active_nqubits_to=nfrom)
    gates3 = build_unitary_gate(params3, p, row, nfrom)
    gates5 = build_unitary_gate(params5, p, row, nto; active_nqubits=nfrom)
    expected5 = kron(Matrix{ComplexF64}(I, 1 << (nto - nfrom), 1 << (nto - nfrom)), gates3[1])
    @test gates5[1] ≈ expected5 atol=1e-10
    @test all(isapprox(g * g', I, atol=1e-6) for g in gates5)

    # Fully active 5-qubit gates do not insert an extra Rx-Rz block before CNOT layer 4.
    params5_active = embed_params(params3, p, nfrom, nto)
    for r in 1:p
        for q in 1:nto
            idx = ppq * nto * (r-1) + ppq * (q-1) + 1
            params5_active[idx] = 0.37 + 0.11q + 0.03r
            params5_active[idx+1] = 0.23 + 0.07q + 0.05r
        end
    end
    gates5_active = build_unitary_gate(params5_active, p, row, nto; active_nqubits=nto)
    expected_active = Matrix{ComplexF64}(I, 1 << nto, 1 << nto)
    rotations_from = function(offset, layer)
        rotations = Matrix{ComplexF64}(undef, 1, 1)
        rotations[1, 1] = 1
        for q in 1:nto
            idx = offset + ppq * nto * (layer-1) + ppq * (q-1) + 1
            θx = params5_active[idx]
            θz = params5_active[idx+1]
            Rx = ComplexF64[cos(θx/2) -im*sin(θx/2); -im*sin(θx/2) cos(θx/2)]
            Rz = ComplexF64[exp(-im * θz/2) 0; 0 exp(im * θz/2)]
            rotations = kron(Rx * Rz, rotations)
        end
        return rotations
    end
    cnot_product = Matrix{ComplexF64}(I, 1 << nto, 1 << nto)
    for (control, target) in cnot_pattern(nto)
        cnot_product *= Matrix(cnot(nto, control, target))
    end
    for r in 1:p
        rotations = rotations_from(0, r)
        expected_active *= rotations * cnot_product
    end
    @test gates5_active[1] ≈ expected_active atol=1e-10
end

@testset "local circuit structure" begin
    @test cnot_pattern(1) == Tuple{Int,Int}[]
    @test cnot_pattern(3) == [(2, 1), (3, 2), (1, 3)]
    @test cnot_pattern(4; max_stride=1) == [(2, 1), (3, 2), (4, 3)]
    @test cnot_pattern(5) == [(2, 1), (3, 2), (1, 3), (4, 1), (4, 2), (5, 3), (5, 4)]
    @test cnot_pattern(5; active_nqubits=3) == [(2, 1), (3, 2), (1, 3)]

    ops = local_circuit_ops(2, 3)
    @test length(ops) == 18
    @test ops[1] == LocalCircuitOp(:rx, (1,), 1, 1)
    @test ops[2] == LocalCircuitOp(:rz, (1,), 1, 2)
    @test ops[5] == LocalCircuitOp(:rx, (3,), 1, 5)
    @test ops[6] == LocalCircuitOp(:rz, (3,), 1, 6)
    @test ops[7:9] == [
        LocalCircuitOp(:cnot, (2, 1), 1, nothing),
        LocalCircuitOp(:cnot, (3, 2), 1, nothing),
        LocalCircuitOp(:cnot, (1, 3), 1, nothing),
    ]
    @test ops[10] == LocalCircuitOp(:rx, (1,), 2, 7)

    embedded_ops = local_circuit_ops(1, 5; active_nqubits=3)
    @test all(op.kind == :cnot || only(op.qubits) <= 3 for op in embedded_ops)
    @test count(op -> op.kind == :rx, embedded_ops) == 3
    @test count(op -> op.kind == :rz, embedded_ops) == 3

    ops5 = local_circuit_ops(1, 5)
    fourth_cnot = findall(op -> op.kind == :cnot, ops5)[4]
    @test count(op -> op.kind == :rx, ops5) == 5
    @test count(op -> op.kind == :rz, ops5) == 5
    @test ops5[fourth_cnot-3:fourth_cnot-1] == [
        LocalCircuitOp(:cnot, (2, 1), 1, nothing),
        LocalCircuitOp(:cnot, (3, 2), 1, nothing),
        LocalCircuitOp(:cnot, (1, 3), 1, nothing),
    ]
    @test ops5[fourth_cnot] == LocalCircuitOp(:cnot, (4, 1), 1, nothing)

    one_qubit_ops = local_circuit_ops(1, 1)
    @test one_qubit_ops == [
        LocalCircuitOp(:rx, (1,), 1, 1),
        LocalCircuitOp(:rz, (1,), 1, 2),
    ]

    tex = circuit_quantikz(1, 3)
    @test occursin("\\begin{quantikz}", tex)
    @test occursin("\\gate{R_x(\\theta_{1})}", tex)
    @test occursin("\\gate{R_z(\\theta_{6})}", tex)
    @test occursin("\\ctrl{-1}", tex)
    @test occursin("\\targ{}", tex)
end

@testset "build_unitary_gate_2x2" begin
    Random.seed!(5678)
    ppq = IsoPEPS.PARAMS_PER_QUBIT_PER_LAYER

    for row in 2:5, p in 1:3, nqubits in [3]
        n_params = gate_parameter_count(p, nqubits; unit_cell=:two_by_two)
        params = rand(n_params) .* 2π
        gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)

        @test length(gates_odd)  == row
        @test length(gates_even) == row

        # All gates should be unitary
        @test all(isapprox(g * g', I, atol=1e-6) for g in vcat(gates_odd, gates_even))

        # Tiling pattern: odd[1]==odd[3], odd[2]==odd[4], even[1]==even[3], etc.
        for i in 1:row-2
            @test gates_odd[i]  === gates_odd[i+2]
            @test gates_even[i] === gates_even[i+2]
        end

        # Odd and even columns should differ (with very high probability for random params)
        @test gates_odd[1] != gates_even[1]
    end

    # Wrong param count should throw
    @test_throws AssertionError build_unitary_gate_2x2([0.1], 2, 1, 3)

    # active_nqubits also applies to every gate in a 2×2 unit cell.
    p = 2; row = 3; nfrom = 3; nto = 5
    chunk3 = ppq * nfrom * p
    params3 = rand(4 * chunk3) .* 2π
    params5 = embed_params(params3, p, nfrom, nto; unit_cell=:two_by_two,
                           active_nqubits_to=nfrom)
    odd3, even3 = build_unitary_gate_2x2(params3, p, row, nfrom)
    odd5, even5 = build_unitary_gate_2x2(params5, p, row, nto; active_nqubits=nfrom)
    idle = Matrix{ComplexF64}(I, 1 << (nto - nfrom), 1 << (nto - nfrom))
    @test odd5[1] ≈ kron(idle, odd3[1]) atol=1e-10
    @test odd5[2] ≈ kron(idle, odd3[2]) atol=1e-10
    @test even5[1] ≈ kron(idle, even3[1]) atol=1e-10
    @test even5[2] ≈ kron(idle, even3[2]) atol=1e-10
end

@testset "embed_params" begin
    Random.seed!(9012)
    ppq = IsoPEPS.PARAMS_PER_QUBIT_PER_LAYER

    for p in [1, 2, 3]
        nfrom = 3; nto = 5
        chunk_from = ppq * nfrom * p

        # Single unit cell
        params_from = rand(chunk_from)
        params_to   = embed_params(params_from, p, nfrom, nto)
        @test length(params_to) == gate_parameter_count(p, nto)

        # Original qubit rotations preserved layer-by-layer
        for r in 1:p, i in 1:nfrom
            old_idx = ppq * nfrom * (r-1) + ppq*(i-1) + 1
            new_idx = ppq * nto   * (r-1) + ppq*(i-1) + 1
            for k in 0:ppq-1
                @test params_from[old_idx+k] == params_to[new_idx+k]
            end
        end

        # Extra qubits (nfrom+1 : nto) default to 0 in every layer
        for r in 1:p, i in nfrom+1:nto
            new_idx = ppq * nto * (r-1) + ppq*(i-1) + 1
            for k in 0:ppq-1
                @test params_to[new_idx+k] == 0.0
            end
        end

        # 2×2 unit cell
        params_from4 = rand(4 * chunk_from)
        params_to4   = embed_params(params_from4, p, nfrom, nto; unit_cell=:two_by_two)
        @test length(params_to4) == gate_parameter_count(p, nto; unit_cell=:two_by_two)

        # nqubits_to > nqubits_from required
        @test_throws ErrorException embed_params(params_from, p, nfrom, nfrom)
        @test_throws ErrorException embed_params(params_from, p, nfrom, nfrom - 1)
    end
end

@testset "compute_tfim_energy" begin
    Random.seed!(2345)
    for row in 3:6
        a = rand()
        b = rand()
        X_samples = a * ones(100)
        Z_samples = b * ones(100)
        g = 1.0
        J = 1.0
        energy = compute_tfim_energy(X_samples, Z_samples, g, J, row)
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
                n_params = gate_parameter_count(p, nqubits)
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
                n_params = gate_parameter_count(p, nqubits)
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
                n_params = gate_parameter_count(p, nqubits)
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
