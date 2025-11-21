using Test
using IsoPEPS.InfPEPS  
using Yao, Optim, Manifolds, Statistics
using LinearAlgebra

@testset "exact" begin
    @testset "contract_Elist" begin
        A = randn(ComplexF64, 2,2,2,2,2)
        for row in 1:3
            code, result = contract_Elist([A for _ in 1:row], [conj(A) for _ in 1:row], row)
            @test result isa Array{ComplexF64, 4*(row+1)}
        end
    end

    @testset "left_eigen" begin
        nqubits = 3;
        for row in [1]
            gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
            A_matrix = [Matrix(gate) for _ in 1:row]
            rho, gap, eigenvalues = exact_left_eigen(A_matrix, row)
            @test LinearAlgebra.tr(rho) ≈ 1.
            @test all(eigenvalues[1:end-1] .< 1.0) 
        end
    end

    @testset "single_transfer" begin
        nqubits = 3;
        for row in 2:4
            gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 4))
            A_matrix = [Matrix(gate) for _ in 1:row]
            rho, gap, eigenvalues = single_transfer(A_matrix, nqubits)
            
            @test LinearAlgebra.tr(rho) ≈ 1.
            @test all(eigenvalues[1:end-1] .< 1.0)
        end
    end

    @testset "cost_single" begin
        nqubits = 3; row = 3
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 4))
        A_matrix = [Matrix(gate) for _ in 1:row]
        rho, gap, eigenvalues = single_transfer(A_matrix, nqubits)
        X_exp = cost_X_single(rho, A_matrix)
        ZZ_exp = cost_ZZ_single(rho, A_matrix)
        @show X_exp, ZZ_exp
    end
end

@testset "gate_and_cost" begin
    @testset "gate construction" begin
        for row in 3:6, p in 2:4, nqubits in [3]
            # sharing parameters: all gates should be equal
            params = rand(2*nqubits*p)
            A_matrix = InfPEPS.build_gate_from_params(params, p, row, nqubits)
            @test all(A_matrix[i] == A_matrix[1] for i in 2:row)
            
            # independent parameters: all gates should be different
            params = rand(2*nqubits*p*row)
            A_matrix = InfPEPS.build_gate_from_params(params, p, row, nqubits; share_params=false)
            @test all(A_matrix[i] != A_matrix[1] for i in 2:row)
        end
    end

    @testset "energy_measure" begin
        for row in 3:6
            a = rand()
            b = rand()
            X_list = a*ones(100)
            Z_list = b*ones(100)
            g = 1.0
            J = 1.0
            energy = InfPEPS.energy_measure(X_list, Z_list, g, J, row)
            @test energy ≈ -a-2*b^2 atol=1e-5
        end
    end
end

@testset "iterate_channel_PEPS" begin
    nqubits = 3; row = 3; niters = 10000
    g = 1.0; J = 1.0
    gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
    rho_iter, Z_list, X_list = iterate_channel_PEPS(gate, row; niters=niters)
    rho_eigen, gap = exact_left_eigen(gate, row)
    cost_x = InfPEPS.cost_X(rho_eigen, row, gate)
    @test mean(X_list[end-length(Z_list):end]) ≈ cost_x atol=1e-2 # TODO: add correlation test, maybe the samples for zz_correlation (1250 each) is not enough 

    z_configs = extract_Z_configurations(Z_list, row)
    energy = energy_measure(X_list, z_configs, g, J, row)
    @test energy isa Float64
end

@testset "refer" begin
    @testset "result_MPSKit" begin
        J=1.0; g=0.25; row=3; d=2; D=8
        E, len_gapped, entrop_gapped, spectrum = result_MPSKit(d, D, g, row)
    end

    @testset "" begin
        J = 1.0; g = 0.01; d = 2; D = 2
        E, len_gapped, entrop_gapped, spectrum = result_1d(d, D, g)
        @show E, len_gapped, entrop_gapped, spectrum
    end
end

