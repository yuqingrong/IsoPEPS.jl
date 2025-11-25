using Test
using IsoPEPS.InfPEPS  
using Yao, Optim, Manifolds, Statistics
using LinearAlgebra
using MPSKit
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
        for row in [1,2,3]
            gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
            A_matrix = [Matrix(gate) for _ in 1:row]
            rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
            @show length(eigenvalues)
            @test LinearAlgebra.tr(rho) ≈ 1.
            @test all(eigenvalues[1:end-1] .< 1.0) 
        end
    end

  

    @testset "cost_X" begin
        nqubits = 3; 
        for row in [1,2,3]
            gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
            A_matrix = [Matrix(gate) for _ in 1:row]
            rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
            X_exp = cost_X(rho, A_matrix, row, nqubits)
            @show X_exp
            @test abs(imag(X_exp)) < 1e-10
            @test -1.0 ≤ real(X_exp) ≤ 1.0
        end
    end

    @testset "cost_ZZ" begin
        nqubits = 3; 
        for row in [1,2,3]
            gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
            A_matrix = [Matrix(gate) for _ in 1:row]
            rho, gap, eigenvalues = exact_left_eigen(A_matrix, row, nqubits)
            ZZ_ver, ZZ_hor = cost_ZZ(rho, A_matrix, row, nqubits)
            @show ZZ_ver, ZZ_hor
            @test abs(imag(ZZ_ver)) < 1e-10
            @test -1.0 ≤ real(ZZ_ver) ≤ 1.0
            @test abs(imag(ZZ_hor)) < 1e-10
            @test -1.0 ≤ real(ZZ_hor) ≤ 1.0
        end
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

    @testset "transfer matrix match MPSKit" begin
        J = 1.0; g = 1.0; d = 2; D = 4; nqubits = 3; row = 1
        E, len_gapped, entrop_gapped, spectrum, psi = result_1d(d, D, g)
        exact_A = Array{ComplexF64}(undef, D, 2, D)
        for (i, j, k) in Iterators.product(1:D, 1:2, 1:D)
            exact_A[i, j, k] = psi.AR.data[1][i, j, k]
        end
        exact_A = reshape(permutedims(exact_A, (2, 1, 3)),(d*D,D))
        nullspace = LinearAlgebra.nullspace(exact_A')
        A_matrix = vcat(exact_A, nullspace)
        A_matrix_list = [A_matrix for _ in 1:row]
        rho, gap, eigenval = exact_left_eigen(A_matrix_list, row, nqubits) 
        @test gap ≈ 1/len_gapped atol=1e-5
    end
end
