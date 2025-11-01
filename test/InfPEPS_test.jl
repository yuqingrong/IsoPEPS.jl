using Test
using IsoPEPS.InfPEPS  
using Yao, Optim, Manifolds, Statistics

@testset "exact_diagonalization" begin
    @testset "contract_Elist" begin
        A = randn(ComplexF64, 2,2,2,2,2)
        for row in 1:3
            code, result = contract_Elist([A for _ in 1:row], [conj(A) for _ in 1:row], row)
            @test result isa Array{ComplexF64, 4*(row+1)}
        end
    end

    @testset "left_eigen" begin
        nqubits = 3; row = 3
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nqubits))
        rho, gap = exact_left_eigen(gate, row)
        @test LinearAlgebra.tr(rho) ≈ 1.
        @test 1>gap > 0
    end
end


# For a random state, observables by circuit measurement should be the same as by exact contraction
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

@testset "cost" begin
    
end

@testset "InfPEPS Submodule Tests" begin
    
    @testset "Cost Functions" begin
        nsites = 3
        row = 3
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
        rho, _, _ = InfPEPS.iterate_channel_PEPS(gate, row; niters=50)
        
        @testset "cost_X and cost_ZZ" begin
            # Test contraction-based cost functions
            cost_x = InfPEPS.cost_X(rho.state, row, gate)
            cost_zz = InfPEPS.cost_ZZ(rho.state, row, gate)
            
            @test cost_x isa Number
            @test cost_zz isa Number
            @test isfinite(cost_x)
            @test isfinite(cost_zz)
        end
        
        @testset "cost_X_circ and cost_ZZ_circ" begin
            # Test measurement-based cost functions
            cost_x_circ = InfPEPS.cost_X_circ(rho, row, gate)
            cost_zz_circ = InfPEPS.cost_ZZ_circ(rho, row, gate)
            
            @test cost_x_circ isa Number
            @test cost_zz_circ isa Number
            @test isfinite(cost_x_circ)
            @test isfinite(cost_zz_circ)
        end
        
        @testset "exact_energy_PEPS benchmark" begin
            d = 2
            D = 2
            g = 1.0
            row = 3
            
            E = InfPEPS.exact_energy_PEPS(d, D, g, row)
            
            @test E isa Real
            @test E < 0  # Ground state energy should be negative
        end
    end
    
    @testset "Gate Construction" begin
        p = 2  # Number of layers
        params = rand(6*p) * 2π
        
        @testset "build_gate_from_params" begin
            gate_matrix = InfPEPS.build_gate_from_params(params, p)
            
            @test size(gate_matrix) == (8, 8)
            @test gate_matrix * gate_matrix' ≈ I  # Unitarity
            @test gate_matrix' * gate_matrix ≈ I
        end
        
        @testset "build_parameterized_gate" begin
            gate_layer = InfPEPS.build_parameterized_gate(params, 1, p)
            
            @test size(gate_layer) == (8, 8)
        end
    end
    
    @testset "Training and Optimization" begin
        @testset "train_energy_circ (quick test)" begin
            J = 1.0
            g = 0.0
            row = 3
            p = 2  # Small for faster testing
            params = rand(6*p) * 2π
            
            # Run short training
            X_history, final_A, final_params, final_cost, Z_lists, X_lists, gaps, param_hist = 
                InfPEPS.train_energy_circ(params, J, g, p, row; maxiter=10)
            
            @test length(X_history) > 0
            @test size(final_A) == (8, 8)
            @test length(final_params) == 6*p
            @test length(gaps) > 0
            @test final_A * final_A' ≈ I  # Final gate is unitary
        end
        
        @testset "train_nocompile" begin
            J = 1.0
            g = 0.0
            nsites = 3
            row = 3
            gate_matrix = YaoBlocks.rand_unitary(ComplexF64, 2^nsites)
            gate = YaoBlocks.matblock(gate_matrix)
            M = Manifolds.Unitary(2^nsites, Manifolds.ℂ)
            
            result, final_energy, final_gate = 
                InfPEPS.train_nocompile(gate_matrix, row, M, J, g; maxiter=5)
            
            @test final_energy isa Real
            @test isfinite(final_energy)
        end
    end
    
    @testset "Analysis" begin
        p = 2
        row = 3
        g = 0.5
        params = rand(6*p) * 2π
        
        @testset "check_gap_sensitivity" begin
            # Test sensitivity analysis for one parameter
            param_idx = 1
            param_range = range(0, 2π, length=10)  # Small range for faster testing
            
            param_vals, gaps, energies = InfPEPS.check_gap_sensitivity(
                params, param_idx, g, row, p;
                param_range=param_range,
                save_path=nothing  # Don't save during test
            )
            
            @test length(param_vals) == length(gaps)
            @test length(param_vals) == length(energies)
            @test all(isfinite, gaps)
            @test all(isfinite, energies)
        end
    end
    
    @testset "Data I/O" begin
        @testset "save_training_data" begin
            # Create dummy data
            Z_list_list = [rand(100) for _ in 1:5]
            X_list_list = [rand(100) for _ in 1:5]
            gap_list = rand(5)
            
            # Save to temporary location
            temp_dir = mktempdir()
            z_file = joinpath(temp_dir, "z_test.dat")
            x_file = joinpath(temp_dir, "x_test.dat")
            gap_file = joinpath(temp_dir, "gap_test.dat")
            
            InfPEPS.save_training_data(
                Z_list_list, X_list_list, gap_list;
                z_file=z_file, x_file=x_file, gap_file=gap_file
            )
            
            @test isfile(z_file)
            @test isfile(x_file)
            @test isfile(gap_file)
            
            # Cleanup
            rm(temp_dir, recursive=true)
        end
    end
    
    @testset "Backward Compatibility" begin
        # Test that functions are still available at top level
        @test isdefined(IsoPEPS, :iterate_channel_PEPS)
        @test isdefined(IsoPEPS, :train_energy_circ)
        @test isdefined(IsoPEPS, :cost_X)
        @test isdefined(IsoPEPS, :exact_left_eigen)
        
        # Test that we can call them without submodule prefix
        nsites = 3
        row = 1
        gate = YaoBlocks.matblock(YaoBlocks.rand_unitary(ComplexF64, 2^nsites))
        
        # Should work without InfPEPS prefix (backward compatibility)
        rho, gap = exact_left_eigen(gate, row)
        @test tr(rho) ≈ 1.0
    end
end

println("\n✅ All InfPEPS submodule tests passed!")

