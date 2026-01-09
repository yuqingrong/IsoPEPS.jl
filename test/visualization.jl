using Test
using IsoPEPS
using CairoMakie: Figure

@testset "save and load results" begin
    # Test CircuitOptimizationResult
    result_circuit = CircuitOptimizationResult(
        [1.0, 0.5, 0.3],
        [zeros(ComplexF64, 8, 8)],
        [0.1, 0.2],
        0.3,
        [0.5, 0.6],
        [0.8, 0.9],
        true
    )
    
    input_args = Dict{Symbol, Any}(
        :g => 2.0, :J => 1.0, :row => 3,
        :initial_params => [0.0, 0.0]
    )
    
    tmpfile = tempname() * ".json"
    save_result(tmpfile, result_circuit, input_args)
    @test isfile(tmpfile)
    
    loaded, loaded_args = load_result(tmpfile; result_type=:circuit)
    @test loaded isa CircuitOptimizationResult
    @test loaded.energy ≈ result_circuit.energy
    @test loaded.converged == result_circuit.converged
    @test loaded_args[:g] == 2.0
    @test loaded_args[:J] == 1.0
    
    rm(tmpfile)
    
    # Test ExactOptimizationResult
    result_exact = ExactOptimizationResult(
        [1.0, 0.5],
        [zeros(ComplexF64, 8, 8)],
        [0.1, 0.2],
        0.5,
        0.2,
        [1.0, 0.8],
        0.9,
        0.85,
        0.88,
        true
    )
    
    input_args = Dict{Symbol, Any}(:g => 2.0, :row => 2)
    
    tmpfile = tempname() * ".json"
    save_result(tmpfile, result_exact, input_args)
    @test isfile(tmpfile)
    
    loaded, loaded_args = load_result(tmpfile; result_type=:exact)
    @test loaded isa ExactOptimizationResult
    @test loaded.gap ≈ result_exact.gap
    @test loaded_args[:g] == 2.0
    
    rm(tmpfile)
end

@testset "plot_correlation_heatmap" begin
    corr = rand(5, 5)
    corr = (corr + corr') / 2  # Make symmetric
    
    fig = plot_correlation_heatmap(corr; title="Test")
    @test fig isa Figure
    
    # Test with Z samples
    Z_samples = randn(100)
    row = 5
    fig2 = plot_correlation_heatmap(Z_samples, row)
    @test fig2 isa Figure
end

@testset "compute_acf" begin
    data = randn(1000)
    lags, acf, acf_err = compute_acf(data; max_lag=50, n_bootstrap=10)
    
    @test length(lags) == 50
    @test length(acf) == 50
    @test length(acf_err) == 50
    @test all(acf_err .>= 0)
end

@testset "fit_acf_exponential" begin
    lags = 0:49
    acf = exp.(-lags ./ 10.0)
    
    A, ξ = fit_acf_exponential(lags, acf)
    @test A ≈ 1.0 atol=0.1
    @test ξ ≈ 10.0 atol=2.0
end

@testset "ACF correlation length matches transfer matrix gap (exact)" begin
    # Test that fitting a perfect exponential ACF = λ₂^k recovers ξ = 1/gap exactly
    # This tests the mathematical relationship: ACF(k) = λ₂^k = exp(-k*gap) = exp(-k/ξ)
    # where gap = -log(λ₂) and ξ = 1/gap = -1/log(λ₂)
    
    for λ2 in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        gap = -log(λ2)
        ξ_theory = 1.0 / gap
        
        # Generate exact ACF data: ACF(k) = λ₂^k
        max_lag = 50
        lags = 0:max_lag
        acf_exact = [λ2^k for k in lags]
        
        # Fit and recover ξ
        A, ξ_fit = fit_acf_exponential(lags, acf_exact; use_log_fit=true)
        
        # Test: ξ from fit should match 1/gap exactly (within numerical precision)
        @test ξ_fit ≈ ξ_theory atol=1e-3
        @test A ≈ 1.0 atol=1e-3
        
        # Also verify the relationship: gap = -log(λ₂) = 1/ξ
        gap_from_fit = 1.0 / ξ_fit
        @test gap_from_fit ≈ gap atol=1e-3
    end
end

@testset "ACF correlation length from channel sampling" begin
   
end

@testset "plot_acf" begin
    lags = 0:49
    acf = exp.(-lags ./ 10.0)
    
    fig = plot_acf(lags, acf; title="Test ACF")
    @test fig isa Figure
    
    # With fit using fit_acf
    fit_params = fit_acf(lags, acf)
    fig2 = plot_acf(lags, acf; fit_params=fit_params)
    @test fig2 isa Figure
end

@testset "plot_training_history" begin
    steps = 1:10
    energies = rand(10)
    
    fig = plot_training_history(steps, energies; ylabel="Energy")
    @test fig isa Figure
    
    # With reference
    fig2 = plot_training_history(steps, energies; reference=-1.0)
    @test fig2 isa Figure
    
    # With result
    result = CircuitOptimizationResult(
        energies, [], [], 0.5, [], [], true
    )
    fig3 = plot_training_history(result)
    @test fig3 isa Figure
end

@testset "plot_variance_vs_samples" begin
    samples = [100, 500, 1000, 5000]
    variances = [0.1, 0.02, 0.01, 0.002]
    
    fig = plot_variance_vs_samples(samples, variances)
    @test fig isa Figure
    
    # With errors
    errors = [0.01, 0.002, 0.001, 0.0002]
    fig2 = plot_variance_vs_samples(samples, variances; errors=errors)
    @test fig2 isa Figure
end

@testset "Sampled expectations match transfer matrix fixed point" begin
    using Statistics
    using Yao, YaoBlocks
    using Random
    using LinearAlgebra
    
    # This is the correct way to verify measurement results match propagation:
    # 1. Compute exact expectation values from transfer matrix fixed point
    # 2. Sample from quantum channel
    # 3. Check sampled means match exact values within statistical error
    
    for seed in [42, 123, 456]
        Random.seed!(seed)
        
        nqubits = 3
        row = 2
        
        # Random gates
        gate_matrix = rand_unitary(ComplexF64, 2^nqubits)
        gates = [gate_matrix for _ in 1:row]
        
        # Exact values from transfer matrix fixed point
        rho, _, _ = compute_transfer_spectrum(gates, row, nqubits)
        X_exact = real(compute_X_expectation(rho, gates, row, nqubits))
        
        # Sample
        n_samples = 500000
        _, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                          conv_step=2000, 
                                                          samples=n_samples)
        
        X_sampled = mean(X_samples)
        X_std = std(X_samples) / sqrt(length(X_samples))  # Standard error
        X_error = abs(X_sampled - X_exact)
        
        # Test: sampled mean should match exact value within 3σ
        @test X_error < 3 * X_std
        
        println("seed=$seed: ⟨X⟩_exact=$(round(X_exact, digits=4)), ⟨X⟩_sampled=$(round(X_sampled, digits=4)) ± $(round(X_std, digits=4))")
    end
end
