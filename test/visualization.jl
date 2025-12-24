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

@testset "plot_acf" begin
    lags = 0:49
    acf = exp.(-lags ./ 10.0)
    
    fig = plot_acf(lags, acf; title="Test ACF")
    @test fig isa Figure
    
    # With fit
    fig2 = plot_acf(lags, acf; fit_params=(1.0, 10.0))
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
