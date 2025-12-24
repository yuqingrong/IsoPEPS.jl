using Test
using IsoPEPS
using CairoMakie

@testset "TrainingData" begin
    data = TrainingData(
        2.0, 1.0, 3, 2, 3,
        [1.0, 0.5, 0.3],
        [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0],
        [0.2, 0.3],
        0.3,
        0.5
    )
    @test data.g == 2.0
    @test data.row == 3
    @test length(data.energy_history) == 3
end

@testset "save and load data" begin
    data = TrainingData(
        2.0, 1.0, 3, 2, 3,
        [1.0, 0.5, 0.3],
        [[0.1, 0.2], [0.15, 0.25]],
        [0.5, 0.6],
        [0.8, 0.9],
        [0.2, 0.3],
        0.3,
        0.5
    )
    
    tmpfile = tempname() * ".json"
    save_data(tmpfile, data)
    @test isfile(tmpfile)
    
    loaded = load_data(tmpfile)
    @test loaded.g == data.g
    @test loaded.energy_history == data.energy_history
    
    rm(tmpfile)
end

@testset "save and load results" begin
    tmpfile = tempname() * ".json"
    
    save_results(tmpfile; g=2.0, energy=[1.0, 0.5], params=[0.1, 0.2])
    @test isfile(tmpfile)
    
    results = load_results(tmpfile)
    @test results["g"] == 2.0
    @test results["energy"] == [1.0, 0.5]
    
    rm(tmpfile)
end

@testset "plot_correlation_heatmap" begin
    corr = rand(5, 5)
    corr = (corr + corr') / 2  # Make symmetric
    
    fig = plot_correlation_heatmap(corr; title="Test")
    @test fig isa Figure
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

@testset "visualize_training integration" begin
    data = TrainingData(
        2.0, 1.0, 3, 2, 3,
        [1.0, 0.5, 0.3],
        [[0.1, 0.2], [0.15, 0.25]],
        ones(100),
        ones(100),
        [0.2, 0.3],
        0.3,
        0.5
    )
    
    tmpfile = tempname() * ".json"
    tmpdir = tempname()
    
    save_data(tmpfile, data)
    visualize_training(tmpfile; save_dir=tmpdir)
    
    @test isdir(tmpdir)
    @test length(readdir(tmpdir)) > 0
    
    rm(tmpfile)
    rm(tmpdir, recursive=true)
end
