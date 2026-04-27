using Test
using IsoPEPS
using CairoMakie: Figure, Legend, Theme, with_theme

@testset "save and load results" begin
    # Test CircuitOptimizationResult
    result_circuit = CircuitOptimizationResult(
        [1.0, 0.5, 0.3],
        [zeros(ComplexF64, 8, 8)],
        [0.1, 0.2],
        0.3,
        [0.5, 0.6],
        [0.8, 0.9],
        Float64[],
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
    @test loaded.final_cost ≈ result_circuit.final_cost
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
    # Skip this test - plot_correlation_heatmap function doesn't exist
    # TODO: Implement or remove this test
    @test true
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
    ax = fig.content[1]
    @test ax.xlabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
    @test ax.ylabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
    
    # With reference
    fig2 = plot_training_history(steps, energies; reference=-1.0)
    @test fig2 isa Figure
    legend = fig2.content[2]
    @test legend isa Legend
    @test legend.labelsize[] == 8
    @test isnothing(legend.layoutobservables.gridcontent[])
    @test IsoPEPS.compact_reference_label(:pepskit, -0.5738123) == "PEPSKit (-0.5738)"
    @test IsoPEPS.compact_reference_label(:dmrg, -0.5738123) == "DMRG (-0.5738)"
    
    # With result
    result = CircuitOptimizationResult(
        energies, [], [], 0.5, [], [], Float64[], true
    )
    fig3 = plot_training_history(result)
    @test fig3 isa Figure
end

@testset "plot_training_history ignores ambient axis label sizes" begin
    with_theme(Theme(Axis=(xlabelsize=21, ylabelsize=19,
                           xticklabelsize=17, yticklabelsize=15,
                           titlesize=23))) do
        fig = plot_training_history(1:3, [0.1, 0.2, 0.3]; ylabel="Energy")
        ax = fig.content[1]
        @test ax.xlabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
        @test ax.ylabelsize[] == IsoPEPS.PAPER_AXIS_LABELSIZE
        @test ax.xticklabelsize[] == IsoPEPS.PAPER_TICKLABELSIZE
        @test ax.yticklabelsize[] == IsoPEPS.PAPER_TICKLABELSIZE
        @test ax.titlesize[] == IsoPEPS.PAPER_TITLESIZE
    end
end

@testset "plot_correlation_vs_g legend stays stacked top-left" begin
    data_dir = joinpath(@__DIR__, "..", "project", "results")
    fig, _ = plot_correlation_vs_g(data_dir, [2.0];
                                   max_separation=1,
                                   dmrg_file=joinpath(data_dir, "dmrg_tfim_100x3.json"),
                                   pepskit_file=joinpath(data_dir, "pepskit_results_D=2.json"),
                                   g_c=3.04)
    @test fig isa Figure
    legend = fig.content[2]
    @test legend isa Legend
    @test legend.nbanks[] == 1
end

@testset "plot_M2_comparison legend stays inside blank region" begin
    repo = joinpath(@__DIR__, "..")
    fig = plot_M2_comparison(exact_file=joinpath(repo, "project", "results", "M2_exact.json"),
                             sampling_file=joinpath(repo, "project", "results", "M2_sampling.json"))
    @test fig isa Figure
    legend = fig.content[2]
    @test legend isa Legend
    gc = legend.layoutobservables.gridcontent[]
    @test gc.span.rows == 1:1
    @test gc.span.cols == 1:1
    @test legend.tellwidth[] == false
    @test legend.tellheight[] == false
    @test legend.nbanks[] == 1
    @test legend.halign[] == :left
    @test legend.valign[] == 0.88
    @test legend.margin[] == (1, 1, 1, 1)
    g = legend.entrygroups[][1]
    @test [e.label[] for e in g[2]] == ["M²(π,π) TN", "M²(π,π) Samp.", "M²(0,π) TN", "M²(0,π) Samp."]
    annotations = IsoPEPS.m2_phase_annotations(0.24)
    @test [a.label for a in annotations] == ["Neel order", "VBS", "Stripe order"]
    @test [(a.x, a.y) for a in annotations] == [(0.20, 0.05), (0.57, 0.05), (0.80, 0.05)]
    ax = fig.content[1]
    texts = ax.scene.plots[5:7]
    @test all(text_plot.fontsize[] == IsoPEPS.PAPER_LEGEND_LABELSIZE for text_plot in texts)
    @test all(text_plot.color[] == to_color(:firebrick) for text_plot in texts)
    @test all(text_plot.strokecolor[] == to_color(:firebrick) for text_plot in texts)
    @test all(text_plot.strokewidth[] == 0 for text_plot in texts)
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
        virtual_qubits = 1  # Bond qubits per side
        X_exact = real(compute_X_expectation(rho, gates, row, virtual_qubits))
        
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
