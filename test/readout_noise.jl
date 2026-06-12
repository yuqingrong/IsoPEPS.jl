using Test
using IsoPEPS
using JSON3
using Random
using Statistics
using CairoMakie: Axis, Errorbars, Figure

@testset "apply_readout_noise" begin
    samples = [1.0, -1.0, 1.0, -1.0]

    unchanged = apply_readout_noise(samples, 0.0; rng=MersenneTwister(1))
    @test unchanged == samples
    @test unchanged !== samples

    flipped = apply_readout_noise(samples, 1.0; rng=MersenneTwister(1))
    @test flipped == -samples
    @test samples == [1.0, -1.0, 1.0, -1.0]

    noisy_a = apply_readout_noise(samples, 0.5; rng=MersenneTwister(42))
    noisy_b = apply_readout_noise(samples, 0.5; rng=MersenneTwister(42))
    @test noisy_a == noisy_b

    @test_throws ArgumentError apply_readout_noise(samples, -0.1)
    @test_throws ArgumentError apply_readout_noise(samples, 1.1)
    @test_throws ArgumentError apply_readout_noise([1.0, 0.0], 0.1)
end

@testset "plot readout-noise energy and bias" begin
    mktempdir() do dir
        results_file = joinpath(dir, "readout_energy.json")
        energy_figure_file = joinpath(dir, "readout_energy.pdf")
        bias_figure_file = joinpath(dir, "readout_bias.pdf")
        save_results(
            results_file;
            p_values=[0.02, 0.0, 0.01],
            energy_mean=[-1.90, -2.00, -1.96],
            energy_std=[0.04, 0.0, 0.02],
            energy_stderr=[0.02, 0.0, 0.01],
            baseline_energy=-2.00,
            repeats=100,
            seed=1234,
        )

        plot_data = IsoPEPS._load_readout_plot_data(results_file)
        @test plot_data.p_percent == [0.0, 1.0, 2.0]
        @test plot_data.energy_mean == [-2.00, -1.96, -1.90]

        energy_fig = plot_readout_energy(
            results_file; save_path=energy_figure_file)
        bias_fig = plot_readout_energy_bias(
            results_file; save_path=bias_figure_file)
        @test energy_fig isa Figure
        @test bias_fig isa Figure
        @test isfile(energy_figure_file)
        @test isfile(bias_figure_file)

        energy_ax = only(filter(content -> content isa Axis, energy_fig.content))
        bias_ax = only(filter(content -> content isa Axis, bias_fig.content))
        @test energy_ax.xlabel[] == "Readout error rate p (%)"
        @test energy_ax.ylabel[] == "Energy density e(p)"
        @test bias_ax.xlabel[] == "Readout error rate p (%)"
        @test bias_ax.ylabel[] == "|e(p) - e(0)|"
        @test count(plot -> plot isa Errorbars, energy_ax.scene.plots) == 1
        @test count(plot -> plot isa Errorbars, bias_ax.scene.plots) == 1

        figures = plot_readout_noise(results_file)
        @test figures.energy_figure isa Figure
        @test figures.bias_figure isa Figure

        broken_file = joinpath(dir, "broken.json")
        save_results(
            broken_file;
            p_values=[0.0],
            energy_mean=[-2.0],
            baseline_energy=-2.0,
        )
        @test_throws ArgumentError plot_readout_noise(broken_file)
    end
end

@testset "readout scaling acts on site outcomes" begin
    p = 0.1
    λ = 1 - 2p
    ideal = ones(200_000)
    noisy = apply_readout_noise(ideal, p; rng=MersenneTwister(7))

    @test mean(noisy) ≈ λ atol=0.01
    @test mean(noisy[1:end-1] .* noisy[2:end]) ≈ λ^2 atol=0.01
end

@testset "TFIM flat sample scan preserves chain boundaries" begin
    mktempdir() do dir
        samples_file = joinpath(dir, "tfim_samples.json")
        save_path = joinpath(dir, "readout_scan.json")

        X_chains = [[1.0, 1.0, 1.0, 1.0],
                    [-1.0, -1.0, -1.0, -1.0]]
        Z_chains = [[1.0, 1.0, 1.0, 1.0],
                    [-1.0, -1.0, -1.0, -1.0]]
        data = Dict(
            "type" => "CircuitOptimizationResult",
            "X_samples" => reduce(vcat, X_chains),
            "Z_samples" => reduce(vcat, Z_chains),
            "Y_samples" => Float64[],
            "input_args" => Dict(
                "model" => "tfim",
                "row" => 2,
                "J" => 1.0,
                "g" => 0.0,
                "n_runs" => 2,
            ),
        )
        open(samples_file, "w") do io
            JSON3.pretty(io, data)
        end

        result_a = compute_readout_energy_scan(
            samples_file;
            p_values=[0.0, 0.25],
            repeats=8,
            seed=123,
            save_path=save_path,
        )
        result_b = compute_readout_energy_scan(
            samples_file;
            p_values=[0.0, 0.25],
            repeats=8,
            seed=123,
        )

        @test result_a.baseline_energy ≈ -2.0
        @test result_a.energy_mean[1] ≈ result_a.baseline_energy
        @test result_a.energy_std[1] ≈ 0.0
        @test result_a.energy_samples == result_b.energy_samples
        @test result_a.n_chains == 2
        @test result_a.samples_per_chain == [4, 4]

        saved = load_results(save_path)
        @test saved["p_values"] == [0.0, 0.25]
        @test saved["baseline_energy"] ≈ -2.0
        @test saved["n_chains"] == 2
    end
end

@testset "Heisenberg nested multi-chain scan" begin
    mktempdir() do dir
        samples_file = joinpath(dir, "heisenberg_samples.json")
        X_chains = [[1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0]]
        Z_chains = [[-1.0, 1.0, -1.0, 1.0],
                    [1.0, -1.0, -1.0, 1.0]]
        Y_chains = [[1.0, 1.0, 1.0, 1.0],
                    [-1.0, -1.0, -1.0, -1.0]]
        data = Dict(
            "sample_layout" => "chains_flat_column_major",
            "model" => "heisenberg_j1j2",
            "row" => 2,
            "conv_step" => 0,
            "n_chains" => 2,
            "J1" => 1.0,
            "J2" => 0.5,
            "X_samples" => X_chains,
            "Z_samples" => Z_chains,
            "Y_samples" => Y_chains,
        )
        open(samples_file, "w") do io
            JSON3.pretty(io, data)
        end

        expected = mean(
            compute_heisenberg_energy(
                X_chains[idx], Z_chains[idx], Y_chains[idx], 1.0, 0.5, 2)
            for idx in eachindex(X_chains)
        )
        result = compute_readout_energy_scan(
            samples_file;
            p_values=[0.0, 1.0],
            repeats=4,
            seed=9,
        )

        @test result.baseline_energy ≈ expected
        @test result.energy_mean[1] ≈ expected
        @test result.energy_mean[2] ≈ expected
        @test all(iszero, result.energy_std)
        @test result.model == "heisenberg_j1j2"
        @test result.n_chains == 2
    end
end
