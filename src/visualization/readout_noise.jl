# =============================================================================
# Readout-noise energy plots
# =============================================================================

function _load_readout_plot_data(results_file::String)
    isfile(results_file) ||
        throw(ArgumentError("readout results file does not exist: $results_file"))

    data = load_results(results_file)
    get_value(key::Symbol) = _readout_get(data, key, nothing)

    p_values = get_value(:p_values)
    energy_mean = get_value(:energy_mean)
    energy_stderr = get_value(:energy_stderr)
    baseline_energy = get_value(:baseline_energy)

    isnothing(p_values) &&
        throw(ArgumentError("readout results file is missing p_values"))
    isnothing(energy_mean) &&
        throw(ArgumentError("readout results file is missing energy_mean"))
    isnothing(energy_stderr) &&
        throw(ArgumentError("readout results file is missing energy_stderr"))
    isnothing(baseline_energy) &&
        throw(ArgumentError("readout results file is missing baseline_energy"))

    probabilities = Float64.(p_values)
    energies = Float64.(energy_mean)
    errors = Float64.(energy_stderr)
    length(probabilities) == length(energies) == length(errors) ||
        throw(DimensionMismatch(
            "p_values, energy_mean, and energy_stderr must have the same length"))
    isempty(probabilities) &&
        throw(ArgumentError("readout results file contains no probability values"))
    all(isfinite, probabilities) ||
        throw(ArgumentError("p_values must be finite"))
    all(probability -> 0.0 <= probability <= 1.0, probabilities) ||
        throw(ArgumentError("p_values must be between 0 and 1"))
    all(isfinite, energies) ||
        throw(ArgumentError("energy_mean must be finite"))
    all(error -> isfinite(error) && error >= 0.0, errors) ||
        throw(ArgumentError("energy_stderr must be finite and non-negative"))

    order = sortperm(probabilities)
    return (
        p_percent=100.0 .* probabilities[order],
        energy_mean=energies[order],
        energy_stderr=errors[order],
        baseline_energy=Float64(baseline_energy),
    )
end

"""
    plot_readout_energy(results_file; figsize=PAPER_FIGSIZE, save_path=nothing)

Plot `e(p)` with standard-error bars and a dashed horizontal line at the
noiseless `p=0` energy. The input JSON must be written by
[`compute_readout_energy_scan`](@ref).

The horizontal axis is displayed as a percentage, so `p=0.005` appears as
`0.5%`.
"""
function plot_readout_energy(
        results_file::String;
        figsize=PAPER_FIGSIZE,
        save_path::Union{Nothing,String}=nothing)
    data = _load_readout_plot_data(results_file)
    p = data.p_percent
    energy = data.energy_mean
    stderr = data.energy_stderr
    baseline = data.baseline_energy

    readout_theme = merge(Theme(Scatter=(strokewidth=0,)), paper_theme())
    fig = with_theme(readout_theme) do
        fig = Figure(size=figsize)

        energy_ax = Axis(
            fig[1, 1];
            xlabel="Readout error rate p (%)",
            ylabel="Energy density e(p)",
        )
        errorbars!(
            energy_ax, p, energy, stderr;
            color=:steelblue,
            whiskerwidth=5,
        )
        scatterlines!(
            energy_ax, p, energy;
            color=:steelblue,
            marker=:circle,
            label="Measured energy",
        )
        hlines!(
            energy_ax, [baseline];
            color=:firebrick,
            linestyle=:dash,
            linewidth=1.0,
            label="p=0 baseline",
        )
        add_paper_legend!(energy_ax; position=:rb)

        if !isnothing(save_path)
            mkpath(dirname(abspath(save_path)))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    return fig
end

"""
    plot_readout_energy_bias(results_file; figsize=PAPER_FIGSIZE, save_path=nothing)

Plot the absolute measured-energy bias `|e(p) - e(0)|` with standard-error bars
from a JSON file written by [`compute_readout_energy_scan`](@ref).

The baseline is fixed by the original samples, so the bias uses the same
standard error as `e(p)`. Its lower error bar is clipped at zero.
"""
function plot_readout_energy_bias(
        results_file::String;
        figsize=PAPER_FIGSIZE,
        save_path::Union{Nothing,String}=nothing)
    data = _load_readout_plot_data(results_file)
    p = data.p_percent
    stderr = data.energy_stderr
    bias = abs.(data.energy_mean .- data.baseline_energy)
    bias_lower_error = min.(stderr, bias)

    readout_theme = merge(Theme(Scatter=(strokewidth=0,)), paper_theme())
    fig = with_theme(readout_theme) do
        fig = Figure(size=figsize)

        bias_ax = Axis(
            fig[1, 1];
            xlabel="Readout error rate p (%)",
            ylabel="|e(p) - e(0)|",
            limits=(nothing, (0.0, nothing)),
        )
        errorbars!(
            bias_ax, p, bias, bias_lower_error, stderr;
            color=:firebrick,
            whiskerwidth=5,
        )
        scatterlines!(
            bias_ax, p, bias;
            color=:firebrick,
            marker=:diamond,
        )

        if !isnothing(save_path)
            mkpath(dirname(abspath(save_path)))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    return fig
end

"""
    plot_readout_noise(results_file; energy_save_path=nothing, bias_save_path=nothing)

Create the readout-energy and absolute-bias plots as two independent figures.
Returns `(energy_figure=..., bias_figure=...)`.
"""
function plot_readout_noise(
        results_file::String;
        figsize=PAPER_FIGSIZE,
        energy_save_path::Union{Nothing,String}=nothing,
        bias_save_path::Union{Nothing,String}=nothing)
    energy_figure = plot_readout_energy(
        results_file; figsize=figsize, save_path=energy_save_path)
    bias_figure = plot_readout_energy_bias(
        results_file; figsize=figsize, save_path=bias_save_path)
    return (energy_figure=energy_figure, bias_figure=bias_figure)
end
