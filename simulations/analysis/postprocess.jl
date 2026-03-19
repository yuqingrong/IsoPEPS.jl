"""
Post-processing for simulation results.

Analyzes result JSON files and saves figures to the model's figures/ directory.

Usage:
    julia --project=simulations simulations/analysis/postprocess.jl <result.json> [result2.json ...]

Examples:
    julia --project=simulations simulations/analysis/postprocess.jl simulations/results/tfim/data/circuit_tfim_g=2.0_row=3_p=3_nqubits=3.json
    julia --project=simulations simulations/analysis/postprocess.jl simulations/results/heisenberg_j1j2/data/*.json
"""

using IsoPEPS
using CairoMakie
using JSON3
using Statistics
using LinearAlgebra
using OMEinsum

"""
    postprocess(filename::String; figures_dir=nothing, use_exact=true)

Analyze a result JSON file: plot training history, expectation values, and save figures.

Figures are saved to `simulations/results/{model}/figures/` by default.
"""
function postprocess(filename::String;
                     figures_dir::Union{String,Nothing}=nothing,
                     use_exact::Bool=true)
    result, input_args = load_result(filename)

    println("=== Post-processing: $(basename(filename)) ===")
    println("Type: ", typeof(result))

    model_str = get(input_args, :model, "tfim")
    row = get(input_args, :row, nothing)
    p = get(input_args, :p, nothing)
    nqubits = get(input_args, :nqubits, nothing)
    g = get(input_args, :g, nothing)
    J = Float64(get(input_args, :J, 1.0))
    J1 = Float64(get(input_args, :J1, 1.0))
    J2 = Float64(get(input_args, :J2, 0.0))

    # Determine figures directory
    if isnothing(figures_dir)
        figures_dir = joinpath(@__DIR__, "..", "results", model_str, "figures")
    end
    mkpath(figures_dir)

    base_name = splitext(basename(filename))[1]

    if result isa CircuitOptimizationResult
        println("Energy: ", result.final_cost)
    end

    # --- Training history ---
    fig_hist = plot_training_history(result; g=g, row=row, nqubits=nqubits)
    hist_path = joinpath(figures_dir, "$(base_name)_training_history.pdf")
    save(hist_path, fig_hist)
    println("Saved: $hist_path")

    # --- Expectation values ---
    skip_resample = (!isnothing(nqubits) && nqubits >= 5)
    fig_exp = plot_expectation_values(result;
        g=g, J=J, row=row, p=p, nqubits=nqubits,
        use_exact=use_exact, model=model_str, J1=J1, J2=J2,
        datafile=skip_resample ? nothing : filename)
    exp_path = joinpath(figures_dir, "$(base_name)_expectation_values.pdf")
    save(exp_path, fig_exp)
    println("Saved: $exp_path")

    println("Done: $(basename(filename))\n")
    return result, input_args
end

# Entry point
if !isempty(ARGS)
    for f in ARGS
        if isfile(f)
            postprocess(f)
        else
            @warn "File not found: $f"
        end
    end
else
    println("Usage: julia --project=simulations simulations/analysis/postprocess.jl <result.json> [...]")
end
