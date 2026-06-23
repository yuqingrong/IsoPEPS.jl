using IsoPEPS
using CairoMakie
set_theme!(IsoPEPS.paper_theme())
using Random
using LinearAlgebra
using JSON3
using Statistics
using OMEinsum
"""
    analyze_result(filename::String; pepskit_results_file=nothing, dmrg_bulk_file=nothing)

Analyze a saved training result from JSON file.

# Arguments
- `filename`: Path to the result JSON file
- `pepskit_results_file`: Path to pepskit results JSON file for reference energy (optional)
- `dmrg_bulk_file`: Path to DMRG bulk model JSON file for reference energy (optional)
"""
function analyze_result(filename::String; pepskit_results_file::Union{String,Nothing}=nothing, dmrg_bulk_file::Union{String,Nothing}=nothing, use_exact::Bool=true, figures_dir::Union{String,Nothing}=nothing)
    result, input_args = load_result(filename)
    
    println("=== Training Result Analysis ===")
    println("Type: ", typeof(result))
    println("Final energy: ", result.final_cost)
    
    # Extract parameters
    g = get(input_args, :g, nothing)
    J = Float64(get(input_args, :J, 1.0))
    row = get(input_args, :row, nothing)
    p = get(input_args, :p, nothing)
    nqubits = get(input_args, :nqubits, nothing)
    share_params = get(input_args, :share_params, true)
    structure = get(input_args, :structure, nothing)
    active_nqubits = get(input_args, :active_nqubits, nqubits)
    unit_cell_value = get(input_args, :unit_cell, nothing)
    unit_cell = if isnothing(unit_cell_value)
        occursin("_2x2", basename(filename)) ? :two_by_two : :single
    else
        Symbol(unit_cell_value)
    end
    conv_step = Int(get(input_args, :conv_step, 100))
    samples = get(input_args, :samples, nothing)
    model = get(input_args, :model, "tfim")
    J1 = Float64(get(input_args, :J1, 1.0))
    J2 = Float64(get(input_args, :J2, 0.0))
    
    if !isnothing(g)
        println("\nModel parameters:")
        println("  g = ", g)
        println("  J = ", J)
        println("  row = ", row)
        println("  p = ", p)
        println("  nqubits = ", nqubits)
    end
    
    # Plot training history with reference energies
    fig = plot_training_history(result;
        g=g,
        row=row,
        p=p,
        nqubits=nqubits,
        model=model,
        J=J,
        J1=J1,
        J2=J2,
        share_params=share_params,
        structure=structure,
        active_nqubits=active_nqubits,
        unit_cell=unit_cell,
        compute_exact=use_exact,
        pepskit_results_file=pepskit_results_file,
        dmrg_bulk_file=dmrg_bulk_file
    )
    display(fig)
    
    # Plot expectation values (using exact contraction if parameters available)
    # Note: passing datafile=filename triggers expensive resampling with 1M samples
    # For nqubits=5, this can take 10-30 minutes. Set datafile=nothing to skip.
    skip_resample = (nqubits >= 5)  # Skip resampling for large systems
    fig_exp = plot_expectation_values(result; g=g, J=J, row=row, p=p, nqubits=nqubits,
                                      share_params=share_params,
                                      structure=structure,
                                      active_nqubits=active_nqubits,
                                      unit_cell=unit_cell,
                                      use_exact=use_exact,
                                      model=model, J1=J1, J2=J2,
                                      datafile=skip_resample ? nothing : filename,
                                      resample_conv_step=conv_step,
                                      resample_samples=samples)
    display(fig_exp)
    
    
    # Save figures (defaults to project/results/figures; override via figures_dir kwarg)
    if isnothing(figures_dir)
        figures_dir = joinpath(@__DIR__, "results", "figures")
    end
    mkpath(figures_dir)
    
    # Generate base filename from input
    base_name = splitext(basename(filename))[1]
    
    # Save training history figure
    training_fig_path = joinpath(figures_dir, "$(base_name)_training_history.pdf")
    save(training_fig_path, fig)
    println("\nSaved training history figure to: $training_fig_path")
    
    # Save expectation values figure
    exp_fig_path = joinpath(figures_dir, "$(base_name)_expectation_values.pdf")
    save(exp_fig_path, fig_exp)
    println("Saved expectation values figure to: $exp_fig_path")
    
    return result, input_args
end


# ============================================================================
# CLI dispatcher
#
# Each plot/data block from the original top-level script is wrapped in a
# no-arg function and registered in TARGETS. Invoke via Makefile or directly:
#     julia --project=.. postprocess.jl <target>
#
# `save_M2_vs_J2` and `save_combined_structure_factor_data` are intentionally
# omitted — their JSON outputs already exist on disk and are treated as static
# inputs by the Makefile.
# ============================================================================

# Resolve CWD to repo root so the hardcoded "project/..." paths below work
# regardless of where julia is invoked from.
const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

function plot_analyze_heisenberg()
    J1=1.0;J2=0.5; row=4; nqubits=3; p=3; virtual_qubits=1; D=2
    data_dir = joinpath(@__DIR__, "results/heisenberg")
    datafile = joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(J2)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json")
    referfile = joinpath(data_dir, "pepskit_results_D=$(D).json")
    analyze_result(datafile;
        pepskit_results_file=referfile,
        dmrg_bulk_file="project/results/reference/dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json",
        figures_dir=joinpath(@__DIR__, "results", "heisenberg", "figures"))
end

function plot_m2_comparison()
    plot_M2_comparison(
        sampling_file="project/results/heisenberg/M2_sampling.json",
        save_path="project/results/heisenberg/figures/M2_comparison.pdf",
        markersize=4,
        show_errorbars=false)
end

function plot_structure_factors_combined()
    fig, _, _ = plot_combined_structure_factors(
        "project/results/heisenberg", [0.0, 0.5, 0.6, 1.0];
        data_file="project/results/heisenberg/sf.json",
        save_path="project/results/heisenberg/figures/structure_factors_combined.pdf",
    )
    fig
end

function plot_bond_energy_exact()
    plot_bond_energy_pattern("project/results/heisenberg/circuit_heisenberg_j1j2_J1=1.0_J2=0.0_row=4_p=3_nqubits=3_2x2.json";
        use_exact=true,
        save_path="project/results/heisenberg/figures/bond_energy_exact.pdf")
end

function plot_energy_vs_g_tfim()
    plot_energy_error_vs_g("project/results/tfim_abc", collect(0.0:0.25:5.0);
        model="tfim",
        J=1.0, row=3, p=3, nqubits=3,
        energy_source=:computed,
        conv_step=300,
        samples=3000000,
        dmrg_file="project/results/reference/dmrg_bulk_tfim_Ly3_D2_gscan.json",
        save_path="project/results/tfim_abc/figures/tfim_energy_vs_g.pdf",
        markersize=6)
end

function plot_energy_vs_J2_heisenberg()
    plot_energy_error_vs_g(
        "project/results/heisenberg",
        collect(0.0:0.1:1.0);
        model="heisenberg_j1j2",
        J1=1.0, row=4, p=3, nqubits=3,
        energy_source=:computed,
        dmrg_file=[
            "project/results/reference/dmrg_bulk_heisenberg_j1j2_Ly4_D32_J2scan.json",
            "project/results/reference/dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json",
        ],
        save_path="project/results/heisenberg/figures/heisenberg_energy_vs_J2.pdf",
        markersize=6,
    )
end

function compute_variance_data()
    compute_variance_vs_samples(
        "project/results/heisenberg/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json",
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000];
        total_samples=nothing,
        conv_step=100,
        n_bootstrap=200,
        n_ci_bootstrap=5000,
        confidence_level=0.68,
        save_path="project/results/heisenberg/heisenberg_variance_vs_samples.json",
    )
end

function plot_variance_vs_samples_target()
    plot_variance_vs_samples(
        "project/results/heisenberg/heisenberg_variance_vs_samples.json";
        fit_scaling=true,
        marker=:circle,
        markersize=4,
        figsize=PAPER_FIGSIZE,
        save_path="project/results/heisenberg/figures/heisenberg_variance_vs_samples_J2=0.5.pdf",
    )
end

function plot_energy_vs_inv_samples_target()
    plot_energy_vs_inv_samples(
        "project/results/tfim_abc/circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x1.json",
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000];
        conv_step=100, n_bootstrap=200,
        save_path="project/results/tfim_abc/figures/tfim_energy_vs_inv_samples_g=3.0.pdf")
end

function plot_connected_corr_vs_g_target()
    plot_connected_corr_vs_g(
        "project/results/tfim_abc",
        collect(0.0:0.25:5.0);
        J=1.0, row=3, p=3, nqubits=3,
        use_exact=true,
        save_path="project/results/tfim_abc/figures/NNconnected_corr_vs_g.pdf")
end

function plot_corr_length_vs_g()
    plot_correlation_vs_g("project/results/tfim_abc", collect(0.5:0.25:5.0);
        row=3, nqubits=3, p=3,
        dmrg_file="project/results/reference/dmrg_bulk_tfim_Ly3_D2_gscan.json",
        pepskit_file="project/results/reference/pepskit_results_D=2.json",
        g_c=3.04,
        spectrum_krylovdim=200,
        spectrum_tol=1e-7,
        spectrum_maxiter=2000,
        save_path="project/results/tfim_abc/figures/corr_length_vs_g_row=3.pdf")
end

function plot_magnetization_vs_g_target()
    plot_magnetization_vs_g(
        "project/results/tfim_abc",
        collect(0.0:0.25:5.0);
        J=1.0, row=3, p=3, nqubits=3,
        conv_step=100, samples=4000000,
        save_path="project/results/tfim_abc/figures/magnetization_vs_g.pdf")
end

function plot_energy_dynamics_tfim()
    plot_energy_dynamics_vs_g("project/results/tfim_abc", collect(0.5:0.5:4.0);
        J=1.0, row=3, p=3, nqubits=3,
        M=10000, shots=20, conv_step=0,
        save_path="project/results/tfim_abc/figures/energy_dynamics_vs_g.pdf")
end

function plot_circuit_block_target()
    plot_circuit_block(3, 5; save_path="project/results/figures/circuit_block_3x5.pdf")
end

function plot_channel_circuit_target()
    plot_channel_circuit(3, 3, 5;
        cycles=2,
        expanded=false,
        save_path="project/results/figures/circuit_full_3x3x5.pdf")
end

function compute_readout_data()
    compute_readout_energy_scan(
        "project/results/tfim_abc/circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x1.json";
        p_values=[0.0, 0.005, 0.01, 0.02, 0.05],
        repeats=100,
        seed=123,
        save_path="project/results/tfim_abc/readout_noise_energ_g=3.0.json",
    )
end

function plot_readout_energy_target()
    plot_readout_energy("project/results/tfim_abc/readout_noise_energ_g=3.0.json";
        markersize=4,
        save_path="project/results/tfim_abc/figures/readout_noise_energ_g=3.0.pdf")
end

function plot_readout_energy_bias_target()
    plot_readout_energy_bias("project/results/tfim_abc/readout_noise_energ_g=3.0.json";
        markersize=4,
        save_path="project/results/tfim_abc/figures/readout_noise_energ_error_g=3.0.pdf")
end

const TARGETS = Dict{String, Function}(
    "analyze-heisenberg"      => plot_analyze_heisenberg,
    "m2-comparison"           => plot_m2_comparison,
    "structure-factors"       => plot_structure_factors_combined,
    "bond-energy-exact"       => plot_bond_energy_exact,
    "energy-vs-g-tfim"        => plot_energy_vs_g_tfim,
    "energy-vs-J2-heisenberg" => plot_energy_vs_J2_heisenberg,
    "variance-data"           => compute_variance_data,
    "variance-vs-samples"     => plot_variance_vs_samples_target,
    "energy-vs-inv-samples"   => plot_energy_vs_inv_samples_target,
    "connected-corr-vs-g"     => plot_connected_corr_vs_g_target,
    "corr-length-vs-g"        => plot_corr_length_vs_g,
    "magnetization-vs-g"      => plot_magnetization_vs_g_target,
    "energy-dynamics-tfim"    => plot_energy_dynamics_tfim,
    "circuit-block"           => plot_circuit_block_target,
    "circuit-full"            => plot_channel_circuit_target,
    "readout-data"            => compute_readout_data,
    "readout-energy"          => plot_readout_energy_target,
    "readout-bias"            => plot_readout_energy_bias_target,
)

function main(args::Vector{String})
    if isempty(args)
        println(stderr, "usage: julia --project=.. postprocess.jl <target>")
        println(stderr, "available targets:")
        for k in sort(collect(keys(TARGETS)))
            println(stderr, "  ", k)
        end
        exit(1)
    end
    target = args[1]
    if !haskey(TARGETS, target)
        println(stderr, "unknown target: ", target)
        println(stderr, "available: ", join(sort(collect(keys(TARGETS))), ", "))
        exit(1)
    end
    cd(REPO_ROOT)
    TARGETS[target]()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
