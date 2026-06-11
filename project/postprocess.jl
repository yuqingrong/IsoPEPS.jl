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
function analyze_result(filename::String; pepskit_results_file::Union{String,Nothing}=nothing, dmrg_bulk_file::Union{String,Nothing}=nothing, use_exact::Bool=true)
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
    unit_cell = Symbol(get(input_args, :unit_cell, :single))
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
        nqubits=nqubits,
        J2=J2,
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
    
    
    # Save figures to project/results/figures
    figures_dir = joinpath(@__DIR__, "results", "figures")
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
# Example usage (commented out)
# ============================================================================
# Uncomment the block below (remove #= and =#) to run analysis examples

# Analyze a single result
J=1.0;g = 2.0; row=3 ; nqubits=3; p=3; virtual_qubits=1;D=2
data_dir = joinpath(@__DIR__, "results_tfim_abc")
datafile = joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1.json")
referfile = joinpath(data_dir, "pepskit_results_D=$(D).json")
result, args = analyze_result(datafile; pepskit_results_file=referfile, dmrg_bulk_file="project/results/dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json")

# M^2(q)
save_M2_vs_J2(      "project/results",
                    J2_values;
                    method=:sampling,  # sampling or exact
                    output_file="project/results/M2_sampling.json",
                    row=4,
                    nqubits=3,
                    p=3,
                    max_separation=20,
                    conv_step=100,
                    samples=1000000,
                    n_bootstrap=200,
                ) 
plot_M2_comparison(
                sampling_file="project/results/M2_sampling.json",
                save_path="project/results/figures/M2_comparison.pdf",
                markersize=4,
                show_errorbars=false)   

# structure factor
save_combined_structure_factor_data("sf.json", "project/results/heisenberg", [0.0, 0.5, 1.0];
      use_exact=false, max_separation_spin=10, max_separation_dimer=10,
      samples_files=Dict(
          0.0 => "project/results/heisenberg/samples_heisenberg_J2=0.0.json",
          0.5 => "project/results/heisenberg/samples_heisenberg_J2=0.5.json",
          1.0 => "project/results/heisenberg/samples_heisenberg_J2=1.0.json",
      ))
 fig, _, _ = plot_combined_structure_factors(
    "project/results", [0.0, 0.5, 1.0];
    data_file="sf.json",
    save_path="project/results/figures/structure_factors_combined.pdf"
)

# spin-spin correlation
fig, data = plot_bond_energy_pattern("project/results/circuit_heisenberg_j1j2_J1=1.0_J2=0.0_row=4_p=3_nqubits=3_2x2.json";
      use_exact=true, save_path="project/results/figures/bond_energy——exact.pdf")

fig, data = plot_bond_energy_pattern("project/results/heisenberg", [0.0, 0.5, 1.0];
      use_exact=false,
      save_path="project/results/figures/bond_energy_sampling.pdf"
    )

# energy vs g
plot_energy_error_vs_g("project/results/tfim", [2.0,3.0,4.0];                            
      model="tfim",                                              
      J=1.0, row=3, p=3, nqubits=3,                        
      energy_source=:computed, # computed or resampled
      conv_step=300,
      samples=3000000,
      dmrg_file="project/results/reference/dmrg_bulk_tfim_Ly3_D2_gscan.json",save_path="project/results/figures/tfim_energy_vs_g.pdf",
      markersize=6)
    
      
# variance vs samples
compute_variance_vs_samples(
    "project/results/heisenberg/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json",
    [1000, 2000, 3000, 4000,5000,6000, 7000,8000,9000, 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000];
    total_samples=nothing,
    conv_step=100,
    n_bootstrap=200,
    n_ci_bootstrap=5000,
    confidence_level=0.68,
    save_path="project/results/heisenberg/heisenberg_variance_vs_samples.json",
)

plot_variance_vs_samples(
    "project/results/heisenberg/heisenberg_variance_vs_samples.json";
    fit_scaling=true,
    marker=:circle,
    markersize=4,
    figsize=PAPER_FIGSIZE,
    save_path="project/results/figures/heisenberg_variance_vs_samples_J2=0.5.pdf",
)

 fig, E_mat = plot_energy_vs_inv_samples(
                "project/results/circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x1.json",
                [1000, 2000, 3000, 4000,5000,6000, 7000,8000,9000, 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000];
                conv_step=100, n_bootstrap=200,
                save_path="project/results/figures/tfim_energy_vs_inv_samples_g=3.0.pdf")
                # total_samples defaults to 20 * 10000 * 4 = 800_000 spins → ~200_000 columns pool

# correlation and magnetization
fig, data = plot_connected_corr_vs_g(
                    "project/results",
                    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0];
                    J=1.0, row=4, p=3, nqubits=3,
                    use_exact=true,
                    save_path="project/results/figures/NNconnected_corr_vs_g.pdf")

plot_correlation_vs_g(data_dir, [2.0, 3.0];row=3, nqubits=5,p=2,dmrg_file=joinpath(data_dir,"dmrg_bulk_tfim_Ly3_D2_gscan.json"),pepskit_file=referfile, g_c=3.04,
                    spectrum_krylovdim=200,
                    spectrum_tol=1e-7,
                    spectrum_maxiter=2000,
                    save_path="project/results/figures/corr_length_vs_g_row=3.pdf")

plot_correlation_vs_J2("project/results", [0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
                    row=4, dmrg_file="project/results/dmrg_j1j2_100x4_D=2.json")       
fig, data = plot_magnetization_vs_g(
    "project/results",
    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0];
    J=1.0, row=3, p=3, nqubits=3,
    conv_step=100, samples=40000,
    save_path="project/results/figures/magnetization_vs_g.pdf")

fig, data = plot_correlation_function(datafile;
                                   max_separation=14,
                                   conv_step=100,
                                   samples=4000000,
                                   save_path="project/results/figures/correlation_function_heisenberg_2x2_J1=$(J)_J2=0.5.pdf")

# energy dynamic
fig = plot_energy_dynamics_vs_g("project/results", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
J=1.0, row=3, p=3, nqubits=5,                                                                                                
M=10000, shots=20, conv_step=0, save_path="project/results/figures/energy_dynamics_vs_g_D=4.pdf")

fig = plot_local_xz_dynamics_vs_g("project/results", [0.5];
    J=1.0, row=3, p=3, nqubits=5,
    M=10000, shots=20, conv_step=0,
    save_path="project/results/figures/local_xz_dynamics_vs_g_D=4.pdf")

fig = plot_energy_dynamics_vs_g("project/results", [5.0];
    J=1.0, row=3, p=3, nqubits=3,
    M=10000, shots=100, conv_step=0,
    parameter_source=:random,
    random_seed=234,
    save_path="project/results/figures/energy_dynamics_vs_g_random.pdf")

fig = plot_local_xz_dynamics_vs_g("project/results", [4.0];
    J=1.0, row=3, p=3, nqubits=5,
    M=10000, shots=100, conv_step=0,
    parameter_source=:random,
    random_seed=123,
    save_path="project/results/figures/local_xz_dynamics_vs_g_random.pdf")

# circuit and gate structure
fig = plot_circuit_block(3, 5; save_path="project/results/figures/circuit_block_3x5.pdf")
display(fig)

plot_channel_circuit(3, 3, 5;
    cycles=2,
    expanded=false,
    save_path="project/results/figures/circuit_full_3x3x5.pdf")
