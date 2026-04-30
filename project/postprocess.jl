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
    #fig_exp = plot_expectation_values(result; g=g, J=J, row=row, p=p, nqubits=nqubits, use_exact=use_exact,
    #                                  model=model, J1=J1, J2=J2,
    #                                  datafile=skip_resample ? nothing : filename)
    #display(fig_exp)
    
    
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
    #exp_fig_path = joinpath(figures_dir, "$(base_name)_expectation_values.pdf")
    #save(exp_fig_path, fig_exp)
    #println("Saved expectation values figure to: $exp_fig_path")
    
    return result, input_args
end


# ============================================================================
# Example usage (commented out)
# ============================================================================
# Uncomment the block below (remove #= and =#) to run analysis examples

# Analyze a single result
J=1.0;g = 1.0; row=4 ; nqubits=3; p=3; virtual_qubits=1;D=2
data_dir = joinpath(@__DIR__, "results")
datafile = joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J)_J2=0.5_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json")
referfile = joinpath(data_dir, "pepskit_results_D=$(D).json")
result, args = analyze_result(datafile; pepskit_results_file=referfile, dmrg_bulk_file="project/results/dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json")

fig, data = plot_M2_vs_J2(                                                                                                                    
      data_dir,           # directory with saved JSON result files
      [0.0];   # J2 values to scan                                                                                          
      J1=1.0, row=4, nqubits=3, p=3,                    
      samples=1000000,
      max_separation=10,
      save_path="project/results/figures/M2_vs_J2.pdf"  # optional
  )
display(fig)

save_M2_vs_J2("project/results", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,0.58,0.59,0.6,0.7,0.8,0.9,1.0];                                                                              
                method=:exact, output_file="project/results/M2_exact.json",                                                                     
                row=4, nqubits=3, p=3, max_separation=20) 

save_M2_vs_J2("project/results", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,0.58,0.59,0.6,0.7,0.8,0.9,1.0];                                                                              
                method=:sampling, output_file="project/results/M2_sampling.json",
                row=4, nqubits=3, p=3, max_separation=20) 

plot_M2_comparison(exact_file="project/results/M2_exact.json",
                sampling_file="project/results/M2_sampling.json",
                dmrg_file="dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json",
                save_path="project/results/figures/M2_comparison.pdf")   
 
# Combined spin + dimer structure factor panel (J2 = 0.0, 0.5, 1.0)
fig, spin_mats, dimer_mats = plot_combined_structure_factors(
    "project/results", [0.0, 0.5, 1.0];
    row=4, p=3, nqubits=3, nq=50,
    max_separation_spin=10, max_separation_dimer=10,
    use_exact=true,
    save_path="project/results/figures/structure_factors_combined.pdf"
)
display(fig)

fig, data = plot_bond_energy_pattern("project/results/circuit_heisenberg_j1j2_J1=1.0_J2=0.0_row=4_p=3_nqubits=3_2x2.json";
      use_exact=true, save_path="project/results/figures/bond_energy——exact.pdf")

fig, data = plot_bond_energy_pattern("project/results/circuit_heisenberg_j1j2_J1=1.0_J2=1.0_row=4_p=3_nqubits=3_2x2.json";
      use_exact=false,
      samples=1000000,
      conv_step=100,
      save_path="project/results/figures/bond_energy_sampling.pdf"
    )
display(fig)

# Save the heavy computation once:
save_combined_structure_factor_data(
     "project/results/structure_factors_sampling.json",
     "project/results", [0.0, 0.5, 1.0];
     use_exact=false, conv_step=100, samples=1000000
 )

# Re-plot from saved data without recomputing:
fig, _, _ = plot_combined_structure_factors(
      "project/results", [0.0, 0.5, 1.0];
      data_file="project/results/structure_factors_sampling.json",
      save_path="project/results/figures/structure_factors_combined.pdf"
  )
 display(fig)

# Reconstruct gates and analyze
plot_energy_error_vs_g("project/results", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];                            
      model="tfim",                                              
      J1=1.0, row=3, p=3, nqubits=3,                        
      dmrg_file="project/results/dmrg_bulk_tfim_Ly3_D2_gscan.json",save_path="project/results/figures/tfim_energy_vs_g.pdf")


ns, vars, errs = compute_variance_vs_samples(
        "project/results/circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x1.json",
        [1000, 2000, 3000, 4000,5000,6000, 7000,8000,9000, 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000];
        conv_step=100, n_bootstrap=200,
        save_path="project/results/tfim_variance_vs_samples.json"   # optional
    )
# Step 2 — plot
fig = plot_variance_vs_samples(ns, vars; errors=errs,
              save_path="project/results/figures/tfim_variance_vs_samples_g=3.0.pdf")

fig, E_mat = plot_energy_vs_inv_samples(
                "project/results/circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x1.json",
                [1000, 2000, 3000, 4000,5000,6000, 7000,8000,9000, 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000];
                conv_step=100, n_bootstrap=200,
                save_path="project/results/figures/tfim_energy_vs_inv_samples_g=3.0.pdf")
                # total_samples defaults to 20 * 10000 * 4 = 800_000 spins → ~200_000 columns pool

fig, data = plot_connected_corr_vs_g(
                    "project/results",
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.75, 4.0, 4.25,4.5, 5.0];
                    J=1.0, row=3, p=3, nqubits=3,
                    use_exact=true,
                    save_path="project/results/figures/NNconnected_corr_vs_g.pdf")

fig, data = plot_magnetization_vs_g(
    "project/results",
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
    J=1.0, row=3, p=3, nqubits=3,
    conv_step=100, samples=40000,
    save_path="project/results/figures/magnetization_vs_g.pdf")
display(fig)
                   
 
 plot_correlation_vs_g(data_dir, [0.5, 0.75,1.0,1.25, 1.5,1.75, 2.0,2.25,2.5, 2.75];dmrg_file=joinpath(data_dir,"dmrg_bulk_tfim_Ly3_D2_gscan.json"),pepskit_file=referfile, g_c=3.04,
save_path="project/results/figures/corr_length_vs_g.pdf")

fig, data = plot_correlation_vs_J2("project/results", [0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
row=4, dmrg_file="project/results/dmrg_j1j2_100x4_D=2.json")
display(fig)

fig, data = plot_correlation_function(datafile;
                                   max_separation=14,
                                   conv_step=100,
                                   samples=4000000,
                                   save_path="project/results/figures/correlation_function_heisenberg_2x2_J1=$(J)_J2=0.5.pdf")

datafile = joinpath(data_dir, "circuit_J=$(J)_g=1.0_row=3_p=3_nqubits=$(nqubits).json")
fig = plot_observable_convergence(datafile; save_path="convergence.pdf")
display(fig)

fig = plot_energy_convergence_vs_g("project/results", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];                                                      
      J=1.0, row=3, p=3, nqubits=3, conv_step=100, save_path="project/results/figures/energy_convergence_vs_g.pdf")          

fig = plot_energy_dynamics(datafile;                               
      M=1000, shots=150, conv_step=0, save_path="project/results/figures/energy_dynamics.pdf")
display(fig)
fig = plot_energy_dynamics_vs_g("project/results", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
J=1.0, row=3, p=3, nqubits=3,                                                                                                
M=1000, shots=200, conv_step=0, save_path="project/results/figures/energy_dynamics_vs_g.pdf")