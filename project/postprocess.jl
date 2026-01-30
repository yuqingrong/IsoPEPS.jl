using IsoPEPS
using CairoMakie
using Random
using LinearAlgebra
using JSON3
using Statistics
using OMEinsum
"""
    analyze_result(filename::String; pepskit_results_file=nothing)

Analyze a saved training result from JSON file.

# Arguments
- `filename`: Path to the result JSON file
- `pepskit_results_file`: Path to pepskit results JSON file for reference energy (optional)
"""
function analyze_result(filename::String; pepskit_results_file::Union{String,Nothing}=nothing, use_exact::Bool=true)
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
    
    if !isnothing(g)
        println("\nModel parameters:")
        println("  g = ", g)
        println("  J = ", J)
        println("  row = ", row)
        println("  p = ", p)
        println("  nqubits = ", nqubits)
    end
    
    # Plot training history with reference from pepskit results
    fig = plot_training_history(result;
        g=g,
        row=row,
        nqubits=nqubits,
        pepskit_results_file=pepskit_results_file
    )
    display(fig)
    
    # Plot expectation values (using exact contraction if parameters available)
    fig_exp = plot_expectation_values(result; g=g, J=J, row=row, p=p, nqubits=nqubits, use_exact=use_exact)
    display(fig_exp)
    
    # Compute entropy and channel gaps if parameters are available
    fig_channels = nothing
    analysis = nothing
    if !isnothing(row) && !isnothing(p) && !isnothing(nqubits) && !isempty(result.final_params)
        println("\n=== Channel Analysis ===")
        
        # Use plot_channel_analysis from visualization.jl
        fig_channels, analysis = plot_channel_analysis(result;
            row=row, p=p, nqubits=nqubits, g=g, share_params=share_params)
        
        println("Virtual bond entropy: ", round(analysis.S_virtual, digits=4))
        println("Virtual channel gap: ", round(analysis.gap_virtual, digits=4))
        println("Physical entropy: ", round(analysis.S_physical, digits=4))
        println("Physical channel gap: ", round(analysis.gap_physical, digits=4))
        
        display(fig_channels)
    end
    
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
    
    # Save channel analysis figure if computed
    if !isnothing(fig_channels)
        channels_fig_path = joinpath(figures_dir, "$(base_name)_channel_analysis.pdf")
        save(channels_fig_path, fig_channels)
        println("Saved channel analysis figure to: $channels_fig_path")
    end
    
    return result, input_args
end

"""
    compare_results(file1::String, file2::String; labels=nothing)

Compare two optimization results from JSON files.

# Arguments
- `file1`: Path to first result JSON file
- `file2`: Path to second result JSON file
- `labels`: Tuple of (label1, label2) for plot legend (default: use filenames)
"""
function compare_results(file1::String, file2::String; labels=nothing)
    result1, args1 = load_result(file1)
    result2, args2 = load_result(file2)
    
    if isnothing(labels)
        labels = (basename(file1), basename(file2))
    end
    
    println("\n=== Results Comparison ===")
    println("Result 1 ($(labels[1])):")
    println("  Type: ", typeof(result1))
    println("  Energy: ", result1.energy)
    println("  Converged: ", result1.converged)
    
    println("\nResult 2 ($(labels[2])):")
    println("  Type: ", typeof(result2))
    println("  Energy: ", result2.energy)
    println("  Converged: ", result2.converged)
    
    # Plot comparison
    fig = Figure(size=(1200, 500))
    
    ax1 = Axis(fig[1, 1], xlabel="Iteration", ylabel="Energy", title=labels[1])
    lines!(ax1, 1:length(result1.energy_history), result1.energy_history, linewidth=2)
    
    ax2 = Axis(fig[1, 2], xlabel="Iteration", ylabel="Energy", title=labels[2])
    lines!(ax2, 1:length(result2.energy_history), result2.energy_history, linewidth=2)
    
    display(fig)
    
    return result1, result2
end

"""
    visualize_correlation(filename::String)

Visualize spin-spin correlation from circuit optimization result stored in JSON file.

# Arguments
- `filename`: Path to JSON result file

# Returns
- Figure object or nothing if no Z samples available
"""
function visualize_correlation(filename::String)
    result, input_args = load_result(filename)
    
    if !(result isa CircuitOptimizationResult)
        @warn "Result is not CircuitOptimizationResult, correlation not available"
        return nothing
    end
    
    if isempty(result.final_Z_samples)
        @warn "No Z samples in result"
        return nothing
    end
    
    row = input_args[:row]
    fig = plot_correlation_heatmap(result.final_Z_samples, row; 
                                    title="Spin-Spin Correlation ($(basename(filename)))")
    display(fig)
    
    return fig
end

"""
    analyze_acf(filename::String; max_lag=100, basis=:Z, resample=false, conv_step=1000, samples=100000)

Analyze autocorrelation function of samples from result stored in JSON file.

# Arguments
- `filename`: Path to JSON result file
- `max_lag`: Maximum lag for ACF computation (default: 100)
- `basis`: Which samples to use, :Z or :X (default: :Z)
- `resample`: If true, generate fresh samples instead of using saved ones (default: false)
- `conv_step`: Convergence steps before sampling when resampling (default: 1000)
- `samples`: Number of samples to generate when resampling (default: 100000)

# Returns
- Tuple of (lags, acf, ξ) where ξ is the correlation length

# Example
```julia
# Analyze ACF using saved samples
lags, acf, ξ = analyze_acf("results/circuit_J=1.0_g=2.0_row=6.json")

# Analyze ACF with fresh resampled data
lags, acf, ξ = analyze_acf("results/circuit_J=1.0_g=2.0_row=6.json"; resample=true, samples=50000)
```
"""
function analyze_acf(filename::String,row::Int; max_lag=100,basis=:Z, resample=true, conv_step=1000, samples=1000000)
    result, input_args = load_result(filename)
    
    if !(result isa CircuitOptimizationResult)
        @warn "Result is not CircuitOptimizationResult, samples not available"
        return nothing
    end
    
    # Get samples - either from resampling or from saved result
    if resample
        println("Generating fresh samples for ACF analysis...")
        rho, Z_samples_new, X_samples_new, params, gates = resample_circuit(filename; 
                                                                              conv_step=conv_step, 
                                                                              samples=samples)
        sample_data = basis == :Z ? Z_samples_new : X_samples_new
        sample_source = "resampled"
        
        # Compute and display energy from resampled data
        # Make sure samples have same length
        min_len = min(length(X_samples_new), length(Z_samples_new))
        try
            energy = compute_energy(X_samples_new[1:min_len], Z_samples_new[1:min_len], 
                                   input_args[:g], input_args[:J], input_args[:row])
            println("\n=== Energy from Resampled Data ===")
            println("Energy: $energy")
            println("g=$(input_args[:g]), J=$(input_args[:J]), row=$(input_args[:row])")
        catch e
            println("Error computing energy: $e")
            @show length(X_samples_new), length(Z_samples_new)
        end
    else
        # Samples are stored as matrices (n_chains × n_samples_per_chain)
        # Keep as matrix to compute ACF per chain and average
        sample_matrix = basis == :Z ? result.final_Z_samples : result.final_X_samples
        sample_data = sample_matrix
        sample_source = "saved"
        n_chains = size(sample_matrix, 1)
        
        # Compute and display energy from saved samples (need to flatten for energy computation)
        Z_vec = vec(result.final_Z_samples)
        X_vec = vec(result.final_X_samples)
        min_len = min(length(X_vec), length(Z_vec))
        try
            energy_computed = compute_energy(X_vec[1:min_len], Z_vec[1:min_len], 
                                            input_args[:g], input_args[:J], input_args[:row])
            println("\n=== Energy from Saved Samples ===")
            println("Energy (recomputed): $energy_computed")
            println("Energy (stored):     $(result.final_cost)")
            println("Difference:          $(abs(energy_computed - result.final_cost))")
        catch e
            println("Error computing energy: $e")
            @show length(X_vec), length(Z_vec)
        end
    end
    
    if isempty(sample_data)
        @warn "No $(basis) samples available"
        return nothing
    end
    
    println("\n=== ACF Analysis ($(sample_source) samples) ===")
    println("Basis: $basis")
    if sample_data isa Matrix
        println("Number of chains: $(size(sample_data, 1))")
        println("Samples per chain: $(size(sample_data, 2))")
        println("Total samples: $(length(sample_data))")
    else
        println("Number of samples: $(length(sample_data))")
    end
    
    # Compute transfer matrix gap for comparison
    p = input_args[:p]
    nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)
    _, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    ξ_transfer = 1.0 / gap  # Theoretical correlation length from transfer matrix
    
    # Subsample every `row` steps so each lag = one full layer
    # This should match the transfer matrix which also describes one layer
    if sample_data isa Matrix
        # For matrix: subsample each chain independently
        subsampled = sample_data[:, 100:row:end]
        lags, acf, acf_err = compute_acf(subsampled; max_lag=max_lag, n_bootstrap=50)
    else
        # For vector: subsample directly, reshape to 1×N matrix for compute_acf
        subsampled = reshape(sample_data[100:row:end], 1, :)
        lags, acf, acf_err = compute_acf(subsampled; max_lag=max_lag, n_bootstrap=50)
    end
    @show acf, acf_err
    fit_params = fit_acf(lags, acf; include_zero=true)
    
    println("\n--- Correlation Length Comparison ---")
    println("Transfer matrix gap:        gap = $(round(gap, digits=4))")
    println("Transfer matrix ξ:          1/gap = $(round(ξ_transfer, digits=4))")
    println("ACF correlation length:     ξ_ACF = $(round(fit_params.ξ, digits=4))")
    println("Ratio (ξ_ACF / ξ_transfer): $(round(fit_params.ξ / ξ_transfer, digits=4))")
    println("Second eigenvalue |λ₂|:     $(round(eigenvalues[2], digits=6))")
    println("Fitted |λ₂|:                $(round(fit_params.λ₂, digits=6))")
    println("Amplitude A:                $(round(fit_params.A, digits=4))")
    
    title_text = "ACF $(basename(filename))\nξ_ACF=$(round(fit_params.ξ, digits=2)), ξ_transfer=$(round(ξ_transfer, digits=2))"
    if resample
        title_text *= " [resampled]"
    end
    
    fig = plot_acf(lags, acf; 
                    acf_err=acf_err,
                    fit_params=fit_params,
                    logscale=true,
                    title=title_text)
    display(fig)
    
    # Save figure to file
    output_filename = replace(basename(filename), ".json" => "_acf.pdf")
    output_path = joinpath(dirname(filename), output_filename)
    save(output_path, fig)
    println("Figure saved to: $output_path")
    
    # Correlation length computed but not saved to file
    println("Correlation length ξ=$(round(fit_params.ξ, digits=3))")
    
    return lags, acf, fit_params
end

"""
    resample_circuit(filename::String; conv_step=1000, samples=100000, measure_first=nothing)

Extract final parameters from a saved result and re-run the circuit to generate new samples.

# Arguments
- `filename`: Path to JSON result file containing CircuitOptimizationResult
- `conv_step`: Number of convergence steps before sampling (default: 1000)
- `samples`: Number of samples to collect (default: 100000)
- `measure_first`: Which observable to measure first, :X or :Z (default: use value from saved result)

# Returns
- Tuple of (rho, Z_samples, X_samples, params, gates) where:
  - `rho`: Final quantum state
  - `Z_samples`: Vector of Z measurement outcomes
  - `X_samples`: Vector of X measurement outcomes
  - `params`: Parameters used (from the saved result)
  - `gates`: Gates reconstructed from parameters

# Example
```julia
rho, Z_samples, X_samples, params, gates = resample_circuit("results/circuit_J=1.0_g=2.0_row=6.json"; samples=50000)
```
"""
function resample_circuit(filename::String; conv_step=1000, samples=100000, measure_first=nothing)
    result, input_args = load_result(filename)
    
    if !(result isa CircuitOptimizationResult)
        @warn "Result is not CircuitOptimizationResult, cannot resample"
        return nothing
    end
    
    # Extract parameters from result
    params = result.params
    
    # Extract circuit configuration from input_args
    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    
    # Use measure_first from result if not specified
    if isnothing(measure_first)
        measure_first = Symbol(get(input_args, :measure_first, "Z"))
    end
    
    println("=== Resampling Circuit ===")
    println("File: ", basename(filename))
    println("Parameters: $(length(params)) params")
    println("Configuration: p=$p, row=$row, nqubits=$nqubits")
    println("Share params: $share_params")
    println("Measure first: $measure_first")
    println("Conv steps: $conv_step, Samples: $samples")
    
    # Reconstruct gates from parameters
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    
    # Run the quantum channel to generate new samples
    println("\nGenerating new samples...")
    rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                        conv_step=conv_step, 
                                                        samples=samples,
                                                        measure_first=measure_first)
    
    println("Generated $(length(Z_samples)) Z samples and $(length(X_samples)) X samples")
    
    return rho, Z_samples, X_samples, params, gates
end

"""
    plot_energy_vs_g(data_dir::String; J=1.0, row=3)

Load all results matching the pattern and plot energy vs g.

# Arguments
- `data_dir`: Directory containing JSON result files
- `J`: Filter results by J value (default: 1.0)
- `row`: Filter results by row value (default: 3)

# Returns
- Figure object
"""
function plot_energy_vs_g(data_dir::String; J=1.0, row=3)
    # Find all matching files
    pattern = r"circuit_J=([\d\.]+)_g=([\d\.]+)_row=(\d+)\.json"
    files = filter(readdir(data_dir, join=true)) do f
        m = match(pattern, basename(f))
        !isnothing(m) && parse(Float64, m.captures[1]) == J && parse(Int, m.captures[3]) == row
    end
    
    if isempty(files)
        @warn "No matching files found in $data_dir"
        return nothing
    end
    
    # Load results and extract g, energy pairs
    data = map(files) do file
        result, input_args = load_result(file)
        (g=input_args[:g], energy=result.final_cost, converged=result.converged)
    end
    
    # Sort by g
    sort!(data, by=x->x.g)
    
    # Plot
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="Transverse field (g)", 
              ylabel="Ground state energy",
              title="TFIM Energy Landscape (J=$J, row=$row)")
    
    g_values = [d.g for d in data]
    energies = [d.energy for d in data]
    converged = [d.converged for d in data]
    
    # Plot converged and non-converged separately
    conv_mask = converged
    scatter!(ax, g_values[conv_mask], energies[conv_mask], 
             marker=:circle, markersize=12, color=:blue, label="Converged")
    if !all(conv_mask)
        scatter!(ax, g_values[.!conv_mask], energies[.!conv_mask], 
                 marker=:xcross, markersize=12, color=:red, label="Not converged")
    end
    
    lines!(ax, g_values, energies, linewidth=2, alpha=0.5)
    
    axislegend(ax, position=:rt)
    
    return fig
end

"""
    plot_training_history_comparison(data_dir::String; J=1.0, row=3, selected_g=nothing)

Plot training history for multiple g values from saved JSON files.

# Arguments
- `data_dir`: Directory containing JSON result files
- `J`: Filter results by J value (default: 1.0)
- `row`: Filter results by row value (default: 3)
- `selected_g`: Vector of g values to plot (default: plot all)

# Returns
- Figure object
"""
function plot_training_history_comparison(data_dir::String; J=1.0, row=3, selected_g=nothing)
    # Find all matching files
    pattern = r"circuit_J=([\d\.]+)_g=([\d\.]+)_row=(\d+)\.json"
    files = filter(readdir(data_dir, join=true)) do f
        m = match(pattern, basename(f))
        if isnothing(m)
            return false
        end
        file_J = parse(Float64, m.captures[1])
        file_g = parse(Float64, m.captures[2])
        file_row = parse(Int, m.captures[3])
        
        matches_filter = file_J == J && file_row == row
        if !isnothing(selected_g)
            return matches_filter && file_g in selected_g
        end
        return matches_filter
    end
    
    if isempty(files)
        @warn "No matching files found in $data_dir"
        return nothing
    end
    
    # Load results
    data = map(files) do file
        result, input_args = load_result(file)
        (g=input_args[:g], energy_history=result.energy_history, converged=result.converged)
    end
    
    # Sort by g
    sort!(data, by=x->x.g)
    
    # Plot
    fig = Figure(size=(1000, 700))
    ax = Axis(fig[1, 1], 
              xlabel="Iteration", 
              ylabel="Energy",
              title="Training History (J=$J, row=$row)")
    
    for d in data
        label = "g=$(d.g)" * (d.converged ? "" : " (NC)")
        lines!(ax, 1:length(d.energy_history), d.energy_history, 
               linewidth=2, label=label)
    end
    
    axislegend(ax, position=:rt)
    
    return fig
end

"""
    extract_correlation_lengths(data_dir::String; J=1.0, g=3.0, nqubits=3, selected_rows=nothing)

Extract correlation lengths (ξ) from multiple result files.

# Arguments
- `data_dir`: Directory containing JSON result files
- `J`: Filter results by J value (default: 1.0)
- `g`: Filter results by g value (default: 3.0)
- `nqubits`: Filter results by nqubits value (default: 3)
- `selected_rows`: Vector of row values to extract (default: extract all)

# Returns
- Tuple of (row_values, ξ_values) sorted by row

# Example
```julia
row, ξ = extract_correlation_lengths("project/results"; J=1.0, g=3.0, nqubits=3)
plot_corr_scale(row, ξ; save_path="project/results/corr_scaling.pdf")
```
"""
function extract_correlation_lengths(data_dir::String; J=1.0, g=3.0, nqubits=3, selected_rows=nothing)
    # Find all matching files
    pattern = r"circuit_J=([\d\.]+)_g=([\d\.]+)_row=(\d+)_nqubits=(\d+)\.json"
    files = filter(readdir(data_dir, join=true)) do f
        m = match(pattern, basename(f))
        if isnothing(m)
            return false
        end
        file_J = parse(Float64, m.captures[1])
        file_g = parse(Float64, m.captures[2])
        file_row = parse(Int, m.captures[3])
        file_nqubits = parse(Int, m.captures[4])
        
        matches_filter = file_J == J && file_g == g && file_nqubits == nqubits
        if !isnothing(selected_rows)
            return matches_filter && file_row in selected_rows
        end
        return matches_filter
    end
    
    if isempty(files)
        @warn "No matching files found in $data_dir"
        return nothing, nothing
    end
    
    # Extract row and ξ from each file
    data = []
    for file in files
        json_data = open(file, "r") do io
            JSON3.read(io, Dict)
        end
        
        # Extract row from input_args
        input_args = get(json_data, "input_args", get(json_data, :input_args, Dict()))
        row = get(input_args, "row", get(input_args, :row, nothing))
        
        # Extract correlation_length if it exists
        ξ = get(json_data, "correlation_length", get(json_data, :correlation_length, nothing))
        
        if !isnothing(row) && !isnothing(ξ)
            push!(data, (row=row, ξ=ξ))
            println("Found: row=$row, ξ=$ξ ($(basename(file)))")
        else
            @warn "Missing correlation_length in $(basename(file)). Run analyze_acf first."
        end
    end
    
    if isempty(data)
        @warn "No files with correlation_length found. Run analyze_acf on the result files first."
        return nothing, nothing
    end
    
    # Sort by row
    sort!(data, by=x->x.row)
    
    row_values = [d.row for d in data]
    ξ_values = [d.ξ for d in data]
    
    return row_values, ξ_values
end

"""
    run_energy_evolution(file1::String, file2::String; n_runs=50, conv_step=100, samples=10000, labels=nothing)

Load parameters from two files, run the quantum channel multiple times for each, and plot energy comparison.

# Arguments
- `file1`: Path to first JSON result file
- `file2`: Path to second JSON result file
- `n_runs`: Number of independent runs per file (default: 50)
- `conv_step`: Convergence steps before sampling (default: 100)
- `samples`: Number of samples per run (default: 10000)
- `labels`: Tuple of (label1, label2) for plot legend (default: use filenames)

# Returns
- `fig`: Figure object with energy comparison plot
- `energies1`: Energy values from file1
- `energies2`: Energy values from file2
"""
function run_energy_evolution(file1::String, file2::String; n_runs=50, conv_step=100, samples=10000, labels=nothing)
    if isnothing(labels)
        labels = (basename(file1), basename(file2))
    end
    
    # Helper function to run circuits and get energies
    function get_energies(filename, label_name)
        result, input_args = load_result(filename)
        
        params = result.final_params
        p = input_args[:p]
        row = input_args[:row]
        nqubits = input_args[:nqubits]
        g = input_args[:g]
        J = input_args[:J]
        share_params = get(input_args, :share_params, true)
        
        println("=== $label_name ===")
        println("File: ", basename(filename))
        println("Configuration: p=$p, row=$row, nqubits=$nqubits, g=$g, J=$J")
        println("Running $n_runs independent circuits...")
        
        gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
        
        energies = zeros(n_runs)
        Threads.@threads for run_idx in 1:n_runs
            _, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                              conv_step=conv_step, 
                                                              samples=samples,
                                                              measure_first=:Z)
            # Z_samples includes burn-in period, X_samples does not (collected in second phase)
            Z_valid = Z_samples[conv_step+1:end]
            X_valid = X_samples  # Already post burn-in
            energies[run_idx] = compute_energy(X_valid, Z_valid, g, J, row)
        end
        
        return energies, g, J, row
    end
    
    # Get energies from both files
    energies1, g, J, row = get_energies(file1, labels[1])
    energies2, _, _, _ = get_energies(file2, labels[2])
    
    # Create plot
    fig = Figure(size=(900, 500))
    ax = Axis(fig[1, 1],
              xlabel="Run Number",
              ylabel="Energy",
              title="Energy Comparison (g=$g, J=$J, row=$row, samples=$samples)")
    
    # Plot energies for each file
    scatter!(ax, 1:n_runs, energies1, markersize=8, color=:blue, label=labels[1])
    scatter!(ax, 1:n_runs, energies2, markersize=8, color=:orange, label=labels[2])
    
    # Add horizontal lines for mean energies
    mean1, std1 = mean(energies1), std(energies1)
    mean2, std2 = mean(energies2), std(energies2)
    
    hlines!(ax, [mean1], color=:blue, linewidth=2, linestyle=:dash,
            label="Mean1: $(round(mean1, digits=4)) ± $(round(std1, digits=4))")
    hlines!(ax, [mean2], color=:orange, linewidth=2, linestyle=:dash,
            label="Mean2: $(round(mean2, digits=4)) ± $(round(std2, digits=4))")
    
    axislegend(ax, position=:rt)
    
    display(fig)
    
    # Save figure
    output_filename = "energy_comparison_g=$(g)_row=$(row).pdf"
    output_path = joinpath(dirname(file1), output_filename)
    save(output_path, fig)
    println("Figure saved to: $output_path")
    
    # Print summary
    println("\n--- Energy Comparison ---")
    println("$(labels[1]): $(round(mean1, digits=4)) ± $(round(std1, digits=4))")
    println("$(labels[2]): $(round(mean2, digits=4)) ± $(round(std2, digits=4))")
    println("Difference: $(round(mean1 - mean2, digits=4))")
    
    return fig, energies1, energies2
end
# Example usage (commented out)
# Analyze a single result
J=1.0;g = 1.0; row=2 ; nqubits=3; p=3; virtual_qubits=1;D=2
data_dir = joinpath(@__DIR__, "results")
datafile = joinpath(data_dir, "circuit_J=1.0_g=$(g)_row=$(row)_nqubits=$(nqubits).json")
referfile = joinpath(data_dir, "pepskit_results_D=$(D).json")
result, args = analyze_result(datafile; pepskit_results_file=referfile)
# Reconstruct gates and analyze

gates, rho, gap, eigenvalues = reconstruct_gates(datafile; use_iterative=false, matrix_free=false)
eigenvalues
tensors = gates_to_tensors(gates, row, virtual_qubits)
A = reshape(tensors[1], 2,4,4)
E = get_physical_channel(gates, row, virtual_qubits, rho)
eigenvalues = eigvals(E)
gap = -log(abs(eigenvalues[end-1]/eigenvalues[end]))


S, σ = mps_physical_entanglement_infinite(A; tol=1e-12)
S, spectrum,_ = multiline_mps_physical_entanglement_infinite(gates, row; nqubits=nqubits)


rho, Z_samples, X_samples=sample_quantum_channel(gates, row, nqubits; conv_step=100, samples=100000, measure_first=:Z)
energy = compute_energy(X_samples[100:end], Z_samples[100:end], g, J, row) |> println
 
# Save the plot
save(joinpath(dirname(datafile), replace(basename(datafile), ".json" => "_eigenvalues.pdf")), fig)
println("Energy: $energy")
# Analyze autocorrelation (using saved samples)
lags, acf, fit_params = analyze_acf(datafile, row; max_lag=100, resample=false, samples=1000000)

data_dir = joinpath(@__DIR__, "results")
datafile1 = joinpath(data_dir, "circuit_J=1.0_g=2.0_row=2_nqubits=3_ones.json")
datafile2 = joinpath(data_dir, "circuit_J=1.0_g=2.0_row=2_nqubits=5.json")

run_energy_evolution(datafile1, datafile2; n_runs=100, conv_step=100, samples=40000)

# Compare two results
# exact_datafile = joinpath(data_dir, "exact_J=1.0_g=2.0_row=3.json")
# result1, result2 = compare_results(datafile, exact_datafile)

# Visualize correlation
visualize_correlation(datafile)



# Analyze autocorrelation with fresh resampled data
# lags, acf, ξ = analyze_acf(datafile; max_lag=10, resample=true, samples=50000)

# Or resample circuit separately to get the raw samples
# rho, Z_samples, X_samples, params, gates = resample_circuit(datafile; conv_step=1000, samples=50000)

# Plot energy landscape
fig = plot_energy_vs_g(data_dir; J=1.0, row=3)
display(fig)

# Plot training history comparison
fig = plot_training_history_comparison(data_dir; J=1.0, row=3, selected_g=[1.0, 2.0, 3.0])
display(fig)



# ============================================================================
# Batch process ACF and correlation length scaling
# ============================================================================

# Step 1: Compute and save ξ for all result files (run once for each file)
# J_val = 1.0; g_val = 3.0; nqubits_val = 3
# for row_val in [2, 4, 6, 7, 8, 10, 12]
#     datafile = joinpath(data_dir, "circuit_J=$(J_val)_g=$(g_val)_row=$(row_val)_nqubits=$(nqubits_val).json")
#     if isfile(datafile)
#         println("\n=== Processing J=$J_val, g=$g_val, row=$row_val, nqubits=$nqubits_val ===")
#         lags, acf, ξ = analyze_acf(datafile; max_lag=6, resample=true, samples=1000000)
#     end
# end

# Step 2: Extract correlation lengths from all result files and plot scaling
row, ξ = extract_correlation_lengths(data_dir; J=1.0, g=3.0, nqubits=5)
if !isnothing(row) && !isnothing(ξ)
    fig = plot_corr_scale(row, ξ;
                          title="Correlation Length Scaling, g=3.0, p=4, virtual_bond=4",
                          save_path="project/results/corr_scaling_g3.pdf")
    display(fig)
end