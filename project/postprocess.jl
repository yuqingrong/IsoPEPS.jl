using IsoPEPS
using CairoMakie
using Random
using LinearAlgebra
using JSON3

"""
    analyze_result(filename::String)

Analyze a saved training result from JSON file.
"""
function analyze_result(filename::String)
    result, input_args = load_result(filename)
    
    println("=== Training Result Analysis ===")
    println("Type: ", typeof(result))
    println("Final energy: ", result.final_cost)
    
    if haskey(input_args, :g)
        println("\nModel parameters:")
        println("  g = ", input_args[:g])
        println("  J = ", get(input_args, :J, "N/A"))
        println("  row = ", get(input_args, :row, "N/A"))
        println("  p = ", get(input_args, :p, "N/A"))
        println("  nqubits = ", get(input_args, :nqubits, "N/A"))
    end
    
    # Plot training history
    fig = plot_training_history(result; title="Training History")
    display(fig)
    
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
        # For vector: subsample directly
        lags, acf, acf_err = compute_acf(sample_data[100:row:end]; max_lag=max_lag, n_bootstrap=50)
    end
    @show acf, acf_err
    fit_params = fit_acf(lags, acf)
    
    println("\n--- Correlation Length Comparison ---")
    println("Transfer matrix gap:        gap = $(round(gap, digits=4))")
    println("Transfer matrix ξ:          1/gap = $(round(ξ_transfer, digits=4))")
    println("ACF correlation length:     ξ_ACF = $(round(fit_params.ξ, digits=4))")
    println("Ratio (ξ_ACF / ξ_transfer): $(round(fit_params.ξ / ξ_transfer, digits=4))")
    println("Second eigenvalue |λ₂|:     $(round(eigenvalues[2], digits=6))")
    println("Fitted |λ₂|:                $(round(fit_params.λ₂, digits=6))")
    println("Oscillation frequency k:    $(round(fit_params.k, digits=4))")
    println("Phase φ:                    $(round(fit_params.φ, digits=4))")
    println("Amplitude A:                $(round(fit_params.A, digits=4))")
    
    title_text = "ACF $(basename(filename))\nξ=$(round(fit_params.ξ, digits=2)), k=$(round(fit_params.k, digits=3)), 1/gap=$(round(ξ_transfer, digits=2))"
    if resample
        title_text *= " [resampled]"
    end
    
    fig = plot_acf(lags, acf; 
                    acf_err=acf_err,
                    fit_params=fit_params,
                    logscale=false,
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

# Example usage (commented out)
# Analyze a single result
J=1.0;g = 3.0; row=2 ; nqubits=3; p=4
data_dir = joinpath(@__DIR__, "results")
datafile = joinpath(data_dir, "circuit_J=1.0_g=$(g)_row=$(row)_nqubits=$(nqubits).json")
result, args = analyze_result(datafile)
# Reconstruct gates and analyze
gates, rho, gap, eigenvalues = reconstruct_gates(datafile; use_iterative=false, matrix_free=false)
@show gates[1]
rho, Z_samples, X_samples=sample_quantum_channel(gates, row, nqubits; conv_step=100, samples=100000, measure_first=:Z)
compute_energy(X_samples[100:end], Z_samples[100:end], g, J, row) |> println

# Save the plot
save(joinpath(dirname(datafile), replace(basename(datafile), ".json" => "_eigenvalues.pdf")), fig)

# Analyze autocorrelation (using saved samples)
lags, acf, fit_params = analyze_acf(datafile, row; max_lag=9, resample=false, samples=1000000)


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