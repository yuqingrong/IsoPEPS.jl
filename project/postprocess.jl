using IsoPEPS
using CairoMakie
using Random
using LinearAlgebra

"""
    analyze_result(filename::String)

Analyze a saved training result from JSON file.
"""
function analyze_result(filename::String)
    result, input_args = load_result(filename)
    
    println("=== Training Result Analysis ===")
    println("Type: ", typeof(result))
    println("Final energy: ", result.energy)
    println("Converged: ", result.converged)
    println("Iterations: ", length(result.energy_history))
    
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
    reconstruct_gates(filename::String; share_params=true)

Reconstruct gates from optimization result stored in JSON file and analyze transfer spectrum.

# Arguments
- `filename`: Path to JSON result file
- `share_params`: Share parameters across circuit layers (default: true)

# Returns
- Tuple of (gates, rho, gap, eigenvalues)
"""
function reconstruct_gates(filename::String; share_params=true)
    result, input_args = load_result(filename)
    
    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    
    gates = build_unitary_gate(result.params, p, row, nqubits; share_params=share_params)
    
    # Compute transfer spectrum
    rho, gap, eigenvalues = compute_transfer_spectrum(gates, row, nqubits)
    
    println("=== Gate Analysis for $(basename(filename)) ===")
    println("Spectral gap: ", gap)
    println("Largest eigenvalue: ", maximum(abs.(eigenvalues)))
    println("Second largest eigenvalue: ", sort(abs.(eigenvalues))[end-1])
    
    return gates, rho, gap, eigenvalues
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
    
    if isempty(result.Z_samples)
        @warn "No Z samples in result"
        return nothing
    end
    
    row = input_args[:row]
    fig = plot_correlation_heatmap(result.Z_samples, row; 
                                    title="Spin-Spin Correlation ($(basename(filename)))")
    display(fig)
    
    return fig
end

"""
    analyze_acf(filename::String; max_lag=100, basis=:Z)

Analyze autocorrelation function of samples from result stored in JSON file.

# Arguments
- `filename`: Path to JSON result file
- `max_lag`: Maximum lag for ACF computation (default: 100)
- `basis`: Which samples to use, :Z or :X (default: :Z)

# Returns
- Tuple of (lags, acf, ξ) where ξ is the correlation length
"""
function analyze_acf(filename::String; max_lag=100, basis=:Z)
    result, input_args = load_result(filename)
    
    if !(result isa CircuitOptimizationResult)
        @warn "Result is not CircuitOptimizationResult, samples not available"
        return nothing
    end
    
    samples = basis == :Z ? result.Z_samples : result.X_samples
    
    if isempty(samples)
        @warn "No $(basis) samples in result"
        return nothing
    end
    
    lags, acf, acf_err = compute_acf(samples; max_lag=max_lag, n_bootstrap=50)
    
    A, ξ = fit_acf_exponential(lags, acf)
    println("Autocorrelation length ($(basis)-basis): ξ = ", ξ)
    
    fig = plot_acf(lags, acf; 
                    acf_err=acf_err,
                    fit_params=(A, ξ),
                    title="ACF $(basename(filename)) (ξ = $(round(ξ, digits=2)))")
    display(fig)
    
    return lags, acf, ξ
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
        (g=input_args[:g], energy=result.energy, converged=result.converged)
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

# Example usage (commented out)
# Analyze a single result
g = 3.0
data_dir = joinpath(@__DIR__, "results")
datafile = joinpath(data_dir, "circuit_J=1.0_g=$(g)_row=3.json")
result, args = analyze_result(datafile)

# Compare two results
# exact_datafile = joinpath(data_dir, "exact_J=1.0_g=2.0_row=3.json")
# result1, result2 = compare_results(datafile, exact_datafile)

# Reconstruct gates and analyze
gates, rho, gap, eigenvalues = reconstruct_gates(datafile)

# Visualize correlation
visualize_correlation(datafile)

# Analyze autocorrelation
lags, acf, ξ = analyze_acf(datafile; max_lag=100)

# Plot energy landscape
fig = plot_energy_vs_g(data_dir; J=1.0, row=3)
display(fig)

# Plot training history comparison
fig = plot_training_history_comparison(data_dir; J=1.0, row=3, selected_g=[1.0, 2.0, 3.0])
display(fig)


