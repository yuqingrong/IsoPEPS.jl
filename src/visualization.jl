# ============================================================================
# Data I/O with JSON
# ============================================================================

"""
    save_result(filename::String, result, input_args::Dict)

Save optimization result to JSON file with input arguments.

# Arguments
- `filename`: Path to save file
- `result`: Optimization result (CircuitOptimizationResult, ExactOptimizationResult, or ManifoldOptimizationResult)
- `input_args`: Dictionary of input arguments/metadata

# Example
```julia
result = optimize_circuit(...)
input_args = Dict(
    :g => 2.0, :J => 1.0, :row => 3,
    :initial_params => params,
    :maxiter => 5000
)
save_result("data/result.json", result, input_args)
```
"""
function save_result(filename::String, result::CircuitOptimizationResult, input_args::Dict)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    
    # Convert matrices to nested arrays (array of arrays) for proper JSON serialization
    # Each row (chain) becomes a separate array
    Z_nested = [result.final_Z_samples[i, :] for i in 1:size(result.final_Z_samples, 1)]
    X_nested = [result.final_X_samples[i, :] for i in 1:size(result.final_X_samples, 1)]
    
    data = Dict{Symbol, Any}(
        :type => "CircuitOptimizationResult",
        :energy_history => result.energy_history,
        :params => result.final_params,
        :energy => result.final_cost,
        :converged => result.converged,
        :Z_samples => Z_nested,
        :X_samples => X_nested,
        :sample_shape => size(result.final_Z_samples),  # Store original shape for reference
        :input_args => input_args
    )
    
    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    @info "Result saved to $filename"
end

function save_result(filename::String, result::ExactOptimizationResult, input_args::Dict)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    
    data = Dict{Symbol, Any}(
        :type => "ExactOptimizationResult",
        :energy_history => result.energy_history,
        :params => result.params,
        :energy => result.energy,
        :gap => result.gap,
        :eigenvalues => result.eigenvalues,
        :X_expectation => result.X_expectation,
        :ZZ_vertical => result.ZZ_vertical,
        :ZZ_horizontal => result.ZZ_horizontal,
        :converged => result.converged,
        :input_args => input_args
    )
    
    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    @info "Result saved to $filename"
end

function save_result(filename::String, result::ManifoldOptimizationResult, input_args::Dict)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    
    data = Dict{Symbol, Any}(
        :type => "ManifoldOptimizationResult",
        :energy_history => result.energy_history,
        :gate => result.gate,
        :energy => result.energy,
        :gap_history => result.gap_history,
        :converged => result.converged,
        :input_args => input_args
    )
    
    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    @info "Result saved to $filename"
end

"""
    load_result(filename::String; result_type=:auto)

Load optimization result from JSON file.

# Arguments
- `filename`: Path to JSON file
- `result_type`: Result type to load (`:auto`, `:circuit`, `:exact`, or `:manifold`)

# Returns
Tuple of (result, input_args_dict)
"""
function load_result(filename::String; result_type::Symbol=:auto)
    data = open(filename, "r") do io
        JSON3.read(io, Dict)
    end
    
    # Determine type
    if result_type == :auto
        result_type_str = get(data, "type", get(data, :type, ""))
        result_type = Symbol(lowercase(replace(result_type_str, "OptimizationResult" => "")))
    end
    
    # Extract input_args and convert string keys to symbols
    input_args_raw = get(data, "input_args", get(data, :input_args, Dict()))
    input_args = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(input_args_raw))
    
    # Helper function to get data with both string and symbol key fallback
    function get_data(dict, key)
        get(dict, string(key), get(dict, Symbol(key), nothing))
    end
    
    # Helper to convert samples to matrix
    function samples_to_matrix(samples_data, input_args)
        if samples_data === nothing || isempty(samples_data)
            return Matrix{Float64}(undef, 0, 0)
        end
        
        # Check if data is nested array (new format) or flat vector (old format)
        if samples_data isa Vector && !isempty(samples_data) && samples_data[1] isa Vector
            # New format: nested array (array of arrays)
            # Each element is a row (chain) in the matrix
            return reduce(vcat, [Vector{Float64}(row)' for row in samples_data])
        else
            # Old format: flat vector - need to reshape
            samples_vec = Vector{Float64}(samples_data)
            n_parallel_runs = get(input_args, :n_parallel_runs, 1)
            
            # Infer actual samples per run from the data size
            actual_samples_per_run = div(length(samples_vec), n_parallel_runs)
            
            # Reshape: flat vector -> matrix (n_parallel_runs × actual_samples_per_run)
            # The samples are stored in row-major order
            return reshape(samples_vec, actual_samples_per_run, n_parallel_runs)'
        end
    end
    
    # Reconstruct result based on type
    if result_type == :circuit
        Z_samples_data = get(data, "Z_samples", get(data, :Z_samples, nothing))
        X_samples_data = get(data, "X_samples", get(data, :X_samples, nothing))
        
        energy_history = get_data(data, :energy_history)
        params = get_data(data, :params)
        energy = get_data(data, :energy)
        converged = get_data(data, :converged)
        
        # Convert samples to vectors (flatten if matrix)
        Z_samples = samples_to_matrix(Z_samples_data, input_args)
        X_samples = samples_to_matrix(X_samples_data, input_args)
        Z_samples_vec = Z_samples isa AbstractMatrix ? vec(collect(Z_samples)) : Vector{Float64}(collect(Z_samples))
        X_samples_vec = X_samples isa AbstractMatrix ? vec(collect(X_samples)) : Vector{Float64}(collect(X_samples))
        
        result = CircuitOptimizationResult(
            Vector{Float64}(energy_history === nothing ? Float64[] : energy_history),
            Vector{Matrix{ComplexF64}}[],  # Gates not saved to JSON
            Vector{Float64}(params === nothing ? Float64[] : params),
            Float64(energy === nothing ? 0.0 : energy),
            Z_samples_vec,
            X_samples_vec,
            Bool(converged === nothing ? false : converged)
        )
    elseif result_type == :exact
        result = ExactOptimizationResult(
            Vector{Float64}(get_data(data, :energy_history)),
            Vector{Matrix{ComplexF64}}[],  # Gates not saved to JSON
            Vector{Float64}(get_data(data, :params)),
            Float64(get_data(data, :energy)),
            Float64(get_data(data, :gap)),
            Vector{Float64}(get_data(data, :eigenvalues)),
            Float64(get_data(data, :X_expectation)),
            Float64(get_data(data, :ZZ_vertical)),
            Float64(get_data(data, :ZZ_horizontal)),
            Bool(get_data(data, :converged))
        )
    elseif result_type == :manifold
        result = ManifoldOptimizationResult(
            Vector{Float64}(get_data(data, :energy_history)),
            Matrix{ComplexF64}(get_data(data, :gate)),
            Float64(get_data(data, :energy)),
            Vector{Float64}(get_data(data, :gap_history)),
            Bool(get_data(data, :converged))
        )
    else
        error("Unknown result type: $result_type")
    end
    
    return result, input_args
end

"""
    save_results(filename::String; kwargs...)

Save arbitrary results to a JSON file. Accepts keyword arguments for flexibility.

# Example
```julia
save_results("results.json"; 
    g=2.0, row=3, 
    energy_history=[1.0, 0.5, 0.3],
    correlation_matrix=corr_mat
)
```
"""
function save_results(filename::String; kwargs...)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    
    open(filename, "w") do io
        JSON3.pretty(io, Dict(kwargs))
    end
    @info "Results saved to $filename"
end

"""
    load_results(filename::String) -> Dict

Load results from a JSON file.
"""
function load_results(filename::String)
    open(filename, "r") do io
        JSON3.read(io, Dict)
    end
end

# ============================================================================
# Visualization Functions
# ============================================================================
"""
    reconstruct_gates(filename::String; share_params=true, plot=true, save_plot=false)

Reconstruct gates from optimization result stored in JSON file and analyze transfer spectrum.

# Arguments
- `filename`: Path to JSON result file
- `share_params`: Share parameters across circuit layers (default: true)
- `plot`: Display eigenvalue spectrum plot (default: true)
- `save_plot`: Save plot to PDF file (default: false)

# Returns
- Tuple of (gates, rho, gap, eigenvalues)

# Example
```julia
# With visualization (default)
gates, rho, gap, eigenvalues = reconstruct_gates("result.json")

# Without visualization
gates, rho, gap, eigenvalues = reconstruct_gates("result.json"; plot=false)

# Save the plot
gates, rho, gap, eigenvalues = reconstruct_gates("result.json"; save_plot=true)
```
"""
function reconstruct_gates(filename::String; share_params=true, plot=true, save_plot=true, use_iterative=:auto, matrix_free=:auto)
    result, input_args = load_result(filename)
    
    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    
    gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)
    
    # Compute transfer spectrum
    rho, gap, eigenvalues, eigenvalues_raw = compute_transfer_spectrum(gates, row, nqubits; use_iterative=use_iterative, matrix_free=matrix_free)
    
    println("=== Gate Analysis for $(basename(filename)) ===")
    println("Spectral gap: ", gap)
    println("Largest eigenvalue: ", maximum(abs.(eigenvalues)))
    println("Second largest eigenvalue: ", eigenvalues[2])
    println("Correlation length ξ: ", round(1/gap, digits=2))
    
    # Count eigenvalues near 1
    n_near_one = sum(eigenvalues .> 0.99)
    println("Eigenvalues > 0.99: $n_near_one / $(length(eigenvalues))")
    
    # Status indicator
    if gap > 0.1
        println("Status: ✓ Good spectral gap")
    elseif gap > 0.01
        println("Status: ⚠ Poor spectral gap")
    else
        println("Status: ✗ Very small spectral gap - optimization issue likely")
    end
    
    # Plot eigenvalue spectrum if requested
    if plot
        save_path = save_plot ? replace(filename, ".json" => "_eigenvalues.pdf") : nothing
        fig = plot_eigenvalue_spectrum(eigenvalues_raw; 
                                        title=basename(filename), 
                                        save_path=save_path,
                                        show_gap=true)
        display(fig)
        
        if save_plot
            println("Plot saved to: $save_path")
        end
    end
    
    return gates, rho, gap, eigenvalues
end

"""
    plot_eigenvalue_spectrum(eigenvalues_raw; title="", save_path=nothing, show_gap=true)

Visualize the transfer matrix eigenvalue spectrum.

# Arguments
- `eigenvalues_raw`: Vector of raw complex eigenvalues (sorted by magnitude, descending)
- `title`: Plot title (default: "")
- `save_path`: Path to save figure (default: nothing, no save)
- `show_gap`: Annotate spectral gap on plot (default: true)

# Returns
- `Figure` object

# Example
```julia
gates, rho, gap, eigenvalues = reconstruct_gates("result.json")
fig = plot_eigenvalue_spectrum(eigenvalues; title="My Circuit")
```
"""
function plot_eigenvalue_spectrum(eigenvalues_raw::AbstractVector{<:Complex}; 
                                   title::String="",
                                   save_path::Union{String,Nothing}=nothing,
                                   show_gap::Bool=true)
    # Sort by magnitude (descending) and compute magnitudes
    sorted_indices = sortperm(abs.(eigenvalues_raw), rev=true)
    sorted_raw = eigenvalues_raw[sorted_indices]
    sorted_eigs = abs.(sorted_raw)
    n = length(sorted_eigs)
    
    # Compute gap
    λ₁ = sorted_eigs[1]
    λ₂ = sorted_eigs[2]
    gap = -log(λ₂)
    ξ = 1 / gap
    
    # Create figure with two panels
    fig = Figure(size=(1200, 500))
    
    # Left panel: Bar plot of all eigenvalues (sorted descending)
    ax1 = Axis(fig[1, 1], 
               xlabel="Eigenvalue index (sorted)", 
               ylabel="Eigenvalue magnitude |λ|",
               title=isempty(title) ? "Eigenvalue Spectrum" : title)
    
    # Color by distance from 1
    colors = [λ > 0.99 ? :red : (λ > 0.9 ? :orange : :steelblue) for λ in sorted_eigs]
    
    barplot!(ax1, 1:n, sorted_eigs, color=colors, strokewidth=0.5, strokecolor=:black)
    
    # Reference line at 1.0
    hlines!(ax1, [1.0], color=:black, linestyle=:dash, linewidth=1.5, label="λ=1")
    hlines!(ax1, [0.99], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="λ=0.99")
    
    # Highlight λ₁ and λ₂
    scatter!(ax1, [1], [λ₁], markersize=15, color=:green, marker=:star5, label="λ₁=$(round(λ₁, digits=4))")
    scatter!(ax1, [2], [λ₂], markersize=12, color=:purple, marker=:diamond, label="λ₂=$(round(λ₂, digits=4))")
    
    axislegend(ax1, position=:rb)
    
    # Right panel: Eigenvalues in the complex plane (unit disc)
    ax2 = Axis(fig[1, 2],
               xlabel="Re(λ)",
               ylabel="Im(λ)",
               title="Eigenvalues in Complex Plane",
               aspect=DataAspect())
    
    # Draw unit circle
    θ = range(0, 2π, length=100)
    lines!(ax2, cos.(θ), sin.(θ), color=:black, linestyle=:dash, linewidth=1.5, label="Unit circle")
    
    # Draw |λ|=0.99 circle
    lines!(ax2, 0.99 .* cos.(θ), 0.99 .* sin.(θ), color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="|λ|=0.99")
    
    # Plot all eigenvalues
    re_parts = real.(sorted_raw)
    im_parts = imag.(sorted_raw)
    colors_scatter = [abs(λ) > 0.99 ? :red : (abs(λ) > 0.9 ? :orange : :steelblue) for λ in sorted_raw]
    scatter!(ax2, re_parts, im_parts, color=colors_scatter, markersize=8, strokewidth=0.5, strokecolor=:black)
    
    # Highlight λ₁ and λ₂
    scatter!(ax2, [real(sorted_raw[1])], [imag(sorted_raw[1])], markersize=15, color=:green, marker=:star5, label="λ₁")
    scatter!(ax2, [real(sorted_raw[2])], [imag(sorted_raw[2])], markersize=12, color=:purple, marker=:diamond, label="λ₂")
    
    axislegend(ax2, position=:lt)
    
    # Add text annotation for gap and ξ
    if show_gap
        # Color code the status
        status_color = gap > 0.1 ? :green : (gap > 0.01 ? :orange : :red)
        status_text = gap > 0.1 ? "✓ Good" : (gap > 0.01 ? "⚠ Poor" : "✗ Bad")
        
        Label(fig[0, 1:2], 
              "Spectral Gap: $(round(gap, digits=6))  |  ξ = $(round(ξ, digits=2))  |  $status_text",
              fontsize=16, font=:bold, color=status_color, halign=:center)
    end
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end


"""
    fit_acf(lags::AbstractVector, acf::AbstractVector; fit_range=nothing, include_zero=false)

Fit ACF data using exponential decay model: |C(r)| = A·exp(-r/ξ)

# Arguments
- `lags`: Lag values
- `acf`: ACF values (absolute values will be used for fitting)
- `fit_range`: (start_lag, end_lag) range for fitting (default: use all data)
- `include_zero`: Whether to include lag=0 in fitting (default: false)

# Returns
- `NamedTuple` with fields:
  - `A`: Amplitude
  - `ξ`: Correlation length (decay constant)
  - `λ₂`: Effective second eigenvalue exp(-1/ξ)
  - `fit_lags`: Lags used for fitting
  - `model`: Fitted model function
"""
function fit_acf(lags::AbstractVector, acf::AbstractVector; 
                 fit_range::Union{Tuple{Int,Int},Nothing}=nothing,
                 include_zero::Bool=false)
    lags_vec = collect(Float64, lags)
    acf_vec = collect(Float64, acf)
    
    # Optionally exclude first value (lag=0) from fitting
    if include_zero
        mask = trues(length(lags_vec))
    else
        mask = lags_vec .> 0
    end
    
    # Apply fit range if specified
    if !isnothing(fit_range)
        start_lag, end_lag = fit_range
        range_mask = (lags_vec .>= start_lag) .& (lags_vec .<= end_lag)
        mask = mask .& range_mask
    end
    
    fit_lags = lags_vec[mask]
    fit_acf_vals = abs.(acf_vec[mask])  # Fit to |C(r)|
    
    if length(fit_lags) < 2
        error("Not enough valid data points for fitting (need at least 2)")
    end
    
    # Define the exponential decay model: |C(r)| = A·exp(-r/ξ)
    model(r, p) = p[1] .* exp.(-r ./ p[2])
    
    # Initial parameter guesses: [A, ξ]
    A_init = fit_acf_vals[1]
    
    # Estimate decay constant from log-linear fit
    valid_idx = fit_acf_vals .> 1e-15
    if sum(valid_idx) >= 2
        log_acf = log.(fit_acf_vals[valid_idx])
        valid_lags = fit_lags[valid_idx]
        # Linear fit: log|C| ≈ log(A) - r/ξ
        slope = (log_acf[end] - log_acf[1]) / (valid_lags[end] - valid_lags[1])
        ξ_init = -1.0 / slope
        ξ_init = clamp(ξ_init, 0.1, length(fit_lags) * 10.0)
    else
        ξ_init = length(fit_lags) / 2.0
    end
    
    p0 = [A_init, ξ_init]
    
    # Set parameter bounds: A > 0, ξ > 0
    lower = [0.0, 0.01]
    upper = [Inf, Inf]
    
    try
        # Nonlinear least squares fit
        fit_result = curve_fit(model, fit_lags, fit_acf_vals, p0, lower=lower, upper=upper)
        A, ξ = coef(fit_result)
        
        # Compute effective second eigenvalue
        λ₂ = exp(-1.0/ξ)
        
        return (A=A, ξ=ξ, λ₂=λ₂, fit_lags=fit_lags, 
                model=(r) -> model(r, [A, ξ]))
    catch e
        @warn "Nonlinear fit failed: $e. Using initial guess."
        A, ξ = p0
        λ₂ = exp(-1.0/ξ)
        return (A=A, ξ=ξ, λ₂=λ₂, fit_lags=fit_lags,
                model=(r) -> model(r, [A, ξ]))
    end
end

"""
    plot_acf(lags::AbstractVector, acf::AbstractVector; kwargs...)

Plot autocorrelation function with optional exponential fit.

# Arguments
- `lags`: Lag values
- `acf`: ACF values
- `acf_err`: Error bars (optional)
- `fit_range`: (start_lag, end_lag) for automatic fitting (optional)
- `fit_params`: Pre-computed fit params from `fit_acf` (optional, overrides fit_range)
- `title`: Plot title
- `logscale`: Use log scale for y-axis (default: true)
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object
"""
function plot_acf(lags::AbstractVector, acf::AbstractVector;
                  acf_err::Union{AbstractVector,Nothing}=nothing,
                  fit_range::Union{Tuple{Int,Int},Nothing}=nothing,
                  fit_params::Union{NamedTuple,Nothing}=nothing,
                  title::String="Autocorrelation Function",
                  logscale::Bool=true,
                  save_path::Union{String,Nothing}=nothing)
    
    fig = Figure(size=(500, 350))
    
    # Prepare data - use absolute values for log scale
    abs_acf = abs.(collect(acf))
    lags_vec = collect(lags)
    
    if logscale
        # Set minimum threshold to avoid log10(0)
        min_threshold = max(maximum(abs_acf) * 1e-10, 1e-15)
        abs_acf = max.(abs_acf, min_threshold)
        plot_y = abs_acf
        ylabel = "|C(lag)|"
    else
        plot_y = collect(acf)
        ylabel = "C(lag)"
    end
    
    ax = Axis(fig[1, 1],
              xlabel="Lag", ylabel=ylabel,
              title=title,
              yscale=logscale ? log10 : identity)
    
    # Plot data with error bars (only for linear scale)
    if !isnothing(acf_err) && !logscale
        errorbars!(ax, lags_vec, plot_y, collect(acf_err), color=:gray)
    end
    scatter!(ax, lags_vec, plot_y, markersize=8, label="Data")
    
    # Compute fit if fit_range provided but no fit_params
    if isnothing(fit_params) && !isnothing(fit_range)
        fit_params = fit_acf(lags, acf; fit_range=fit_range)
    end
    
    # Plot fit: |C(r)| = A·exp(-r/ξ)
    if !isnothing(fit_params)
        A = fit_params.A
        ξ = fit_params.ξ
        λ₂ = fit_params.λ₂
        
        # Evaluate fit curve
        fit_curve = A .* exp.(-lags_vec ./ ξ)
        
        if logscale
            # Clamp to min threshold to avoid log10(0)
            fit_curve = max.(fit_curve, min_threshold)
        end
        
        # Format label with key parameters
        label_text = "Fit: ξ=$(round(ξ, digits=2)), λ₂=$(round(λ₂, digits=4))"
        
        lines!(ax, lags_vec, fit_curve, linewidth=2, linestyle=:dash, color=:red,
               label=label_text)
    end
    
    axislegend(ax, position=:rt)
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    compute_acf(data; max_lag::Int=100, n_bootstrap::Int=100, normalize::Bool=true)

Compute autocorrelation function with error estimates.

Accepts either:
- `Matrix{Float64}`: Each row is an independent chain (preferred for parallel sampling)
- `Vector{Float64}`: Single time series (uses bootstrap for errors)

When given a matrix (multiple chains), uses ALL pairs from ALL chains to compute ACF.
For example, with 2 chains of length 5, lag=1 uses pairs:
  chain1: (1,2), (2,3), (3,4), (4,5)
  chain2: (1,2), (2,3), (3,4), (4,5)
Total: 8 pairs for better statistics.

Error bars from standard error across chains (each chain contributes one ACF estimate).

# Arguments
- `data`: Time series data (Vector or Matrix)
- `max_lag`: Maximum lag to compute (default: 100)
- `n_bootstrap`: Number of bootstrap samples for error estimation (only used for Vector input)
- `normalize`: Whether to normalize by variance (default: true)

# Returns
- `lags`: Lag values (0 to max_lag-1)
- `acf`: Autocorrelation at each lag (using all pairs from all chains)
- `acf_err`: Standard error at each lag (from variance across chains)
"""
function compute_acf(data::Matrix{Float64}; max_lag::Int=100, n_bootstrap::Int=100, normalize::Bool=true)
    n_chains, n_samples = size(data)
    max_lag = min(max_lag, div(n_samples, 2))
    
    # Compute global mean and variance using all data
    μ_global = mean(data)
    centered_global = data .- μ_global
    var_global = mean(centered_global.^2)
    
    acf = zeros(max_lag)
    acf_per_chain = zeros(n_chains, max_lag)
    
    # Compute ACF using all pairs from all chains
    for k in 1:max_lag
        lag = k - 1
        
        # Collect all pairs from all chains
        all_products = Float64[]
        
        for i in 1:n_chains
            chain_centered = centered_global[i, :]
            for j in 1:(n_samples - lag)
                push!(all_products, chain_centered[j] * chain_centered[j + lag])
            end
        end
        
        # Mean over all pairs
        raw_acf = mean(all_products)
        acf[k] = normalize ? raw_acf / var_global : raw_acf
        
        # Also compute ACF per chain for error estimation
        for i in 1:n_chains
            chain_centered = centered_global[i, :]
            n_pairs = n_samples - lag
            raw_acf_chain = mean(chain_centered[j] * chain_centered[j + lag] for j in 1:n_pairs)
            acf_per_chain[i, k] = normalize ? raw_acf_chain / var_global : raw_acf_chain
        end
    end
    
    # Standard error from variance across chains
    acf_err = vec(std(acf_per_chain, dims=1) / sqrt(n_chains))
    
    return 0:(max_lag-1), acf, acf_err
end

"""
    plot_training_history(steps, values; kwargs...)

Plot training loss/energy/observable vs training steps.

# Arguments
- `steps`: Step indices or iteration numbers
- `values`: Values at each step (energy, loss, observable, etc.)
- `reference`: Reference value (e.g., exact energy) shown as horizontal line (optional)
- `ylabel`: Y-axis label (default: "Energy")
- `title`: Plot title
- `logscale`: Use log scale (default: false)
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object
"""
function plot_training_history(steps::AbstractVector, values::AbstractVector;
                                reference::Union{Real,Nothing}=nothing,
                                ylabel::String="Energy",
                                title::String="Training History",
                                logscale::Bool=false,
                                save_path::Union{String,Nothing}=nothing)
    
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1],
              xlabel="Step", ylabel=ylabel,
              title=title,
              yscale=logscale ? log10 : identity)
    
    lines!(ax, collect(steps), collect(values), linewidth=2, label=ylabel)
    
    if !isnothing(reference)
        hlines!(ax, [reference], linestyle=:dash, color=:red, linewidth=1.5, label="Reference")
        axislegend(ax, position=:rt)
    end
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    plot_training_history(result; kwargs...)

Plot training history from optimization result.
"""
function plot_training_history(result::Union{CircuitOptimizationResult, ExactOptimizationResult, ManifoldOptimizationResult}; kwargs...)
    n = length(result.energy_history)
    plot_training_history(1:n, result.energy_history; ylabel="Energy", kwargs...)
end

"""
    plot_variance_vs_samples(sample_sizes, variances; kwargs...)

Plot energy variance vs number of samples.

# Arguments
- `sample_sizes`: Sample sizes
- `variances`: Variance values
- `fit_scaling`: Show 1/N scaling fit (default: true)
- `title`: Plot title
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object
"""
function plot_variance_vs_samples(sample_sizes::AbstractVector, variances::AbstractVector;
                                   errors::Union{AbstractVector,Nothing}=nothing,
                                   fit_scaling::Bool=true,
                                   title::String="Variance vs Samples",
                                   save_path::Union{String,Nothing}=nothing)
    
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1],
              xlabel="Number of Samples", ylabel="Variance",
              title=title,
              xscale=log10, yscale=log10)
    
    # Plot data
    if !isnothing(errors)
        errorbars!(ax, collect(sample_sizes), collect(variances), collect(errors), color=:gray)
    end
    scatter!(ax, collect(sample_sizes), collect(variances), markersize=10, label="Data")
    
    # Fit and plot 1/N scaling
    if fit_scaling && length(sample_sizes) > 1
        # Fit: var = a/N → log(var) = log(a) - log(N)
        log_N = log.(sample_sizes)
        log_var = log.(variances)
        # Simple linear regression: log_var = c - log_N
        c = mean(log_var .+ log_N)
        a = exp(c)
        
        N_range = range(minimum(sample_sizes), maximum(sample_sizes), length=50)
        fit_line = a ./ N_range
        lines!(ax, collect(N_range), fit_line, linewidth=2, linestyle=:dash, color=:red,
               label="1/N scaling (a=$(round(a, digits=2)))")
    end
    
    axislegend(ax, position=:rt)
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    plot_corr_scale(row, corr_length; kwargs...)

Plot inverse correlation length vs inverse row to analyze finite-size scaling.

# Arguments
- `row`: Array of system sizes (row values)
- `corr_length`: Array of correlation lengths (ξ) corresponding to each row
- `title`: Plot title (default: "Correlation Length Scaling")
- `fit_line`: Show linear fit (default: true)
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object

# Example
```julia
row = [3, 4, 5, 6, 7, 8, 10]
corr_length = [2.5, 3.2, 4.1, 5.3, 6.8, 8.5, 11.2]
fig = plot_corr_scale(row, corr_length; save_path="corr_scaling.png")
```
"""
function plot_corr_scale(row::AbstractVector, corr_length::AbstractVector;
    title::String="Correlation Length Scaling",
    save_path::Union{String,Nothing}=nothing)

# Check input validity
if length(row) != length(corr_length)
error("row and corr_length must have the same length")
end

if any(x -> x <= 0, row) || any(x -> x <= 0, corr_length)
error("All values must be positive for inverse scaling plot")
end

# Compute inverse quantities
inv_row = 1 ./ row
inv_corr_length = 1 ./ corr_length

# Create figure
fig = Figure(size=(500, 350))
ax = Axis(fig[1, 1],
xlabel="1/row", ylabel="1/ξ",
title=title)

# Plot data points with connecting lines
lines!(ax, collect(inv_row), collect(inv_corr_length), 
linewidth=2, color=:blue, label="Data")
scatter!(ax, collect(inv_row), collect(inv_corr_length), 
markersize=12, color=:blue)

axislegend(ax, position=:rb)

if !isnothing(save_path)
save(save_path, fig)
@info "Figure saved to $save_path"
end

return fig
end

"""
    plot_diagnosis(diag::NamedTuple; title::String="", save_path::Union{String,Nothing}=nothing)

Visualize the transfer channel diagnosis results.

# Arguments
- `diag`: Named tuple returned by `diagnose_transfer_channel`
- `title`: Plot title (default: "")
- `save_path`: Path to save figure (default: nothing, no save)

# Returns
- `Figure` object

# Example
```julia
diag = diagnose_transfer_channel(gates, row, virtual_qubits)
fig = plot_diagnosis(diag; title="Channel Diagnosis", save_path="diagnosis.pdf")
```
"""
function plot_diagnosis(diag::NamedTuple; 
                        title::String="",
                        save_path::Union{String,Nothing}=nothing)
    # Extract data from diagnosis
    eigenvalues_raw = diag.eigenvalues_complex
    gap = diag.gap
    unitality = diag.unitality
    is_unital = diag.is_unital
    is_unitary_channel = diag.is_unitary_channel
    dist_to_identity = diag.dist_to_identity
    
    # Sort eigenvalues by magnitude
    sorted_indices = sortperm(abs.(eigenvalues_raw), rev=true)
    sorted_raw = eigenvalues_raw[sorted_indices]
    sorted_eigs = abs.(sorted_raw)
    n = length(sorted_eigs)
    
    # Compute correlation length
    ξ = gap > 0 ? 1 / gap : Inf
    
    # Create figure with three panels
    fig = Figure(size=(1400, 500))
    
    # === Left panel: Bar plot of eigenvalue magnitudes ===
    ax1 = Axis(fig[1, 1], 
               xlabel="Eigenvalue index (sorted)", 
               ylabel="Eigenvalue magnitude |λ|",
               title="Eigenvalue Spectrum")
    
    # Color by distance from 1
    colors = [λ > 0.99 ? :red : (λ > 0.9 ? :orange : :steelblue) for λ in sorted_eigs]
    
    barplot!(ax1, 1:n, sorted_eigs, color=colors, strokewidth=0.5, strokecolor=:black)
    
    # Reference lines
    hlines!(ax1, [1.0], color=:black, linestyle=:dash, linewidth=1.5, label="λ=1")
    hlines!(ax1, [0.99], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="λ=0.99")
    
    # Highlight λ₁ and λ₂
    λ₁ = sorted_eigs[1]
    λ₂ = length(sorted_eigs) > 1 ? sorted_eigs[2] : 0.0
    scatter!(ax1, [1], [λ₁], markersize=15, color=:green, marker=:star5, label="λ₁=$(round(λ₁, digits=4))")
    if length(sorted_eigs) > 1
        scatter!(ax1, [2], [λ₂], markersize=12, color=:purple, marker=:diamond, label="λ₂=$(round(λ₂, digits=4))")
    end
    
    axislegend(ax1, position=:rb)
    
    # === Middle panel: Complex plane ===
    ax2 = Axis(fig[1, 2],
               xlabel="Re(λ)",
               ylabel="Im(λ)",
               title="Eigenvalues in Complex Plane",
               aspect=DataAspect())
    
    # Draw unit circle
    θ = range(0, 2π, length=100)
    lines!(ax2, cos.(θ), sin.(θ), color=:black, linestyle=:dash, linewidth=1.5, label="Unit circle")
    lines!(ax2, 0.99 .* cos.(θ), 0.99 .* sin.(θ), color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="|λ|=0.99")
    
    # Plot all eigenvalues
    re_parts = real.(sorted_raw)
    im_parts = imag.(sorted_raw)
    colors_scatter = [abs(λ) > 0.99 ? :red : (abs(λ) > 0.9 ? :orange : :steelblue) for λ in sorted_raw]
    scatter!(ax2, re_parts, im_parts, color=colors_scatter, markersize=8, strokewidth=0.5, strokecolor=:black)
    
    # Highlight λ₁ and λ₂
    scatter!(ax2, [real(sorted_raw[1])], [imag(sorted_raw[1])], markersize=15, color=:green, marker=:star5, label="λ₁")
    if length(sorted_raw) > 1
        scatter!(ax2, [real(sorted_raw[2])], [imag(sorted_raw[2])], markersize=12, color=:purple, marker=:diamond, label="λ₂")
    end
    
    axislegend(ax2, position=:lt)
    
    # === Right panel: Diagnosis summary ===
    ax3 = Axis(fig[1, 3],
               title="Channel Properties",
               limits=(0, 1, 0, 1))
    hidedecorations!(ax3)
    hidespines!(ax3)
    
    # Build summary text
    status_gap = gap > 0.1 ? "✓ Good" : (gap > 0.01 ? "⚠ Poor" : "✗ Bad")
    status_color = gap > 0.1 ? :green : (gap > 0.01 ? :orange : :red)
    
    unital_status = is_unital ? "⚠ Yes (preserves I/d)" : "✓ No"
    unitary_status = is_unitary_channel ? "⚠ Yes (problematic)" : "✓ No"
    
    n_near_1 = count(x -> x > 0.99, sorted_eigs)
    n_near_095 = count(x -> x > 0.95, sorted_eigs)
    
    lines = [
        ("Spectral Gap:", "$(round(gap, digits=4)) ($status_gap)", status_color),
        ("Correlation Length ξ:", "$(round(ξ, digits=2))", :black),
        ("", "", :black),
        ("λ₁ (largest):", "$(round(λ₁, digits=6))", :black),
        ("λ₂ (2nd largest):", "$(round(λ₂, digits=6))", :black),
        ("", "", :black),
        ("|λ| > 0.99:", "$n_near_1 / $n", n_near_1 > 1 ? :orange : :black),
        ("|λ| > 0.95:", "$n_near_095 / $n", :black),
        ("", "", :black),
        ("Unital:", unital_status, is_unital ? :orange : :green),
        ("Unitary Channel:", unitary_status, is_unitary_channel ? :red : :green),
        ("Unitality Deviation:", "$(round(unitality, digits=4))", :black),
        ("Dist to Identity:", "$(round(dist_to_identity, digits=4))", :black),
    ]
    
    y_pos = 0.95
    for (label, value, color) in lines
        if !isempty(label)
            text!(ax3, 0.05, y_pos, text=label, fontsize=12, align=(:left, :center))
            text!(ax3, 0.55, y_pos, text=value, fontsize=12, align=(:left, :center), color=color)
        end
        y_pos -= 0.07
    end
    
    # Add title at top
    if !isempty(title)
        Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)
    end
    
    # Save if requested
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end