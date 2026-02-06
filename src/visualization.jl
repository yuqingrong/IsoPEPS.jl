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
function resample_circuit(filename::String; conv_step=100, samples=1000, measure_first=nothing)
    result, input_args = load_result(filename)
    
    if !(result isa CircuitOptimizationResult)
        @warn "Result is not CircuitOptimizationResult, cannot resample"
        return nothing
    end
    
    # Extract parameters from result
    params = result.final_params
    
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
    fit_acf_oscillatory(lags::AbstractVector, acf::AbstractVector; fit_range=nothing, include_zero=false)

Fit ACF data using damped oscillatory model for complex eigenvalues:
    C(r) = A · exp(-r/ξ) · cos(k·r + φ)

This corresponds to a complex second eigenvalue λ₂ = |λ₂|·e^{iθ} where:
- |λ₂| = exp(-1/ξ)
- θ = k (wavevector)

# Arguments
- `lags`: Lag values
- `acf`: ACF values (real part will be used for fitting)
- `fit_range`: (start_lag, end_lag) range for fitting (default: use all data)
- `include_zero`: Whether to include lag=0 in fitting (default: false)

# Returns
- `NamedTuple` with fields:
  - `A`: Amplitude
  - `ξ`: Correlation length (decay constant)
  - `k`: Wavevector (oscillation frequency)
  - `phase`: Phase offset φ
  - `λ₂_magnitude`: |λ₂| = exp(-1/ξ)
  - `λ₂_phase`: θ = k (eigenvalue phase)
  - `fit_lags`: Lags used for fitting
  - `model`: Fitted model function

# Example
```julia
lags, acf, acf_err = compute_acf(samples; max_lag=50)
fit_params = fit_acf_oscillatory(lags, acf)
# Check if correlation has oscillatory component: fit_params.k ≠ 0
```
"""
function fit_acf_oscillatory(lags::AbstractVector, acf::AbstractVector; 
                              fit_range::Union{Tuple{Int,Int},Nothing}=nothing,
                              include_zero::Bool=false)
    lags_vec = collect(Float64, lags)
    acf_vec = collect(Float64, real.(acf))  # Use real part
    
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
    fit_acf_vals = acf_vec[mask]
    
    if length(fit_lags) < 4
        error("Not enough valid data points for oscillatory fitting (need at least 4)")
    end
    
    # Define the damped oscillatory model: C(r) = A·exp(-r/ξ)·cos(k·r + φ)
    model(r, p) = p[1] .* exp.(-r ./ p[2]) .* cos.(p[3] .* r .+ p[4])
    
    # Initial parameter guesses: [A, ξ, k, φ]
    A_init = abs(fit_acf_vals[1])
    
    # Estimate decay constant from envelope of |acf|
    abs_acf = abs.(fit_acf_vals)
    valid_idx = abs_acf .> 1e-15
    if sum(valid_idx) >= 2
        log_acf = log.(abs_acf[valid_idx])
        valid_lags_temp = fit_lags[valid_idx]
        slope = (log_acf[end] - log_acf[1]) / (valid_lags_temp[end] - valid_lags_temp[1])
        ξ_init = -1.0 / slope
        ξ_init = clamp(ξ_init, 0.1, length(fit_lags) * 10.0)
    else
        ξ_init = length(fit_lags) / 2.0
    end
    
    # Estimate frequency from zero crossings or sign changes
    sign_changes = findall(diff(sign.(fit_acf_vals)) .!= 0)
    if length(sign_changes) >= 2
        avg_period = 2 * mean(diff(fit_lags[sign_changes]))
        k_init = 2π / avg_period
    else
        k_init = 0.0  # No oscillation detected
    end
    
    # Phase: start with 0 or π depending on sign of first value
    φ_init = fit_acf_vals[1] < 0 ? π : 0.0
    
    p0 = [A_init, ξ_init, k_init, φ_init]
    
    # Set parameter bounds
    lower = [0.0, 0.01, -π, -2π]
    upper = [Inf, Inf, π, 2π]
    
    try
        # Nonlinear least squares fit
        fit_result = curve_fit(model, fit_lags, fit_acf_vals, p0, lower=lower, upper=upper)
        A, ξ, k, φ = coef(fit_result)
        
        # Compute effective eigenvalue magnitude and phase
        λ₂_magnitude = exp(-1.0/ξ)
        λ₂_phase = k
        
        return (A=A, ξ=ξ, k=k, phase=φ, 
                λ₂_magnitude=λ₂_magnitude, λ₂_phase=λ₂_phase,
                fit_lags=fit_lags, 
                model=(r) -> model(r, [A, ξ, k, φ]))
    catch e
        @warn "Oscillatory fit failed: $e. Falling back to simple exponential fit."
        # Fall back to simple exponential
        simple_fit = fit_acf(lags, abs.(acf); fit_range=fit_range, include_zero=include_zero)
        return (A=simple_fit.A, ξ=simple_fit.ξ, k=0.0, phase=0.0,
                λ₂_magnitude=simple_fit.λ₂, λ₂_phase=0.0,
                fit_lags=simple_fit.fit_lags,
                model=simple_fit.model)
    end
end

"""
    plot_acf(lags::AbstractVector, acf::AbstractVector; kwargs...)

Plot autocorrelation function with optional exponential fit and theoretical decay comparison.

# Arguments
- `lags`: Lag values
- `acf`: ACF values
- `acf_err`: Error bars (optional)
- `fit_range`: (start_lag, end_lag) for automatic fitting (optional)
- `fit_params`: Pre-computed fit params from `fit_acf` (optional, overrides fit_range)
- `fit_oscillatory`: If true, use oscillatory model `A·exp(-r/ξ)·cos(kr+φ)` (default: false)
- `theoretical_decay`: Dominant mode decay A·|λ_α|^r (optional, green solid line)
- `exact_decay`: Exact decay from sum of all modes Σ c_α λ_α^(r-1) (optional, purple dotted line)
- `title`: Plot title
- `logscale`: Use log scale for y-axis (default: true)
- `save_path`: Path to save figure (optional)
- `show_lambda_eff`: If true, show second panel with λ_eff(r) = C(r+1)/C(r) (default: false)
- `lambda_theory`: Theoretical λ₂ from transfer matrix to show in λ_eff panel (optional)

# Returns
- `Figure` object

# Example with λ_eff panel
```julia
# Compute correlation coefficients and get theoretical λ₂
eigenvalues, coefficients, ξ = compute_correlation_coefficients(gates, row, virtual_qubits, Matrix(Z))
λ₂_theory = abs(eigenvalues[2])

# Plot with λ_eff panel
lags, acf, acf_err = compute_acf(samples; max_lag=50)
fig = plot_acf(lags, acf; show_lambda_eff=true, lambda_theory=λ₂_theory)
```
"""
function plot_acf(lags::AbstractVector, acf::AbstractVector;
                  acf_err::Union{AbstractVector,Nothing}=nothing,
                  fit_range::Union{Tuple{Int,Int},Nothing}=nothing,
                  fit_params::Union{NamedTuple,Nothing}=nothing,
                  fit_oscillatory::Bool=false,
                  theoretical_decay::Union{Tuple{AbstractVector,AbstractVector},Nothing}=nothing,
                  exact_decay::Union{Tuple{AbstractVector,AbstractVector},Nothing}=nothing,
                  dominant_decay::Union{Tuple{AbstractVector,AbstractVector},Nothing}=nothing,
                  title::String="Autocorrelation Function",
                  logscale::Bool=true,
                  save_path::Union{String,Nothing}=nothing,
                  show_lambda_eff::Bool=false,
                  lambda_theory::Union{Real,Nothing}=nothing,
                  lambda_eff_theory::Union{Tuple{AbstractVector,AbstractVector},Nothing}=nothing,
                  lambda_eff_dominant::Union{Tuple{AbstractVector,AbstractVector},Nothing}=nothing)
    
    # Determine figure size based on whether we show λ_eff panel
    fig_height = show_lambda_eff ? 600 : 400
    fig = Figure(size=(600, fig_height))
    
    # Prepare data - use absolute values for log scale
    abs_acf = collect(acf)
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
    scatter!(ax, lags_vec, plot_y, markersize=8, label="Data", color=:steelblue)
    
    # Compute fit if fit_range provided but no fit_params
    if isnothing(fit_params) && !isnothing(fit_range)
        if fit_oscillatory
            fit_params = fit_acf_oscillatory(lags, acf; fit_range=fit_range)
        else
            fit_params = fit_acf(lags, acf; fit_range=fit_range)
        end
    end
    
    # Plot fit curve
    if !isnothing(fit_params)
        A = fit_params.A
        ξ = fit_params.ξ
        
        # Check if this is an oscillatory fit
        has_oscillation = haskey(fit_params, :k) && abs(fit_params.k) > 1e-6
        
        if has_oscillation
            # Oscillatory model: A·exp(-r/ξ)·cos(k·r + φ)
            k = fit_params.k
            φ = fit_params.phase
            λ₂_mag = fit_params.λ₂_magnitude
            
            fit_curve = A .* exp.(-lags_vec ./ ξ) .* cos.(k .* lags_vec .+ φ)
            label_text = "Fit: ξ=$(round(ξ, digits=2)), |λ₂|=$(round(λ₂_mag, digits=4)), k=$(round(k, digits=3))"
            
            if logscale
                fit_curve = abs.(fit_curve)
                fit_curve = max.(fit_curve, min_threshold)
            end
        else
            # Simple exponential: A·exp(-r/ξ)
            λ₂ = haskey(fit_params, :λ₂) ? fit_params.λ₂ : fit_params.λ₂_magnitude
            fit_curve = A .* exp.(-lags_vec ./ ξ)
            label_text = "Fit: ξ=$(round(ξ, digits=2)), λ₂=$(round(λ₂, digits=4))"
            
            if logscale
                fit_curve = max.(fit_curve, min_threshold)
            end
        end
        
        lines!(ax, lags_vec, fit_curve, linewidth=2, linestyle=:dash, color=:red,
               label=label_text)
    end
    
    # Plot theoretical decay from eigenmode analysis (single dominant mode)
    if !isnothing(theoretical_decay)
        theory_lags, theory_corr = theoretical_decay
        theory_lags_vec = collect(theory_lags)
        
        if logscale
            theory_y = abs.(collect(theory_corr))
            theory_y = max.(theory_y, min_threshold)
        else
            theory_y = real.(collect(theory_corr))
        end
        
        lines!(ax, theory_lags_vec, theory_y, linewidth=2, linestyle=:solid, color=:green,
               label="Theory (dominant)")
    end
    
    # Plot exact decay from sum of all modes: Σ c_α λ_α^(r-1)
    if !isnothing(exact_decay)
        exact_lags, exact_corr = exact_decay
        exact_lags_vec = collect(exact_lags)
        
        if logscale
            exact_y = abs.(collect(exact_corr))
            exact_y = max.(exact_y, min_threshold)
        else
            exact_y = real.(collect(exact_corr))
        end
        
        lines!(ax, exact_lags_vec, exact_y, linewidth=2, linestyle=:dot, color=:purple,
               label="Exact (all modes)")
    end
    
    # Plot dominant modes decay: C(r) from only the few dominant modes
    if !isnothing(dominant_decay)
        dom_lags, dom_corr = dominant_decay
        dom_lags_vec = collect(dom_lags)
        
        if logscale
            dom_y = abs.(collect(dom_corr))
            dom_y = max.(dom_y, min_threshold)
        else
            dom_y = real.(collect(dom_corr))
        end
        
        lines!(ax, dom_lags_vec, dom_y, linewidth=2.5, linestyle=:dashdot, color=:orange,
               label="Dominant modes")
    end
    
    axislegend(ax, position=:rt)
    
    # Add λ_eff(r) = C(r+1)/C(r) panel if requested
    if show_lambda_eff && length(acf) > 2
        ax2 = Axis(fig[2, 1],
                   xlabel="Lag r", ylabel="λ_eff(r) = C(r+1)/C(r)",
                   title="Effective Eigenvalue")
        
        # Compute λ_eff(r) = C(r+1) / C(r)
        # For exponential decay C(r) ~ λ^r, this gives λ_eff = λ
        acf_vals = abs.(collect(acf))
        lambda_eff_lags = collect(lags)[1:end-1]
        lambda_eff = acf_vals[2:end] ./ acf_vals[1:end-1]
        
        # Filter out infinities and NaNs
        valid_mask = isfinite.(lambda_eff) .& (lambda_eff .> 0) .& (lambda_eff .< 10)
        if any(valid_mask)
            scatter!(ax2, lambda_eff_lags[valid_mask], lambda_eff[valid_mask], 
                    markersize=6, label="λ_eff(r) from data", color=:steelblue)
        end
        
        # Add theoretical λ_eff from mixed-mode analysis if provided
        # This is the weighted average: λ_eff(r) = Σ w_α(r) λ_α / Σ w_α(r)
        if !isnothing(lambda_eff_theory)
            theory_lags, theory_lambda_eff = lambda_eff_theory
            theory_lags_vec = collect(theory_lags)
            theory_vals = abs.(collect(theory_lambda_eff))
            
            # Filter to valid range
            valid_theory = isfinite.(theory_vals) .& (theory_vals .> 0) .& (theory_vals .< 10)
            if any(valid_theory)
                lines!(ax2, theory_lags_vec[valid_theory], theory_vals[valid_theory], 
                      linewidth=2.5, linestyle=:solid, color=:purple,
                      label="λ_eff (all modes)")
            end
        end
        
        # Add λ_eff from dominant modes only (e.g., 3 largest coefficient modes)
        if !isnothing(lambda_eff_dominant)
            dom_lags, dom_lambda_eff = lambda_eff_dominant
            dom_lags_vec = collect(dom_lags)
            dom_vals = abs.(collect(dom_lambda_eff))
            
            # Filter to valid range
            valid_dom = isfinite.(dom_vals) .& (dom_vals .> 0) .& (dom_vals .< 10)
            if any(valid_dom)
                lines!(ax2, dom_lags_vec[valid_dom], dom_vals[valid_dom], 
                      linewidth=2.5, linestyle=:dashdot, color=:orange,
                      label="λ_eff (dominant modes)")
            end
        end
        
        # Add single dominant eigenvalue from transfer matrix if provided
        if !isnothing(lambda_theory)
            hlines!(ax2, [lambda_theory], color=:green, linestyle=:dash, linewidth=2,
                   label="λ_slow (contributing) = $(round(lambda_theory, digits=4))")
        end
        
        # Add theoretical λ from fit if available
        if !isnothing(fit_params)
            λ_fit = haskey(fit_params, :λ₂) ? fit_params.λ₂ : 
                    (haskey(fit_params, :λ₂_magnitude) ? fit_params.λ₂_magnitude : exp(-1/fit_params.ξ))
            hlines!(ax2, [λ_fit], color=:red, linestyle=:dot, linewidth=2,
                   label="λ₂ (single exp fit) = $(round(λ_fit, digits=4))")
        end
        
        # Set y-axis limits to reasonable range
        if any(valid_mask)
            ymin = max(0.8 * minimum(lambda_eff[valid_mask]), 0.9)
            ymax = min(1.2 * maximum(lambda_eff[valid_mask]), 1.1)
            ylims!(ax2, ymin, ymax)
        end
        
        axislegend(ax2, position=:rt)
    end
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    plot_training_history(steps, values; kwargs...)

Plot training loss/energy/observable vs training steps.

# Arguments
- `steps`: Step indices or iteration numbers
- `values`: Values at each step (energy, loss, observable, etc.)
- `reference`: Reference value (e.g., exact energy) shown as horizontal line (optional, overridden by pepskit_results_file if provided)
- `ylabel`: Y-axis label (default: "Energy")
- `title`: Plot title (default: "Training History", overridden if g, row, nqubits provided)
- `logscale`: Use log scale (default: false)
- `save_path`: Path to save figure (optional)
- `g`: Transverse field value (optional, for loading reference from pepskit results and title)
- `row`: Number of rows in the circuit (optional, for title)
- `nqubits`: Number of qubits (optional, for title)
- `pepskit_results_file`: Path to pepskit results JSON file (optional, to load reference energy)

# Returns
- `Figure` object
"""
function plot_training_history(steps::AbstractVector, values::AbstractVector;
                                reference::Union{Real,Nothing}=nothing,
                                ylabel::String="Energy",
                                title::String="Training History",
                                logscale::Bool=false,
                                save_path::Union{String,Nothing}=nothing,
                                g::Union{Real,Nothing}=nothing,
                                row::Union{Int,Nothing}=nothing,
                                nqubits::Union{Int,Nothing}=nothing,
                                pepskit_results_file::Union{String,Nothing}=nothing)
    
    # Load reference energy from pepskit results if file and g are provided
    ref_energy = reference
    if !isnothing(pepskit_results_file) && !isnothing(g) && isfile(pepskit_results_file)
        pepskit_data = open(pepskit_results_file, "r") do io
            JSON3.read(io, Dict)
        end
        g_values = pepskit_data["g_values"]
        energies = pepskit_data["energies"]
        
        # Find the closest g value
        idx = argmin(abs.(g_values .- g))
        if abs(g_values[idx] - g) < 1e-6
            ref_energy = energies[idx]
        else
            @warn "g=$g not found in pepskit results, using closest value g=$(g_values[idx])"
            ref_energy = energies[idx]
        end
    end
    
    # Generate title if g, row, nqubits are provided
    plot_title = title
    if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
        plot_title = "Training History: row=$row, g=$g, nqubits=$nqubits"
    elseif !isnothing(g) && !isnothing(row)
        plot_title = "row=$row, g=$g"
    elseif !isnothing(g)
        plot_title = "g=$g"
    end
    
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1],
              xlabel="Step", ylabel=ylabel,
              title=plot_title,
              yscale=logscale ? log10 : identity)
    
    lines!(ax, collect(steps), collect(values), linewidth=2, label=ylabel)
    
    if !isnothing(ref_energy)
        ref_label = "PEPSKit.jl (E=$(round(ref_energy, digits=4)))"
        hlines!(ax, [ref_energy], linestyle=:dash, color=:red, linewidth=1.5, label=ref_label)
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
    plot_expectation_values(; X=nothing, Z=nothing, ZZ_vert=nothing, ZZ_horiz=nothing, kwargs...)

Plot expectation values ⟨X⟩, ⟨Z⟩, ⟨ZZ⟩ and energy as a bar chart with labeled values.

# Arguments
- `energy`: Energy value (optional)
- `X`: ⟨X⟩ expectation value (optional)
- `Z`: ⟨Z⟩ expectation value (optional)
- `ZZ_vert`: ⟨ZZ⟩ vertical bond expectation value (optional)
- `ZZ_horiz`: ⟨ZZ⟩ horizontal bond expectation value (optional)
- `ZZ_connected`: ⟨ZZ⟩_c = ⟨ZZ⟩ - ⟨Z⟩² connected correlation (optional, auto-computed if not provided)
- `title`: Plot title (default: "Expectation Values")
- `g`: Transverse field value (optional, for title)
- `row`: Number of rows (optional, for title)
- `nqubits`: Number of qubits (optional, for title)
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object
"""
function plot_expectation_values(; energy::Union{Real,Nothing}=nothing,
                                   X::Union{Real,Nothing}=nothing,
                                   Z::Union{Real,Nothing}=nothing,
                                   ZZ_vert::Union{Real,Nothing}=nothing,
                                   ZZ_horiz::Union{Real,Nothing}=nothing,
                                   ZZ_connected::Union{Real,Nothing}=nothing,
                                   title::String="Expectation Values",
                                   g::Union{Real,Nothing}=nothing,
                                   row::Union{Int,Nothing}=nothing,
                                   nqubits::Union{Int,Nothing}=nothing,
                                   save_path::Union{String,Nothing}=nothing)
    
    # Collect available values
    labels = String[]
    values = Float64[]
    
    if !isnothing(energy)
        push!(labels, "E")
        push!(values, energy)
    end
    if !isnothing(X)
        push!(labels, "⟨X⟩")
        push!(values, X)
    end
    if !isnothing(Z)
        push!(labels, "⟨Z⟩")
        push!(values, Z)
    end
    if !isnothing(ZZ_vert)
        push!(labels, "⟨ZZ⟩ᵥ")
        push!(values, ZZ_vert)
    end
    if !isnothing(ZZ_horiz)
        push!(labels, "⟨ZZ⟩ₕ")
        push!(values, ZZ_horiz)
    end
    if isempty(values)
        @warn "No expectation values provided"
        return nothing
    end
    
    # Generate title
    plot_title = title
    if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
        plot_title = "Expectation Values: row=$row, g=$g, nqubits=$nqubits"
    elseif !isnothing(g) && !isnothing(row)
        plot_title = "Expectation Values: row=$row, g=$g"
    elseif !isnothing(g)
        plot_title = "Expectation Values: g=$g"
    end
    
    # Create figure
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1],
              xlabel="Observable",
              ylabel="Value",
              title=plot_title,
              xticks=(1:length(labels), labels))
    
    # Color bars based on sign
    colors = [v >= 0 ? :steelblue : :coral for v in values]
    
    barplot!(ax, 1:length(values), values, color=colors, strokewidth=1, strokecolor=:black)
    
    # Add value labels on bars
    for (i, v) in enumerate(values)
        offset = v >= 0 ? 0.05 : -0.05
        align = v >= 0 ? (:center, :bottom) : (:center, :top)
        text!(ax, i, v + offset; text=string(round(v, digits=4)), align=align, fontsize=12)
    end
    
    # Reference line at 0
    hlines!(ax, [0], color=:black, linewidth=1)
    
    # Set y-axis limits with some padding
    ymin = min(minimum(values), 0) * 1.2
    ymax = max(maximum(values), 0) * 1.2
    if ymin == ymax
        ymin, ymax = -1, 1
    end
    ylims!(ax, ymin - 0.1, ymax + 0.1)
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    plot_expectation_values(result::ExactOptimizationResult; kwargs...)

Plot expectation values from ExactOptimizationResult.
"""
function plot_expectation_values(result::ExactOptimizationResult; kwargs...)
    plot_expectation_values(;
        energy=result.energy,
        X=result.X_expectation,
        ZZ_vert=result.ZZ_vertical,
        ZZ_horiz=result.ZZ_horizontal,
        kwargs...
    )
end

"""
    plot_expectation_values(result::CircuitOptimizationResult; row=nothing, p=nothing, nqubits=nothing, J=1.0, g=nothing, kwargs...)

Plot expectation values from CircuitOptimizationResult, showing both sample-based and exact transfer-matrix values.

# Arguments
- `row`: Number of rows
- `p`: Number of layers (needed for exact contraction)
- `nqubits`: Number of qubits per gate (needed for exact contraction)
- `J`: Ising coupling strength (default: 1.0)
- `g`: Transverse field strength (needed for energy calculation)
- `title`: Plot title (optional)
- `save_path`: Path to save figure (optional)
- `datafile`: Optional result JSON path; if provided, resamples using optimized parameters

Shows grouped bar chart comparing sample-based estimation vs exact tensor contraction.
Sample values include error bars showing standard error (SE = std/√n).
"""
function plot_expectation_values(result::CircuitOptimizationResult; 
                                  row::Union{Int,Nothing}=nothing,
                                  p::Union{Int,Nothing}=nothing,
                                  nqubits::Union{Int,Nothing}=nothing,
                                  J::Float64=1.0,
                                  g::Union{Real,Nothing}=nothing,
                                  title::String="Expectation Values",
                                  save_path::Union{String,Nothing}=nothing,
                                  datafile::Union{String,Nothing}=nothing,
                                  kwargs...)
    
    # Compute sample-based values
    Z_samples = result.final_Z_samples
    X_samples = result.final_X_samples
    if !isnothing(datafile)
        if isfile(datafile)
            resampled = resample_circuit(datafile; conv_step=1000, samples=1000000, measure_first=nothing)
            if !isnothing(resampled)
                _, Z_samples, X_samples, _, _ = resampled
            else
                @warn "Resampling failed for $datafile; using samples in result"
            end
        else
            @warn "Resample datafile not found: $datafile; using samples in result"
        end
    end
    
    Z_sample = isempty(Z_samples) ? nothing : mean(Z_samples)
    X_sample = isempty(X_samples) ? nothing : mean(X_samples)
    
    # Calculate standard errors
    Z_stderr = isempty(Z_samples) ? nothing : std(Z_samples) / sqrt(length(Z_samples))
    X_stderr = isempty(X_samples) ? nothing : std(X_samples) / sqrt(length(X_samples))
    
    ZZ_vert_sample = nothing
    ZZ_horiz_sample = nothing
    energy_sample = nothing
    ZZ_vert_stderr = nothing
    ZZ_horiz_stderr = nothing
    energy_stderr = nothing
    
    N = length(Z_samples)
    if !isnothing(row) && row > 1 && N > 1
        # Vertical bonds: Z[i]*Z[i+1] only within columns (skip cross-column pairs)
        ZZ_vert_pairs = [Z_samples[i] * Z_samples[i+1] for i in 1:N-1 if i % row != 0]
        ZZ_vert_sample = mean(ZZ_vert_pairs)
        ZZ_vert_stderr = std(ZZ_vert_pairs) / sqrt(length(ZZ_vert_pairs))
    elseif N > 1
        # row=1 case: no vertical bonds, skip ZZ_vert_sample
    end
    if !isnothing(row) && N > row
        ZZ_horiz_pairs = [Z_samples[i] * Z_samples[i+row] for i in 1:N-row]
        ZZ_horiz_sample = mean(ZZ_horiz_pairs)
        ZZ_horiz_stderr = std(ZZ_horiz_pairs) / sqrt(length(ZZ_horiz_pairs))
    end
    if !isnothing(g) && !isnothing(row) && N > 0 && !isempty(X_samples)
        energy_sample = compute_energy(X_samples, Z_samples, g, J, row)
        
        # Calculate energy standard error using error propagation
        # E = -g*⟨X⟩ - J*⟨ZZ⟩, where measurements are independent
        # σ_E = sqrt(g² * σ_X² + J² * σ_ZZ²)
        if row == 1
            # Only horizontal bonds
            ZZ_stderr_total = isnothing(ZZ_horiz_stderr) ? 0.0 : ZZ_horiz_stderr
        else
            # Both vertical and horizontal bonds contribute
            # For uncorrelated measurements: σ_(A+B)² = σ_A² + σ_B²
            ZZ_vert_se = isnothing(ZZ_vert_stderr) ? 0.0 : ZZ_vert_stderr
            ZZ_horiz_se = isnothing(ZZ_horiz_stderr) ? 0.0 : ZZ_horiz_stderr
            ZZ_stderr_total = sqrt(ZZ_vert_se^2 + ZZ_horiz_se^2)
        end
        X_se = isnothing(X_stderr) ? 0.0 : X_stderr
        energy_stderr = sqrt(g^2 * X_se^2 + J^2 * ZZ_stderr_total^2)
    end
    
    # Compute exact values if parameters are available
    X_exact = nothing
    Z_exact = nothing
    ZZ_vert_exact = nothing
    ZZ_horiz_exact = nothing
    energy_exact = nothing
    
    can_compute_exact = !isnothing(row) && !isnothing(p) && !isnothing(nqubits) && !isempty(result.final_params)
    
    if can_compute_exact
        virtual_qubits = (nqubits - 1) ÷ 2
        
        gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=true)
        
        # Average over all positions (samples average over all rows)
        X_exact = mean(real(expect(gates, row, virtual_qubits, :X; position=i)) for i in 1:row)
        Z_exact = mean(real(expect(gates, row, virtual_qubits, :Z; position=i)) for i in 1:row)
        
        # Average ZZ_vert over all vertical bonds (row-1 bonds per column)
        if row > 1
            ZZ_vert_exact = mean(real(expect(gates, row, virtual_qubits, Dict(i => :Z, i+1 => :Z))) for i in 1:row-1)
        end
        
        # ZZ_horiz: average correlation at separation=1 over all starting positions
        ZZ_horiz_vals = Float64[]
        for pos in 1:row
            correlations = correlation_function(gates, row, virtual_qubits, :Z, 1; position=pos)
            if haskey(correlations, 1)
                push!(ZZ_horiz_vals, real(correlations[1]))
            end
        end
        if !isempty(ZZ_horiz_vals)
            ZZ_horiz_exact = mean(ZZ_horiz_vals)
        end
        
        if !isnothing(g) && !isnothing(ZZ_horiz_exact) && (row == 1 || !isnothing(ZZ_vert_exact))
            energy_exact = -g*X_exact - J*(row == 1 ? ZZ_horiz_exact : ZZ_vert_exact + ZZ_horiz_exact)
        end
    end
    
    # Generate title
    plot_title = title
    if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
        plot_title = "Expectation Values: row=$row, g=$g, nqubits=$nqubits"
    elseif !isnothing(g) && !isnothing(row)
        plot_title = "Expectation Values: row=$row, g=$g"
    end
    
    # Build data for grouped bar chart
    labels = String[]
    sample_values = Float64[]
    exact_values = Float64[]
    sample_errors = Float64[]  # Standard errors for sample values
    
    # Energy
    if !isnothing(energy_sample)
        push!(labels, "E")
        push!(sample_values, energy_sample)
        push!(exact_values, isnothing(energy_exact) ? NaN : energy_exact)
        push!(sample_errors, isnothing(energy_stderr) ? 0.0 : energy_stderr)
    end
    
    # X expectation
    if !isnothing(X_sample) || !isnothing(X_exact)
        push!(labels, "⟨X⟩")
        push!(sample_values, isnothing(X_sample) ? NaN : X_sample)
        push!(exact_values, isnothing(X_exact) ? NaN : X_exact)
        push!(sample_errors, isnothing(X_stderr) ? 0.0 : X_stderr)
    end
    
    # Z expectation
    if !isnothing(Z_sample) || !isnothing(Z_exact)
        push!(labels, "⟨Z⟩")
        push!(sample_values, isnothing(Z_sample) ? NaN : Z_sample)
        push!(exact_values, isnothing(Z_exact) ? NaN : Z_exact)
        push!(sample_errors, isnothing(Z_stderr) ? 0.0 : Z_stderr)
    end
    
    # ZZ vertical
    if !isnothing(ZZ_vert_sample) || !isnothing(ZZ_vert_exact)
        push!(labels, "⟨ZZ⟩ᵥ")
        push!(sample_values, isnothing(ZZ_vert_sample) ? NaN : ZZ_vert_sample)
        push!(exact_values, isnothing(ZZ_vert_exact) ? NaN : ZZ_vert_exact)
        push!(sample_errors, isnothing(ZZ_vert_stderr) ? 0.0 : ZZ_vert_stderr)
    end
    
    # ZZ horizontal
    if !isnothing(ZZ_horiz_sample) || !isnothing(ZZ_horiz_exact)
        push!(labels, "⟨ZZ⟩ₕ")
        push!(sample_values, isnothing(ZZ_horiz_sample) ? NaN : ZZ_horiz_sample)
        push!(exact_values, isnothing(ZZ_horiz_exact) ? NaN : ZZ_horiz_exact)
        push!(sample_errors, isnothing(ZZ_horiz_stderr) ? 0.0 : ZZ_horiz_stderr)
    end
    
    if isempty(labels)
        @warn "No expectation values to plot"
        return nothing
    end
    
    # Create grouped bar chart
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              xlabel="Observable",
              ylabel="Value",
              title=plot_title,
              xticks=(1:length(labels), labels))
    
    n = length(labels)
    bar_width = 0.35
    
    # Sample bars (left)
    positions_sample = collect(1:n) .- bar_width/2
    valid_sample = .!isnan.(sample_values)
    if any(valid_sample)
        barplot!(ax, positions_sample[valid_sample], sample_values[valid_sample], 
                 width=bar_width, color=:steelblue, strokewidth=1, strokecolor=:black,
                 label="Sample")
        
        # Add error bars
        valid_errors = sample_errors[valid_sample]
        if any(valid_errors .> 0)
            errorbars!(ax, positions_sample[valid_sample], sample_values[valid_sample], 
                      valid_errors, 
                      color=:black, linewidth=1.5, whiskerwidth=8)
        end
        
        # Add value labels (adjust offset to account for error bars)
        for (i, (pos, val, err)) in enumerate(zip(positions_sample, sample_values, sample_errors))
            if !isnan(val)
                offset = val >= 0 ? (0.03 + err) : (-0.03 - err)
                align = val >= 0 ? (:center, :bottom) : (:center, :top)
                text!(ax, pos, val + offset; text=string(round(val, digits=3)), align=align, fontsize=9)
            end
        end
    end
    
    # Exact bars (right) - only if we have exact values
    if can_compute_exact
        positions_exact = collect(1:n) .+ bar_width/2
        valid_exact = .!isnan.(exact_values)
        if any(valid_exact)
            barplot!(ax, positions_exact[valid_exact], exact_values[valid_exact], 
                     width=bar_width, color=:coral, strokewidth=1, strokecolor=:black,
                     label="Exact")
            # Add value labels
            for (i, (pos, val)) in enumerate(zip(positions_exact, exact_values))
                if !isnan(val)
                    offset = val >= 0 ? 0.03 : -0.03
                    align = val >= 0 ? (:center, :bottom) : (:center, :top)
                    text!(ax, pos, val + offset; text=string(round(val, digits=3)), align=align, fontsize=9)
                end
            end
        end
    end
    
    # Reference line at 0
    hlines!(ax, [0], color=:black, linewidth=1)
    
    # Legend at right bottom
    axislegend(ax, position=:rb)
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
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
    plot_correlation_function(filename; max_separation=20, conv_step=1000, samples=100000, save_path=nothing)

Plot two-point correlation functions comparing exact (transfer matrix) and sampling methods.

Shows both full correlation ⟨Z_i Z_{i+r}⟩ and connected correlation ⟨Z_i Z_{i+r}⟩ - ⟨Z⟩² 
on a log scale for different separations r.

# Arguments
- `filename`: Path to JSON file with circuit optimization results
- `max_separation`: Maximum separation r to compute (default: 20)
- `conv_step`: Convergence steps for sampling (default: 1000)
- `samples`: Number of samples to generate (default: 100000)
- `save_path`: Path to save figure (optional)

# Returns
- `fig`: CairoMakie Figure object
- `data`: NamedTuple with computed correlation data

# Example
```julia
fig, data = plot_correlation_function("results/circuit.json"; max_separation=30)
```
"""
function plot_correlation_function(filename::String; 
                                   max_separation::Int=20,
                                   conv_step::Int=1000, 
                                   samples::Int=100000,
                                   save_path::Union{String,Nothing}=nothing)
    # Load data and extract configuration
    result, input_args = load_result(filename)
    
    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    g = get(input_args, :g, NaN)
    virtual_qubits = (nqubits - 1) ÷ 2
    share_params = get(input_args, :share_params, true)
    
    # Reconstruct gates from parameters
    params = result.final_params
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    
    println("=== Correlation Function Analysis ===")
    println("File: ", basename(filename))
    println("Configuration: g=$g, row=$row, nqubits=$nqubits")
    
    # Compute exact correlations using transfer matrix
    println("\nComputing exact correlations (transfer matrix)...")
    exact_full = correlation_function(gates, row, virtual_qubits, :Z, 1:max_separation; connected=false)
    exact_connected = correlation_function(gates, row, virtual_qubits, :Z, 1:max_separation; connected=true)
    
    # Extract values in order
    separations = collect(1:max_separation)
    exact_full_vals = [real(exact_full[r]) for r in separations]
    exact_connected_vals = [real(exact_connected[r]) for r in separations]
    
    # Compute correlation length from transfer matrix spectrum
    _, gap, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    correlation_length = 1 / gap
    println("Correlation length ξ = $(round(correlation_length, digits=2))")
    
    # Compute sample-based correlations
    println("\nGenerating samples (conv_step=$conv_step, samples=$samples)...")
    rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits; 
                                                        conv_step=conv_step, 
                                                        samples=samples,
                                                        measure_first=:Z)
    
    # Compute ACF on samples
    # With row parameter, lag in subsampled data = horizontal separation directly
    Z_vec = Z_samples[conv_step+1:end]  # Discard burn-in
    
    lags, acf, acf_err, corr_full, corr_err, corr_connected, corr_connected_err = compute_acf(
        reshape(Float64.(Z_vec), 1, :); max_lag=max_separation+1, row=row
    )
    
    # Extract correlations and their errors
    sample_full = corr_full
    sample_full_err = corr_err
    sample_connected = corr_connected
    sample_connected_err = corr_connected_err
    
    # Limit to available data
    n_sample_seps = min(length(sample_full) - 1, max_separation)
    sample_seps = 1:n_sample_seps
    sample_full_vals = sample_full[2:n_sample_seps+1]  # index 2 is separation 1
    sample_full_err_vals = sample_full_err[2:n_sample_seps+1]
    sample_connected_vals = sample_connected[2:n_sample_seps+1]
    sample_connected_err_vals = sample_connected_err[2:n_sample_seps+1]
    
    println("Sampling std errors range (full): $(round(minimum(sample_full_err_vals), sigdigits=2)) - $(round(maximum(sample_full_err_vals), sigdigits=2))")
    println("Sampling std errors range (connected): $(round(minimum(sample_connected_err_vals), sigdigits=2)) - $(round(maximum(sample_connected_err_vals), sigdigits=2))")
    
    # Compute error between exact and sample
    common_seps = min(length(exact_full_vals), length(sample_full_vals))
    error_full = abs.(exact_full_vals[1:common_seps] .- sample_full_vals[1:common_seps])
    error_connected = abs.(exact_connected_vals[1:common_seps] .- sample_connected_vals[1:common_seps])
    mean_error_full = mean(error_full)
    mean_error_connected = mean(error_connected)
    println("Mean |exact - sample| error (full): $(round(mean_error_full, digits=6))")
    println("Mean |exact - sample| error (connected): $(round(mean_error_connected, digits=6))")
    
    # Create figure with 2x1 layout
    fig = Figure(size=(800, 700))
    
    # Minimum value for log scale (avoid log of zero/negative)
    min_val = 1e-15
    
    # Title with g, row, nqubits info
    title_str = "Correlation Function: g=$g, row=$row, nqubits=$nqubits, ξ=$(round(correlation_length, digits=2))"
    Label(fig[0, 1], title_str, fontsize=16, font=:bold)
    
    # Top panel: Full correlation
    ax1 = Axis(fig[1, 1],
               xlabel="Separation r",
               ylabel="|⟨Z_i Z_{i+r}⟩|",
               title="Full Correlation",
               yscale=log10)
               
    
    # Prepare data for log scale (clamp to min_val)
    exact_full_abs = max.(abs.(exact_full_vals), min_val)
    sample_full_abs = max.(abs.(sample_full_vals), min_val)
    error_full_plot = max.(error_full, min_val)
    
    # Plot exact values
    lines!(ax1, separations, exact_full_abs, 
           label="Exact contraction", color=:blue, linewidth=2)
    scatter!(ax1, separations, exact_full_abs, 
             color=:blue, markersize=8)
    
    # Plot sample values with error bars (per-lag standard errors)
    # Use asymmetric error bars to avoid negative values on log scale
    err_low = min.(sample_full_err_vals, sample_full_abs .- min_val)  # Clamp lower bound
    err_high = sample_full_err_vals
    scatter!(ax1, collect(sample_seps), sample_full_abs, 
             label="Sampling ± std err", color=:red, markersize=8, marker=:diamond)
    errorbars!(ax1, collect(sample_seps), sample_full_abs, err_low, err_high,
               color=:red, whiskerwidth=6)
    
    axislegend(ax1, position=:rt)
    
    # Bottom panel: Connected correlation
    ax2 = Axis(fig[2, 1],
               xlabel="Separation r",
               ylabel="|⟨Z_i Z_{i+r}⟩_c|",
               title="Connected Correlation",
               yscale=log10)
    
    # Prepare data for log scale (clamp to min_val)
    exact_connected_abs = max.(abs.(exact_connected_vals), min_val)
    sample_connected_abs = max.(abs.(sample_connected_vals), min_val)
    error_connected_plot = max.(error_connected, min_val)
    
    # Plot exact values
    lines!(ax2, separations, exact_connected_abs, 
           label="Exact contraction", color=:blue, linewidth=2)
    scatter!(ax2, separations, exact_connected_abs, 
             color=:blue, markersize=8)
    
    # Plot sample values with error bars (per-lag standard errors)
    # Use asymmetric error bars to avoid negative values on log scale
    err_low_conn = min.(sample_connected_err_vals, sample_connected_abs .- min_val)  # Clamp lower bound
    err_high_conn = sample_connected_err_vals
    scatter!(ax2, collect(sample_seps), sample_connected_abs, 
             label="Sampling ± std err", color=:red, markersize=8, marker=:diamond)
    errorbars!(ax2, collect(sample_seps), sample_connected_abs, err_low_conn, err_high_conn,
               color=:red, whiskerwidth=6)
    
    axislegend(ax2, position=:rt)
    
    # Save if path provided
    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end
    
    # Return data for further analysis
    data = (
        separations = separations,
        exact_full = exact_full_vals,
        exact_connected = exact_connected_vals,
        sample_seps = collect(sample_seps),
        sample_full = sample_full_vals,
        sample_full_err = sample_full_err_vals,
        sample_connected = sample_connected_vals,
        sample_connected_err = sample_connected_err_vals,
        error_full = error_full,
        error_connected = error_connected,
        mean_error_full = mean_error_full,
        mean_error_connected = mean_error_connected,
        correlation_length = correlation_length,
        g = g,
        row = row,
        nqubits = nqubits
    )
    
    return fig, data
end

"""
    plot_MI(mi_matrix::Matrix; title="", save_path=nothing, show_values=true, unit=:nats)
    plot_MI(filename::String; conv_step=100, samples=1000, title="", save_path=nothing, show_values=true, unit=:nats)

Plot mutual information matrix as a heatmap.

# Arguments
- `mi_matrix`: n×n mutual information matrix (from `mutual_information`)
- `filename`: Path to result file (alternative: compute MI from file)
- `conv_step`: Convergence steps for MI computation (default: 100)
- `samples`: Samples for MI computation (default: 1000)
- `title`: Plot title (default: auto-generated or empty)
- `save_path`: Path to save figure (default: nothing, no save)
- `show_values`: Show MI values in each cell (default: true)
- `unit`: Unit for display, :nats or :bits (default: :nats)

# Returns
- `fig`: Figure object
- `mi_matrix`: The mutual information matrix (useful when computed from filename)

# Example
```julia
# From pre-computed matrix
mi_matrix, rho = mutual_information("result.json")
fig, _ = plot_MI(mi_matrix; title="My Circuit")

# Directly from file
fig, mi_matrix = plot_MI("result.json"; conv_step=1000)

# Save the plot
fig, _ = plot_MI(mi_matrix; save_path="mi_plot.pdf")
```
"""
function plot_MI(filename::String; conv_step=100, samples=1000, 
                 title::String="", save_path::Union{String,Nothing}=nothing,
                 show_values::Bool=true, unit::Symbol=:nats)
    # Compute mutual information from file
    mi_matrix, rho = mutual_information(filename; conv_step=conv_step, samples=samples)
    
    # Auto-generate title if not provided
    if isempty(title)
        title = "Mutual Information: $(basename(filename))"
    end
    
    fig = plot_MI(mi_matrix; title=title, save_path=save_path, show_values=show_values, unit=unit)
    return fig, mi_matrix
end

function plot_MI(mi_matrix::Matrix; title::String="", 
                 save_path::Union{String,Nothing}=nothing,
                 show_values::Bool=true, unit::Symbol=:nats)
    n = size(mi_matrix, 1)
    
    # Convert to bits if requested
    display_matrix = unit == :bits ? mi_matrix ./ log(2) : mi_matrix
    unit_label = unit == :bits ? "bits" : "nats"
    
    # Create figure
    fig = Figure(size=(700, 600))
    
    # Main heatmap
    ax = Axis(fig[1, 1],
              xlabel="Qubit j",
              ylabel="Qubit i",
              title=isempty(title) ? "Mutual Information I(i:j)" : title,
              xticks=1:n,
              yticks=1:n,
              yreversed=true,  # So (1,1) is top-left
              aspect=DataAspect())
    
    # Plot heatmap
    max_val = maximum(display_matrix)
    hm = heatmap!(ax, 1:n, 1:n, display_matrix',  # Transpose for correct orientation
                  colormap=:viridis,
                  colorrange=(0, max_val))
    
    # Add colorbar
    Colorbar(fig[1, 2], hm, label="I(i:j) [$unit_label]")
    
    # Add text annotations with values
    if show_values
        for i in 1:n
            for j in 1:n
                val = display_matrix[i, j]
                # Choose text color based on background brightness
                text_color = val > max_val * 0.5 ? :white : :black
                text!(ax, j, i, text=string(round(val, digits=3)),
                      align=(:center, :center),
                      fontsize=10,
                      color=text_color)
            end
        end
    end
    
    # Add summary statistics
    # Extract off-diagonal elements for statistics
    off_diag = [mi_matrix[i,j] for i in 1:n for j in 1:n if i != j]
    if !isempty(off_diag)
        max_mi = maximum(off_diag)
        mean_mi = mean(off_diag)
        max_mi_display = unit == :bits ? max_mi / log(2) : max_mi
        mean_mi_display = unit == :bits ? mean_mi / log(2) : mean_mi
        
        Label(fig[2, 1:2], 
              "Max MI: $(round(max_mi_display, digits=4)) $unit_label  |  Mean MI: $(round(mean_mi_display, digits=4)) $unit_label  |  Qubits: $n",
              fontsize=12, halign=:center)
    end
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

