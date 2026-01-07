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
    
    data = Dict{Symbol, Any}(
        :type => "CircuitOptimizationResult",
        :energy_history => result.energy_history,
        :params => result.params,
        :energy => result.energy,
        :converged => result.converged,
        :Z_samples => result.Z_samples,
        :X_samples => result.X_samples,
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
    
    # Reconstruct result based on type
    if result_type == :circuit
        result = CircuitOptimizationResult(
            Vector{Float64}(get_data(data, :energy_history)),
            Vector{Matrix{ComplexF64}}[],  # Gates not saved to JSON
            Vector{Float64}(get_data(data, :params)),
            Float64(get_data(data, :energy)),
            Vector{Float64}(get(data, "Z_samples", get(data, :Z_samples, Float64[]))),
            Vector{Float64}(get(data, "X_samples", get(data, :X_samples, Float64[]))),
            Bool(get_data(data, :converged))
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
    plot_correlation_heatmap(corr_matrix::Matrix; kwargs...)

Plot spin-spin correlation matrix as a heatmap.

# Arguments
- `corr_matrix`: Correlation matrix ⟨σᵢσⱼ⟩
- `title`: Plot title (default: "Spin Correlation")
- `colormap`: Colormap (default: :RdBu)
- `colorrange`: Color range (default: auto)
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object
"""
function plot_correlation_heatmap(corr_matrix::Matrix;
                                   title::String="Spin Correlation",
                                   colormap=:RdBu,
                                   colorrange=nothing,
                                   save_path::Union{String,Nothing}=nothing)
    n = size(corr_matrix, 1)
    
    fig = Figure(size=(550, 500))
    ax = Axis(fig[1, 1], 
              xlabel="Site i", ylabel="Site j",
              title=title,
              xticks=1:n, yticks=1:n,
              aspect=DataAspect())
    
    cr = isnothing(colorrange) ? extrema(corr_matrix) : colorrange
    hm = heatmap!(ax, 1:n, 1:n, corr_matrix, colormap=colormap, colorrange=cr)
    Colorbar(fig[1, 2], hm, label="⟨σᵢσⱼ⟩")
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    plot_correlation_heatmap(Z_samples::Vector{Float64}, row::Int; kwargs...)

Compute and plot spin correlation heatmap from Z measurement samples.
"""
function plot_correlation_heatmap(Z_samples::Vector{Float64}, row::Int; kwargs...)
    # Build correlation matrix
    n_steps = div(length(Z_samples) - row, row)
    endpoint = row + n_steps * row
    
    corr = zeros(row, row)
    for i in 1:row
        for j in i:row
            Zi = Z_samples[i:row:endpoint]
            Zj = Z_samples[j:row:endpoint]
            corr[i,j] = mean(Zi .* Zj)
            corr[j,i] = corr[i,j]
        end
    end
    
    plot_correlation_heatmap(corr; kwargs...)
end

"""
    plot_correlation_heatmap(result::CircuitOptimizationResult; kwargs...)

Plot correlation heatmap from circuit optimization result.
"""
function plot_correlation_heatmap(result::CircuitOptimizationResult; kwargs...)
    if isempty(result.Z_samples)
        @warn "No Z samples in result"
        return nothing
    end
    
    # Infer row from samples length (this is approximate)
    # For now, just use the samples directly
    @warn "Row size not stored in result, correlation plot may be incorrect"
    return nothing
end

"""
    plot_acf(lags::AbstractVector, acf::AbstractVector; kwargs...)

Plot autocorrelation function with optional exponential fit.

# Arguments
- `lags`: Lag values
- `acf`: ACF values
- `acf_err`: Error bars (optional)
- `fit_params`: (A, ξ) for exponential fit A·exp(-lag/ξ) (optional)
- `title`: Plot title
- `logscale`: Use log scale for y-axis (default: true)
- `save_path`: Path to save figure (optional)

# Returns
- `Figure` object
"""
function plot_acf(lags::AbstractVector, acf::AbstractVector;
                  acf_err::Union{AbstractVector,Nothing}=nothing,
                  fit_params::Union{Tuple{<:Real,<:Real},Nothing}=nothing,
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
    
    # Plot exponential fit
    if !isnothing(fit_params)
        A, ξ = fit_params
        A_pos = abs(A)
        fit_curve = A_pos .* exp.(-lags_vec ./ ξ)
        
        if logscale
            # Clamp to min threshold to avoid log10(0)
            fit_curve = max.(fit_curve, min_threshold)
        end
        
        lines!(ax, lags_vec, fit_curve, linewidth=2, linestyle=:dash, color=:red,
               label="Fit: A·exp(-lag/$(round(ξ, digits=2)))")
    end
    
    axislegend(ax, position=:rt)
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

"""
    compute_acf(data::Vector{Float64}; max_lag::Int=100, normalize::Bool=true) -> (lags, acf, acf_err)

Compute normalized autocorrelation function with bootstrap error estimation.

# Arguments
- `data`: Time series data
- `max_lag`: Maximum lag to compute (default: 100)
- `n_bootstrap`: Number of bootstrap samples for error estimation (default: 100)
- `normalize`: If true, normalize so ACF(0) = 1 (default: true)

# Returns
- `lags`: Lag values (0 to max_lag-1)
- `acf`: Autocorrelation values (normalized if normalize=true)
- `acf_err`: Bootstrap standard errors
"""
function compute_acf(data::Vector{Float64}; max_lag::Int=100, n_bootstrap::Int=100, normalize::Bool=true)
    N = length(data)
    max_lag = min(max_lag, div(N, 2))
    
    μ = mean(data)
    centered = data .- μ
    variance = mean(centered.^2)  # Variance for normalization
    
    acf = zeros(max_lag)
    acf_err = zeros(max_lag)
    
    for k in 1:max_lag
        lag = k - 1
        n_pairs = N - lag
        # Compute ACF without abs() to preserve sign (important for oscillating correlations)
        raw_acf = mean(centered[i] * centered[i + lag] for i in 1:n_pairs)
        acf[k] = normalize ? raw_acf / variance : raw_acf
        
        # Bootstrap error estimation
        block_size = min(50, div(N, 10))
        bootstrap_vals = zeros(n_bootstrap)
        for b in 1:n_bootstrap
            n_blocks = div(N, block_size)
            boot_sample = Float64[]
            for _ in 1:n_blocks
                start = rand(1:N)
                for j in 0:(block_size-1)
                    push!(boot_sample, data[mod1(start + j, N)])
                end
            end
            if length(boot_sample) > lag
                boot_μ = mean(boot_sample)
                boot_centered = boot_sample .- boot_μ
                boot_var = mean(boot_centered.^2)
                n_boot_pairs = min(length(boot_sample) - lag, n_pairs)
                if n_boot_pairs > 0 && boot_var > 0
                    raw_boot = mean(boot_centered[i] * boot_centered[i + lag] for i in 1:n_boot_pairs)
                    bootstrap_vals[b] = normalize ? raw_boot / boot_var : raw_boot
                end
            end
        end
        acf_err[k] = std(bootstrap_vals)
    end
    
    return 0:(max_lag-1), acf, acf_err
end

"""
    fit_acf_exponential(lags, acf; use_log_fit::Bool=true) -> (A, ξ)

Fit ACF to exponential decay A·exp(-lag/ξ).

For normalized ACF (ACF(0)=1), use use_log_fit=true for more robust fitting
by fitting log(|ACF|) vs lag linearly.

# Arguments
- `lags`: Lag values
- `acf`: Autocorrelation values
- `use_log_fit`: If true, use linear fit on log scale (more robust for exponential decay)

# Returns
- `A`: Amplitude (should be ≈1 for normalized ACF)
- `ξ`: Correlation length (= 1/gap from transfer matrix theory)
"""
function fit_acf_exponential(lags::AbstractVector, acf::AbstractVector; use_log_fit::Bool=true)
    abs_acf = abs.(acf)
    
    if use_log_fit
        # Use linear fit on log scale: log|ACF| = log(A) - lag/ξ
        # This is more robust for exponential decay
        # Skip lag=0 if ACF(0)=1 (normalized), and skip very small values
        valid_mask = (abs_acf .> 1e-10) .& (collect(lags) .> 0)
        if sum(valid_mask) < 2
            # Fall back to nonlinear fit if not enough valid points
            valid_mask = abs_acf .> 1e-10
        end
        
        valid_lags = collect(lags)[valid_mask]
        valid_log_acf = log.(abs_acf[valid_mask])
        
        if length(valid_lags) >= 2
            # Linear regression: log|ACF| = a + b*lag, where b = -1/ξ
            X = hcat(ones(length(valid_lags)), valid_lags)
            coeffs = X \ valid_log_acf
            a, b = coeffs
            A = exp(a)
            ξ = -1.0 / b
            
            # Sanity check: ξ should be positive
            if ξ > 0
                return (A, ξ)
            end
        end
    end
    
    # Fallback: nonlinear least squares fit
    model(x, p) = p[1] .* exp.(-x ./ p[2])
    
    A_init = abs_acf[1]
    # Estimate ξ from where |ACF| drops to 1/e
    ξ_init = 10.0
    for i in 2:length(abs_acf)
        if abs_acf[i] < abs_acf[1] / ℯ
            ξ_init = Float64(i - 1)
            break
        end
    end
    ξ_init = clamp(ξ_init, 0.1, length(lags) * 2.0)
    
    try
        fit = curve_fit(model, collect(lags), collect(abs_acf), [A_init, ξ_init])
        A, ξ = coef(fit)
        return (abs(A), abs(ξ))
    catch
        # If fit fails, return rough estimate
        return (A_init, ξ_init)
    end
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