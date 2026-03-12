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
        :params => result.final_params,
        :energy => result.final_cost,
        :converged => result.converged,
        :Z_samples => result.final_Z_samples,
        :X_samples => result.final_X_samples,
        :Y_samples => result.final_Y_samples,
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
        Y_samples_data = get(data, "Y_samples", get(data, :Y_samples, nothing))

        energy_history = get_data(data, :energy_history)
        params = get_data(data, :params)
        energy = get_data(data, :energy)
        converged = get_data(data, :converged)

        # Convert samples to vectors (flatten if matrix)
        Z_samples = samples_to_matrix(Z_samples_data, input_args)
        X_samples = samples_to_matrix(X_samples_data, input_args)
        Z_samples_vec = Z_samples isa AbstractMatrix ? vec(collect(Z_samples)) : Vector{Float64}(collect(Z_samples))
        X_samples_vec = X_samples isa AbstractMatrix ? vec(collect(X_samples)) : Vector{Float64}(collect(X_samples))
        Y_samples_vec = Y_samples_data === nothing ? Float64[] : Vector{Float64}(collect(Y_samples_data))

        result = CircuitOptimizationResult(
            Vector{Float64}(energy_history === nothing ? Float64[] : energy_history),
            Vector{Matrix{ComplexF64}}[],  # Gates not saved to JSON
            Vector{Float64}(params === nothing ? Float64[] : params),
            Float64(energy === nothing ? 0.0 : energy),
            Z_samples_vec,
            X_samples_vec,
            Y_samples_vec,
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

# ============================================================================
# use the optimized parameters and resample
# ============================================================================

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
# ============================================================================
# fit_acf
# ============================================================================

function fit_acf(lags::AbstractVector, acf::AbstractVector;
                 fit_range::Union{Tuple{Int,Int},Nothing}=nothing,
                 include_zero::Bool=false)
    lags_vec = collect(Float64, lags)
    acf_vec = collect(Float64, acf)

    if include_zero
        mask = trues(length(lags_vec))
    else
        mask = lags_vec .> 0
    end

    if !isnothing(fit_range)
        start_lag, end_lag = fit_range
        range_mask = (lags_vec .>= start_lag) .& (lags_vec .<= end_lag)
        mask = mask .& range_mask
    end

    fit_lags = lags_vec[mask]
    fit_acf_vals = acf_vec[mask]

    if length(fit_lags) < 2
        error("Not enough valid data points for fitting (need at least 2)")
    end

    # Determine the sign of the correlation from the mean of first few points
    sign_sample = fit_acf_vals[1:min(3, length(fit_acf_vals))]
    correlation_sign = sign(mean(sign_sample))
    if iszero(correlation_sign)
        correlation_sign = 1.0
    end

    fit_acf_abs = abs.(fit_acf_vals)

    # Log-space OLS: fit log|C(r)| = log(A) - r/ξ
    # This weights all points equally on the log scale so the fitted line
    # visually matches the data on log-scale plots (unlike linear-space
    # least squares, which is dominated by large small-r values).
    valid_idx = fit_acf_abs .> 1e-15
    A_abs = fit_acf_abs[1]
    ξ = fit_lags[end] / 2.0

    if sum(valid_idx) >= 2
        log_abs    = log.(fit_acf_abs[valid_idx])
        valid_lags = fit_lags[valid_idx]
        # Design matrix: log|C| = b - r/ξ  →  [1, -r] * [b, 1/ξ]ᵀ
        X = hcat(ones(length(valid_lags)), -valid_lags)
        coeffs    = X \ log_abs
        log_A_abs = coeffs[1]
        inv_xi    = coeffs[2]
        if inv_xi > 1e-10
            A_abs = exp(log_A_abs)
            ξ     = clamp(1.0 / inv_xi, 0.01, Inf)
        else
            # Fallback: two-point slope estimate
            ξ     = clamp(-(valid_lags[end] - valid_lags[1]) /
                           (log_abs[end]   - log_abs[1]),   0.01, Inf)
            A_abs = exp(log_abs[1] + valid_lags[1] / ξ)
        end
    end

    A  = correlation_sign * A_abs
    λ₂ = exp(-1.0 / ξ)
    return (A=A, ξ=ξ, λ₂=λ₂, fit_lags=fit_lags,
            model=(r) -> A .* exp.(-r ./ ξ))
end

# ============================================================================
# fit_acf_oscillatory
# ============================================================================

function fit_acf_oscillatory(lags::AbstractVector, acf::AbstractVector;
                              fit_range::Union{Tuple{Int,Int},Nothing}=nothing,
                              include_zero::Bool=false)
    lags_vec = collect(Float64, lags)
    acf_vec = collect(Float64, real.(acf))

    if include_zero
        mask = trues(length(lags_vec))
    else
        mask = lags_vec .> 0
    end

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

    model(r, p) = p[1] .* exp.(-r ./ p[2]) .* cos.(p[3] .* r .+ p[4])

    A_init = abs(fit_acf_vals[1])
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

    sign_changes = findall(diff(sign.(fit_acf_vals)) .!= 0)
    if length(sign_changes) >= 2
        avg_period = 2 * mean(diff(fit_lags[sign_changes]))
        k_init = 2π / avg_period
    else
        k_init = 0.0
    end

    φ_init = fit_acf_vals[1] < 0 ? π : 0.0
    p0 = [A_init, ξ_init, k_init, φ_init]
    lower = [0.0, 0.01, -π, -2π]
    upper = [Inf, Inf, π, 2π]

    try
        fit_result = curve_fit(model, fit_lags, fit_acf_vals, p0, lower=lower, upper=upper)
        A, ξ, k, φ = coef(fit_result)
        λ₂_magnitude = exp(-1.0/ξ)
        λ₂_phase = k
        return (A=A, ξ=ξ, k=k, phase=φ,
                λ₂_magnitude=λ₂_magnitude, λ₂_phase=λ₂_phase,
                fit_lags=fit_lags,
                model=(r) -> model(r, [A, ξ, k, φ]))
    catch e
        @warn "Oscillatory fit failed: $e. Falling back to simple exponential fit."
        simple_fit = fit_acf(lags, abs.(acf); fit_range=fit_range, include_zero=include_zero)
        return (A=simple_fit.A, ξ=simple_fit.ξ, k=0.0, phase=0.0,
                λ₂_magnitude=simple_fit.λ₂, λ₂_phase=0.0,
                fit_lags=simple_fit.fit_lags,
                model=simple_fit.model)
    end
end

# ============================================================================
# plot_eigenvalue_spectrum
# ============================================================================

function plot_eigenvalue_spectrum(eigenvalues_raw::AbstractVector{<:Complex};
                                   title::String="",
                                   save_path::Union{String,Nothing}=nothing,
                                   show_gap::Bool=true)
    sorted_indices = sortperm(abs.(eigenvalues_raw), rev=true)
    sorted_raw = eigenvalues_raw[sorted_indices]
    sorted_eigs = abs.(sorted_raw)
    n = length(sorted_eigs)

    λ₁ = sorted_eigs[1]
    λ₂ = sorted_eigs[2]
    gap = -log(λ₂)
    ξ = 1 / gap

    fig = Figure(size=(1200, 500))

    ax1 = Axis(fig[1, 1],
               xlabel="Eigenvalue index (sorted)",
               ylabel="Eigenvalue magnitude |λ|",
               title=isempty(title) ? "Eigenvalue Spectrum" : title)

    colors = [λ > 0.99 ? :red : (λ > 0.9 ? :orange : :steelblue) for λ in sorted_eigs]
    barplot!(ax1, 1:n, sorted_eigs, color=colors, strokewidth=0.5, strokecolor=:black)
    hlines!(ax1, [1.0], color=:black, linestyle=:dash, linewidth=1.5, label="λ=1")
    hlines!(ax1, [0.99], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="λ=0.99")
    scatter!(ax1, [1], [λ₁], markersize=15, color=:green, marker=:star5, label="λ₁=$(round(λ₁, digits=4))")
    scatter!(ax1, [2], [λ₂], markersize=12, color=:purple, marker=:diamond, label="λ₂=$(round(λ₂, digits=4))")
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2],
               xlabel="Re(λ)",
               ylabel="Im(λ)",
               title="Eigenvalues in Complex Plane",
               aspect=DataAspect())

    θ = range(0, 2π, length=100)
    lines!(ax2, cos.(θ), sin.(θ), color=:black, linestyle=:dash, linewidth=1.5, label="Unit circle")
    lines!(ax2, 0.99 .* cos.(θ), 0.99 .* sin.(θ), color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="|λ|=0.99")

    re_parts = real.(sorted_raw)
    im_parts = imag.(sorted_raw)
    colors_scatter = [abs(λ) > 0.99 ? :red : (abs(λ) > 0.9 ? :orange : :steelblue) for λ in sorted_raw]
    scatter!(ax2, re_parts, im_parts, color=colors_scatter, markersize=8, strokewidth=0.5, strokecolor=:black)
    scatter!(ax2, [real(sorted_raw[1])], [imag(sorted_raw[1])], markersize=15, color=:green, marker=:star5, label="λ₁")
    scatter!(ax2, [real(sorted_raw[2])], [imag(sorted_raw[2])], markersize=12, color=:purple, marker=:diamond, label="λ₂")
    axislegend(ax2, position=:lt)

    if show_gap
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


# ============================================================================
# plot_acf
# ============================================================================

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

    fig_height = show_lambda_eff ? 600 : 400
    fig = Figure(size=(600, fig_height))

    abs_acf = collect(acf)
    lags_vec = collect(lags)

    min_threshold = max(maximum(abs_acf) * 1e-10, 1e-15)

    if logscale
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

    if !isnothing(acf_err) && !logscale
        errorbars!(ax, lags_vec, plot_y, collect(acf_err), color=:gray)
    end
    scatter!(ax, lags_vec, plot_y, markersize=8, label="Data", color=:steelblue)

    if isnothing(fit_params) && !isnothing(fit_range)
        if fit_oscillatory
            fit_params = fit_acf_oscillatory(lags, acf; fit_range=fit_range)
        else
            fit_params = fit_acf(lags, acf; fit_range=fit_range)
        end
    end

    if !isnothing(fit_params)
        A = fit_params.A
        ξ = fit_params.ξ
        has_oscillation = haskey(fit_params, :k) && abs(fit_params.k) > 1e-6

        if has_oscillation
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
            λ₂ = haskey(fit_params, :λ₂) ? fit_params.λ₂ : fit_params.λ₂_magnitude
            fit_curve = A .* exp.(-lags_vec ./ ξ)
            label_text = "Fit: ξ=$(round(ξ, digits=2)), λ₂=$(round(λ₂, digits=4))"
            if logscale
                fit_curve = max.(abs.(fit_curve), min_threshold)
            end
        end

        lines!(ax, lags_vec, fit_curve, linewidth=2, linestyle=:dash, color=:red, label=label_text)
    end

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

    if show_lambda_eff && length(acf) > 2
        ax2 = Axis(fig[2, 1],
                   xlabel="Lag r", ylabel="λ_eff(r) = C(r+1)/C(r)",
                   title="Effective Eigenvalue")

        acf_vals = abs.(collect(acf))
        lambda_eff_lags = collect(lags)[1:end-1]
        lambda_eff = acf_vals[2:end] ./ acf_vals[1:end-1]

        valid_mask = isfinite.(lambda_eff) .& (lambda_eff .> 0) .& (lambda_eff .< 10)
        if any(valid_mask)
            scatter!(ax2, lambda_eff_lags[valid_mask], lambda_eff[valid_mask],
                    markersize=6, label="λ_eff(r) from data", color=:steelblue)
        end

        if !isnothing(lambda_eff_theory)
            theory_lags, theory_lambda_eff = lambda_eff_theory
            theory_lags_vec = collect(theory_lags)
            theory_vals = abs.(collect(theory_lambda_eff))
            valid_theory = isfinite.(theory_vals) .& (theory_vals .> 0) .& (theory_vals .< 10)
            if any(valid_theory)
                lines!(ax2, theory_lags_vec[valid_theory], theory_vals[valid_theory],
                      linewidth=2.5, linestyle=:solid, color=:purple,
                      label="λ_eff (all modes)")
            end
        end

        if !isnothing(lambda_eff_dominant)
            dom_lags, dom_lambda_eff = lambda_eff_dominant
            dom_lags_vec = collect(dom_lags)
            dom_vals = abs.(collect(dom_lambda_eff))
            valid_dom = isfinite.(dom_vals) .& (dom_vals .> 0) .& (dom_vals .< 10)
            if any(valid_dom)
                lines!(ax2, dom_lags_vec[valid_dom], dom_vals[valid_dom],
                      linewidth=2.5, linestyle=:dashdot, color=:orange,
                      label="λ_eff (dominant modes)")
            end
        end

        if !isnothing(lambda_theory)
            hlines!(ax2, [lambda_theory], color=:green, linestyle=:dash, linewidth=2,
                   label="λ_slow (contributing) = $(round(lambda_theory, digits=4))")
        end

        if !isnothing(fit_params)
            λ_fit = haskey(fit_params, :λ₂) ? fit_params.λ₂ :
                    (haskey(fit_params, :λ₂_magnitude) ? fit_params.λ₂_magnitude : exp(-1/fit_params.ξ))
            hlines!(ax2, [λ_fit], color=:red, linestyle=:dot, linewidth=2,
                   label="λ₂ (single exp fit) = $(round(λ_fit, digits=4))")
        end

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

# ============================================================================
# plot_training_history (two methods)
# ============================================================================

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

    ref_energy = reference
    if !isnothing(pepskit_results_file) && !isnothing(g) && isfile(pepskit_results_file)
        pepskit_data = open(pepskit_results_file, "r") do io
            JSON3.read(io, Dict)
        end
        g_values = pepskit_data["g_values"]
        energies = pepskit_data["energies"]
        idx = argmin(abs.(g_values .- g))
        if abs(g_values[idx] - g) < 1e-6
            ref_energy = energies[idx]
        else
            @warn "g=$g not found in pepskit results, using closest value g=$(g_values[idx])"
            ref_energy = energies[idx]
        end
    end

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

function plot_training_history(result::Union{CircuitOptimizationResult, ExactOptimizationResult, ManifoldOptimizationResult}; kwargs...)
    n = length(result.energy_history)
    plot_training_history(1:n, result.energy_history; ylabel="Energy", kwargs...)
end

# ============================================================================
# plot_expectation_values (3 methods)
# ============================================================================

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

    labels = String[]
    values = Float64[]

    !isnothing(energy)   && (push!(labels, "E");      push!(values, energy))
    !isnothing(X)        && (push!(labels, "⟨X⟩");   push!(values, X))
    !isnothing(Z)        && (push!(labels, "⟨Z⟩");   push!(values, Z))
    !isnothing(ZZ_vert)  && (push!(labels, "⟨ZZ⟩ᵥ"); push!(values, ZZ_vert))
    !isnothing(ZZ_horiz) && (push!(labels, "⟨ZZ⟩ₕ"); push!(values, ZZ_horiz))

    if isempty(values)
        @warn "No expectation values provided"
        return nothing
    end

    plot_title = title
    if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
        plot_title = "Expectation Values: row=$row, g=$g, nqubits=$nqubits"
    elseif !isnothing(g) && !isnothing(row)
        plot_title = "Expectation Values: row=$row, g=$g"
    elseif !isnothing(g)
        plot_title = "Expectation Values: g=$g"
    end

    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1],
              xlabel="Observable", ylabel="Value",
              title=plot_title,
              xticks=(1:length(labels), labels))

    colors = [v >= 0 ? :steelblue : :coral for v in values]
    barplot!(ax, 1:length(values), values, color=colors, strokewidth=1, strokecolor=:black)

    for (i, v) in enumerate(values)
        offset = v >= 0 ? 0.05 : -0.05
        align = v >= 0 ? (:center, :bottom) : (:center, :top)
        text!(ax, i, v + offset; text=string(round(v, digits=4)), align=align, fontsize=12)
    end

    hlines!(ax, [0], color=:black, linewidth=1)

    ymin = min(minimum(values), 0) * 1.2
    ymax = max(maximum(values), 0) * 1.2
    if ymin == ymax; ymin, ymax = -1, 1; end
    ylims!(ax, ymin - 0.1, ymax + 0.1)

    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    return fig
end

function plot_expectation_values(result::ExactOptimizationResult; kwargs...)
    plot_expectation_values(;
        energy=result.energy,
        X=result.X_expectation,
        ZZ_vert=result.ZZ_vertical,
        ZZ_horiz=result.ZZ_horizontal,
        kwargs...
    )
end

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

    Z_samples = result.final_Z_samples
    X_samples = result.final_X_samples
    if !isnothing(datafile)
        if isfile(datafile)
            # Adaptive sampling: reduce samples for large nqubits (expensive to simulate)
            # nqubits=3: 100K samples (~fast)
            # nqubits=5: 50K samples (~moderate, 2^5=32 dim state)
            # nqubits=7: 20K samples (~slow, 2^7=128 dim state)
            adaptive_samples = if !isnothing(nqubits)
                if nqubits <= 3
                    1000000
                elseif nqubits == 5
                    1000000
                else
                    20000
                end
            else
                100000  # default
            end

            resampled = resample_circuit(datafile; conv_step=1000, samples=adaptive_samples, measure_first=nothing)
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
    Z_stderr = isempty(Z_samples) ? nothing : std(Z_samples) / sqrt(length(Z_samples))
    X_stderr = isempty(X_samples) ? nothing : std(X_samples) / sqrt(length(X_samples))

    ZZ_vert_sample = nothing; ZZ_horiz_sample = nothing; energy_sample = nothing
    ZZ_vert_stderr = nothing; ZZ_horiz_stderr = nothing; energy_stderr = nothing

    N = length(Z_samples)
    if !isnothing(row) && row > 1 && N > 1
        ZZ_vert_pairs = [Z_samples[i] * Z_samples[i+1] for i in 1:N-1 if i % row != 0]
        ZZ_vert_sample = mean(ZZ_vert_pairs)
        ZZ_vert_stderr = std(ZZ_vert_pairs) / sqrt(length(ZZ_vert_pairs))
    end
    if !isnothing(row) && N > row
        ZZ_horiz_pairs = [Z_samples[i] * Z_samples[i+row] for i in 1:N-row]
        ZZ_horiz_sample = mean(ZZ_horiz_pairs)
        ZZ_horiz_stderr = std(ZZ_horiz_pairs) / sqrt(length(ZZ_horiz_pairs))
    end
    if !isnothing(g) && !isnothing(row) && N > 0 && !isempty(X_samples)
        energy_sample = compute_energy(X_samples, Z_samples, g, J, row)
        if row == 1
            ZZ_stderr_total = isnothing(ZZ_horiz_stderr) ? 0.0 : ZZ_horiz_stderr
        else
            ZZ_vert_se = isnothing(ZZ_vert_stderr) ? 0.0 : ZZ_vert_stderr
            ZZ_horiz_se = isnothing(ZZ_horiz_stderr) ? 0.0 : ZZ_horiz_stderr
            ZZ_stderr_total = sqrt(ZZ_vert_se^2 + ZZ_horiz_se^2)
        end
        X_se = isnothing(X_stderr) ? 0.0 : X_stderr
        energy_stderr = sqrt(g^2 * X_se^2 + J^2 * ZZ_stderr_total^2)
    end

    X_exact = nothing; Z_exact = nothing
    ZZ_vert_exact = nothing; ZZ_horiz_exact = nothing; energy_exact = nothing

    # Skip exact computation for large systems (nqubits >= 5) as it's very slow
    # correlation_function becomes expensive for large Hilbert spaces
    can_compute_exact = !isnothing(row) && !isnothing(p) && !isnothing(nqubits) && !isempty(result.final_params)
    skip_exact_for_large_system = !isnothing(nqubits) && nqubits >= 5

    if can_compute_exact && !skip_exact_for_large_system
        virtual_qubits = (nqubits - 1) ÷ 2
        gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=true)
        X_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, :X; position=i)) for i in 1:row)
        Z_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, :Z; position=i)) for i in 1:row)
        if row > 1
            ZZ_vert_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, Dict(i => :Z, i+1 => :Z))) for i in 1:row-1)
        end
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

    plot_title = title
    if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
        plot_title = "Expectation Values: row=$row, g=$g, nqubits=$nqubits"
    elseif !isnothing(g) && !isnothing(row)
        plot_title = "Expectation Values: row=$row, g=$g"
    end

    labels = String[]; sample_values = Float64[]; exact_values = Float64[]; sample_errors = Float64[]

    if !isnothing(energy_sample)
        push!(labels, "E")
        push!(sample_values, energy_sample)
        push!(exact_values, isnothing(energy_exact) ? NaN : energy_exact)
        push!(sample_errors, isnothing(energy_stderr) ? 0.0 : energy_stderr)
    end
    if !isnothing(X_sample) || !isnothing(X_exact)
        push!(labels, "⟨X⟩")
        push!(sample_values, isnothing(X_sample) ? NaN : X_sample)
        push!(exact_values, isnothing(X_exact) ? NaN : X_exact)
        push!(sample_errors, isnothing(X_stderr) ? 0.0 : X_stderr)
    end
    if !isnothing(Z_sample) || !isnothing(Z_exact)
        push!(labels, "⟨Z⟩")
        push!(sample_values, isnothing(Z_sample) ? NaN : Z_sample)
        push!(exact_values, isnothing(Z_exact) ? NaN : Z_exact)
        push!(sample_errors, isnothing(Z_stderr) ? 0.0 : Z_stderr)
    end
    if !isnothing(ZZ_vert_sample) || !isnothing(ZZ_vert_exact)
        push!(labels, "⟨ZZ⟩ᵥ")
        push!(sample_values, isnothing(ZZ_vert_sample) ? NaN : ZZ_vert_sample)
        push!(exact_values, isnothing(ZZ_vert_exact) ? NaN : ZZ_vert_exact)
        push!(sample_errors, isnothing(ZZ_vert_stderr) ? 0.0 : ZZ_vert_stderr)
    end
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

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              xlabel="Observable", ylabel="Value",
              title=plot_title,
              xticks=(1:length(labels), labels))

    n = length(labels)
    bar_width = 0.35

    positions_sample = collect(1:n) .- bar_width/2
    valid_sample = .!isnan.(sample_values)
    if any(valid_sample)
        barplot!(ax, positions_sample[valid_sample], sample_values[valid_sample],
                 width=bar_width, color=:steelblue, strokewidth=1, strokecolor=:black,
                 label="Sample")
        valid_errors = sample_errors[valid_sample]
        if any(valid_errors .> 0)
            errorbars!(ax, positions_sample[valid_sample], sample_values[valid_sample],
                      valid_errors, color=:black, linewidth=1.5, whiskerwidth=8)
        end
        for (i, (pos, val, err)) in enumerate(zip(positions_sample, sample_values, sample_errors))
            if !isnan(val)
                offset = val >= 0 ? (0.03 + err) : (-0.03 - err)
                align = val >= 0 ? (:center, :bottom) : (:center, :top)
                text!(ax, pos, val + offset; text=string(round(val, digits=3)), align=align, fontsize=9)
            end
        end
    end

    if can_compute_exact
        positions_exact = collect(1:n) .+ bar_width/2
        valid_exact = .!isnan.(exact_values)
        if any(valid_exact)
            barplot!(ax, positions_exact[valid_exact], exact_values[valid_exact],
                     width=bar_width, color=:coral, strokewidth=1, strokecolor=:black,
                     label="Exact")
            for (i, (pos, val)) in enumerate(zip(positions_exact, exact_values))
                if !isnan(val)
                    offset = val >= 0 ? 0.03 : -0.03
                    align = val >= 0 ? (:center, :bottom) : (:center, :top)
                    text!(ax, pos, val + offset; text=string(round(val, digits=3)), align=align, fontsize=9)
                end
            end
        end
    end

    hlines!(ax, [0], color=:black, linewidth=1)
    axislegend(ax, position=:rb)

    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    return fig
end

# ============================================================================
# plot_variance_vs_samples
# ============================================================================

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

    if !isnothing(errors)
        errorbars!(ax, collect(sample_sizes), collect(variances), collect(errors), color=:gray)
    end
    scatter!(ax, collect(sample_sizes), collect(variances), markersize=10, label="Data")

    if fit_scaling && length(sample_sizes) > 1
        log_N = log.(sample_sizes)
        log_var = log.(variances)
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

# ============================================================================
# plot_correlation_function
# ============================================================================

function plot_correlation_function(filename::String;
                                   max_separation::Int=20,
                                   conv_step::Int=1000,
                                   samples::Int=100000,
                                   save_path::Union{String,Nothing}=nothing)

    result, input_args = load_result(filename)

    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    g = get(input_args, :g, NaN)
    virtual_qubits = (nqubits - 1) ÷ 2
    share_params = get(input_args, :share_params, true)

    params = result.final_params
    gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)

    println("=== Correlation Function Analysis ===")
    println("File: ", basename(filename))
    println("Configuration: g=$g, row=$row, nqubits=$nqubits")

    println("\nComputing exact correlations (transfer matrix)...")
    println("Averaging over all positions 1 to $row...")

    # Compute correlations for each position and average
    separations = collect(1:max_separation)
    exact_full_vals = zeros(Float64, max_separation)
    exact_connected_vals = zeros(Float64, max_separation)

    for pos in 1:row
        exact_full_pos = correlation_function(gates, row, virtual_qubits, :Z, 1:max_separation;
                                              position=pos, connected=false)
        exact_connected_pos = correlation_function(gates, row, virtual_qubits, :Z, 1:max_separation;
                                                   position=pos, connected=true)

        for r in separations
            exact_full_vals[r] += real(exact_full_pos[r])
            exact_connected_vals[r] += real(exact_connected_pos[r])
        end
    end

    # Average over positions
    exact_full_vals ./= row
    exact_connected_vals ./= row

    _, gap, _, _ = compute_transfer_spectrum(gates, row, nqubits)
    correlation_length = 1 / gap
    println("Correlation length ξ = $(round(correlation_length, digits=2))")

    println("\nGenerating samples (conv_step=$conv_step, samples=$samples)...")
    rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits;
                                                        conv_step=conv_step,
                                                        samples=samples,
                                                        measure_first=:Z)

    Z_vec = Z_samples[conv_step+1:end]

    lags, acf, acf_err, corr_full, corr_err, corr_connected, corr_connected_err = compute_acf(
        reshape(Float64.(Z_vec), 1, :); max_lag=max_separation+1, row=row
    )

    @show corr_err
    @show corr_connected_err

    sample_full = corr_full
    sample_full_err = corr_err
    sample_connected = corr_connected
    sample_connected_err = corr_connected_err

    n_sample_seps = min(length(sample_full) - 1, max_separation)
    sample_seps = 1:n_sample_seps
    sample_full_vals = sample_full[2:n_sample_seps+1]
    sample_full_err_vals = sample_full_err[2:n_sample_seps+1]
    sample_connected_vals = sample_connected[2:n_sample_seps+1]
    sample_connected_err_vals = sample_connected_err[2:n_sample_seps+1]

    println("Sampling std errors range (full): $(round(minimum(sample_full_err_vals), sigdigits=2)) - $(round(maximum(sample_full_err_vals), sigdigits=2))")
    println("Sampling std errors range (connected): $(round(minimum(sample_connected_err_vals), sigdigits=2)) - $(round(maximum(sample_connected_err_vals), sigdigits=2))")

    common_seps = min(length(exact_full_vals), length(sample_full_vals))
    error_full = abs.(exact_full_vals[1:common_seps] .- sample_full_vals[1:common_seps])
    error_connected = abs.(exact_connected_vals[1:common_seps] .- sample_connected_vals[1:common_seps])
    mean_error_full = mean(error_full)
    mean_error_connected = mean(error_connected)
    println("Mean |exact - sample| error (full): $(round(mean_error_full, digits=6))")
    println("Mean |exact - sample| error (connected): $(round(mean_error_connected, digits=6))")

    fig = Figure(size=(800, 700))
    min_val = 1e-15

    title_str = "Correlation Function: g=$g, row=$row, nqubits=$nqubits, ξ=$(round(correlation_length, digits=2))"
    Label(fig[0, 1], title_str, fontsize=16, font=:bold)

    ax1 = Axis(fig[1, 1],
               xlabel="Separation r",
               ylabel="|⟨Z_i Z_{i+r}⟩|",
               title="Full Correlation",
               yscale=log10)

    exact_full_abs = max.(abs.(exact_full_vals), min_val)
    sample_full_abs = max.(abs.(sample_full_vals), min_val)
    error_full_plot = max.(error_full, min_val)

    lines!(ax1, separations, exact_full_abs, label="Exact contraction", color=:blue, linewidth=2)
    scatter!(ax1, separations, exact_full_abs, color=:blue, markersize=8)

    err_low = min.(sample_full_err_vals, sample_full_abs .- min_val)
    err_high = sample_full_err_vals
    scatter!(ax1, collect(sample_seps), sample_full_abs,
             label="Sampling ± std err", color=:red, markersize=8, marker=:diamond)
    errorbars!(ax1, collect(sample_seps), sample_full_abs, err_low, err_high,
               color=:red, whiskerwidth=6)
    axislegend(ax1, position=:rt)

    ax2 = Axis(fig[2, 1],
               xlabel="Separation r",
               ylabel="|⟨Z_i Z_{i+r}⟩_c|",
               title="Connected Correlation",
               yscale=log10)

    exact_connected_abs = max.(abs.(exact_connected_vals), min_val)
    sample_connected_abs = max.(abs.(sample_connected_vals), min_val)

    # Plot exact data first
    lines!(ax2, separations, exact_connected_abs, label="Exact contraction", color=:blue, linewidth=2)
    scatter!(ax2, separations, exact_connected_abs, color=:blue, markersize=8)

    err_low_conn = min.(sample_connected_err_vals, sample_connected_abs .- min_val)
    err_high_conn = sample_connected_err_vals
    scatter!(ax2, collect(sample_seps), sample_connected_abs,
             label="Sampling ± std err", color=:red, markersize=8, marker=:diamond)
    errorbars!(ax2, collect(sample_seps), sample_connected_abs, err_low_conn, err_high_conn,
               color=:red, whiskerwidth=6)

    # Fit connected correlation to extract correlation length
    println("\nFitting connected correlation to A*exp(-r/ξ)...")
    println("Data to fit:")
    for (i, r) in enumerate(separations[1:min(5, length(separations))])
        println("  r=$r: $(exact_connected_vals[i]) -> |val|=$(abs(exact_connected_vals[i]))")
    end

    ξ_fitted = nothing
    A_fitted = nothing
    try
        # Pass raw values to fit_acf (not absolute values)
        # fit_acf will handle absolute values internally
        fit_params = fit_acf(separations, exact_connected_vals; include_zero=false)
        ξ_fitted = fit_params.ξ
        A_fitted = fit_params.A
        println("Fitted correlation length ξ = $(round(ξ_fitted, digits=3))")
        println("Fitted amplitude A = $(round(A_fitted, digits=4))")
        println("Transfer matrix ξ = $(round(correlation_length, digits=3))")
        println("Ratio ξ_fitted/ξ_transfer = $(round(ξ_fitted/correlation_length, digits=3))")

        # Check fit quality
        r_check = separations[1:min(10, length(separations))]
        fitted_check = abs(A_fitted) .* exp.(-r_check ./ ξ_fitted)
        data_check = abs.(exact_connected_vals[1:length(r_check)])
        println("Fit check (first $(length(r_check)) points):")
        for (i, r) in enumerate(r_check)
            rel_err = abs(fitted_check[i] - data_check[i]) / data_check[i] * 100
            println("  r=$r: data=$(data_check[i]), fit=$(fitted_check[i]), rel_err=$(round(rel_err, digits=1))%")
        end

        # Plot fitted curve ON TOP of data with high visibility
        r_fit = range(1, max_separation, length=100)
        fitted_curve = abs(A_fitted) .* exp.(-r_fit ./ ξ_fitted)
        fitted_curve_plot = max.(fitted_curve, min_val)
        lines!(ax2, r_fit, fitted_curve_plot,
               label="Fit: |A|*exp(-r/$(round(ξ_fitted, digits=2)))",
               color=:green, linewidth=3, linestyle=:dash)

        # Update title with fitted ξ
        ax2.title = "Connected Correlation (ξ_fit=$(round(ξ_fitted, digits=2)), ξ_TM=$(round(correlation_length, digits=2)))"
    catch e
        @warn "Fitting failed: $e"
        println("Skipping fit overlay")
        println("Exception: ", e)
        println("Stacktrace: ")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end

    axislegend(ax2, position=:rt)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end

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
        correlation_length_fitted = ξ_fitted,
        fitted_amplitude = A_fitted,
        g = g, row = row, nqubits = nqubits
    )

    return fig, data
end

# ============================================================================
# Multi-g comparison plots
# ============================================================================

"""
    plot_energy_error_vs_g(data_dir::String, g_values::Vector{Float64};
                           J=1.0, row=3, nqubits=5, p=3,
                           pepskit_file::Union{String,Nothing}=nothing,
                           dmrg_file::Union{String,Nothing}=nothing,
                           save_path::Union{String,Nothing}=nothing)

Plot energy error (exact contraction - PEPSKit reference) for different g values.

# Arguments
- `data_dir`: Directory containing result JSON files
- `g_values`: Vector of g values to plot
- `J`: Coupling strength (default: 1.0)
- `row`: Number of rows (default: 3)
- `nqubits`: Number of qubits (default: 5)
- `p`: Circuit depth (default: 3)
- `pepskit_file`: Path to PEPSKit reference results JSON (optional)
- `dmrg_file`: Path to DMRG results JSON (optional)
- `save_path`: Path to save figure (optional)

# Returns
- `fig`: Makie Figure object
- `data`: NamedTuple with (g_values, energies_exact, energies_ref, energies_dmrg, errors)

# Example
```julia
g_vals = [1.0, 2.0, 3.0, 4.0]
fig, data = plot_energy_error_vs_g("project/results", g_vals;
                                   pepskit_file="project/results/pepskit_results_D=2.json",
                                   dmrg_file="project/results/dmrg_tfim_100x3.json",
                                   save_path="energy_error_vs_g.pdf")
```
"""
function plot_energy_error_vs_g(data_dir::String, g_values::Vector{Float64};
                                J=1.0, row=3, nqubits=3, p=3,
                                pepskit_file::Union{String,Nothing}=nothing,
                                dmrg_file::Union{String,Nothing}=nothing,
                                save_path::Union{String,Nothing}=nothing)

    println("="^70)
    println("Energy Error vs g Analysis")
    println("="^70)

    # Load PEPSKit reference energies if provided
    ref_energies = Dict{Float64, Float64}()
    if !isnothing(pepskit_file) && isfile(pepskit_file)
        println("Loading PEPSKit reference from: $(basename(pepskit_file))")
        ref_data = open(pepskit_file, "r") do io
            JSON3.read(io, Dict)
        end

        # Extract reference energies for each g (parallel arrays format)
        g_key   = haskey(ref_data, "g_values") ? "g_values" : (haskey(ref_data, :g_values) ? :g_values : nothing)
        e_key   = haskey(ref_data, "energies")  ? "energies"  : (haskey(ref_data, :energies)  ? :energies  : nothing)
        if !isnothing(g_key) && !isnothing(e_key)
            g_refs = Float64.(ref_data[g_key])
            e_refs = Float64.(ref_data[e_key])
            for (g_ref, energy_ref) in zip(g_refs, e_refs)
                ref_energies[g_ref] = energy_ref
            end
        else
            @warn "PEPSKit JSON does not contain expected 'g_values'/'energies' arrays; skipping."
        end
        println("Loaded $(length(ref_energies)) reference energies")
    else
        println("No PEPSKit reference file provided - will only show exact energies")
    end

    # Load DMRG energies if provided
    dmrg_energies = Dict{Float64, Float64}()
    if !isnothing(dmrg_file) && isfile(dmrg_file)
        println("Loading DMRG results from: $(basename(dmrg_file))")
        dmrg_data = open(dmrg_file, "r") do io
            JSON3.read(io, Dict)
        end

        # Extract DMRG energies for each g
        g_key = haskey(dmrg_data, "scan_values") ? "scan_values" : (haskey(dmrg_data, :scan_values) ? :scan_values : nothing)
        e_key = haskey(dmrg_data, "energies_per_site") ? "energies_per_site" : (haskey(dmrg_data, :energies_per_site) ? :energies_per_site : nothing)
        if !isnothing(g_key) && !isnothing(e_key)
            g_dmrg = Float64.(dmrg_data[g_key])
            e_dmrg = Float64.(dmrg_data[e_key])
            for (g_val, energy_val) in zip(g_dmrg, e_dmrg)
                dmrg_energies[g_val] = energy_val
            end
        else
            @warn "DMRG JSON does not contain expected 'g_values'/'energies_per_site' arrays; skipping."
        end
        println("Loaded $(length(dmrg_energies)) DMRG energies")
    else
        println("No DMRG file provided")
    end

    # Load circuit optimization results and compute exact energies
    energies_exact = Float64[]
    energies_ref = Float64[]
    energies_dmrg = Float64[]
    errors = Float64[]
    g_vals_found = Float64[]

    for g in g_values
        filename = joinpath(data_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")

        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping g=$g"
            continue
        end

        # Load result
        result, input_args = load_result(filename)

        # Compute exact energy from optimized parameters
        virtual_qubits = (nqubits - 1) ÷ 2
        share_params = get(input_args, :share_params, true)
        gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)

        X_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, :X; position=i)) for i in 1:row)

        if row > 1
            ZZ_vert_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, Dict(i => :Z, i+1 => :Z))) for i in 1:row-1)
        else
            ZZ_vert_exact = 0.0
        end

        ZZ_horiz_vals = Float64[]
        for pos in 1:row
            correlations = correlation_function(gates, row, virtual_qubits, :Z, 1; position=pos)
            if haskey(correlations, 1)
                push!(ZZ_horiz_vals, real(correlations[1]))
            end
        end
        ZZ_horiz_exact = isempty(ZZ_horiz_vals) ? 0.0 : mean(ZZ_horiz_vals)

        energy_exact = -g*X_exact - J*(row == 1 ? ZZ_horiz_exact : ZZ_vert_exact + ZZ_horiz_exact)

        push!(g_vals_found, g)
        push!(energies_exact, energy_exact)

        # Get reference energy and compute error
        if haskey(ref_energies, g)
            energy_ref = ref_energies[g]
            error = abs(energy_exact - energy_ref)
            push!(energies_ref, energy_ref)
            push!(errors, error)
            println("g=$g: E_exact=$(round(energy_exact, digits=6)), E_ref=$(round(energy_ref, digits=6)), Error=$(round(error, digits=6))")
        else
            push!(energies_ref, NaN)
            push!(errors, NaN)
            println("g=$g: E_exact=$(round(energy_exact, digits=6)), No reference")
        end

        # Get DMRG energy
        if haskey(dmrg_energies, g)
            push!(energies_dmrg, dmrg_energies[g])
            println("       E_dmrg=$(round(dmrg_energies[g], digits=6))")
        else
            push!(energies_dmrg, NaN)
        end
    end

    if isempty(g_vals_found)
        error("No valid results found for any g value")
    end

    # Create figure
    fig = Figure(size=(1000, 800))

    # Plot 1: Energies comparison
    ax1 = Axis(fig[1, 1],
               xlabel="Transverse field g",
               ylabel="Energy",
               title="Ground State Energy: Exact vs Reference")

    lines!(ax1, g_vals_found, energies_exact, label="sampling based optimization",
           color=:blue, linewidth=2)
    scatter!(ax1, g_vals_found, energies_exact, color=:blue, markersize=12)

    if !all(isnan.(energies_ref))
        valid_mask = .!isnan.(energies_ref)
        lines!(ax1, g_vals_found[valid_mask], energies_ref[valid_mask],
               label="iPEPS", color=:red, linewidth=2, linestyle=:dash)
        scatter!(ax1, g_vals_found[valid_mask], energies_ref[valid_mask],
                color=:red, markersize=12, marker=:diamond)
    end

    if !all(isnan.(energies_dmrg))
        valid_mask = .!isnan.(energies_dmrg)
        lines!(ax1, g_vals_found[valid_mask], energies_dmrg[valid_mask],
               label="DMRG", color=:green, linewidth=2, linestyle=:dot)
        scatter!(ax1, g_vals_found[valid_mask], energies_dmrg[valid_mask],
                color=:green, markersize=12, marker=:star5)
    end

    axislegend(ax1, position=:rb)

    # Plot 2: Energy errors
    ax2 = Axis(fig[2, 1],
               xlabel="Transverse field g",
               ylabel="Energy Error",
               title="Energy Error vs g",
               yscale=log10)

    # Error between exact and PEPSKit reference
    if !all(isnan.(errors))
        valid_mask = .!isnan.(errors)
        lines!(ax2, g_vals_found[valid_mask], errors[valid_mask],
               label="|E_optimized - E_iPEPS|", color=:red, linewidth=2)
        scatter!(ax2, g_vals_found[valid_mask], errors[valid_mask],
                color=:red, markersize=12)
    end

    # Error between exact and DMRG
    if !all(isnan.(energies_dmrg))
        valid_mask = .!isnan.(energies_dmrg)
        errors_dmrg = abs.(energies_exact[valid_mask] .- energies_dmrg[valid_mask])
        lines!(ax2, g_vals_found[valid_mask], errors_dmrg,
               label="|E_optimized - E_DMRG|", color=:green, linewidth=2, linestyle=:dash)
        scatter!(ax2, g_vals_found[valid_mask], errors_dmrg,
                color=:green, markersize=12, marker=:star5)
    end

    if !all(isnan.(errors)) || !all(isnan.(energies_dmrg))
        axislegend(ax2, position=:lt)
    else
        text!(ax2, 0.5, 0.5, text="No reference data available",
              align=(:center, :center), space=:relative)
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end

    data = (
        g_values = g_vals_found,
        energies_exact = energies_exact,
        energies_ref = energies_ref,
        energies_dmrg = energies_dmrg,
        errors = errors
    )

    return fig, data
end

"""
    plot_correlation_vs_g(data_dir::String, g_values::Vector{Float64};
                          J=1.0, row=3, nqubits=5, p=3,
                          max_separation=20,
                          connected=true,
                          save_path::Union{String,Nothing}=nothing)

Plot correlation function for different g values on the same plot.

# Arguments
- `data_dir`: Directory containing result JSON files
- `g_values`: Vector of g values to plot
- `J`: Coupling strength (default: 1.0)
- `row`: Number of rows (default: 3)
- `nqubits`: Number of qubits (default: 5)
- `p`: Circuit depth (default: 3)
- `max_separation`: Maximum separation to plot (default: 20)
- `connected`: Plot connected correlation (default: true)
- `save_path`: Path to save figure (optional)

# Returns
- `fig`: Makie Figure object
- `data`: Dict mapping g values to correlation data

# Example
```julia
g_vals = [1.0, 2.0, 3.0, 4.0]
fig, data = plot_correlation_vs_g("project/results", g_vals;
                                  max_separation=30,
                                  save_path="correlation_vs_g.pdf")
```
"""
function plot_correlation_vs_g(data_dir::String, g_values::Vector{Float64};
                               J=1.0, row=3, nqubits=3, p=3,
                               max_separation=20,
                               connected=true,
                               dmrg_file::Union{String,Nothing}=nothing,
                               pepskit_file::Union{String,Nothing}=nothing,
                               g_c::Union{Float64,Nothing}=nothing,
                               save_path::Union{String,Nothing}=nothing)

    println("="^70)
    println("Correlation Function vs g Analysis")
    println("="^70)
    println("Connected: $connected")

    # Load results and compute correlations
    correlation_data = Dict{Float64, NamedTuple}()
    colors = [:blue, :green, :red, :orange, :purple, :brown, :pink, :gray]

    for (idx, g) in enumerate(g_values)
        filename = joinpath(data_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")

        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping g=$g"
            continue
        end

        println("\nProcessing g=$g...")

        # Load result
        result, input_args = load_result(filename)

        # Reconstruct gates
        virtual_qubits = (nqubits - 1) ÷ 2
        share_params = get(input_args, :share_params, true)
        gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)

        # Compute correlation function
        separations = 1:max_separation
        corr_vals = Float64[]

        for r in separations
            corr = correlation_function(gates, row, virtual_qubits, :Z, r; connected=connected)
            push!(corr_vals, real(corr[r]))
        end

        # Compute correlation length from transfer matrix
        _, gap, _, _ = compute_transfer_spectrum(gates, row, nqubits)
        ξ = 1.0 / gap

        # Fit correlation to get fitted ξ
        ξ_fitted = nothing
        try
            fit_params = fit_acf(collect(separations), corr_vals; include_zero=false)
            ξ_fitted = fit_params.ξ
            println("  ξ_transfer = $(round(ξ, digits=3)), ξ_fitted = $(round(ξ_fitted, digits=3))")
        catch e
            println("  ξ_transfer = $(round(ξ, digits=3)), fitting failed")
        end

        correlation_data[g] = (
            separations = collect(separations),
            correlations = corr_vals,
            correlation_length = ξ,
            correlation_length_fitted = ξ_fitted,
            color = colors[mod1(idx, length(colors))]
        )
    end

    if isempty(correlation_data)
        error("No valid results found for any g value")
    end

    # Create figure
    fig = Figure(size=(800, 600))

    ax = Axis(fig[1, 1],
              xlabel="g",
              ylabel="Correlation Length ξ",
              title="Correlation Length vs g (J=$J, row=$row, D=$(nqubits-1))")

    # Extract and plot correlation lengths from transfer matrix
    g_sorted = sort(collect(keys(correlation_data)))
    ξ_transfer = [correlation_data[g].correlation_length for g in g_sorted]

    lines!(ax, g_sorted, ξ_transfer,
           color=:blue, linewidth=2, label="Sampling based optimization (transfer matrix)")
    scatter!(ax, g_sorted, ξ_transfer,
             color=:blue, markersize=12)

    # Overlay DMRG correlation lengths if a file is provided
    if dmrg_file !== nothing && isfile(dmrg_file)
        println("\nLoading DMRG reference from: $dmrg_file")
        dmrg_data = open(dmrg_file, "r") do io
            JSON3.read(io)
        end

        dmrg_g = Float64.(collect(dmrg_data.scan_values))
        dmrg_ξ = collect(dmrg_data.correlation_lengths)

        # Filter out null/nothing entries and unreasonable values
        valid = [i for i in eachindex(dmrg_g)
                 if dmrg_ξ[i] !== nothing && isfinite(Float64(dmrg_ξ[i])) && Float64(dmrg_ξ[i]) < 1e5]
        dmrg_g_valid = dmrg_g[valid]
        dmrg_ξ_valid = Float64.(dmrg_ξ[valid])

        lines!(ax, dmrg_g_valid, dmrg_ξ_valid,
               color=:red, linewidth=2, linestyle=:dash, label="DMRG reference")
        scatter!(ax, dmrg_g_valid, dmrg_ξ_valid,
                 color=:red, markersize=8, marker=:diamond)

        println("  DMRG: $(length(dmrg_g_valid)) valid g points loaded")
    elseif dmrg_file !== nothing
        @warn "DMRG file not found: $dmrg_file"
    end

    # Overlay PEPSKit correlation lengths if a file is provided
    if pepskit_file !== nothing && isfile(pepskit_file)
        println("\nLoading PEPSKit reference from: $pepskit_file")
        peps_data = open(pepskit_file, "r") do io
            JSON3.read(io)
        end

        peps_g = Float64.(collect(peps_data.g_values))
        peps_ξ = collect(peps_data.correlation_lengths)

        # Filter out null/nothing entries and unreasonable values
        valid = [i for i in eachindex(peps_g)
                 if peps_ξ[i] !== nothing && isfinite(Float64(peps_ξ[i])) && Float64(peps_ξ[i]) < 1e5]
        peps_g_valid = peps_g[valid]
        peps_ξ_valid = Float64.(peps_ξ[valid])

        D_label = haskey(peps_data, :parameters) && haskey(peps_data.parameters, :D) ?
            " (D=$(peps_data.parameters.D))" : ""
        lines!(ax, peps_g_valid, peps_ξ_valid,
               color=:green, linewidth=2, linestyle=:dashdot, label="iPEPS reference")
        scatter!(ax, peps_g_valid, peps_ξ_valid,
                 color=:green, markersize=8, marker=:utriangle)

        println("  PEPSKit: $(length(peps_g_valid)) valid g points loaded")
    elseif pepskit_file !== nothing
        @warn "PEPSKit file not found: $pepskit_file"
    end

    # Mark critical point
    if g_c !== nothing
        vlines!(ax, [g_c], color=:black, linestyle=:dot, linewidth=1.5, label="g_c ≈ $g_c")
    end

    axislegend(ax, position=:lt)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end

    return fig, correlation_data
end

