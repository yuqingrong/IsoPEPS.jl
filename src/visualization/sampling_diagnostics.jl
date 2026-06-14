# ============================================================================
# compute_variance_vs_samples
# ============================================================================

"""
    compute_variance_vs_samples(filename, sample_sizes; kwargs...)

Run the circuit once at `maximum(sample_sizes)` samples, then estimate the
variance of the energy estimator at each target sample size via bootstrap
subsampling. A second bootstrap over those energy estimates gives a percentile
confidence interval for the variance. The returned vectors can be passed directly to
`plot_variance_vs_samples`.

# Arguments
- `filename`: Path to a saved optimization result JSON
- `sample_sizes`: Vector of sample counts to evaluate (e.g. `[1_000, 10_000, 100_000]`)
- `conv_step`: Thermalization steps discarded from the front of the chain (default 100)
- `n_bootstrap`: Number of independent energy estimates per sample size (default 200)
- `n_ci_bootstrap`: Number of bootstrap resamples used for the variance CI (default 5000)
- `confidence_level`: Coverage of the percentile confidence interval (default 0.68)
- `save_path`: If provided, saves the results as JSON for later use

# Returns
- `(sample_sizes, variances, errors)`, where `errors[i]` is
  `(variances[i] - lower_ci[i], upper_ci[i] - variances[i])`.

# Example
```julia
sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
ns, vars, errs = compute_variance_vs_samples(
    "project/results/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json",
    sizes; conv_step=100, n_bootstrap=200)
fig = plot_variance_vs_samples(ns, vars; errors=errs, confidence_level=0.68)
```
"""
function _variance_bootstrap_ci(energies::AbstractVector{<:Real};
                                n_resamples::Int=5000,
                                confidence_level::Real=0.68)
    length(energies) >= 2 ||
        throw(ArgumentError("at least two energy estimates are required"))
    n_resamples >= 2 || throw(ArgumentError("n_resamples must be at least 2"))
    0 < confidence_level < 1 ||
        throw(ArgumentError("confidence_level must be between 0 and 1"))

    bootstrap_sample = bootstrap(var, energies, BasicSampling(n_resamples))
    estimate, lower, upper =
        only(confint(bootstrap_sample, PercentileConfInt(confidence_level)))
    return Float64(estimate), Float64(lower), Float64(upper)
end

function compute_variance_vs_samples(filename::String,
                                     sample_sizes::AbstractVector{Int};
                                     total_samples::Union{Int,Nothing}=nothing,
                                     conv_step::Int=100,
                                     n_bootstrap::Int=200,
                                     n_ci_bootstrap::Int=5000,
                                     confidence_level::Real=0.68,
                                     save_path::Union{String,Nothing}=nothing)

    isempty(sample_sizes) && throw(ArgumentError("sample_sizes must not be empty"))
    n_bootstrap >= 2 || throw(ArgumentError("n_bootstrap must be at least 2"))
    n_ci_bootstrap >= 2 || throw(ArgumentError("n_ci_bootstrap must be at least 2"))
    0 < confidence_level < 1 ||
        throw(ArgumentError("confidence_level must be between 0 and 1"))

    _result, input_args = load_result(filename)
    model_str  = get(input_args, :model, "tfim")
    is_heisenberg = (model_str == "heisenberg_j1j2")
    row        = Int(input_args[:row])
    J1         = Float64(get(input_args, :J1, get(input_args, :J, 1.0)))
    J2         = Float64(get(input_args, :J2, 0.0))
    g          = Float64(get(input_args, :g,  1.0))
    J          = Float64(get(input_args, :J,  1.0))

    sorted_sizes = sort(sample_sizes)
    # Pool must be >> max(sample_sizes) so bootstrap blocks can start at different positions.
    _pool_samples = isnothing(total_samples) ? 20 * last(sorted_sizes) : total_samples
    println("=== compute_variance_vs_samples ===")
    println("Model: $model_str  |  pool: $_pool_samples spins  |  conv_step: $conv_step")

    # Run circuit once at pool size (>> max sample size so bootstrap draws can vary)
    resample_result = resample_circuit(filename;
                                       conv_step=conv_step,
                                       samples=_pool_samples,
                                       measure_y=is_heisenberg)
    if isnothing(resample_result)
        error("resample_circuit failed for $filename")
    end

    if is_heisenberg
        _rho, Z_all, X_all, Y_all, _params, _gates = resample_result
        Z_pool = _discard_burnin(Z_all, row, conv_step; requested_samples=_pool_samples)
        X_pool = _discard_burnin(X_all, row, conv_step; requested_samples=_pool_samples)
        Y_pool = _discard_burnin(Y_all, row, conv_step; requested_samples=_pool_samples)
    else
        _rho, Z_all, X_all, _params, _gates = resample_result
        Z_pool = _discard_burnin(Z_all, row, conv_step; requested_samples=_pool_samples)
        X_pool = _discard_burnin(X_all, row, conv_step; requested_samples=_pool_samples)
    end

    # Each MCMC step emits `row` consecutive spin values (one per row position).
    # Horizontal bonds in compute_heisenberg_energy require SPATIALLY ADJACENT
    # columns, i.e., consecutive MCMC steps.  We therefore use a moving-block
    # bootstrap: pick a random start position and take n_cols CONSECUTIVE columns.
    pool_cols   = length(Z_pool) ÷ row
    pool_cols_X = length(X_pool) ÷ row
    pool_cols_Y = is_heisenberg ? length(Y_pool) ÷ row : 0
    println("Pool size after thermalization: $(length(Z_pool)) spins  ($pool_cols columns × row=$row)")

    _block(pool, start_col, n_cols) = @view pool[(start_col-1)*row+1 : (start_col+n_cols-1)*row]

    variances = Vector{Float64}(undef, length(sorted_sizes))
    ci_lower = similar(variances)
    ci_upper = similar(variances)
    errors = Vector{Tuple{Float64,Float64}}(undef, length(sorted_sizes))

    for (k, n_spins) in enumerate(sorted_sizes)
        # n_spins is number of spin measurements; convert to column count
        n_cols = min(n_spins ÷ row, pool_cols - 1)
        energies = Vector{Float64}(undef, n_bootstrap)

        max_start_Z = pool_cols   - n_cols
        max_start_X = pool_cols_X - n_cols
        max_start_Y = is_heisenberg ? pool_cols_Y - n_cols : 1

        for b in 1:n_bootstrap
            sz = rand(1:max_start_Z)
            sx = rand(1:max_start_X)
            if is_heisenberg
                sy = rand(1:max_start_Y)
                energies[b] = compute_heisenberg_energy(
                    _block(X_pool, sx, n_cols),
                    _block(Z_pool, sz, n_cols),
                    _block(Y_pool, sy, n_cols),
                    J1, J2, row)
            else
                energies[b] = compute_tfim_energy(
                    _block(X_pool, sx, n_cols),
                    _block(Z_pool, sz, n_cols),
                    g, J, row)
            end
        end

        v, lower, upper = _variance_bootstrap_ci(
            energies;
            n_resamples=n_ci_bootstrap,
            confidence_level=confidence_level)
        variances[k] = v
        ci_lower[k] = lower
        ci_upper[k] = upper
        errors[k] = (v - lower, upper - v)
        println("  n=$n_spins ($(n_cols) cols)  →  E̅ = $(round(mean(energies), sigdigits=6))  Var(E) = $(round(v, sigdigits=4))  CI = ($(round(lower, sigdigits=4)), $(round(upper, sigdigits=4)))")
    end

    if !isnothing(save_path)
        save_results(save_path;
                     sample_sizes=collect(Int, sorted_sizes),
                     variances=variances,
                     variance_ci_lower=ci_lower,
                     variance_ci_upper=ci_upper,
                     confidence_level=Float64(confidence_level),
                     model=model_str,
                     conv_step=conv_step,
                     n_bootstrap=n_bootstrap,
                     n_ci_bootstrap=n_ci_bootstrap)
        println("Data saved to: $save_path")
    end

    return collect(Float64, sorted_sizes), variances, errors
end

# ============================================================================
# plot_energy_vs_inv_samples
# ============================================================================

"""
    plot_energy_vs_inv_samples(filename, sample_sizes; kwargs...)

Scatter-plot all bootstrap energy estimates versus 1/N for each sample size.
Each vertical cluster of dots represents the distribution of energy estimates
from `n_bootstrap` random subsamples of size N; as N → ∞ (1/N → 0) the
spread narrows, visually confirming the 1/√N standard-error scaling.

# Arguments
- `filename`: Path to a saved optimization result JSON
- `sample_sizes`: Vector of column counts to evaluate (e.g. `[1_000, 5_000, …]`).
  Must be much smaller than `total_samples ÷ row` for the bootstrap to show spread.
- `total_samples`: Total spin measurements to collect as the pool (default:
  `20 * maximum(sample_sizes) * row`). Must satisfy `total_samples >> max(sample_sizes)`.
- `conv_step`: Thermalization steps to discard (default 100)
- `n_bootstrap`: Bootstrap draws per sample size (default 200)
- `figsize`: Override figure size (default `PAPER_FIGSIZE`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, energies_matrix)` where `energies_matrix` is a
  `length(sample_sizes) × n_bootstrap` matrix of energy estimates
"""
function plot_energy_vs_inv_samples(filename::String,
                                    sample_sizes::AbstractVector{Int};
                                    total_samples::Union{Int,Nothing}=nothing,
                                    conv_step::Int=100,
                                    n_bootstrap::Int=200,
                                    figsize=nothing,
                                    save_path::Union{String,Nothing}=nothing)

    result, input_args = load_result(filename)
    model_str     = get(input_args, :model, "tfim")
    is_heisenberg = (model_str == "heisenberg_j1j2")
    row   = Int(input_args[:row])
    J1    = Float64(get(input_args, :J1, get(input_args, :J, 1.0)))
    J2    = Float64(get(input_args, :J2, 0.0))
    g     = Float64(get(input_args, :g,  1.0))
    J     = Float64(get(input_args, :J,  1.0))

    sorted_sizes = sort(sample_sizes)
    # Pool must be >> max(sample_sizes) so different bootstrap blocks exist.
    # Default: 20× the largest requested spin count.
    _pool_samples = isnothing(total_samples) ? 20 * last(sorted_sizes) : total_samples

    println("=== plot_energy_vs_inv_samples ===")
    println("Model: $model_str  |  pool: $_pool_samples spins  |  conv_step: $conv_step  |  n_bootstrap: $n_bootstrap")

    resample_result = resample_circuit(filename;
                                       conv_step=conv_step,
                                       samples=_pool_samples,
                                       measure_y=is_heisenberg)
    isnothing(resample_result) && error("resample_circuit failed for $filename")

    if is_heisenberg
        _rho, Z_all, X_all, Y_all, _params, _gates = resample_result
        Z_pool = _discard_burnin(Z_all, row, conv_step; requested_samples=_pool_samples)
        X_pool = _discard_burnin(X_all, row, conv_step; requested_samples=_pool_samples)
        Y_pool = _discard_burnin(Y_all, row, conv_step; requested_samples=_pool_samples)
    else
        _rho, Z_all, X_all, _params, _gates = resample_result
        Z_pool = _discard_burnin(Z_all, row, conv_step; requested_samples=_pool_samples)
        X_pool = _discard_burnin(X_all, row, conv_step; requested_samples=_pool_samples)
    end
    pool_cols   = length(Z_pool) ÷ row
    pool_cols_X = length(X_pool) ÷ row
    pool_cols_Y = is_heisenberg ? length(Y_pool) ÷ row : 0
    println("Pool size after thermalization: $(length(Z_pool)) spins  ($pool_cols columns × row=$row)")

    _block(pool, start_col, n_cols) = @view pool[(start_col-1)*row+1 : (start_col+n_cols-1)*row]

    n_sizes         = length(sorted_sizes)
    energies_matrix = Matrix{Float64}(undef, n_sizes, n_bootstrap)

    for (k, n_spins) in enumerate(sorted_sizes)
        # n_spins is number of spin measurements; convert to column count
        n_cols      = min(n_spins ÷ row, pool_cols - 1)
        max_start_Z = pool_cols   - n_cols
        max_start_X = pool_cols_X - n_cols
        max_start_Y = is_heisenberg ? pool_cols_Y - n_cols : 1
        for b in 1:n_bootstrap
            sz = rand(1:max_start_Z)
            sx = rand(1:max_start_X)
            if is_heisenberg
                sy = rand(1:max_start_Y)
                energies_matrix[k, b] = compute_heisenberg_energy(
                    _block(X_pool, sx, n_cols),
                    _block(Z_pool, sz, n_cols),
                    _block(Y_pool, sy, n_cols),
                    J1, J2, row)
            else
                energies_matrix[k, b] = compute_tfim_energy(
                    _block(X_pool, sx, n_cols),
                    _block(Z_pool, sz, n_cols),
                    g, J, row)
            end
        end
        println("  n=$n_spins ($(n_cols) cols)  →  E = $(round(mean(energies_matrix[k,:]), sigdigits=6)) ± $(round(std(energies_matrix[k,:]), sigdigits=3))")
    end

    # ── Figure ───────────────────────────────────────────────────────────────
    _figsize = isnothing(figsize) ? PAPER_FIGSIZE : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel = "1 / N (samples)",
                  ylabel = "E / site",
                  xscale = log10)

        for (k, n) in enumerate(sorted_sizes)
            x = fill(1.0 / n, n_bootstrap)
            scatter!(ax, x, energies_matrix[k, :];
                     color=(:steelblue, 0.35), markersize=4, strokewidth=0)
        end

        # overlay mean only
        means = vec(mean(energies_matrix; dims=2))
        xs    = 1.0 ./ sorted_sizes
        scatter!(ax, Float64.(xs), means;
                 color=:firebrick, marker=:circle, markersize=5,
                 label="mean")

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    return fig, energies_matrix
end

# ============================================================================
# plot_variance_vs_samples
# ============================================================================

"""
    plot_variance_vs_samples(sample_sizes, variances; kwargs...)
    plot_variance_vs_samples(results_file; kwargs...)

Plot bootstrap variance estimates against sample count on logarithmic axes.
`errors` may contain symmetric scalar errors or `(lower, upper)` error pairs.
The file-based method reads the JSON produced by `compute_variance_vs_samples`.
Use `marker` and `markersize` to customize the data markers.
"""
function plot_variance_vs_samples(sample_sizes::AbstractVector, variances::AbstractVector;
                                   errors::Union{AbstractVector,Nothing}=nothing,
                                   confidence_level::Union{Real,Nothing}=nothing,
                                   fit_scaling::Bool=true,
                                   marker=:circle,
                                   markersize::Real=6,
                                   figsize=nothing,
                                   save_path::Union{String,Nothing}=nothing)

    ns   = collect(Float64, sample_sizes)
    vars = collect(Float64, variances)

    _figsize = isnothing(figsize) ? PAPER_FIGSIZE : figsize
    variance_theme = merge(Theme(Scatter=(strokewidth=0,)), paper_theme())

    fig = with_theme(variance_theme) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel   = L"N_{\mathrm{shot}}",
                  ylabel   = "Var(E)",
                  xscale   = log10,
                  yscale   = log10)

        if !isnothing(errors)
            length(errors) == length(vars) ||
                throw(DimensionMismatch("errors and variances must have the same length"))
            if all(error -> error isa Real, errors)
                errorbars!(ax, ns, vars, Float64.(errors);
                           color=:steelblue, whiskerwidth=6)
            elseif all(error -> error isa Tuple && length(error) == 2, errors)
                lower_errors = Float64[first(error) for error in errors]
                upper_errors = Float64[last(error) for error in errors]
                errorbars!(ax, ns, vars, lower_errors, upper_errors;
                           color=:steelblue, whiskerwidth=6)
            else
                throw(ArgumentError(
                    "errors must contain either scalar values or (lower, upper) pairs"))
            end
        end
        estimate_label = isnothing(confidence_level) ?
            "Bootstrap estimate" :
            "Bootstrap $(round(Int, 100 * confidence_level))% CI"
        scatter!(ax, ns, vars;
                 label=estimate_label, color=:steelblue, marker=marker,
                 markersize=markersize, strokewidth=0)

        if fit_scaling && length(ns) > 1
            log_N   = log.(ns)
            log_var = log.(vars)
            a = exp(mean(log_var .+ log_N))
            N_fit = range(minimum(ns), maximum(ns); length=200)
            lines!(ax, collect(N_fit), a ./ collect(N_fit);
                   linestyle=:dash, color=:firebrick,
                   label="∝ 1/N  (a=$(round(a, sigdigits=3)))")
        end

        add_paper_legend!(ax; position=:rt)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    return fig
end

function plot_variance_vs_samples(results_file::String;
                                  confidence_level::Union{Real,Nothing}=nothing,
                                  kwargs...)
    data = load_results(results_file)
    get_value(key) = get(data, key, get(data, Symbol(key), nothing))

    sample_sizes = get_value("sample_sizes")
    variances = get_value("variances")
    ci_lower = get_value("variance_ci_lower")
    ci_upper = get_value("variance_ci_upper")
    saved_confidence_level = get_value("confidence_level")

    isnothing(sample_sizes) &&
        throw(ArgumentError("results file is missing sample_sizes"))
    isnothing(variances) &&
        throw(ArgumentError("results file is missing variances"))
    isnothing(ci_lower) &&
        throw(ArgumentError("results file is missing variance_ci_lower"))
    isnothing(ci_upper) &&
        throw(ArgumentError("results file is missing variance_ci_upper"))

    vars = Float64.(variances)
    lower = Float64.(ci_lower)
    upper = Float64.(ci_upper)
    length(vars) == length(lower) == length(upper) ||
        throw(DimensionMismatch("variances and confidence interval bounds must match"))
    errors = collect(zip(vars .- lower, upper .- vars))
    label_confidence_level = isnothing(confidence_level) ?
        saved_confidence_level : confidence_level

    return plot_variance_vs_samples(
        sample_sizes,
        vars;
        errors=errors,
        confidence_level=isnothing(label_confidence_level) ?
            nothing : Float64(label_confidence_level),
        kwargs...)
end
