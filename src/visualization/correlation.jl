# ============================================================================
# corr_exp_fit
# ============================================================================

function corr_exp_fit(lags::AbstractVector, acf::AbstractVector;
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

# plot_correlation_function
# ============================================================================

function plot_correlation_function(filename::String;
                                   max_separation::Int=20,
                                   conv_step::Int=1000,
                                   samples::Int=100000,
                                   exact_method::Symbol=:auto,
                                   include_sampling::Bool=true,
                                   sampling_trajectories::Int=1,
                                   spectrum_krylovdim::Int=60,
                                   spectrum_tol::Real=1e-8,
                                   spectrum_maxiter::Int=1000,
                                   figsize=nothing,
                                   save_path::Union{String,Nothing}=nothing)

    result, input_args = load_result(filename)

    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    g = get(input_args, :g, NaN)
    virtual_qubits = (nqubits - 1) ÷ 2
    share_params = get(input_args, :share_params, true)
    model_str = get(input_args, :model, "tfim")
    is_heisenberg = model_str == "heisenberg_j1j2"

    params = result.final_params

    # Detect 2x2 unit cell
    is_2x2 = is_heisenberg &&
        (length(params) == gate_parameter_count(p, nqubits; unit_cell=:two_by_two))

    if is_2x2
        gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
        op = TransferOperator(gates_odd, gates_even, row, nqubits)
    else
        gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
        op = TransferOperator(gates, row, nqubits)
    end
    N_cols = length(op.columns)

    println("=== Correlation Function Analysis ===")
    println("File: ", basename(filename))
    println("Configuration: g=$g, row=$row, nqubits=$nqubits, unit_cell=$(N_cols)x1")

    sz = matrix_size(op)
    use_matrix_free = exact_method == :matrix_free ||
        (exact_method == :auto && sz > 1024)
    exact_method in (:auto, :dense, :matrix_free) ||
        throw(ArgumentError("exact_method must be :auto, :dense, or :matrix_free"))

    println("\nComputing exact correlations ($(use_matrix_free ? "matrix-free transfer contraction" : "dense transfer matrix"))...")
    println("Averaging over all positions 1 to $row...")

    # Compute correlations for each position and average
    # Separations are in period units; exact values at r*N_cols columns
    max_sep_periods = cld(max_separation, N_cols)
    separations = [r * N_cols for r in 1:max_sep_periods]
    exact_full_vals = zeros(Float64, max_sep_periods)
    exact_connected_vals = zeros(Float64, max_sep_periods)
    rho = nothing
    gap = nothing

    if use_matrix_free
        rho, gap, _, _ = compute_transfer_spectrum(op;
                                                   matrix_free=:always,
                                                   krylovdim=spectrum_krylovdim,
                                                   tol=spectrum_tol,
                                                   maxiter=spectrum_maxiter)
    end

    for pos in 1:row
        exact_full_pos = if use_matrix_free
            correlation_function_matrix_free(op, :Z, 1:max_sep_periods;
                                             position=pos, connected=false,
                                             rho=rho)
        else
            correlation_function(op, :Z, 1:max_sep_periods;
                                 position=pos, connected=false)
        end
        exact_connected_pos = if use_matrix_free
            correlation_function_matrix_free(op, :Z, 1:max_sep_periods;
                                             position=pos, connected=true,
                                             rho=rho)
        else
            correlation_function(op, :Z, 1:max_sep_periods;
                                 position=pos, connected=true)
        end

        for (i, r) in enumerate(1:max_sep_periods)
            exact_full_vals[i] += real(exact_full_pos[r])
            exact_connected_vals[i] += real(exact_connected_pos[r])
        end
    end

    # Average over positions
    exact_full_vals ./= row
    exact_connected_vals ./= row

    if isnothing(gap)
        _, gap, _, _ = compute_transfer_spectrum(
            op;
            matrix_free=:auto,
            krylovdim=spectrum_krylovdim,
            tol=spectrum_tol,
            maxiter=spectrum_maxiter,
        )
    end
    correlation_length = N_cols / gap
    println("Correlation length ξ = $(round(correlation_length, digits=2)) columns")

    sample_seps = Int[]
    sample_full_vals = Float64[]
    sample_full_err_vals = Float64[]
    sample_connected_vals = Float64[]
    sample_connected_err_vals = Float64[]
    error_full = Float64[]
    error_connected = Float64[]
    mean_error_full = NaN
    mean_error_connected = NaN
    sample_full_trajectories = Matrix{Float64}(undef, 0, 0)
    sample_connected_trajectories = Matrix{Float64}(undef, 0, 0)

    if include_sampling
        sampling_trajectories >= 1 ||
            throw(ArgumentError("sampling_trajectories must be at least 1"))

        println("\nGenerating samples (conv_step=$conv_step, samples=$samples, trajectories=$sampling_trajectories)...")

        z_trajectories = Vector{Vector{Float64}}(undef, sampling_trajectories)
        for traj in 1:sampling_trajectories
            if sampling_trajectories > 1
                println("  Sampling trajectory $traj / $sampling_trajectories")
            end

            if is_2x2
                _, Z_samples, _ = sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                                          conv_step=conv_step,
                                                          samples=samples)
            else
                gates = op.columns[1]
                _, Z_samples, _ = sample_quantum_channel(gates, row, nqubits;
                                                          conv_step=conv_step,
                                                          samples=samples)
            end
            z_trajectories[traj] = Float64.(Z_samples[conv_step+1:end])
        end

        min_len = minimum(length.(z_trajectories))
        if any(length(z) != min_len for z in z_trajectories)
            @warn "Sampling trajectories have unequal lengths; truncating all to $min_len samples"
        end
        Z_mat = Matrix{Float64}(undef, sampling_trajectories, min_len)
        for traj in 1:sampling_trajectories
            Z_mat[traj, :] .= @view z_trajectories[traj][1:min_len]
        end

        # Use row*N_cols for subsampling so each subchain has same-column-type measurements
        # (for 2x2 unit cells, row alone mixes odd/even column types)
        acf_row = row * N_cols
        lags, acf, acf_err, corr_full, corr_err, corr_connected, corr_connected_err = compute_acf(
            Z_mat; max_lag=max_sep_periods+1, row=acf_row
        )

        # Keep per-trajectory curves for diagnostics in the returned data.
        n_acf_lags = length(corr_full)
        sample_full_trajectories = Matrix{Float64}(undef, sampling_trajectories, n_acf_lags)
        sample_connected_trajectories = Matrix{Float64}(undef, sampling_trajectories, n_acf_lags)
        for traj in 1:sampling_trajectories
            _, _, _, corr_t, _, corr_connected_t, _ = compute_acf(
                reshape(Z_mat[traj, :], 1, :); max_lag=n_acf_lags, row=acf_row
            )
            sample_full_trajectories[traj, :] .= corr_t
            sample_connected_trajectories[traj, :] .= corr_connected_t
        end

        sample_full = corr_full
        sample_full_err = corr_err
        sample_connected = corr_connected
        sample_connected_err = corr_connected_err

        n_sample_lags = min(length(sample_full) - 1, max_sep_periods)
        # Sample lags are in period units; convert to column separations
        sample_seps = [k * N_cols for k in 1:n_sample_lags]
        sample_full_vals = sample_full[2:n_sample_lags+1]
        sample_full_err_vals = sample_full_err[2:n_sample_lags+1]
        sample_connected_vals = sample_connected[2:n_sample_lags+1]
        sample_connected_err_vals = sample_connected_err[2:n_sample_lags+1]
        sample_full_trajectories = sample_full_trajectories[:, 2:n_sample_lags+1]
        sample_connected_trajectories = sample_connected_trajectories[:, 2:n_sample_lags+1]

        println("Sampling std errors range (full): $(round(minimum(sample_full_err_vals), sigdigits=2)) - $(round(maximum(sample_full_err_vals), sigdigits=2))")
        println("Sampling std errors range (connected): $(round(minimum(sample_connected_err_vals), sigdigits=2)) - $(round(maximum(sample_connected_err_vals), sigdigits=2))")

        # Both exact and sample are now at the same separations (N_cols, 2*N_cols, ...)
        common_seps = min(length(exact_full_vals), length(sample_full_vals))
        error_full = abs.(exact_full_vals[1:common_seps] .- sample_full_vals[1:common_seps])
        error_connected = abs.(exact_connected_vals[1:common_seps] .- sample_connected_vals[1:common_seps])
        mean_error_full = mean(error_full)
        mean_error_connected = mean(error_connected)
        println("Mean |exact - sample| error (full): $(round(mean_error_full, digits=6))")
        println("Mean |exact - sample| error (connected): $(round(mean_error_connected, digits=6))")
    end

    min_val = 1e-15

    # Fit connected correlation to A·exp(−r/ξ)
    println("\nFitting connected correlation to A·exp(−r/ξ)...")
    ξ_fitted = nothing
    A_fitted = nothing
    try
        fit_params = corr_exp_fit(separations, exact_connected_vals; include_zero=false)
        ξ_fitted = fit_params.ξ
        A_fitted = fit_params.A
        println("  ξ_fit = $(round(ξ_fitted, digits=3))   ξ_TM = $(round(correlation_length, digits=3))")
    catch e
        @warn "Exponential fitting failed: $e"
    end

    _w, _h = PAPER_FIGSIZE
    _figsize = isnothing(figsize) ? (_w, 2_h - 20) : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        exact_full_abs       = max.(abs.(exact_full_vals),      min_val)
        sample_full_abs      = max.(abs.(sample_full_vals),     min_val)
        exact_connected_abs  = max.(abs.(exact_connected_vals), min_val)
        sample_connected_abs = max.(abs.(sample_connected_vals),min_val)

        # ── Panel (a): full correlator ────────────────────────────────────
        ax1 = Axis(fig[1, 1];
                   ylabel = "|⟨ZᵢZᵢ₊ᵣ⟩|",
                   yscale = log10,
                   xticklabelsvisible = false)

        scatterlines!(ax1, separations, exact_full_abs;
                      color=:steelblue, marker=:circle, label="TN contraction")

        if include_sampling
            err_low = min.(sample_full_err_vals, sample_full_abs .- min_val)
            errorbars!(ax1, collect(sample_seps), sample_full_abs,
                       err_low, sample_full_err_vals;
                       color=:firebrick, whiskerwidth=4)
            scatter!(ax1, collect(sample_seps), sample_full_abs;
                     color=:firebrick, marker=:diamond, label="Sampling")
        end

        text!(ax1, 0.03, 0.97; text="(a)", space=:relative,
              align=(:left, :top), fontsize=PAPER_TITLESIZE, font=:bold)
        add_paper_legend!(ax1; position=(:left, 0.82))

        # ── Panel (b): connected correlator + fit ─────────────────────────
        ax2 = Axis(fig[2, 1];
                   xlabel = "r",
                   ylabel = "|⟨ZᵢZᵢ₊ᵣ⟩_c|",
                   yscale = log10)

        scatterlines!(ax2, separations, exact_connected_abs;
                      color=:steelblue, marker=:circle, label="TN contraction")

        if include_sampling
            err_low_c = min.(sample_connected_err_vals, sample_connected_abs .- min_val)
            errorbars!(ax2, collect(sample_seps), sample_connected_abs,
                       err_low_c, sample_connected_err_vals;
                       color=:firebrick, whiskerwidth=4)
            scatter!(ax2, collect(sample_seps), sample_connected_abs;
                     color=:firebrick, marker=:diamond, label="Sampling")
        end

        if !isnothing(ξ_fitted) && !isnothing(A_fitted)
            r_fit = range(first(separations), last(separations); length=200)
            fitted = max.(abs(A_fitted) .* exp.(-r_fit ./ ξ_fitted), min_val)
            lines!(ax2, r_fit, fitted;
                   color=:seagreen, linestyle=:dash,
                   label="∝ e^{−r/$(round(ξ_fitted, digits=2))}")
        end

        text!(ax2, 0.03, 0.97; text="(b)", space=:relative,
              align=(:left, :top), fontsize=PAPER_TITLESIZE, font=:bold)
        add_paper_legend!(ax2; position=(:left, 0.82))

        rowgap!(fig.layout, 4)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("\nFigure saved to: $save_path")
        end

        fig
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
        sample_full_trajectories = sample_full_trajectories,
        sample_connected_trajectories = sample_connected_trajectories,
        sampling_trajectories = sampling_trajectories,
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
