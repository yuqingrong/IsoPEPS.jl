# Visualization: plotting and fitting functions

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
# fit_acf_power — power-law decay: C(r) = A * r^(-η)
# ============================================================================

function fit_acf_power(lags::AbstractVector, acf::AbstractVector;
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

    # Log-space OLS: fit log|C(r)| = log(A) - η*log(r)
    valid_idx = fit_acf_abs .> 1e-15
    A_abs = fit_acf_abs[1]
    η = 1.0

    if sum(valid_idx) >= 2
        log_abs    = log.(fit_acf_abs[valid_idx])
        log_lags   = log.(fit_lags[valid_idx])
        # Design matrix: log|C| = log(A) - η*log(r)  →  [1, -log(r)] * [log(A), η]ᵀ
        X = hcat(ones(length(log_lags)), -log_lags)
        coeffs = X \ log_abs
        log_A_abs = coeffs[1]
        η_fit     = coeffs[2]
        if η_fit > 0
            A_abs = exp(log_A_abs)
            η     = η_fit
        else
            # Fallback: two-point slope estimate
            η     = clamp(-(log_abs[end] - log_abs[1]) /
                           (log_lags[end] - log_lags[1]), 0.01, Inf)
            A_abs = exp(log_abs[1] + η * log_lags[1])
        end
    end

    A = correlation_sign * A_abs
    return (A=A, η=η, fit_lags=fit_lags,
            model=(r) -> A .* r .^ (-η))
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
                                  model::String="tfim",
                                  J1::Float64=1.0,
                                  J2::Float64=0.0,
                                  title::String="Expectation Values",
                                  save_path::Union{String,Nothing}=nothing,
                                  datafile::Union{String,Nothing}=nothing,
                                  kwargs...)

    Z_samples = result.final_Z_samples
    X_samples = result.final_X_samples
    Y_samples = hasproperty(result, :final_Y_samples) ? result.final_Y_samples : Float64[]
    m = _construct_model(model, Dict{Symbol,Any}(:J => J, :g => something(g, 1.0), :J1 => J1, :J2 => J2))
    need_y = needs_y_measurement(m)
    if !isnothing(datafile)
        if isfile(datafile)
            # Adaptive sampling: reduce samples for large nqubits (expensive to simulate)
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

            resampled = resample_circuit(datafile; conv_step=100, samples=adaptive_samples)
            if !isnothing(resampled)
                if need_y
                    _, Z_samples, X_samples, Y_samples, _, _ = resampled
                    Z_samples = Z_samples[101:end] 
                    X_samples = X_samples[101:end]
                    Y_samples = Y_samples[101:end]
                else
                    _, Z_samples, X_samples, _, _ = resampled
                    Z_samples = Z_samples[101:end]
                    X_samples = X_samples[101:end]
                end
            else
                @warn "Resampling failed for $datafile; using samples in result"
            end
        else
            @warn "Resample datafile not found: $datafile; using samples in result"
        end
    end

    # --- Shared: reconstruct gates + TransferOperator for exact computation ---
    can_compute_exact = !isnothing(row) && !isnothing(p) && !isnothing(nqubits) && !isempty(result.final_params)
    skip_exact = can_compute_exact && nqubits >= 5
    op = nothing

    if can_compute_exact && !skip_exact
        is_two_by_two = (m isa HeisenbergJ1J2) &&
            (length(result.final_params) == 4 * PARAMS_PER_QUBIT_PER_LAYER * nqubits * p)
        if is_two_by_two
            gates_odd, gates_even = build_unitary_gate_2x2(result.final_params, p, row, nqubits)
            op = TransferOperator(gates_odd, gates_even, row, nqubits)
        else
            gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=true)
            op = TransferOperator(gates, row, nqubits)
        end
    else
        can_compute_exact = false
    end

    labels = String[]; sample_values = Float64[]; exact_values = Float64[]; sample_errors = Float64[]

    if m isa HeisenbergJ1J2
        N = length(Z_samples)

        # Helper: compute NN correlations for a single Pauli component
        function _nn_correlations(S, row)
            N_s = length(S)
            if row == 1
                vert_val = nothing; vert_err = nothing
                horiz_pairs = [S[i] * S[i+1] for i in 1:N_s-1]
                horiz_val = mean(horiz_pairs)
                horiz_err = std(horiz_pairs) / sqrt(length(horiz_pairs))
            else
                vert_pairs = [S[i] * S[i+1] for i in 1:N_s-1 if i % row != 0]
                vert_val = mean(vert_pairs)
                vert_err = std(vert_pairs) / sqrt(length(vert_pairs))
                horiz_pairs = [S[i] * S[i+row] for i in 1:N_s-row]
                horiz_val = mean(horiz_pairs)
                horiz_err = std(horiz_pairs) / sqrt(length(horiz_pairs))
            end
            return (vert_val, vert_err, horiz_val, horiz_err)
        end

        # Helper: compute NNN diagonal and anti-diagonal correlations separately
        function _diag_correlations(S, row)
            N_s = length(S)
            # Diagonal ↘: (pos, col) -> (pos+1, col+1)
            diag1 = [S[i] * S[i+row+1] for i in 1:N_s-row-1 if i % row != 0]
            # Anti-diagonal ↗: (pos, col) -> (pos-1, col+1)
            diag2 = [S[i] * S[i+row-1] for i in 1:N_s-row+1 if (i-1) % row != 0]
            d1_val = mean(diag1); d1_err = std(diag1) / sqrt(length(diag1))
            d2_val = mean(diag2); d2_err = std(diag2) / sqrt(length(diag2))
            return (d1_val, d1_err, d2_val, d2_err)
        end

        # Energy
        energy_sample = compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)

        # --- Exact correlations (precompute shared quantities) ---
        energy_exact = NaN
        exact_vert = Dict{Symbol, Float64}()
        exact_horiz = Dict{Symbol, Float64}()
        exact_diag_down = Dict{Symbol, Float64}()
        exact_diag_up = Dict{Symbol, Float64}()

        if can_compute_exact
            N_cols = length(op.columns)
            vq = op.virtual_qubits
            T_cols = _column_transfer_matrices(op)
            T_combined = reduce(*, T_cols)
            l_vec, r_vec, nf, _ = _fixed_points(T_combined)
            l_pre, r_suf = _precompute_shifted_vectors(T_cols, l_vec, r_vec)

            pauli_syms = [:X, :Y, :Z]
            pauli_mats = [_resolve_op(s) for s in pauli_syms]

            # Precompute single-operator transfer matrices
            E_O = Dict{Tuple{Int,Int,Int}, Matrix{ComplexF64}}()
            for c in 1:N_cols, (si, σ) in enumerate(pauli_mats), pos in 1:row
                E_O[(c, si, pos)] = get_transfer_matrix_with_operator(
                    op.columns[c], row, vq, σ; position=pos, optimizer=GreedyMethod())
            end

            for (si, σ) in enumerate(pauli_mats)
                sym = pauli_syms[si]

                # Vertical: open boundary (row-1 bonds), all columns
                if row > 1
                    vert_vals = Float64[]
                    for c in 1:N_cols, i in 1:row-1
                        E_OO = get_transfer_matrix_with_operator(
                            op.columns[c], row, vq, Dict(i => σ, i+1 => σ);
                            optimizer=GreedyMethod())
                        push!(vert_vals, real(dot(l_pre[c], E_OO * r_suf[c]) / nf))
                    end
                    exact_vert[sym] = mean(vert_vals)
                end

                # Horizontal: within-period + cross-period, all columns
                horiz_vals = Float64[]
                for c in 1:(N_cols-1), pos in 1:row
                    val = dot(l_pre[c],
                              E_O[(c, si, pos)] * E_O[(c+1, si, pos)] * r_suf[c+1]) / nf
                    push!(horiz_vals, real(val))
                end
                # Cross-period: col N → col 1'
                for pos in 1:row
                    val = dot(l_pre[N_cols],
                              E_O[(N_cols, si, pos)] * E_O[(1, si, pos)] * r_suf[1]) / nf
                    push!(horiz_vals, real(val))
                end
                exact_horiz[sym] = mean(horiz_vals)

                # NNN diagonal (if needed)
                if J2 != 0.0 && row > 1
                    diag_down_vals = Float64[]
                    diag_up_vals = Float64[]

                    # Diagonal-down ↘: (i, c) → (i+1, c+1), open vertical boundary
                    for i in 1:row-1
                        j_down = i + 1
                        for c in 1:(N_cols-1)
                            val_d = dot(l_pre[c],
                                        E_O[(c, si, i)] * E_O[(c+1, si, j_down)] * r_suf[c+1]) / nf
                            push!(diag_down_vals, real(val_d))
                        end
                        # Cross-period
                        val_d = dot(l_pre[N_cols],
                                    E_O[(N_cols, si, i)] * E_O[(1, si, j_down)] * r_suf[1]) / nf
                        push!(diag_down_vals, real(val_d))
                    end

                    # Diagonal-up ↗: (i, c) → (i-1, c+1), open vertical boundary
                    for i in 2:row
                        j_up = i - 1
                        for c in 1:(N_cols-1)
                            val_u = dot(l_pre[c],
                                        E_O[(c, si, i)] * E_O[(c+1, si, j_up)] * r_suf[c+1]) / nf
                            push!(diag_up_vals, real(val_u))
                        end
                        # Cross-period
                        val_u = dot(l_pre[N_cols],
                                    E_O[(N_cols, si, i)] * E_O[(1, si, j_up)] * r_suf[1]) / nf
                        push!(diag_up_vals, real(val_u))
                    end

                    exact_diag_down[sym] = mean(diag_down_vals)
                    exact_diag_up[sym] = mean(diag_up_vals)
                end
            end

            # Energy: compute_exact_heisenberg_energy returns per-column sums,
            # sampling compute_heisenberg_energy returns per-bond means.
            # Reconstruct energy from per-bond exact values to match sampling convention.
            SS_vert_exact = sum(get(exact_vert, s, 0.0) for s in pauli_syms)
            SS_horiz_exact = sum(get(exact_horiz, s, 0.0) for s in pauli_syms)
            energy_exact = J1 * (row == 1 ? SS_horiz_exact : SS_vert_exact + SS_horiz_exact) / 4.0
            if J2 != 0.0 && row > 1
                SS_diag_exact = sum(get(exact_diag_down, s, 0.0) + get(exact_diag_up, s, 0.0)
                                    for s in pauli_syms)
                energy_exact += J2 * SS_diag_exact / 4.0
            end
        end

        push!(labels, "E"); push!(sample_values, energy_sample)
        push!(exact_values, energy_exact); push!(sample_errors, 0.0)

        # Per-component NN correlations
        for (name, S, pauli) in [("XX", X_samples, :X), ("YY", Y_samples, :Y), ("ZZ", Z_samples, :Z)]
            vv, ve, hv, he = _nn_correlations(S, row)

            vert_ex = get(exact_vert, pauli, NaN)
            horiz_ex = get(exact_horiz, pauli, NaN)

            if !isnothing(vv)
                push!(labels, "⟨$(name)⟩ᵥ"); push!(sample_values, vv)
                push!(exact_values, vert_ex); push!(sample_errors, ve)
            end
            push!(labels, "⟨$(name)⟩ₕ"); push!(sample_values, hv)
            push!(exact_values, horiz_ex); push!(sample_errors, he)
        end

        # NNN diagonal correlations (only when J2 ≠ 0 and row > 1)
        if J2 != 0.0 && !isnothing(row) && row > 1
            for (name, S, pauli) in [("XX", X_samples, :X), ("YY", Y_samples, :Y), ("ZZ", Z_samples, :Z)]
                d1v, d1e, d2v, d2e = _diag_correlations(S, row)

                d1_exact = get(exact_diag_down, pauli, NaN)
                d2_exact = get(exact_diag_up, pauli, NaN)

                push!(labels, "⟨$(name)⟩↘"); push!(sample_values, d1v)
                push!(exact_values, d1_exact); push!(sample_errors, d1e)
                push!(labels, "⟨$(name)⟩↗"); push!(sample_values, d2v)
                push!(exact_values, d2_exact); push!(sample_errors, d2e)
            end
        end

        plot_title = title
        if !isnothing(row) && !isnothing(nqubits)
            j2_str = J2 != 0.0 ? ", J2=$J2" : ""
            plot_title = "Heisenberg Correlations: row=$row, J1=$J1$(j2_str), nqubits=$nqubits"
        end

    else
        # --- TFIM branch (original logic) ---
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
            energy_sample = compute_tfim_energy(X_samples, Z_samples, g, J, row)
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

        if can_compute_exact
            energy_raw, X_total, ZZ_vert_e, ZZ_horiz_e = compute_exact_energy(m, op)
            # compute_exact_energy returns per-column sums; divide by row for per-site averages
            # to match sampling convention (compute_tfim_energy uses per-site/per-bond means)
            energy_exact = energy_raw / row
            X_exact = X_total / row
            ZZ_horiz_exact = ZZ_horiz_e / row
            Z_exact = mean(real(expect(op, :Z; col=1, position=i)) for i in 1:row)
            if row > 1
                ZZ_vert_exact = ZZ_vert_e / row
            end
        end

        plot_title = title
        if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
            plot_title = "Expectation Values: row=$row, g=$g, nqubits=$nqubits"
        elseif !isnothing(g) && !isnothing(row)
            plot_title = "Expectation Values: row=$row, g=$g"
        end

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
    end  # model branch

    if isempty(labels)
        @warn "No expectation values to plot"
        return nothing
    end

    fig = Figure(size=(800, 400))
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
    model_str = get(input_args, :model, "tfim")
    is_heisenberg = model_str == "heisenberg_j1j2"

    params = result.final_params

    # Detect 2x2 unit cell
    is_2x2 = is_heisenberg &&
        (length(params) == 4 * PARAMS_PER_QUBIT_PER_LAYER * nqubits * p)

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

    println("\nComputing exact correlations (transfer matrix)...")
    println("Averaging over all positions 1 to $row...")

    # Compute correlations for each position and average
    # Separations are in period units; exact values at r*N_cols columns
    max_sep_periods = cld(max_separation, N_cols)
    separations = [r * N_cols for r in 1:max_sep_periods]
    exact_full_vals = zeros(Float64, max_sep_periods)
    exact_connected_vals = zeros(Float64, max_sep_periods)

    for pos in 1:row
        exact_full_pos = correlation_function(op, :Z, 1:max_sep_periods;
                                              position=pos, connected=false)
        exact_connected_pos = correlation_function(op, :Z, 1:max_sep_periods;
                                                   position=pos, connected=true)

        for (i, r) in enumerate(1:max_sep_periods)
            exact_full_vals[i] += real(exact_full_pos[r])
            exact_connected_vals[i] += real(exact_connected_pos[r])
        end
    end

    # Average over positions
    exact_full_vals ./= row
    exact_connected_vals ./= row

    _, gap, _, _ = compute_transfer_spectrum(op)
    correlation_length = N_cols / gap
    println("Correlation length ξ = $(round(correlation_length, digits=2)) columns")

    println("\nGenerating samples (conv_step=$conv_step, samples=$samples)...")
    if is_2x2
        rho, Z_samples, X_samples = sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                                            conv_step=conv_step,
                                                            samples=samples)
    else
        gates = op.columns[1]
        rho, Z_samples, X_samples = sample_quantum_channel(gates, row, nqubits;
                                                            conv_step=conv_step,
                                                            samples=samples)
    end

    Z_vec = Z_samples[conv_step+1:end]

    # Use row*N_cols for subsampling so each subchain has same-column-type measurements
    # (for 2x2 unit cells, row alone mixes odd/even column types)
    acf_row = row * N_cols
    lags, acf, acf_err, corr_full, corr_err, corr_connected, corr_connected_err = compute_acf(
        reshape(Float64.(Z_vec), 1, :); max_lag=max_sep_periods+1, row=acf_row
    )

    @show corr_err
    @show corr_connected_err

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

    fig = Figure(size=(800, is_heisenberg ? 1000 : 700))
    min_val = 1e-15

    if is_heisenberg
        J2 = get(input_args, :J2, 0.0)
        title_str = "Correlation Function: J2=$J2, row=$row, nqubits=$nqubits, ξ=$(round(correlation_length, digits=2))"
    else
        title_str = "Correlation Function: g=$g, row=$row, nqubits=$nqubits, ξ=$(round(correlation_length, digits=2))"
    end
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
        fit_params = fit_acf(separations, exact_connected_vals; include_zero=false)
        ξ_fitted = fit_params.ξ
        A_fitted = fit_params.A
        println("Fitted correlation length ξ = $(round(ξ_fitted, digits=3))")
        println("Fitted amplitude A = $(round(A_fitted, digits=4))")
        println("Transfer matrix ξ = $(round(correlation_length, digits=3))")
        println("Ratio ξ_fitted/ξ_transfer = $(round(ξ_fitted/correlation_length, digits=3))")

        r_check = separations[1:min(10, length(separations))]
        fitted_check = abs(A_fitted) .* exp.(-r_check ./ ξ_fitted)
        data_check = abs.(exact_connected_vals[1:length(r_check)])
        println("Fit check (first $(length(r_check)) points):")
        for (i, r) in enumerate(r_check)
            rel_err = abs(fitted_check[i] - data_check[i]) / data_check[i] * 100
            println("  r=$r: data=$(data_check[i]), fit=$(fitted_check[i]), rel_err=$(round(rel_err, digits=1))%")
        end

        r_fit = range(1, max_separation, length=100)
        fitted_curve = abs(A_fitted) .* exp.(-r_fit ./ ξ_fitted)
        fitted_curve_plot = max.(fitted_curve, min_val)
        lines!(ax2, r_fit, fitted_curve_plot,
               label="Fit: |A|*exp(-r/$(round(ξ_fitted, digits=2)))",
               color=:green, linewidth=3, linestyle=:dash)

        ax2.title = "Connected Correlation (ξ_fit=$(round(ξ_fitted, digits=2)), ξ_TM=$(round(correlation_length, digits=2)))"
    catch e
        @warn "Exponential fitting failed: $e"
        println("Skipping exponential fit overlay")
    end

    axislegend(ax2, position=:rt)

    # Fit power-law decay: C(r) = A * r^(-η)
    η_fitted = nothing
    A_power_fitted = nothing

    if is_heisenberg
        # Heisenberg: power-law fit on a separate log-log panel
        ax3 = Axis(fig[3, 1],
                   xlabel="Separation r",
                   ylabel="|⟨Z_i Z_{i+r}⟩_c|",
                   title="Connected Correlation (power-law fit)",
                   xscale=log10, yscale=log10)

        lines!(ax3, separations, exact_connected_abs, label="Exact contraction", color=:blue, linewidth=2)
        scatter!(ax3, separations, exact_connected_abs, color=:blue, markersize=8)
        scatter!(ax3, collect(sample_seps), sample_connected_abs,
                 label="Sampling ± std err", color=:red, markersize=8, marker=:diamond)
        errorbars!(ax3, collect(sample_seps), sample_connected_abs, err_low_conn, err_high_conn,
                   color=:red, whiskerwidth=6)

        try
            println("\nFitting connected correlation to A*r^(-η)...")
            power_fit_range = (1, min(max_separation, max(5, ceil(Int, 2 * correlation_length))))
            power_params = fit_acf_power(separations, exact_connected_vals; include_zero=false, fit_range=power_fit_range)
            η_fitted = power_params.η
            A_power_fitted = power_params.A
            println("Fitted exponent η = $(round(η_fitted, digits=3))")
            println("Fitted amplitude A = $(round(A_power_fitted, digits=4))")

            r_fit = range(1, max_separation, length=100)
            power_curve = abs(A_power_fitted) .* r_fit .^ (-η_fitted)
            power_curve_plot = max.(power_curve, min_val)
            lines!(ax3, r_fit, power_curve_plot,
                   label="Fit: |A|*r^(-$(round(η_fitted, digits=2)))",
                   color=:purple, linewidth=3, linestyle=:dot)
        catch e
            @warn "Power-law fitting failed: $e"
            println("Skipping power-law fit overlay")
        end

        axislegend(ax3, position=:rt)
    else
        # TFIM: power-law fit on the same panel as exponential
        try
            println("\nFitting connected correlation to A*r^(-η)...")
            power_fit_range = (1, min(max_separation, max(5, ceil(Int, 2 * correlation_length))))
            power_params = fit_acf_power(separations, exact_connected_vals; include_zero=false, fit_range=power_fit_range)
            η_fitted = power_params.η
            A_power_fitted = power_params.A
            println("Fitted exponent η = $(round(η_fitted, digits=3))")
            println("Fitted amplitude A = $(round(A_power_fitted, digits=4))")

            r_fit = range(1, max_separation, length=100)
            power_curve = abs(A_power_fitted) .* r_fit .^ (-η_fitted)
            power_curve_plot = max.(power_curve, min_val)
            lines!(ax2, r_fit, power_curve_plot,
                   label="Fit: |A|*r^(-$(round(η_fitted, digits=2)))",
                   color=:purple, linewidth=3, linestyle=:dot)
        catch e
            @warn "Power-law fitting failed: $e"
            println("Skipping power-law fit overlay")
        end
    end

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
        power_law_exponent = η_fitted,
        power_law_amplitude = A_power_fitted,
        g = g, row = row, nqubits = nqubits
    )

    return fig, data
end

# ============================================================================
# Multi-g comparison plots
# ============================================================================

"""
    plot_energy_error_vs_g(data_dir::String, scan_values::Vector{Float64};
                           model="tfim", J=1.0, J1=1.0, row=3, nqubits=5, p=3,
                           conv_step=100, samples=1000,
                           pepskit_file=nothing, dmrg_file=nothing, save_path=nothing)

Plot energy error for different scan parameter values. Supports TFIM (scan over g)
and Heisenberg J1-J2 (scan over J2).

# Arguments
- `data_dir`: Directory containing result JSON files
- `scan_values`: Vector of scan parameter values (g for TFIM, J2 for Heisenberg)
- `model`: Model type, `"tfim"` (default) or `"heisenberg_j1j2"`
- `J`: Coupling strength for TFIM (default: 1.0)
- `J1`: J1 coupling for Heisenberg (default: 1.0)
- `row`: Number of rows (default: 3)
- `nqubits`: Number of qubits (default: 5)
- `p`: Circuit depth (default: 3)
- `conv_step`: Convergence steps for resampling (default: 100)
- `samples`: Number of samples for resampling (default: 1000)
- `pepskit_file`: Path to PEPSKit reference results JSON (optional)
- `dmrg_file`: Path to DMRG results JSON (optional)
- `save_path`: Path to save figure (optional)

# Returns
- `fig`: Makie Figure object
- `data`: NamedTuple with (scan_values, energies_exact, energies_ref, energies_dmrg, errors)

# Example
```julia
# TFIM
fig, data = plot_energy_error_vs_g("project/results", [1.0, 2.0, 3.0, 4.0];
                                   dmrg_file="project/results/dmrg_tfim_100x3.json")

# Heisenberg J1-J2
fig, data = plot_energy_error_vs_g("project/results", [0.0, 0.1, 0.2, 0.5];
                                   model="heisenberg_j1j2", J1=1.0, row=4, p=3, nqubits=3,
                                   dmrg_file="project/results/dmrg_j1j2_100x4.json")
```
"""
function plot_energy_error_vs_g(data_dir::String, scan_values::Vector{Float64};
                                model::String="tfim",
                                J=1.0, J1::Float64=1.0, row=3, nqubits=3, p=3,
                                conv_step::Int=100, samples::Int=1000000,
                                pepskit_file::Union{String,Nothing}=nothing,
                                dmrg_file::Union{String,Nothing}=nothing,
                                save_path::Union{String,Nothing}=nothing)

    m = _construct_model(model, Dict{Symbol,Any}(:J => Float64(J), :g => 1.0, :J1 => J1, :J2 => 0.0))
    is_heisenberg = m isa HeisenbergJ1J2
    scan_label = is_heisenberg ? "J2" : "g"

    println("="^70)
    println("Energy Error vs $scan_label Analysis (model=$model)")
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

    for val in scan_values
        # Build filename based on model
        if is_heisenberg
            candidates = [
                joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
                joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
                joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            ]
            filename = ""
            for c in candidates
                if isfile(c)
                    filename = c
                    break
                end
            end
            if isempty(filename)
                @warn "No file found for J2=$val, tried $(length(candidates)) patterns, skipping"
                continue
            end
        else
            filename = joinpath(data_dir, "circuit_J=$(J)_g=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")
        end

        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping $scan_label=$val"
            continue
        end

        # Load result
        result, input_args = load_result(filename)

        # Compute exact energy from optimized parameters
        virtual_qubits = (nqubits - 1) ÷ 2
        share_params = get(input_args, :share_params, true)

        if is_heisenberg
            # Use resample-based energy for Heisenberg
            resample_result = resample_circuit(filename; conv_step=conv_step, samples=samples, measure_y=true)
            if isnothing(resample_result)
                @warn "Resampling failed for $scan_label=$val, skipping"
                continue
            end
            _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
            Z_samples = Z_samples[conv_step+1:end] 
            X_samples = X_samples[conv_step+1:end]
            Y_samples = Y_samples[conv_step+1:end]
            energy_exact = compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, val, row)
        else
            g = val
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
        end

        push!(g_vals_found, val)
        push!(energies_exact, energy_exact)

        # Get reference energy and compute error
        if haskey(ref_energies, val)
            energy_ref = ref_energies[val]
            error = abs(energy_exact - energy_ref)
            push!(energies_ref, energy_ref)
            push!(errors, error)
            println("$scan_label=$val: E_exact=$(round(energy_exact, digits=6)), E_ref=$(round(energy_ref, digits=6)), Error=$(round(error, digits=6))")
        else
            push!(energies_ref, NaN)
            push!(errors, NaN)
            println("$scan_label=$val: E_exact=$(round(energy_exact, digits=6)), No reference")
        end

        # Get DMRG energy
        if haskey(dmrg_energies, val)
            push!(energies_dmrg, dmrg_energies[val])
            println("       E_dmrg=$(round(dmrg_energies[val], digits=6))")
        else
            push!(energies_dmrg, NaN)
        end
    end

    if isempty(g_vals_found)
        error("No valid results found for any $scan_label value")
    end

    # Axis labels and titles based on model
    xlabel_str = is_heisenberg ? "J2 / J1" : "Transverse field g"
    title_energy = is_heisenberg ? "Heisenberg J1-J2: Energy vs J2 (row=$row, p=$p)" : "Ground State Energy: Exact vs Reference"
    title_error = is_heisenberg ? "Energy Error vs J2" : "Energy Error vs g"

    # Create figure
    fig = Figure(size=(1000, 800))

    # Plot 1: Energies comparison
    ax1 = Axis(fig[1, 1],
               xlabel=xlabel_str,
               ylabel="Energy",
               title=title_energy)

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
               xlabel=xlabel_str,
               ylabel="Energy Error",
               title=title_error,
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
        scan_values = g_vals_found,
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
    skipped_g = Float64[]

    for (idx, g) in enumerate(g_values)
        filename = joinpath(data_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")

        if !isfile(filename)
            push!(skipped_g, g)
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
            exp_fit_range = (1, min(max_separation, max(5, ceil(Int, 2 * ξ))))
            fit_params = fit_acf(collect(separations), corr_vals; include_zero=false, fit_range=exp_fit_range)
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

    if !isempty(skipped_g)
        println("\nSkipped $(length(skipped_g)) missing g values: $skipped_g")
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

# ============================================================================
# plot_correlation_vs_J2
# ============================================================================

"""
    plot_correlation_vs_J2(data_dir::String, J2_values::Vector{Float64};
                           J1=1.0, row=3, nqubits=3, p=3,
                           max_separation=20,
                           connected=true,
                           dmrg_file=nothing,
                           save_path=nothing)

Plot correlation length ξ vs J2 for the Heisenberg J1-J2 model.

# Arguments
- `data_dir`: Directory containing result JSON files
- `J2_values`: Vector of J2 values to scan
- `J1`: Nearest-neighbor coupling (default: 1.0)
- `row`: Number of rows (default: 3)
- `nqubits`: Number of qubits (default: 3)
- `p`: Circuit depth (default: 3)
- `max_separation`: Maximum separation for correlation function (default: 20)
- `connected`: Use connected correlation (default: true)
- `dmrg_file`: Optional JSON file with DMRG reference (expected keys: `J2_values`, `correlation_lengths`)
- `save_path`: Path to save figure (optional)

# Returns
- `fig`: Makie Figure object
- `data`: Dict mapping J2 values to correlation data
"""
function plot_correlation_vs_J2(data_dir::String, J2_values::Vector{Float64};
                                J1=1.0, row=3, nqubits=3, p=3,
                                max_separation=20,
                                connected=true,
                                dmrg_file::Union{String,Nothing}=nothing,
                                save_path::Union{String,Nothing}=nothing)

    println("="^70)
    println("Correlation Length vs J2 Analysis")
    println("="^70)
    println("Connected: $connected, J1=$J1, row=$row, nqubits=$nqubits, p=$p")

    correlation_data = Dict{Float64, NamedTuple}()
    colors = [:blue, :green, :red, :orange, :purple, :brown, :pink, :gray]
    skipped_J2 = Float64[]

    for (idx, val) in enumerate(J2_values)
        # Find file (prefer 2x2 over 1x1)
        candidates = [
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
        ]
        filename = ""
        for c in candidates
            if isfile(c)
                filename = c
                break
            end
        end
        if isempty(filename)
            push!(skipped_J2, val)
            continue
        end

        println("\nProcessing J2=$val  →  $(basename(filename))")

        result, input_args = load_result(filename)
        virtual_qubits = (nqubits - 1) ÷ 2
        share_params = get(input_args, :share_params, true)

        # Detect 2x2 unit cell from filename
        is_2x2 = endswith(filename, "_2x2.json")

        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(result.final_params, p, row, nqubits)
            op = TransferOperator(gates_odd, gates_even, row, nqubits)
        else
            gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)
            op = TransferOperator(gates, row, nqubits)
        end

        # Compute correlation length from transfer matrix
        # gap is per period; multiply by N_cols to get ξ in column units
        N_cols = length(op.columns)
        _, gap, _, _ = compute_transfer_spectrum(op)
        ξ = N_cols / gap

        # Compute correlation function at every column separation
        T_cols = _column_transfer_matrices(op)
        T_combined = reduce(*, T_cols)
        l_vec, r_vec, nf_corr, _ = _fixed_points(T_combined)
        _, r_suf = _precompute_shifted_vectors(T_cols, l_vec, r_vec)
        σz = _resolve_op(:Z)

        E_O_col = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
        for c in 1:N_cols, pos in 1:row
            E_O_col[(c, pos)] = get_transfer_matrix_with_operator(
                op.columns[c], op.row, op.virtual_qubits, Dict(pos => σz);
                optimizer=GreedyMethod())
        end

        corr_vals = zeros(Float64, max_separation)
        for pos in 1:row
            l_side = E_O_col[(1, pos)]' * l_vec
            dim = size(T_cols[1], 1)
            middle = Matrix{ComplexF64}(I, dim, dim)
            for d in 1:max_separation
                target_type = (d % N_cols) + 1
                right_part = E_O_col[(target_type, pos)] * r_suf[target_type]
                full_val = real(dot(l_side, middle * right_part) / nf_corr)
                if connected
                    one_pt = real(dot(l_vec, E_O_col[(1, pos)] * r_suf[1]) / nf_corr)
                    corr_vals[d] += full_val - one_pt^2
                else
                    corr_vals[d] += full_val
                end
                middle = middle * T_cols[target_type]
            end
        end
        corr_vals ./= row
        col_seps = collect(Float64, 1:max_separation)

        # Fit with limited range
        ξ_fitted = nothing
        try
            max_fit_col = min(max_separation, max(5, ceil(Int, 2 * ξ)))
            exp_fit_range = (1, max_fit_col)
            fit_params = fit_acf(col_seps, corr_vals; include_zero=false, fit_range=exp_fit_range)
            ξ_fitted = fit_params.ξ
            println("  ξ_transfer = $(round(ξ, digits=3)), ξ_fitted = $(round(ξ_fitted, digits=3))")
        catch e
            println("  ξ_transfer = $(round(ξ, digits=3)), fitting failed")
        end

        correlation_data[val] = (
            separations = col_seps,
            correlations = corr_vals,
            correlation_length = ξ,
            correlation_length_fitted = ξ_fitted,
            color = colors[mod1(idx, length(colors))]
        )
    end

    if !isempty(skipped_J2)
        println("\nSkipped $(length(skipped_J2)) missing J2 values: $skipped_J2")
    end

    if isempty(correlation_data)
        error("No valid results found for any J2 value")
    end

    # Create figure
    fig = Figure(size=(800, 600))

    ax = Axis(fig[1, 1],
              xlabel="J₂ / J₁",
              ylabel="Correlation Length ξ",
              title="Correlation Length vs J₂ (J₁=$J1, row=$row, D=$(nqubits-1))")

    # Plot correlation lengths from transfer matrix
    J2_sorted = sort(collect(keys(correlation_data)))
    ξ_transfer = [correlation_data[j].correlation_length for j in J2_sorted]

    lines!(ax, J2_sorted, ξ_transfer,
           color=:blue, linewidth=2, label="IsoPEPS (transfer matrix)")
    scatter!(ax, J2_sorted, ξ_transfer,
             color=:blue, markersize=12)

    # Overlay DMRG correlation lengths if provided
    if dmrg_file !== nothing && isfile(dmrg_file)
        println("\nLoading DMRG reference from: $dmrg_file")
        dmrg_data = open(dmrg_file, "r") do io
            JSON3.read(io)
        end

        if haskey(dmrg_data, :correlation_lengths)
            j2_key = haskey(dmrg_data, :scan_values) ? :scan_values : :J2_values
            dmrg_J2 = Float64.(collect(dmrg_data[j2_key]))
            dmrg_ξ = collect(dmrg_data.correlation_lengths)

            valid = [i for i in eachindex(dmrg_J2)
                     if dmrg_ξ[i] !== nothing && isfinite(Float64(dmrg_ξ[i])) && Float64(dmrg_ξ[i]) < 1e5]
            dmrg_J2_valid = dmrg_J2[valid]
            dmrg_ξ_valid = Float64.(dmrg_ξ[valid])

            lines!(ax, dmrg_J2_valid, dmrg_ξ_valid,
                   color=:red, linewidth=2, linestyle=:dash, label="DMRG reference")
            scatter!(ax, dmrg_J2_valid, dmrg_ξ_valid,
                     color=:red, markersize=8, marker=:diamond)
            println("  DMRG: $(length(dmrg_J2_valid)) valid J2 points loaded")
        end
    elseif dmrg_file !== nothing
        println("DMRG file not found: $dmrg_file")
    end

    axislegend(ax, position=:lt)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end

    return fig, correlation_data
end

# ============================================================================
# plot_M2_vs_J2
# ============================================================================

"""
    plot_M2_vs_J2(data_dir::String, J2_values::Vector{Float64};
                  J1=1.0, row=3, nqubits=3, p=3,
                  conv_step=100, samples=1000000,
                  max_separation=20,
                  dmrg_file=nothing,
                  save_path=nothing)

Plot magnetic order parameter squared M²(q) vs J2 for the Heisenberg J1-J2 model.

Computes and plots M²(q0) with q0=(π,π) (Néel order) and M²(q1) with q1=(π,0)
(stripe order) as functions of J2. The crossing/transition between the two
order parameters signals the phase boundary.

# Arguments
- `data_dir`: Directory containing result JSON files
- `J2_values`: Vector of J2 values to scan
- `J1`: Nearest-neighbor coupling (default: 1.0)
- `row`, `nqubits`, `p`: Circuit parameters
- `conv_step`: Burn-in steps for the quantum channel Markov chain
- `samples`: Number of measurement samples
- `max_separation`: Max column separation for structure factor sum
- `dmrg_file`: Optional JSON file with DMRG reference M² data
  (expected keys: `J2_values`, `M2_neel`, `M2_stripe`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, data)` tuple
"""
function plot_M2_vs_J2(data_dir::String, J2_values::Vector{Float64};
                       J1=1.0, row=3, nqubits=3, p=3,
                       conv_step=100, samples=1000000,
                       max_separation::Int=20,
                       dmrg_file=nothing,
                       save_path=nothing)

    q0 = (Float64(π), Float64(π))   # Néel
    q1 = (Float64(π), 0.0)          # Stripe

    J2_found = Float64[]
    M2_neel = Float64[]
    M2_stripe = Float64[]

    println("=== M²(q) vs J2 ===")
    println("q0 = (π,π) [Néel],  q1 = (π,0) [Stripe]")
    println("row=$row, nqubits=$nqubits, p=$p, max_sep=$max_separation")

    for val in J2_values
        # Find file (same pattern as plot_energy_error_vs_g)
        candidates = [
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
        ]
        filename = ""
        for c in candidates
            if isfile(c)
                filename = c
                break
            end
        end
        if isempty(filename)
            @warn "No file found for J2=$val, tried $(length(candidates)) patterns, skipping"
            continue
        end

        println("\nJ2=$val  →  $(basename(filename))")

        resample_result = resample_circuit(filename; conv_step=conv_step,
                                           samples=samples, measure_y=true)
        if isnothing(resample_result)
            @warn "Resampling failed for J2=$val, skipping"
            continue
        end
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = Y_samples[conv_step+1:end]

        m2_neel   = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q0;
                                           max_separation=max_separation)
        m2_stripe = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q1;
                                           max_separation=max_separation)

        push!(J2_found, val)
        push!(M2_neel, m2_neel)
        push!(M2_stripe, m2_stripe)

        println("  M²(π,π) = $(round(m2_neel, digits=6)),  M²(π,0) = $(round(m2_stripe, digits=6))")
    end

    if isempty(J2_found)
        error("No data found for any J2 value")
    end

    # --- Plot ---
    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
              xlabel="J₂ / J₁",
              ylabel="M²(q)",
              title="Magnetic Order: row=$row, nqubits=$nqubits, p=$p")

    scatterlines!(ax, J2_found, M2_neel,
                  label="M²(π,π) Néel", color=:blue, marker=:circle, markersize=10, linewidth=2)
    scatterlines!(ax, J2_found, M2_stripe,
                  label="M²(π,0) Stripe", color=:red, marker=:diamond, markersize=10, linewidth=2)

    # Overlay DMRG reference if provided
    if dmrg_file !== nothing && isfile(dmrg_file)
        dmrg_data = JSON3.read(read(dmrg_file, String))
        if haskey(dmrg_data, :J2_values) && haskey(dmrg_data, :M2_neel)
            dmrg_J2 = Float64.(dmrg_data[:J2_values])
            dmrg_neel = Float64.(dmrg_data[:M2_neel])
            scatterlines!(ax, dmrg_J2, dmrg_neel,
                          label="DMRG M²(π,π)", color=:blue, linestyle=:dash,
                          marker=:utriangle, markersize=8, linewidth=1.5)
        end
        if haskey(dmrg_data, :J2_values) && haskey(dmrg_data, :M2_stripe)
            dmrg_J2 = Float64.(dmrg_data[:J2_values])
            dmrg_stripe = Float64.(dmrg_data[:M2_stripe])
            scatterlines!(ax, dmrg_J2, dmrg_stripe,
                          label="DMRG M²(π,0)", color=:red, linestyle=:dash,
                          marker=:utriangle, markersize=8, linewidth=1.5)
        end
    elseif dmrg_file !== nothing
        @warn "DMRG file not found: $dmrg_file"
    end

    axislegend(ax, position=:rt)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end

    data = (
        J2_values = J2_found,
        M2_neel = M2_neel,
        M2_stripe = M2_stripe,
        row = row, nqubits = nqubits, p = p
    )

    return fig, data
end

