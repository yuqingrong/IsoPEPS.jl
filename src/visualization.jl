# Visualization: plotting and fitting functions

# ============================================================================
# Paper-quality theme
# ============================================================================

const PAPER_FIGSIZE = (246, 170)         # PRL/PRX single column (3.375 in × 2.36 in)
const PAPER_FIGSIZE_WIDE = (510, 200)    # PRX/Nature double column (7.08 in × 2.78 in)
const PAPER_FONT = "Helvetica"
const PAPER_FONTSIZE = 10
const PAPER_AXIS_LABELSIZE = 10
const PAPER_TICKLABELSIZE = 9
const PAPER_TITLESIZE = 11
const PAPER_LEGEND_LABELSIZE = 8

"""
    paper_theme()

Science/PRX-style Makie theme: Helvetica sans-serif, compact margins, framed
legends, light grid. Apply with `set_theme!(paper_theme())` or
`with_theme(paper_theme()) do ... end`.
"""
function paper_theme()
    Theme(
        fontsize = PAPER_FONTSIZE,
        font = PAPER_FONT,
        figure_padding = 6,
        palette = (color = [:steelblue, :firebrick, :seagreen, :darkorange,
                            :purple, :saddlebrown, :hotpink, :teal, :gray],),
        Axis = (
            xlabelsize = PAPER_AXIS_LABELSIZE, ylabelsize = PAPER_AXIS_LABELSIZE,
            xticklabelsize = PAPER_TICKLABELSIZE, yticklabelsize = PAPER_TICKLABELSIZE,
            titlesize = PAPER_TITLESIZE,
            xgridvisible = true, ygridvisible = true,
            xgridcolor = (:gray, 0.25), ygridcolor = (:gray, 0.25),
            xgridwidth = 0.5, ygridwidth = 0.5,
            spinewidth = 0.8,
            xtickwidth = 0.8, ytickwidth = 0.8,
        ),
        Legend = (
            framevisible = true, framewidth = 0.5,
            labelsize = PAPER_LEGEND_LABELSIZE, padding = (3, 3, 3, 3),
            rowgap = 1,
        ),
        Lines = (linewidth = 1.0, cycle = [:color]),
        Scatter = (markersize = 6, strokewidth = 0.5, cycle = [:color]),
        ScatterLines = (linewidth = 1.0, markersize = 6, cycle = [:color]),
        Errorbars = (linewidth = 0.8, whiskerwidth = 4),
    )
end

function compact_reference_label(kind::Symbol, value::Real)
    rounded_value = round(value, digits=4)
    if kind === :pepskit
        return "PEPSKit ($rounded_value)"
    elseif kind === :dmrg
        return "DMRG ($rounded_value)"
    else
        throw(ArgumentError("unknown reference label kind: $kind"))
    end
end

function m2_phase_annotations(ymax::Real)
    [
        (x=0.20, y=0.05, label="Neel order", align=(:center, :center)),
        (x=0.57, y=0.05, label="VBS", align=(:center, :center)),
        (x=0.80, y=0.05, label="Stripe order", align=(:center, :center)),
    ]
end

function add_paper_legend!(ax::Axis; position=:rt, nbanks::Int=1)
    axislegend(ax;
               position=position,
               nbanks=nbanks,
               labelsize=PAPER_LEGEND_LABELSIZE,
               padding=(1, 1, 1, 1),
               margin=(1, 1, 1, 1),
               framevisible=false)
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
    fig = Figure(size=PAPER_FIGSIZE)

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
    scatter!(ax, lags_vec, plot_y, label="Data", color=:steelblue)

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

        lines!(ax, lags_vec, fit_curve, linestyle=:dash, color=:firebrick, label=label_text)
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
        lines!(ax, theory_lags_vec, theory_y, linestyle=:solid, color=:seagreen,
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
        lines!(ax, exact_lags_vec, exact_y, linestyle=:dot, color=:purple,
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
        lines!(ax, dom_lags_vec, dom_y, linestyle=:dashdot, color=:darkorange,
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
                    label="λ_eff(r) from data", color=:steelblue)
        end

        if !isnothing(lambda_eff_theory)
            theory_lags, theory_lambda_eff = lambda_eff_theory
            theory_lags_vec = collect(theory_lags)
            theory_vals = abs.(collect(theory_lambda_eff))
            valid_theory = isfinite.(theory_vals) .& (theory_vals .> 0) .& (theory_vals .< 10)
            if any(valid_theory)
                lines!(ax2, theory_lags_vec[valid_theory], theory_vals[valid_theory],
                      linestyle=:solid, color=:purple,
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
                      linestyle=:dashdot, color=:darkorange,
                      label="λ_eff (dominant modes)")
            end
        end

        if !isnothing(lambda_theory)
            hlines!(ax2, [lambda_theory], color=:seagreen, linestyle=:dash,
                   label="λ_slow (contributing) = $(round(lambda_theory, digits=4))")
        end

        if !isnothing(fit_params)
            λ_fit = haskey(fit_params, :λ₂) ? fit_params.λ₂ :
                    (haskey(fit_params, :λ₂_magnitude) ? fit_params.λ₂_magnitude : exp(-1/fit_params.ξ))
            hlines!(ax2, [λ_fit], color=:firebrick, linestyle=:dot,
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
                                ylabel::String="E/site",
                                title::String="",
                                logscale::Bool=false,
                                ylims::Union{Tuple{Real,Real},Nothing}=(-0.7, 0.2),
                                save_path::Union{String,Nothing}=nothing,
                                g::Union{Real,Nothing}=nothing,
                                row::Union{Int,Nothing}=nothing,
                                nqubits::Union{Int,Nothing}=nothing,
                                pepskit_results_file::Union{String,Nothing}=nothing,
                                dmrg_bulk_file::Union{String,Nothing}=nothing,
                                exact_energy::Union{Real,Nothing}=nothing,
                                J2::Union{Real,Nothing}=nothing)

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

    dmrg_ref = nothing
    if !isnothing(dmrg_bulk_file) && isfile(dmrg_bulk_file)
        dmrg_data = open(dmrg_bulk_file, "r") do io
            JSON3.read(io, Dict)
        end
        scan_vals = Float64.(collect(get(dmrg_data, "scan_values", get(dmrg_data, "J2_values", nothing))))
        e_bulk = Float64.(collect(dmrg_data["e_bulk_values"]))
        scan_param = if haskey(dmrg_data["parameters"], "scan_param")
            string(dmrg_data["parameters"]["scan_param"])
        elseif haskey(dmrg_data, "J2_values")
            "J2"
        else
            "g"
        end

        scan_target = if scan_param == "g" && !isnothing(g)
            Float64(g)
        elseif scan_param == "J2" && !isnothing(J2)
            Float64(J2)
        else
            nothing
        end

        if !isnothing(scan_target)
            idx = argmin(abs.(scan_vals .- scan_target))
            dmrg_ref = e_bulk[idx]
            if abs(scan_vals[idx] - scan_target) > 1e-6
                @warn "$scan_param=$scan_target not found in DMRG bulk, using closest $scan_param=$(scan_vals[idx])"
            end
        end
    end

    plot_title = title
    if !isnothing(g) && !isnothing(row) && !isnothing(nqubits)
        plot_title = ""
    elseif !isnothing(g) && !isnothing(row)
        plot_title = ""
    elseif !isnothing(g)
        plot_title = ""
    end

    fig = with_theme(paper_theme()) do
        fig = Figure(size=PAPER_FIGSIZE)
        ax = Axis(fig[1, 1];
                  xlabel = "Optimization step",
                  ylabel = ylabel,
                  title  = plot_title,
                  yscale = logscale ? log10 : identity,
                  yticks = -1.0:0.1:0.1,
                  limits = (nothing, isnothing(ylims) ? nothing : (Float64(ylims[1]), Float64(ylims[2]))))

        lines!(ax, collect(steps), collect(values), label="IsoPEPS (sampling)")

        if !isnothing(ref_energy)
            ref_label = compact_reference_label(:pepskit, ref_energy)
            hlines!(ax, [ref_energy], linestyle=:dash, color=:firebrick, label=ref_label)
        end

        if !isnothing(dmrg_ref)
            dmrg_label = "DMRG (4x1000 bulk)"
            hlines!(ax, [dmrg_ref], linestyle=:dash, color=:darkorange, label=dmrg_label)
        end

        if !isnothing(exact_energy)
            hlines!(ax, [exact_energy], linestyle=:dash, color=:royalblue,
                    label="Exact contraction")
        end

        hlines!(ax, [-0.495530], linestyle=:dash, color=:forestgreen,
                linewidth=2, label="DMRG (10×10)")
        hlines!(ax, [-0.49755], linestyle=:dot, color=:purple,
                linewidth=2, label="VMC (10×10)")

        add_paper_legend!(ax; position=:rt)

        if !isnothing(save_path)
            save(save_path, fig)
            @info "Figure saved to $save_path"
        end

        fig
    end

    return fig
end

function plot_training_history(result::Union{CircuitOptimizationResult, ExactOptimizationResult, ManifoldOptimizationResult}; kwargs...)
    n = length(result.energy_history)
    plot_training_history(1:n, result.energy_history; ylabel="E/site", kwargs...)
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

    fig = Figure(size=PAPER_FIGSIZE)
    ax = Axis(fig[1, 1],
              xlabel="Observable", ylabel="Value",
              title=plot_title,
              xticks=(1:length(labels), labels))

    colors = [v >= 0 ? :steelblue : :coral for v in values]
    barplot!(ax, 1:length(values), values, color=colors, strokewidth=1, strokecolor=:black)

    for (i, v) in enumerate(values)
        offset = v >= 0 ? 0.05 : -0.05
        align = v >= 0 ? (:center, :bottom) : (:center, :top)
        text!(ax, i, v + offset; text=string(round(v, digits=4)), align=align)
    end

    hlines!(ax, [0], color=:black)

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
                n_cols = div(N_s, row)
                # Open vertical bonds within each column: (pos, pos+1)
                vert_pairs = [S[i] * S[i+1] for i in 1:N_s-1 if i % row != 0]
                # Periodic wrap bonds: (row, col) <-> (1, col)
                wrap_pairs = [S[c*row] * S[(c-1)*row+1] for c in 1:n_cols]
                all_vert = vcat(vert_pairs, wrap_pairs)
                vert_val = mean(all_vert)
                vert_err = std(all_vert) / sqrt(length(all_vert))
                horiz_pairs = [S[i] * S[i+row] for i in 1:N_s-row]
                horiz_val = mean(horiz_pairs)
                horiz_err = std(horiz_pairs) / sqrt(length(horiz_pairs))
            end
            return (vert_val, vert_err, horiz_val, horiz_err)
        end

        # Helper: compute NNN diagonal and anti-diagonal correlations separately
        function _diag_correlations(S, row)
            N_s = length(S)
            n_cols = div(N_s, row)
            # Diagonal-down ↘: (pos, col) -> (pos+1, col+1), open vertical
            diag1 = [S[i] * S[i+row+1] for i in 1:N_s-row-1 if i % row != 0]
            # Periodic wrap ↘: (row, col) -> (1, col+1)
            wrap1 = [S[c*row] * S[c*row+1] for c in 1:n_cols-1]
            all_diag1 = vcat(diag1, wrap1)
            d1_val = mean(all_diag1); d1_err = std(all_diag1) / sqrt(length(all_diag1))
            # Diagonal-up ↗: (pos, col) -> (pos-1, col+1), open vertical
            diag2 = [S[i] * S[i+row-1] for i in 1:N_s-row+1 if (i-1) % row != 0]
            # Periodic wrap ↗: (1, col) -> (row, col+1)
            wrap2 = [S[(c-1)*row+1] * S[(c+1)*row] for c in 1:n_cols-1]
            all_diag2 = vcat(diag2, wrap2)
            d2_val = mean(all_diag2); d2_err = std(all_diag2) / sqrt(length(all_diag2))
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

                # Vertical: periodic boundary (row bonds), all columns
                if row > 1
                    vert_vals = Float64[]
                    for c in 1:N_cols, i in 1:row
                        j = i % row + 1
                        E_OO = get_transfer_matrix_with_operator(
                            op.columns[c], row, vq, Dict(i => σ, j => σ);
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

                    # Diagonal-down ↘: (i, c) → (i%row+1, c+1), periodic vertical
                    for i in 1:row
                        j_down = i % row + 1
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

                    # Diagonal-up ↗: (i, c) → ((i-2+row)%row+1, c+1), periodic vertical
                    for i in 1:row
                        j_up = (i - 2 + row) % row + 1
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
            n_cols = div(N, row)
            # Open vertical bonds within each column
            ZZ_vert_pairs = [Z_samples[i] * Z_samples[i+1] for i in 1:N-1 if i % row != 0]
            # Periodic wrap bonds: (row, col) <-> (1, col)
            ZZ_wrap_pairs = [Z_samples[c*row] * Z_samples[(c-1)*row+1] for c in 1:n_cols]
            ZZ_all_vert = vcat(ZZ_vert_pairs, ZZ_wrap_pairs)
            ZZ_vert_sample = mean(ZZ_all_vert)
            ZZ_vert_stderr = std(ZZ_all_vert) / sqrt(length(ZZ_all_vert))
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

    fig = Figure(size=PAPER_FIGSIZE)
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
                      valid_errors, color=:black)
        end
        for (i, (pos, val, err)) in enumerate(zip(positions_sample, sample_values, sample_errors))
            if !isnan(val)
                offset = val >= 0 ? (0.03 + err) : (-0.03 - err)
                align = val >= 0 ? (:center, :bottom) : (:center, :top)
                text!(ax, pos, val + offset; text=string(round(val, digits=3)), align=align)
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
                    text!(ax, pos, val + offset; text=string(round(val, digits=3)), align=align)
                end
            end
        end
    end

    hlines!(ax, [0], color=:black)
    axislegend(ax, position=:rb)

    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    return fig
end

# ============================================================================
# compute_variance_vs_samples
# ============================================================================

"""
    compute_variance_vs_samples(filename, sample_sizes; kwargs...)

Run the circuit once at `maximum(sample_sizes)` samples, then estimate the
variance of the energy estimator at each target sample size via bootstrap
subsampling.  The returned vectors can be passed directly to
`plot_variance_vs_samples`.

# Arguments
- `filename`: Path to a saved optimization result JSON
- `sample_sizes`: Vector of sample counts to evaluate (e.g. `[1_000, 10_000, 100_000]`)
- `conv_step`: Thermalization steps discarded from the front of the chain (default 100)
- `n_bootstrap`: Number of bootstrap draws per sample size (default 200)
- `save_path`: If provided, saves the results as JSON for later use

# Returns
- `(sample_sizes, variances, errors)` — each is a `Vector{Float64}` of the same
  length.  `errors` is the standard error on each variance estimate.

# Example
```julia
sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
ns, vars, errs = compute_variance_vs_samples(
    "project/results/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json",
    sizes; conv_step=100, n_bootstrap=200)
fig = plot_variance_vs_samples(ns, vars; errors=errs)
```
"""
function compute_variance_vs_samples(filename::String,
                                     sample_sizes::AbstractVector{Int};
                                     total_samples::Union{Int,Nothing}=nothing,
                                     conv_step::Int=100,
                                     n_bootstrap::Int=200,
                                     save_path::Union{String,Nothing}=nothing)

    result, input_args = load_result(filename)
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
        Z_pool = Z_all[conv_step+1:end]
        X_pool = X_all[conv_step+1:end]
        Y_pool = Y_all[conv_step+1:end]
    else
        _rho, Z_all, X_all, _params, _gates = resample_result
        Z_pool = Z_all[conv_step+1:end]
        X_pool = X_all[conv_step+1:end]
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

    variances    = Vector{Float64}(undef, length(sorted_sizes))
    errors       = Vector{Float64}(undef, length(sorted_sizes))

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

        v = var(energies)
        variances[k] = v
        errors[k]    = v * sqrt(2 / (n_bootstrap - 1))
        println("  n=$n_spins ($(n_cols) cols)  →  E̅ = $(round(mean(energies), sigdigits=6))  Var(E) = $(round(v, sigdigits=4))")
    end

    if !isnothing(save_path)
        save_results(save_path;
                     sample_sizes=collect(Int, sorted_sizes),
                     variances=variances,
                     errors=errors,
                     model=model_str, conv_step=conv_step, n_bootstrap=n_bootstrap)
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
        Z_pool = Z_all[conv_step+1:end]
        X_pool = X_all[conv_step+1:end]
        Y_pool = Y_all[conv_step+1:end]
    else
        _rho, Z_all, X_all, _params, _gates = resample_result
        Z_pool = Z_all[conv_step+1:end]
        X_pool = X_all[conv_step+1:end]
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

function plot_variance_vs_samples(sample_sizes::AbstractVector, variances::AbstractVector;
                                   errors::Union{AbstractVector,Nothing}=nothing,
                                   fit_scaling::Bool=true,
                                   figsize=nothing,
                                   save_path::Union{String,Nothing}=nothing)

    ns   = collect(Float64, sample_sizes)
    vars = collect(Float64, variances)

    _figsize = isnothing(figsize) ? PAPER_FIGSIZE : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel   = "N (samples)",
                  ylabel   = "Var(E)",
                  xscale   = log10,
                  yscale   = log10)

        if !isnothing(errors)
            errorbars!(ax, ns, vars, collect(Float64, errors);
                       color=(:steelblue, 0.4), whiskerwidth=4)
        end
        scatter!(ax, ns, vars;
                 label="bootstrap estimate", color=:steelblue, marker=:circle)

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

# ============================================================================
# plot_correlation_function
# ============================================================================

function plot_correlation_function(filename::String;
                                   max_separation::Int=20,
                                   conv_step::Int=1000,
                                   samples::Int=100000,
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

    min_val = 1e-15

    # Fit connected correlation to A·exp(−r/ξ)
    println("\nFitting connected correlation to A·exp(−r/ξ)...")
    ξ_fitted = nothing
    A_fitted = nothing
    try
        fit_params = fit_acf(separations, exact_connected_vals; include_zero=false)
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

        err_low = min.(sample_full_err_vals, sample_full_abs .- min_val)
        errorbars!(ax1, collect(sample_seps), sample_full_abs,
                   err_low, sample_full_err_vals;
                   color=:firebrick, whiskerwidth=4)
        scatter!(ax1, collect(sample_seps), sample_full_abs;
                 color=:firebrick, marker=:diamond, label="Sampling")

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

        err_low_c = min.(sample_connected_err_vals, sample_connected_abs .- min_val)
        errorbars!(ax2, collect(sample_seps), sample_connected_abs,
                   err_low_c, sample_connected_err_vals;
                   color=:firebrick, whiskerwidth=4)
        scatter!(ax2, collect(sample_seps), sample_connected_abs;
                 color=:firebrick, marker=:diamond, label="Sampling")

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

function _dict_from_series_spec(spec)
    if spec isa AbstractString
        return Dict{Symbol,Any}(:file => String(spec))
    elseif spec isa NamedTuple
        return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(spec))
    elseif spec isa AbstractDict
        return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(spec))
    else
        error("Series specs must be strings, NamedTuples, or Dicts; got $(typeof(spec))")
    end
end

function _series_spec_list(specs)
    if isnothing(specs)
        return Dict{Symbol,Any}[]
    elseif specs isa AbstractString || specs isa NamedTuple || specs isa AbstractDict
        return [_dict_from_series_spec(specs)]
    else
        return [_dict_from_series_spec(spec) for spec in specs]
    end
end

function _json_get_any(data, keys; default=nothing)
    for key in keys
        if haskey(data, key)
            return data[key]
        end
        skey = string(key)
        if haskey(data, skey)
            return data[skey]
        end
        symkey = Symbol(key)
        if haskey(data, symkey)
            return data[symkey]
        end
    end
    return default
end

function _auto_reference_label(file::String, fallback::String)
    base = basename(file)
    lower = lowercase(base)
    d_match = match(r"D=?(\d+)", base)
    ly_match = match(r"Ly(\d+)", base)

    if occursin("dmrg", lower)
        label = "DMRG"
        if !isnothing(ly_match)
            label *= " Ly=$(ly_match.captures[1])"
        end
        if !isnothing(d_match)
            label *= " D=$(d_match.captures[1])"
        end
        return label
    elseif occursin("pepskit", lower) || occursin("ipeps", lower)
        label = "iPEPS"
        if !isnothing(d_match)
            label *= " D=$(d_match.captures[1])"
        end
        return label
    end

    return fallback
end

function _unique_label(label::String, used::Set{String})
    if !(label in used)
        push!(used, label)
        return label
    end

    i = 2
    candidate = "$label ($i)"
    while candidate in used
        i += 1
        candidate = "$label ($i)"
    end
    push!(used, candidate)
    return candidate
end

function _load_scan_energy_series(file::String; label::Union{String,Nothing}=nothing,
                                  fallback_label::String="Reference",
                                  kind::Symbol=:reference)
    if !isfile(file)
        @warn "Reference file not found, skipping" file
        return nothing
    end

    data = open(file, "r") do io
        JSON3.read(io, Dict)
    end

    scan_vals = _json_get_any(data, ("scan_values", "J2_values", "g_values"))
    energies = _json_get_any(data, ("energies_per_site", "e_bulk_values", "energies", "Lx2_energies_per_site"))
    if isnothing(scan_vals) || isnothing(energies)
        @warn "Reference JSON does not contain expected scan/energy arrays; skipping." file
        return nothing
    end

    series_label = isnothing(label) ? _auto_reference_label(file, fallback_label) : label
    return (
        label = series_label,
        scan_values = Float64.(collect(scan_vals)),
        energies = Float64.(collect(energies)),
        files = [file],
        kind = kind,
    )
end

function _format_scan_value(val)
    return string(val)
end

function _expand_file_template(template::String, data_dir::String, val)
    path = replace(template,
                   "{val}" => _format_scan_value(val),
                   "{g}" => _format_scan_value(val),
                   "{J2}" => _format_scan_value(val))
    return isabspath(path) ? path : joinpath(data_dir, path)
end

function _find_circuit_result_file(data_dir::String, val, spec::Dict{Symbol,Any})
    if haskey(spec, :file_template)
        path = _expand_file_template(String(spec[:file_template]), data_dir, val)
        return isfile(path) ? path : ""
    end

    model = String(get(spec, :model, "tfim"))
    J = get(spec, :J, 1.0)
    J1 = Float64(get(spec, :J1, 1.0))
    row = Int(get(spec, :row, 3))
    p = Int(get(spec, :p, 3))
    nqubits = Int(get(spec, :nqubits, 3))

    if model == "heisenberg_j1j2"
        suffixes = collect(get(spec, :suffixes, ["_2x2", "_1x1", ""]))
        candidates = String[]
        for suffix in suffixes
            push!(candidates, joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)$(suffix).json"))
        end
        push!(candidates, joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"))

        for candidate in candidates
            if isfile(candidate)
                return candidate
            end
        end
        return ""
    else
        suffixes = collect(get(spec, :suffixes, ["_1x1", "_1x1_100*1000", "_1x1_6w", ""]))
        candidates = String[]
        for suffix in suffixes
            push!(candidates, joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)$(suffix).json"))
        end
        push!(candidates, joinpath(data_dir, "circuit_J=$(J)_g=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"))
        push!(candidates, joinpath(data_dir, "circuit_J=$(J)_g=$(val)_row=$(row)_nqubits=$(nqubits).json"))

        for candidate in candidates
            if isfile(candidate)
                return candidate
            end
        end
        return ""
    end
end

function _circuit_energy_mode(model::String, nqubits::Int, energy_source::Symbol; row::Int=3)
    if energy_source == :saved
        return :saved
    elseif energy_source in (:resampled, :sampled)
        return :sampled
    elseif energy_source != :computed
        error("Unsupported circuit energy_source=$energy_source. Use :computed, :resampled, or :saved.")
    elseif model == "heisenberg_j1j2"
        return :sampled
    end
    virtual_qubits = (nqubits - 1) ÷ 2
    bond_dim = 2^virtual_qubits
    ms = bond_dim^(2 * (row + 1))
    ms <= 20_000 && return :exact
    return :sampled
end

function _resampled_tfim_energy(filename::String, val, J, row::Int;
                                conv_step::Int, samples::Int, repeats::Int=1,
                                result::Union{CircuitOptimizationResult,Nothing}=nothing)
    energies = Float64[]
    for _ in 1:repeats
        resample_result = resample_circuit(filename; conv_step=conv_step, samples=samples)
        if isnothing(resample_result)
            continue
        end
        _rho, Z_samples, X_samples, _params, _gates = resample_result
        Z_samples = Z_samples[conv_step+1:end]
        X_samples = X_samples[conv_step+1:end]
        push!(energies, compute_tfim_energy(X_samples, Z_samples, Float64(val), Float64(J), row))
    end

    return isempty(energies) ? nothing : median(energies)
end

function _compute_circuit_energy_from_result(filename::String, result, input_args,
                                             val, spec::Dict{Symbol,Any},
                                             conv_step::Int, samples::Int)
    model = String(get(spec, :model, get(input_args, :model, "tfim")))
    nqubits = Int(get(spec, :nqubits, get(input_args, :nqubits, 3)))
    row = Int(get(spec, :row, get(input_args, :row, 3)))
    source = _circuit_energy_mode(model, nqubits, Symbol(get(spec, :energy_source, :computed)); row=row)

    if source == :saved
        return result.final_cost
    end

    J = get(spec, :J, get(input_args, :J, 1.0))
    J1 = Float64(get(spec, :J1, get(input_args, :J1, 1.0)))
    p = Int(get(spec, :p, get(input_args, :p, 3)))
    share_params = get(input_args, :share_params, get(spec, :share_params, true))
    virtual_qubits = (nqubits - 1) ÷ 2

    if model == "heisenberg_j1j2"
        resample_result = resample_circuit(filename; conv_step=conv_step, samples=samples, measure_y=true)
        if isnothing(resample_result)
            return nothing
        end
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_samples = Z_samples[conv_step+1:end]
        X_samples = X_samples[conv_step+1:end]
        Y_samples = Y_samples[conv_step+1:end]
        return compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, Float64(val), row)
    elseif source == :sampled
        resample_repeats = Int(get(spec, :resample_repeats, 1))
        return _resampled_tfim_energy(filename, val, J, row;
                                      conv_step=conv_step, samples=samples,
                                      repeats=resample_repeats, result=result)
    else
        gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)
        X_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, :X; position=i)) for i in 1:row)

        if row > 1
            ZZ_vert_exact = mean(real(IsoPEPS.expect(gates, row, virtual_qubits, Dict(i => :Z, i % row + 1 => :Z))) for i in 1:row)
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

        g = Float64(val)
        return -g * X_exact - Float64(J) * (row == 1 ? ZZ_horiz_exact : ZZ_vert_exact + ZZ_horiz_exact)
    end
end

function _load_circuit_energy_series(data_dir::String, scan_values::Vector{Float64},
                                     spec::Dict{Symbol,Any}, conv_step::Int, samples::Int)
    label = String(get(spec, :label, "IsoPEPS"))
    found_vals = Float64[]
    energies = Float64[]
    files = String[]

    for val in scan_values
        filename = _find_circuit_result_file(data_dir, val, spec)
        if isempty(filename)
            @warn "No circuit file found, skipping" label val
            continue
        end

        result, input_args = load_result(filename)
        energy = _compute_circuit_energy_from_result(filename, result, input_args, val, spec, conv_step, samples)
        if isnothing(energy)
            @warn "Energy computation failed, skipping" label val filename
            continue
        end

        push!(found_vals, Float64(val))
        push!(energies, Float64(energy))
        push!(files, filename)
    end

    if isempty(found_vals)
        return nothing
    end

    return (
        label = label,
        scan_values = found_vals,
        energies = energies,
        files = files,
        kind = :circuit,
    )
end

function _lookup_series_energy(series, val; atol=1e-8)
    idx = findfirst(x -> isapprox(x, val; atol=atol, rtol=0), series.scan_values)
    return isnothing(idx) ? NaN : series.energies[idx]
end

function _series_errors(circuit_series, reference_series)
    errors = Float64[]
    vals = Float64[]
    for (val, energy) in zip(circuit_series.scan_values, circuit_series.energies)
        ref_energy = _lookup_series_energy(reference_series, val)
        if isnan(ref_energy)
            push!(errors, NaN)
        else
            push!(errors, abs(energy - ref_energy))
        end
        push!(vals, val)
    end
    return (scan_values=vals, errors=errors)
end

"""
    plot_energy_error_vs_g(data_dir::String, scan_values::Vector{Float64};
                           model="tfim", J=1.0, J1=1.0, row=3, nqubits=5, p=3,
                           conv_step=100, samples=1000, resample_repeats=1,
                           pepskit_file=nothing, dmrg_file=nothing,
                           circuit_series=[], energy_source=:computed,
                           save_path=nothing)

Plot energy error for different scan parameter values. Supports TFIM (scan over g)
and Heisenberg J1-J2 (scan over J2). The default call plots one IsoPEPS
series, with optional PEPSKit and DMRG references. Additional circuit or
reference series can be overlaid with `circuit_series`, vector-valued
`dmrg_file`/`pepskit_file`, or `reference_files`.

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
- `resample_repeats`: Number of independent resampling runs to median for TFIM sampled energies (default: 1)
- `pepskit_file`: Path, vector of paths, or specs `(file=..., label=...)` for PEPSKit references
- `dmrg_file`: Path, vector of paths, or specs `(file=..., label=...)` for DMRG references
- `circuit_series`: Additional circuit specs, e.g. `(label="IsoPEPS χ=5", nqubits=5, suffixes=["_1x1_6w"])`
- `energy_source`: `:computed` recomputes energies; TFIM uses exact contraction up to `nqubits=3` and resampling from optimized parameters for `nqubits>=5`. `:resampled` always resamples from optimized parameters. `:saved` reads JSON `energy`
- `save_path`: Path to save figure (optional)

# Returns
- `fig`: Makie Figure object
- `data`: NamedTuple with legacy fields and richer `series`/`errors_by_reference` dictionaries

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
                                conv_step::Int=102, samples::Int=300000,
                                resample_repeats::Int=1,
                                pepskit_file=nothing,
                                dmrg_file=nothing,
                                reference_files=nothing,
                                circuit_series=[],
                                include_default_circuit::Bool=true,
                                energy_source::Symbol=:computed,
                                error_reference=:all,
                                figsize=nothing,
                                save_path::Union{String,Nothing}=nothing)

    m = _construct_model(model, Dict{Symbol,Any}(:J => Float64(J), :g => 1.0, :J1 => J1, :J2 => 0.0))
    is_heisenberg = m isa HeisenbergJ1J2
    scan_label = is_heisenberg ? "J2" : "g"

    println("="^70)
    println("Energy Error vs $scan_label Analysis (model=$model)")
    println("="^70)

    used_labels = Set{String}()

    circuit_specs = Dict{Symbol,Any}[]
    if include_default_circuit
        push!(circuit_specs, Dict{Symbol,Any}(
            :label => "IsoPEPS",
            :model => model,
            :J => J,
            :J1 => J1,
            :row => row,
            :p => p,
            :nqubits => nqubits,
            :energy_source => energy_source,
            :resample_repeats => resample_repeats,
        ))
    end
    append!(circuit_specs, _series_spec_list(circuit_series))
    for spec in circuit_specs
        spec[:model] = get(spec, :model, model)
        spec[:J] = get(spec, :J, J)
        spec[:J1] = get(spec, :J1, J1)
        spec[:row] = get(spec, :row, row)
        spec[:p] = get(spec, :p, p)
        spec[:nqubits] = get(spec, :nqubits, nqubits)
        spec[:energy_source] = get(spec, :energy_source, energy_source)
        spec[:resample_repeats] = get(spec, :resample_repeats, resample_repeats)
        spec[:label] = _unique_label(String(get(spec, :label, "IsoPEPS")), used_labels)
    end

    circuits = []
    for spec in circuit_specs
        println("Loading circuit series: $(spec[:label])")
        series = _load_circuit_energy_series(data_dir, scan_values, spec, conv_step, samples)
        if !isnothing(series)
            push!(circuits, series)
            println("  loaded $(length(series.scan_values)) points")
        end
    end

    references = []
    for spec in _series_spec_list(pepskit_file)
        file = String(spec[:file])
        label = haskey(spec, :label) ? String(spec[:label]) : nothing
        series = _load_scan_energy_series(file; label=label, fallback_label="iPEPS", kind=:reference)
        if !isnothing(series)
            label = _unique_label(series.label, used_labels)
            push!(references, (label=label, scan_values=series.scan_values,
                              energies=series.energies, files=series.files,
                              kind=series.kind))
            println("Loaded reference $label with $(length(series.scan_values)) points")
        end
    end

    for spec in _series_spec_list(dmrg_file)
        file = String(spec[:file])
        label = haskey(spec, :label) ? String(spec[:label]) : nothing
        series = _load_scan_energy_series(file; label=label, fallback_label="DMRG", kind=:reference)
        if !isnothing(series)
            label = _unique_label(series.label, used_labels)
            push!(references, (label=label, scan_values=series.scan_values,
                              energies=series.energies, files=series.files,
                              kind=series.kind))
            println("Loaded reference $label with $(length(series.scan_values)) points")
        end
    end

    for spec in _series_spec_list(reference_files)
        file = String(spec[:file])
        label = haskey(spec, :label) ? String(spec[:label]) : nothing
        fallback = String(get(spec, :fallback_label, "Reference"))
        series = _load_scan_energy_series(file; label=label, fallback_label=fallback, kind=:reference)
        if !isnothing(series)
            label = _unique_label(series.label, used_labels)
            push!(references, (label=label, scan_values=series.scan_values,
                              energies=series.energies, files=series.files,
                              kind=series.kind))
            println("Loaded reference $label with $(length(series.scan_values)) points")
        end
    end

    if isempty(circuits)
        error("No valid results found for any $scan_label value")
    end

    primary = first(circuits)
    primary_ref = isempty(references) ? nothing : first(references)
    primary_dmrg = findfirst(s -> startswith(s.label, "DMRG"), references)
    primary_dmrg_series = isnothing(primary_dmrg) ? nothing : references[primary_dmrg]

    g_vals_found = primary.scan_values
    energies_exact = primary.energies
    energies_ref = isnothing(primary_ref) ? fill(NaN, length(g_vals_found)) :
                   [_lookup_series_energy(primary_ref, val) for val in g_vals_found]
    errors = isnothing(primary_ref) ? fill(NaN, length(g_vals_found)) :
             [isnan(ref) ? NaN : abs(energy - ref) for (energy, ref) in zip(energies_exact, energies_ref)]
    energies_dmrg = isnothing(primary_dmrg_series) ? fill(NaN, length(g_vals_found)) :
                    [_lookup_series_energy(primary_dmrg_series, val) for val in g_vals_found]

    reference_subset = if error_reference == :all
        references
    elseif error_reference == :first
        isempty(references) ? [] : [first(references)]
    else
        wanted = String(error_reference)
        filter(s -> s.label == wanted, references)
    end

    errors_by_reference = Dict{String,Any}()
    for circuit in circuits
        for reference in reference_subset
            err = _series_errors(circuit, reference)
            if !all(isnan.(err.errors))
                errors_by_reference["$(circuit.label) − $(reference.label)"] = err
            end
        end
    end

    xlabel_str = is_heisenberg ? "J₂ / J₁" : "g"

    _w, _h = PAPER_FIGSIZE                       # 246 × 170 pt
    _figsize = isnothing(figsize) ? (_w, 2_h - 20) : figsize   # ~(246, 320) for 2 panels

    colors = [:steelblue, :darkorange, :purple, :teal, :brown, :gray40, :dodgerblue4, :tomato3]
    markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :cross, :xcross, :star5]

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        # ── Panel (a): energy per site ──────────────────────────────────────
        ax1 = Axis(fig[1, 1];
                   ylabel      = "E / site",
                   xticklabelsvisible = false,
                   xgridvisible = true,
                   ygridvisible = true)

        for (idx, series) in enumerate(circuits)
            scatterlines!(ax1, series.scan_values, series.energies;
                          label=series.label,
                          color=colors[mod1(idx, length(colors))],
                          marker=markers[mod1(idx, length(markers))],
                          linestyle=:solid)
        end

        for (idx, series) in enumerate(references)
            ref_color = colors[mod1(idx + length(circuits), length(colors))]
            ref_style = startswith(series.label, "DMRG") ? :dot : :dash
            scatterlines!(ax1, series.scan_values, series.energies;
                          label=series.label,
                          color=ref_color,
                          marker=markers[mod1(idx + length(circuits), length(markers))],
                          linestyle=ref_style)
        end

        text!(ax1, 0.03, 0.97; text="(a)", space=:relative,
              align=(:left, :top), fontsize=PAPER_TITLESIZE, font=:bold)
        add_paper_legend!(ax1; position=:lb)

        # ── Panel (b): energy error (log scale) ─────────────────────────────
        ax2 = Axis(fig[2, 1];
                   xlabel      = xlabel_str,
                   ylabel      = "|ΔE / site|",
                   yscale      = log10,
                   xgridvisible = true,
                   ygridvisible = true)

        has_error = false

        for (idx, (label, err)) in enumerate(sort(collect(errors_by_reference); by=first))
            mask = .!isnan.(err.errors)
            scatterlines!(ax2, err.scan_values[mask], err.errors[mask];
                          label=label,
                          color=colors[mod1(idx, length(colors))],
                          marker=markers[mod1(idx, length(markers))],
                          linestyle=:solid)
            has_error = true
        end

        if has_error
            add_paper_legend!(ax2; position=:rb)
        else
            text!(ax2, 0.5, 0.5; text="No reference data",
                  align=(:center, :center), space=:relative,
                  fontsize=PAPER_AXIS_LABELSIZE, color=:gray)
        end

        text!(ax2, 0.03, 0.97; text="(b)", space=:relative,
              align=(:left, :top), fontsize=PAPER_TITLESIZE, font=:bold)

        rowgap!(fig.layout, 4)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("\nFigure saved to: $save_path")
        end

        fig
    end

    series_data = Dict{String,Any}()
    for series in vcat(circuits, references)
        series_data[series.label] = series
    end

    data = (
        scan_values = g_vals_found,
        energies_exact = energies_exact,
        energies_ref = energies_ref,
        energies_dmrg = energies_dmrg,
        errors = errors,
        series = series_data,
        errors_by_reference = errors_by_reference,
    )

    return fig, data
end

"""
    plot_magnetization_vs_g(data_dir, g_values; kwargs...)

Plot ⟨Z⟩ (longitudinal) and ⟨X⟩ (transverse) magnetisation per site vs
transverse field g for the TFIM, using sampling data.

In the ordered phase (g < g_c ≈ 3.04) ⟨Z⟩ > 0 and ⟨X⟩ is small;
in the disordered phase (g > g_c) the roles reverse.

# Arguments
- `data_dir`: Directory containing result JSON files
- `g_values`: Vector of g values to scan
- `J`: Coupling (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters
- `conv_step`: Thermalization steps (default 100)
- `samples`: Number of spin measurements per g (default 500_000)
- `figsize`: Override figure size (default `PAPER_FIGSIZE`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, data)` where `data` is a `Dict` mapping each g to
  `(mZ=..., mX=...)`.
"""
function plot_magnetization_vs_g(data_dir::String, g_values::Vector{Float64};
                                  J::Float64=1.0,
                                  row::Int=3, p::Int=3, nqubits::Int=3,
                                  conv_step::Int=100,
                                  samples::Int=500_000,
                                  figsize=nothing,
                                  save_path::Union{String,Nothing}=nothing)

    sorted_g = sort(g_values)
    mZ_vals  = Float64[]
    mX_vals  = Float64[]
    g_found  = Float64[]

    println("=== plot_magnetization_vs_g ===")

    for g in sorted_g
        candidates = [
            joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json"),
            joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
        ]
        filename = ""
        for c in candidates
            isfile(c) && (filename = c; break)
        end
        if isempty(filename)
            @warn "No file found for g=$g, skipping"
            continue
        end

        println("  g=$g  →  $(basename(filename))")
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                           samples=samples, measure_y=false)
        if isnothing(resample_result)
            @warn "Resampling failed for g=$g, skipping"
            continue
        end
        _rho, Z_all, X_all, _params, _gates = resample_result
        Z_pool = Z_all[conv_step+1:end]
        X_pool = X_all[conv_step+1:end]

        mZ = abs(expect(Z_pool, row))   # |⟨Z⟩| averaged over all sites
        mX = abs(expect(X_pool, row))   # |⟨X⟩|

        push!(mZ_vals, mZ)
        push!(mX_vals, mX)
        push!(g_found, g)
        println("    |⟨Z⟩| = $(round(mZ, sigdigits=4))   |⟨X⟩| = $(round(mX, sigdigits=4))")
    end

    isempty(g_found) && error("No data found for any g value")

    _figsize = isnothing(figsize) ? PAPER_FIGSIZE : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel = "g",
                  ylabel = "Magnetisation per site")

        scatterlines!(ax, g_found, mZ_vals;
                      color=:steelblue, marker=:circle,
                      label="|⟨Z⟩|")
        scatterlines!(ax, g_found, mX_vals;
                      color=:firebrick, marker=:diamond, linestyle=:dash,
                      label="|⟨X⟩|")

        add_paper_legend!(ax; position=:rt)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    data = Dict(g => (mZ=mZ_vals[k], mX=mX_vals[k]) for (k, g) in enumerate(g_found))
    return fig, data
end

"""
    plot_connected_corr_vs_g(data_dir, g_values; kwargs...)

Plot nearest-neighbor C(1) and next-nearest-neighbor C(2) connected ZZ
correlations vs transverse field g for the TFIM, using sampling data.

Connected correlation at column separation r, averaged over all row positions:

    C(r) = ⟨Zᵢ Zᵢ₊ᵣ⟩ − ⟨Zᵢ⟩²

# Arguments
- `data_dir`: Directory containing result JSON files
- `g_values`: Vector of g values to scan
- `J`: Coupling (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters
- `conv_step`: Thermalization steps (default 100)
- `samples`: Number of spin measurements per g (default 500_000)
- `figsize`: Override figure size (default `PAPER_FIGSIZE`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, data)` where `data` is a `Dict` mapping each g to
  `(C1=..., C2=...)`.
"""
function plot_connected_corr_vs_g(data_dir::String, g_values::Vector{Float64};
                                   J::Float64=1.0,
                                   row::Int=4, p::Int=3, nqubits::Int=3,
                                   use_exact::Bool=true,
                                   conv_step::Int=100,
                                   samples::Int=4000000,
                                   figsize=nothing,
                                   save_path::Union{String,Nothing}=nothing)

    sorted_g = sort(g_values)
    C1_vals  = Float64[]
    C2_vals  = Float64[]
    g_found  = Float64[]

    println("=== plot_connected_corr_vs_g ($(use_exact ? "exact" : "sampling")) ===")

    for g in sorted_g
        candidates = [
            joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1.json"),
            joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
        ]
        filename = ""
        for c in candidates
            isfile(c) && (filename = c; break)
        end
        if isempty(filename)
            @warn "No file found for g=$g, skipping"
            continue
        end

        println("  g=$g  →  $(basename(filename))")

        if use_exact
            result, input_args = load_result(filename)
            virtual_qubits = (nqubits - 1) ÷ 2
            share_params   = get(input_args, :share_params, true)
            gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)

            mean_C = r -> mean(abs(correlation_function(gates, row, virtual_qubits, :Z, r;
                                                        connected=true, position=pos)[r])
                               for pos in 1:row)
        else
            resample_result = resample_circuit(filename; conv_step=conv_step,
                                               samples=samples, measure_y=false)
            if isnothing(resample_result)
                @warn "Resampling failed for g=$g, skipping"
                continue
            end
            _rho, Z_all, _X_all, _params, _gates = resample_result
            Z_pool = Z_all[conv_step+1:end]

            mean_C = r -> mean(abs(correlation_function(Z_pool, row, r;
                                                        position=pos, connected=true)[r])
                               for pos in 1:row)
        end

        push!(C1_vals, mean_C(1))
        push!(C2_vals, mean_C(2))
        push!(g_found, g)
        println("    C(1) = $(round(last(C1_vals), sigdigits=4))   C(2) = $(round(last(C2_vals), sigdigits=4))")
    end

    isempty(g_found) && error("No data found for any g value")

    _figsize = isnothing(figsize) ? PAPER_FIGSIZE : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel = "g",
                  ylabel = "C(r)",
                  yscale = log10)

        scatterlines!(ax, g_found, C1_vals;
                      color=:steelblue, marker=:circle,
                      label="C(1) nearest")
        scatterlines!(ax, g_found, C2_vals;
                      color=:firebrick, marker=:diamond, linestyle=:dash,
                      label="C(2) next-nearest")

        add_paper_legend!(ax; position=:lt)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    data = Dict(g => (C1=C1_vals[k], C2=C2_vals[k]) for (k, g) in enumerate(g_found))
    return fig, data
end

"""
    plot_correlation_vs_g(data_dir, g_values; kwargs...)

Plot correlation length ξ vs g for the TFIM from the transfer-matrix spectrum.

# Arguments
- `data_dir`: Directory containing result JSON files
- `g_values`: Vector of g values to plot
- `J`: Coupling strength (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters
- `max_separation`: Accepted for API compatibility; fitted correlations are no longer computed
- `connected`: Accepted for API compatibility; fitted correlations are no longer computed
- `spectrum_krylovdim`, `spectrum_tol`, `spectrum_maxiter`, `spectrum_eager`: Krylov controls for `compute_transfer_spectrum`
- `dmrg_file`, `pepskit_file`: Optional reference data files
- `g_c`: Optional critical field value for annotation
- `save_path`: Path to save figure (optional)
"""
function plot_correlation_vs_g(data_dir::String, g_values::Vector{Float64};
                               J=1.0, row=3, nqubits=3, p=3,
                               max_separation=20,
                               connected=true,
                               spectrum_krylovdim=60,
                               spectrum_tol=1e-8,
                               spectrum_maxiter=1000,
                               spectrum_eager=false,
                               dmrg_file::Union{String,Nothing}=nothing,
                               pepskit_file::Union{String,Nothing}=nothing,
                               g_c::Union{Float64,Nothing}=nothing,
                               save_path::Union{String,Nothing}=nothing)

    println("="^70)
    println("Correlation Length vs g Analysis")
    println("="^70)
    println("Source: transfer-matrix spectrum")

    # Load results and compute transfer-spectrum correlation lengths.
    correlation_data = Dict{Float64, NamedTuple}()
    colors = [:blue, :green, :red, :orange, :purple, :brown, :pink, :gray]
    skipped_g = Float64[]

    for (idx, g) in enumerate(g_values)
        candidates = [
            joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_randomtest123.json"),
            joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json"),
        ]
        filename = ""
        for candidate in candidates
            if isfile(candidate)
                filename = candidate
                break
            end
        end

        if isempty(filename)
            push!(skipped_g, g)
            continue
        end

        println("\nProcessing g=$g...")

        # Load result
        result, input_args = load_result(filename)

        # Reconstruct gates
        share_params = get(input_args, :share_params, true)
        gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)

        # Compute correlation length from transfer matrix
        _, gap, _, _ = compute_transfer_spectrum(
            gates, row, nqubits;
            matrix_free=:always,
            krylovdim=spectrum_krylovdim,
            tol=spectrum_tol,
            maxiter=spectrum_maxiter,
            eager=spectrum_eager,
        )
        ξ = 1.0 / gap

        println("  ξ_transfer = $(round(ξ, digits=3))")

        correlation_data[g] = (
            separations = Int[],
            correlations = Float64[],
            correlation_length = ξ,
            correlation_length_fitted = nothing,
            color = colors[mod1(idx, length(colors))]
        )
    end

    if !isempty(skipped_g)
        println("\nSkipped $(length(skipped_g)) missing g values: $skipped_g")
    end

    if isempty(correlation_data)
        error("No valid results found for any g value")
    end

    # Create figure with the package paper theme so saved plots keep the same label sizes.
    fig = with_theme(paper_theme()) do
        fig = Figure(size=PAPER_FIGSIZE)

        ax = Axis(fig[1, 1],
                  xlabel="g",
                  ylabel="Correlation Length ξ")

        # Extract and plot correlation lengths from transfer matrix
        g_sorted = sort(collect(keys(correlation_data)))
        ξ_transfer = [correlation_data[g].correlation_length for g in g_sorted]

        scatterlines!(ax, g_sorted, ξ_transfer,
                      color=:steelblue, marker=:circle, label="IsoPEPS")

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

            scatterlines!(ax, dmrg_g_valid, dmrg_ξ_valid,
                          color=:firebrick, linestyle=:dash, marker=:diamond, label="DMRG")

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
            scatterlines!(ax, peps_g_valid, peps_ξ_valid,
                          color=:seagreen, linestyle=:dashdot, marker=:utriangle, label="iPEPS")

            println("  PEPSKit: $(length(peps_g_valid)) valid g points loaded")
        elseif pepskit_file !== nothing
            @warn "PEPSKit file not found: $pepskit_file"
        end

        # Mark critical point
        if g_c !== nothing
            vlines!(ax, [g_c], color=:black, linestyle=:dot,
                    label=rich("g", subscript("c"), " ≈ $g_c"))
        end

        add_paper_legend!(ax; position=:lt, nbanks=1)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("\nFigure saved to: $save_path")
        end

        fig
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

    # Create figure with the package paper theme so saved plots keep the same label sizes.
    fig = with_theme(paper_theme()) do
        fig = Figure(size=PAPER_FIGSIZE)

        ax = Axis(fig[1, 1],
                  xlabel="J₂ / J₁",
                  ylabel="Correlation Length ξ",
                  title="Correlation Length vs J₂ (J₁=$J1, row=$row, D=$(nqubits-1))")

        # Plot correlation lengths from transfer matrix
        J2_sorted = sort(collect(keys(correlation_data)))
        ξ_transfer = [correlation_data[j].correlation_length for j in J2_sorted]

        lines!(ax, J2_sorted, ξ_transfer,
               color=:steelblue, label="IsoPEPS (transfer matrix)")
        scatter!(ax, J2_sorted, ξ_transfer, color=:steelblue)

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
                       color=:firebrick, linestyle=:dash, label="DMRG reference")
                scatter!(ax, dmrg_J2_valid, dmrg_ξ_valid,
                         color=:firebrick, marker=:diamond)
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

        fig
    end

    return fig, correlation_data
end

# ============================================================================
# plot_M2_vs_J2
# ============================================================================

"""
    plot_M2_vs_J2(data_dir::String, J2_values::Vector{Float64};
                  J1=1.0, row=3, nqubits=3, p=3,
                  use_exact=true,
                  conv_step=100, samples=1000000,
                  max_separation=20,
                  dmrg_file=nothing,
                  save_path=nothing)

Plot magnetic order parameter squared M²(q) vs J2 for the Heisenberg J1-J2 model.

Computes and plots M²(q0) with q0=(π,π) (Néel order), M²(q1) with q1=(π,0)
(stripe order), and M²(q2) with q2=(0,π) (stripe order) as functions of J2.
The crossing/transition between the order parameters signals the phase boundary.

# Arguments
- `data_dir`: Directory containing result JSON files
- `J2_values`: Vector of J2 values to scan
- `J1`: Nearest-neighbor coupling (default: 1.0)
- `row`, `nqubits`, `p`: Circuit parameters
- `use_exact`: If true (default), compute M² via exact transfer matrix contraction;
  if false, use sampling-based computation
- `conv_step`: Thermalization steps for sampling (only used when `use_exact=false`)
- `samples`: Number of measurement samples (only used when `use_exact=false`)
- `max_separation`: Max column separation for structure factor sum
- `dmrg_file`: Optional JSON file with DMRG reference M² data
  (expected keys: `J2_values`, `M2_neel`, `M2_stripe`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, data)` tuple
"""
function plot_M2_vs_J2(data_dir::String, J2_values::Vector{Float64};
                       J1=1.0, row=3, nqubits=3, p=3,
                       use_exact::Bool=false,
                       conv_step=100, samples=1000000,
                       max_separation::Int=20,
                       dmrg_file=nothing,
                       save_path=nothing)

    q0 = (Float64(π), Float64(π))   # Néel
    q1 = (Float64(π), 0.0)          # Stripe (π,0)
    q2 = (0.0, Float64(π))          # Stripe (0,π)

    J2_found = Float64[]
    M2_neel = Float64[]
    M2_stripe = Float64[]
    M2_stripe_0pi = Float64[]

    method_str = use_exact ? "exact (transfer matrix)" : "sampling"
    println("=== M²(q) vs J2 [$method_str] ===")
    println("q0 = (π,π) [Néel],  q1 = (π,0) [Stripe],  q2 = (0,π) [Stripe]")
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

        local m2_neel, m2_stripe, m2_stripe_0pi

        if use_exact
            result, input_args = load_result(filename)
            params = result isa ExactOptimizationResult ? result.params : result.final_params
            _p = input_args[:p]
            _row = input_args[:row]
            _nqubits = input_args[:nqubits]
            share_params = get(input_args, :share_params, true)

            is_2x2 = endswith(filename, "_2x2.json")
            if is_2x2
                gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
                op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
            else
                gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
                op = TransferOperator(gates, _row, _nqubits)
            end

            m2_neel       = magnetic_order_squared(op, q0; max_separation=max_separation)
            m2_stripe     = magnetic_order_squared(op, q1; max_separation=max_separation)
            m2_stripe_0pi = magnetic_order_squared(op, q2; max_separation=max_separation)
        else
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

            m2_neel       = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q0;
                                                   max_separation=max_separation)
            m2_stripe     = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q1;
                                                   max_separation=max_separation)
            m2_stripe_0pi = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q2;
                                                   max_separation=max_separation)
        end

        push!(J2_found, val)
        push!(M2_neel, m2_neel)
        push!(M2_stripe, m2_stripe)
        push!(M2_stripe_0pi, m2_stripe_0pi)

        println("  M²(π,π) = $(round(m2_neel, digits=6)),  M²(π,0) = $(round(m2_stripe, digits=6)),  M²(0,π) = $(round(m2_stripe_0pi, digits=6))")
    end

    if isempty(J2_found)
        error("No data found for any J2 value")
    end

    # --- Plot ---
    fig = Figure(size=PAPER_FIGSIZE_WIDE)
    ax = Axis(fig[1, 1],
              xlabel="J₂ / J₁",
              ylabel="M²(q)",
              title="Magnetic Order [$method_str]: row=$row, nqubits=$nqubits, p=$p")

    scatterlines!(ax, J2_found, M2_neel,
                  label="M²(π,π) Néel", color=:steelblue, marker=:circle)
    scatterlines!(ax, J2_found, M2_stripe,
                  label="M²(π,0) Stripe", color=:firebrick, marker=:diamond)
    scatterlines!(ax, J2_found, M2_stripe_0pi,
                  label="M²(0,π) Stripe", color=:seagreen, marker=:rect)

    # Overlay DMRG reference if provided
    if dmrg_file !== nothing && isfile(dmrg_file)
        dmrg_data = JSON3.read(read(dmrg_file, String))
        if haskey(dmrg_data, :J2_values) && haskey(dmrg_data, :M2_neel)
            dmrg_J2 = Float64.(dmrg_data[:J2_values])
            dmrg_neel = Float64.(dmrg_data[:M2_neel])
            scatterlines!(ax, dmrg_J2, dmrg_neel,
                          label="DMRG M²(π,π)", color=:steelblue, linestyle=:dash,
                          marker=:utriangle)
        end
        if haskey(dmrg_data, :J2_values) && haskey(dmrg_data, :M2_stripe)
            dmrg_J2 = Float64.(dmrg_data[:J2_values])
            dmrg_stripe = Float64.(dmrg_data[:M2_stripe])
            scatterlines!(ax, dmrg_J2, dmrg_stripe,
                          label="DMRG M²(π,0)", color=:firebrick, linestyle=:dash,
                          marker=:utriangle)
        end
        if haskey(dmrg_data, :J2_values) && haskey(dmrg_data, :M2_stripe_0pi)
            dmrg_J2 = Float64.(dmrg_data[:J2_values])
            dmrg_stripe_0pi = Float64.(dmrg_data[:M2_stripe_0pi])
            scatterlines!(ax, dmrg_J2, dmrg_stripe_0pi,
                          label="DMRG M²(0,π)", color=:seagreen, linestyle=:dash,
                          marker=:utriangle)
        end
    elseif dmrg_file !== nothing
        @warn "DMRG file not found: $dmrg_file"
    end

    Legend(fig[1, 2], ax)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("\nFigure saved to: $save_path")
    end

    data = (
        J2_values = J2_found,
        M2_neel = M2_neel,
        M2_stripe = M2_stripe,
        M2_stripe_0pi = M2_stripe_0pi,
        row = row, nqubits = nqubits, p = p
    )

    return fig, data
end

# ============================================================================
# save_M2_vs_J2 — compute and save M² data to JSON
# ============================================================================

"""
    save_M2_vs_J2(data_dir, J2_values; method=:exact, output_file, ...)

Compute M²(q) for each J2 value and save results to a JSON file.

# Arguments
- `data_dir`: Directory containing circuit result JSON files
- `J2_values`: Vector of J2 values to scan
- `method`: `:exact` (transfer matrix) or `:sampling` (Monte Carlo)
- `output_file`: Path to save the JSON results
- `J1, row, nqubits, p`: Circuit/model parameters
- `max_separation`: Max column separation for structure factor (default: 20)
- `conv_step, samples`: Sampling parameters (only for `method=:sampling`)
"""
function save_M2_vs_J2(data_dir::String, J2_values::Vector{Float64};
                       method::Symbol=:exact,
                       output_file::String,
                       J1=1.0, row=3, nqubits=3, p=3,
                       max_separation::Int=20,
                       conv_step=100, samples=1000000)
    method in (:exact, :sampling) || error("method must be :exact or :sampling")

    q0 = (Float64(π), Float64(π))
    q1 = (Float64(π), 0.0)
    q2 = (0.0, Float64(π))

    J2_found = Float64[]
    M2_neel = Float64[]
    M2_stripe = Float64[]
    M2_stripe_0pi = Float64[]

    println("=== Computing M²(q) vs J2 [$(method)] ===")

    for val in J2_values
        candidates = [
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
        ]
        filename = ""
        for c in candidates
            isfile(c) && (filename = c; break)
        end
        if isempty(filename)
            @warn "No file found for J2=$val, skipping"
            continue
        end

        println("\nJ2=$val  →  $(basename(filename))")

        local m2_neel, m2_stripe, m2_stripe_0pi

        if method == :exact
            result, input_args = load_result(filename)
            params = result isa ExactOptimizationResult ? result.params : result.final_params
            _p = input_args[:p]
            _row = input_args[:row]
            _nqubits = input_args[:nqubits]
            share_params = get(input_args, :share_params, true)

            is_2x2 = endswith(filename, "_2x2.json")
            if is_2x2
                gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
                op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
            else
                gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
                op = TransferOperator(gates, _row, _nqubits)
            end

            m2_neel       = magnetic_order_squared(op, q0; max_separation=max_separation)
            m2_stripe     = magnetic_order_squared(op, q1; max_separation=max_separation)
            m2_stripe_0pi = magnetic_order_squared(op, q2; max_separation=max_separation)
        else
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

            m2_neel       = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q0;
                                                   max_separation=max_separation)
            m2_stripe     = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q1;
                                                   max_separation=max_separation)
            m2_stripe_0pi = magnetic_order_squared(X_vec, Z_vec, Y_vec, row, q2;
                                                   max_separation=max_separation)
        end

        push!(J2_found, val)
        push!(M2_neel, m2_neel)
        push!(M2_stripe, m2_stripe)
        push!(M2_stripe_0pi, m2_stripe_0pi)

        println("  M²(π,π) = $(round(m2_neel, digits=6)),  M²(π,0) = $(round(m2_stripe, digits=6)),  M²(0,π) = $(round(m2_stripe_0pi, digits=6))")
    end

    if isempty(J2_found)
        error("No data found for any J2 value")
    end

    save_results(output_file;
                 method=String(method),
                 J2_values=J2_found,
                 M2_neel=M2_neel,
                 M2_stripe=M2_stripe,
                 M2_stripe_0pi=M2_stripe_0pi,
                 row=row, nqubits=nqubits, p=p,
                 max_separation=max_separation)
    println("\nSaved to: $output_file")
    return (J2_values=J2_found, M2_neel=M2_neel, M2_stripe=M2_stripe, M2_stripe_0pi=M2_stripe_0pi)
end

# ============================================================================
# plot_M2_comparison — overlay exact, sampling, DMRG on one figure
# ============================================================================

"""
    plot_M2_comparison(; exact_file="", sampling_file="", dmrg_file="",
                        save_path=nothing, dmrg_Lx_key="Lx2")

Plot M²(π,π) (Néel) and M²(0,π) together vs J₂/J₁, comparing exact, sampling,
and DMRG methods on a single axis.

Each file is a JSON produced by `save_M2_vs_J2` (for exact/sampling) or
`dmrg_reference.jl` (for DMRG). Pass empty string to skip a method.

# Arguments
- `exact_file`: JSON from `save_M2_vs_J2(...; method=:exact)`
- `sampling_file`: JSON from `save_M2_vs_J2(...; method=:sampling)`
- `dmrg_file`: JSON from DMRG J2 scan (keys: `J2_values`, `M2_neel_Lx2`, etc.)
- `dmrg_Lx_key`: suffix for DMRG keys — `"Lx1"` or `"Lx2"` (default: `"Lx2"`)
- `save_path`: Optional path to save the figure
"""
function plot_M2_comparison(; exact_file::String="",
                              sampling_file::String="",
                              dmrg_file::String="",
                              dmrg_Lx_key::String="Lx2",
                              save_path=nothing)
    # M²(π,π) = Néel, M²(0,π) uses key M2_0pi / M2_stripe_0pi
    q_info = [
        (label="M²(π,π)", std_key="M2_neel",       dmrg_key="M2_neel_$dmrg_Lx_key"),
        (label="M²(0,π)", std_key="M2_stripe_0pi",  dmrg_key="M2_0pi_$dmrg_Lx_key"),
    ]

    # Colors: blue/orange for (π,π)/(0,π); solid/dash/dot for exact/sampling/DMRG
    q_colors = [:blue, :orange]

    fig = Figure(size=PAPER_FIGSIZE)
    ax = Axis(fig[1, 1], xlabel="J₂ / J₁", ylabel="M²(q)")

    function _load(file)
        isempty(file) && return nothing
        !isfile(file) && (@warn "File not found: $file"; return nothing)
        return load_results(file)
    end

    exact_data    = _load(exact_file)
    sampling_data = _load(sampling_file)
    dmrg_data     = _load(dmrg_file)
    all_M2_values = Float64[]

    # Track which method styles have been labelled so each appears once
    method_labelled = Dict{String,Bool}()

    for (iq, qi) in enumerate(q_info)
        color = q_colors[iq]

        if exact_data !== nothing && haskey(exact_data, qi.std_key)
            J2 = Float64.(exact_data["J2_values"])
            M2 = Float64.(exact_data[qi.std_key])
            append!(all_M2_values, M2)
            lbl = get(method_labelled, "exact", false) ? nothing : "TN contraction"
            scatterlines!(ax, J2, M2, label=lbl, color=color, marker=:circle)
            method_labelled["exact"] = true
        end

        if sampling_data !== nothing && haskey(sampling_data, qi.std_key)
            J2 = Float64.(sampling_data["J2_values"])
            M2 = Float64.(sampling_data[qi.std_key])
            append!(all_M2_values, M2)
            lbl = get(method_labelled, "sampling", false) ? nothing : "Sampling"
            scatterlines!(ax, J2, M2, label=lbl, color=color,
                          marker=:rect, linestyle=:dash)
            method_labelled["sampling"] = true
        end

        if dmrg_data !== nothing
            dmrg_key = haskey(dmrg_data, qi.dmrg_key) ? qi.dmrg_key :
                       haskey(dmrg_data, qi.std_key) ? qi.std_key : nothing
            if dmrg_key !== nothing && haskey(dmrg_data, "J2_values")
                J2 = Float64.(dmrg_data["J2_values"])
                M2 = Float64.(dmrg_data[dmrg_key])
                append!(all_M2_values, M2)
                lbl = get(method_labelled, "dmrg", false) ? nothing : "DMRG"
                scatterlines!(ax, J2, M2, label=lbl, color=color,
                              marker=:utriangle, linestyle=:dot)
                method_labelled["dmrg"] = true
            end
        end
    end

    ymax = isempty(all_M2_values) ? 0.25 : max(0.25, 1.12 * maximum(all_M2_values))
    ylims!(ax, 0, ymax)

    # Combined legend: one entry per (q-point, method) pair so color + style are visible together
    all_elems  = []
    all_labels = String[]

    method_style = [
        ("exact",    "TN",      :solid, :circle),
        ("sampling", "Samp.",   :dash,  :rect),
        ("dmrg",     "DMRG",   :dot,   :utriangle),
    ]

    for (iq, qi) in enumerate(q_info)
        color = q_colors[iq]
        for (key, mname, ls, mk) in method_style
            get(method_labelled, key, false) || continue
            push!(all_elems, [LineElement(color=color, linestyle=ls),
                              MarkerElement(color=color, marker=mk)])
            push!(all_labels, "$(qi.label) $(mname)")
        end
    end

    Legend(fig[1, 1], all_elems, all_labels;
           tellwidth=false, tellheight=false,
           halign=:left, valign=0.88,
           nbanks=1,
           margin=(1, 1, 1, 1),
           framevisible=false,
           labelsize=PAPER_LEGEND_LABELSIZE,
           padding=(1, 1, 1, 1))

    for ann in m2_phase_annotations(ymax)
        text!(ax, ann.x, ann.y;
              text=ann.label,
              align=ann.align,
              fontsize=PAPER_LEGEND_LABELSIZE,
              color=:firebrick,
              strokecolor=:firebrick,
              strokewidth=0)
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return fig
end

"""
    plot_dimer_structure_factor(filename; nq=50, dimer_orientation=:vertical,
                                max_separation=20, use_exact=true,
                                conv_step=1000, samples=100000, save_path=nothing)

Brillouin-zone heatmap of the dimer static structure factor S_D(qx, qy).

# Arguments
- `filename`: Path to a saved optimization result JSON
- `nq`: Number of q-points along each axis (total grid: nq × nq)
- `dimer_orientation`: `:vertical` or `:horizontal`
- `max_separation`: Max column separation in the structure factor sum
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, SD)` where `SD` is the nq × nq matrix of S_D values
"""
function plot_dimer_structure_factor(filename::String;
                                     nq::Int=50,
                                     dimer_orientation::Symbol=:vertical,
                                     max_separation::Int=20,
                                     use_exact::Bool=true,
                                     conv_step::Int=1000,
                                     samples::Int=100000,
                                     save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    qvals = range(0.0, 2Float64(π), length=nq)
    SD = zeros(nq, nq)

    method_str = use_exact ? "exact" : "sampling"
    println("=== Dimer Structure Factor S_D(q) [$method_str, $dimer_orientation] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, nq=$nq, max_sep=$max_separation")

    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end

        N_uc = length(op.columns)  # columns per unit cell (1 or 2)
        row = op.row
        vq  = op.virtual_qubits
        paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]

        T_cols = _column_transfer_matrices(op)
        T_combined = reduce(*, T_cols)
        l_vec, r_vec, nf, _ = _fixed_points(T_combined)

        # Build per-(col, pos) single-column dimer TMs and full-period dimer TMs
        println("Building dimer transfer matrices (N_uc=$N_uc)...")
        T_D_col_map  = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
        T_D_period_map = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
        μ_map = Dict{Tuple{Int,Int}, ComplexF64}()

        for κ in 1:N_uc, pos in 1:_row
            pos2 = pos % row + 1
            T_D_κ = zeros(ComplexF64, size(T_cols[1]))
            for σ in paulis
                E = get_transfer_matrix_with_operator(
                    op.columns[κ], row, vq, Dict(pos => σ, pos2 => σ);
                    optimizer=GreedyMethod())
                T_D_κ .+= E
            end
            T_D_κ ./= 4.0
            T_D_col_map[(κ, pos)] = T_D_κ

            T_before = κ > 1 ? reduce(*, T_cols[1:κ-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            T_after  = κ < N_uc ? reduce(*, T_cols[κ+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            T_D = T_before * T_D_κ * T_after
            T_D_period_map[(κ, pos)] = T_D
            μ_map[(κ, pos)] = dot(l_vec, T_D * r_vec) / nf
        end

        # Subtract global mean squared to preserve VBS signal
        # (bond-specific μ1*μ2 removes the alternation between unit cell columns)
        D_avg = real(Statistics.mean(collect(values(μ_map))))
        D_avg_sq = D_avg^2

        # Precompute correlations: corr_cache[(κ1,p1,κ2,p2)][m] = ⟨DD⟩ - D_avg²
        println("Precomputing dimer-dimer correlations...")
        corr_cache = Dict{NTuple{4,Int}, Dict{Int,ComplexF64}}()

        for κ1 in 1:N_uc, pos1 in 1:_row, κ2 in 1:N_uc, pos2 in 1:_row
            corrs = Dict{Int, ComplexF64}()
            pos1b = pos1 % row + 1
            pos2b = pos2 % row + 1

            # --- sep=0 (same period): build combined single-period TM ---
            if κ1 == κ2 && pos1 == pos2
                # ⟨D²⟩ at same bond: 4-operator insertion
                val0 = zero(ComplexF64)
                T_b = κ1 > 1 ? reduce(*, T_cols[1:κ1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_a = κ1 < N_uc ? reduce(*, T_cols[κ1+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                for σa in paulis, σb in paulis
                    O_p  = σa * σb
                    O_p2 = σa * σb
                    E = get_transfer_matrix_with_operator(
                        op.columns[κ1], row, vq, Dict(pos1 => O_p, pos1b => O_p2);
                        optimizer=GreedyMethod())
                    val0 += dot(l_vec, T_b * E * T_a * r_vec) / nf
                end
                corrs[0] = val0 / 16.0 - D_avg_sq

            elseif κ1 == κ2
                # Same column, different positions: 4-operator single-column insertion
                val0 = zero(ComplexF64)
                T_b = κ1 > 1 ? reduce(*, T_cols[1:κ1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_a = κ1 < N_uc ? reduce(*, T_cols[κ1+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                for σa in paulis, σb in paulis
                    ops = Dict{Int, Matrix{ComplexF64}}()
                    for (p, op_mat) in [(pos1, σa), (pos1b, σa), (pos2, σb), (pos2b, σb)]
                        if haskey(ops, p)
                            ops[p] = ops[p] * op_mat
                        else
                            ops[p] = copy(op_mat)
                        end
                    end
                    E = get_transfer_matrix_with_operator(
                        op.columns[κ1], row, vq, ops; optimizer=GreedyMethod())
                    val0 += dot(l_vec, T_b * E * T_a * r_vec) / nf
                end
                corrs[0] = val0 / 16.0 - D_avg_sq

            else
                # Different columns in same period: compose single-column dimer TMs
                lo, hi = minmax(κ1, κ2)
                T_b = lo > 1 ? reduce(*, T_cols[1:lo-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_mid = hi > lo + 1 ? reduce(*, T_cols[lo+1:hi-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_a = hi < N_uc ? reduce(*, T_cols[hi+1:N_uc]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                if κ1 < κ2
                    T_DD = T_b * T_D_col_map[(κ1, pos1)] * T_mid * T_D_col_map[(κ2, pos2)] * T_a
                else
                    T_DD = T_b * T_D_col_map[(κ2, pos2)] * T_mid * T_D_col_map[(κ1, pos1)] * T_a
                end
                corrs[0] = dot(l_vec, T_DD * r_vec) / nf - D_avg_sq
            end

            # --- sep >= 1 (different periods) ---
            T_D1 = T_D_period_map[(κ1, pos1)]
            T_D2 = T_D_period_map[(κ2, pos2)]
            l_TD1 = T_D1' * l_vec
            max_period_sep = max(1, max_separation ÷ N_uc)
            for m in 1:max_period_sep
                current = T_D2 * r_vec
                for _ in 1:(m - 1)
                    current = T_combined * current
                end
                corrs[m] = dot(l_TD1, current) / nf - D_avg_sq
            end

            corr_cache[(κ1, pos1, κ2, pos2)] = corrs
            print("\r  (κ=$κ1,p=$pos1)→(κ=$κ2,p=$pos2)")
        end
        println("\nFourier transforming over $nq × $nq q-grid...")

        # Use max_separation as physical columns (not periods) for consistency with sampling
        max_period = max(1, max_separation ÷ N_uc)
        max_col_sep = max_period * N_uc + (N_uc - 1)
        L_eff = Float64(max_col_sep + 1)  # effective finite system length
        N_d = N_uc * _row
        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                val = 0.0
                for κ1 in 1:N_uc, pos1 in 1:_row, κ2 in 1:N_uc, pos2 in 1:_row
                    Δp = pos2 - pos1
                    corrs = corr_cache[(κ1, pos1, κ2, pos2)]
                    for (m, cval) in corrs
                        Δx = m * N_uc + (κ2 - κ1)
                        if Δx < 0
                            continue
                        end
                        # Bartlett (triangular) window: simulates finite-system weighting
                        w = 1.0 - abs(Δx) / L_eff
                        w <= 0.0 && continue
                        if Δx == 0
                            val += w * cos(qy * Δp) * real(cval)
                        else
                            val += 2.0 * w * cos(qx * Δx + qy * Δp) * real(cval)
                        end
                    end
                end
                SD[i, j] = val / N_d
            end
        end
    else
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                            samples=samples, measure_y=true)
        isnothing(resample_result) && error("Resampling failed for $filename")
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = Y_samples[conv_step+1:end]

        # Precompute dimer values matrix once
        println("Precomputing dimer values from samples...")
        all_samples = (X_vec, Y_vec, Z_vec)
        ncols = length(Z_vec) ÷ _row

        if dimer_orientation == :vertical
            dimer_vals = zeros(_row, ncols)
            for S in all_samples
                for c in 1:ncols, pos in 1:_row
                    pos2 = pos % _row + 1
                    i1 = _row * (c - 1) + pos
                    i2 = _row * (c - 1) + pos2
                    dimer_vals[pos, c] += S[i1] * S[i2] / 4.0
                end
            end
            n_cols_d = ncols
        else  # :horizontal
            dimer_vals = zeros(_row, ncols - 1)
            for S in all_samples
                for c in 1:(ncols - 1), pos in 1:_row
                    i1 = _row * (c - 1) + pos
                    i2 = _row * c + pos
                    dimer_vals[pos, c] += S[i1] * S[i2] / 4.0
                end
            end
            n_cols_d = ncols - 1
        end

        max_sep = min(max_separation, n_cols_d - 1)
        n_pos = _row
        μ = vec(mean(dimer_vals, dims=2))

        # Precompute correlation tables
        println("Precomputing correlation tables...")
        # corr0[p1, p2] = mean over c of dimer_vals[p1,c]*dimer_vals[p2,c]
        corr0 = zeros(n_pos, n_pos)
        for p1 in 1:n_pos, p2 in 1:n_pos
            corr0[p1, p2] = mean(dimer_vals[p1, c] * dimer_vals[p2, c] for c in 1:n_cols_d)
        end
        # corr_dc[Δc][p1, p2]
        corr_dc = Vector{Matrix{Float64}}(undef, max_sep)
        for Δc in 1:max_sep
            m = zeros(n_pos, n_pos)
            for p1 in 1:n_pos, p2 in 1:n_pos
                m[p1, p2] = mean(dimer_vals[p1, c] * dimer_vals[p2, c + Δc] for c in 1:(n_cols_d - Δc))
            end
            corr_dc[Δc] = m
        end

        println("Fourier transforming over $nq × $nq q-grid...")
        μ_avg = mean(μ)
        μ_avg_sq = μ_avg^2
        L_eff_s = Float64(max_sep + 1)
        N_d = n_pos
        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                val = 0.0
                for p1 in 1:n_pos, p2 in 1:n_pos
                    Δp = p2 - p1
                    val += cos(qy * Δp) * (corr0[p1, p2] - μ_avg_sq)
                end
                for Δc in 1:max_sep
                    w = 1.0 - Δc / L_eff_s
                    for p1 in 1:n_pos, p2 in 1:n_pos
                        Δp = p2 - p1
                        val += 2.0 * w * cos(qx * Δc + qy * Δp) * (corr_dc[Δc][p1, p2] - μ_avg_sq)
                    end
                end
                SD[i, j] = val / N_d
            end
        end
    end

    # --- Plot ---
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1],
              xlabel="qₓ",
              ylabel="qᵧ",
              title="Dimer Structure Factor S_D(q) [$dimer_orientation]",
              aspect=DataAspect(),
              xticks=([0, π, 2π], ["0", "π", "2π"]),
              yticks=([0, π, 2π], ["0", "π", "2π"]))

    hm = heatmap!(ax, qvals, qvals, SD, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_D(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, SD)
end

"""
    plot_spin_structure_factor(filename; nq=50, max_separation=20, use_exact=true,
                               conv_step=1000, samples=100000, save_path=nothing)

Brillouin-zone heatmap of the spin-spin static structure factor S_SS(qx, qy).

# Arguments
- `filename`: Path to a saved optimization result JSON
- `nq`: Number of q-points along each axis (total grid: nq × nq)
- `max_separation`: Max column separation in the structure factor sum
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, SSS)` where `SSS` is the nq × nq matrix of S_SS values
"""
function plot_spin_structure_factor(filename::String;
                                    nq::Int=50,
                                    max_separation::Int=10,
                                    use_exact::Bool=true,
                                    conv_step::Int=1000,
                                    samples::Int=100000,
                                    J2::Float64=0.0,
                                    D::Int=2,
                                    save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    qvals = range(0.0, 2Float64(π), length=nq)
    SSS = zeros(nq, nq)

    method_str = use_exact ? "exact" : "sampling"
    println("=== Spin Structure Factor S_SS(q) [$method_str] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, nq=$nq, max_sep=$max_separation")

    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end

        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                SSS[i, j] = spin_spin_structure_factor(op, (qx, qy);
                                max_separation=max_separation)
            end
            print("\r  qx $i/$nq")
        end
        println()
    else
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                            samples=samples, measure_y=true)
        isnothing(resample_result) && error("Resampling failed for $filename")
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = Y_samples[conv_step+1:end]

        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                SSS[i, j] = spin_spin_structure_factor(X_vec, Z_vec, Y_vec, _row, (qx, qy);
                                max_separation=max_separation)
            end
            print("\r  qx $i/$nq")
        end
        println()
    end

    # --- Plot ---
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1],
              xlabel="qₓ",
              ylabel="qᵧ",
              title="S_SS(q)  J₂=$J2, D=$D",
              aspect=DataAspect(),
              xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
              yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))

    hm = heatmap!(ax, qvals, qvals, SSS, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_SS(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, SSS)
end

"""
    save_combined_structure_factor_data(output_file, data_dir, J2_values; kwargs...)

Compute spin and dimer structure factor matrices for each J2 value and save the
results to a JSON file. The saved file can later be passed to
`plot_combined_structure_factors` via the `data_file` keyword to skip
recomputation.

# Arguments
- `output_file`: Path to write the JSON data file
- `data_dir`: Directory containing saved optimization result JSONs
- `J2_values`: Vector of J2 coupling values (e.g., [0.0, 0.5, 1.0])
- `J1`: J1 coupling (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters for filename matching
- `nq`: Number of q-points along each axis (grid: nq × nq)
- `max_separation_spin`: Max column separation for spin structure factor
- `max_separation_dimer`: Max column separation for dimer structure factor
- `dimer_orientation`: `:vertical` or `:horizontal`
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`, `samples`: Sampling parameters (when `use_exact=false`)

# Returns
- `(spin_matrices, dimer_matrices)` — Vectors of nq×nq matrices
"""
function save_combined_structure_factor_data(output_file::String,
        data_dir::String, J2_values::Vector{Float64};
        J1::Float64=1.0,
        row::Int=4, p::Int=3, nqubits::Int=3,
        nq::Int=50,
        max_separation_spin::Int=10,
        max_separation_dimer::Int=20,
        dimer_orientation::Symbol=:vertical,
        use_exact::Bool=true,
        conv_step::Int=1000,
        samples::Int=100000)

    n = length(J2_values)
    spin_matrices = Vector{Matrix{Float64}}(undef, n)
    dimer_matrices = Vector{Matrix{Float64}}(undef, n)
    filenames = Vector{String}(undef, n)

    for (idx, val) in enumerate(J2_values)
        candidates = [
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
            joinpath(data_dir, "exact_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
        ]
        found = ""
        for c in candidates
            if isfile(c)
                found = c
                break
            end
        end
        isempty(found) && error("No file found for J2=$val, tried $(length(candidates)) patterns")
        filenames[idx] = found
        println("J2=$val  →  $(basename(found))")
    end

    for idx in 1:n
        println("\n--- Computing spin structure factor for J2=$(J2_values[idx]) ---")
        _, SSS = plot_spin_structure_factor(filenames[idx];
                    nq=nq, max_separation=max_separation_spin,
                    use_exact=use_exact, conv_step=conv_step, samples=samples)
        spin_matrices[idx] = SSS

        println("\n--- Computing dimer structure factor for J2=$(J2_values[idx]) ---")
        _, SD = plot_dimer_structure_factor(filenames[idx];
                    nq=nq, dimer_orientation=dimer_orientation,
                    max_separation=max_separation_dimer,
                    use_exact=use_exact, conv_step=conv_step, samples=samples)
        dimer_matrices[idx] = SD
    end

    save_results(output_file;
        J2_values=J2_values,
        nq=nq,
        use_exact=use_exact,
        spin_matrices=[collect(eachcol(m)) for m in spin_matrices],
        dimer_matrices=[collect(eachcol(m)) for m in dimer_matrices])
    println("\nData saved to: $output_file")
    return (spin_matrices, dimer_matrices)
end

"""
    plot_combined_structure_factors(data_dir, J2_values; kwargs...)

Combined 2-row × N-column panel figure: spin structure factor S(q) on top,
dimer structure factor Sᴅ(q) on bottom, one column per J2 value, with shared
colorbars per row.

# Arguments
- `data_dir`: Directory containing saved optimization result JSONs
- `J2_values`: Vector of J2 coupling values (e.g., [0.0, 0.5, 1.0])
- `J1`: J1 coupling (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters for filename matching
- `nq`: Number of q-points along each axis (grid: nq × nq)
- `max_separation_spin`: Max column separation for spin structure factor
- `max_separation_dimer`: Max column separation for dimer structure factor
- `dimer_orientation`: `:vertical` or `:horizontal`
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`, `samples`: Sampling parameters (when `use_exact=false`)
- `data_file`: Optional path to a JSON produced by `save_combined_structure_factor_data`.
  When provided the matrices are loaded from disk and no computation is performed;
  `data_dir`, `J2_values`, and all method parameters are ignored.
- `figsize`: `(width, height)` in points. Defaults to double-column APS width (510 pt)
  with height derived from the number of columns so each heatmap stays roughly square.
- `save_path`: Optional path to save the figure

# Returns
- `(fig, spin_matrices, dimer_matrices)` where each `*_matrices` is a Vector of nq×nq matrices
"""
function plot_combined_structure_factors(data_dir::String, J2_values::Vector{Float64};
        J1::Float64=1.0,
        row::Int=4, p::Int=3, nqubits::Int=3,
        nq::Int=50,
        max_separation_spin::Int=10,
        max_separation_dimer::Int=20,
        dimer_orientation::Symbol=:vertical,
        use_exact::Bool=true,
        conv_step::Int=1000,
        samples::Int=100000,
        data_file=nothing,
        figsize=nothing,
        save_path=nothing)

    local spin_matrices, dimer_matrices

    if !isnothing(data_file)
        # --- Load pre-computed data ---
        println("Loading structure factor data from: $data_file")
        d = load_results(data_file)
        J2_values = Float64.(d["J2_values"])
        nq = Int(d["nq"])
        spin_matrices  = [Float64.(hcat(col...)) for col in d["spin_matrices"]]
        dimer_matrices = [Float64.(hcat(col...)) for col in d["dimer_matrices"]]
    else
        spin_matrices = Vector{Matrix{Float64}}(undef, length(J2_values))
        dimer_matrices = Vector{Matrix{Float64}}(undef, length(J2_values))
        filenames = Vector{String}(undef, length(J2_values))

        # --- Find files ---
        for (idx, val) in enumerate(J2_values)
            candidates = [
                joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
                joinpath(data_dir, "circuit_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
                joinpath(data_dir, "circuit_heisenberg_j1j2_J=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json"),
                joinpath(data_dir, "exact_heisenberg_j1j2_J1=$(J1)_J2=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_2x2.json"),
            ]
            found = ""
            for c in candidates
                if isfile(c)
                    found = c
                    break
                end
            end
            if isempty(found)
                error("No file found for J2=$val, tried $(length(candidates)) patterns")
            end
            filenames[idx] = found
            println("J2=$val  →  $(basename(found))")
        end

        # --- Compute structure factor matrices ---
        for idx in 1:length(J2_values)
            println("\n--- Computing spin structure factor for J2=$(J2_values[idx]) ---")
            _, SSS = plot_spin_structure_factor(filenames[idx];
                        nq=nq, max_separation=max_separation_spin,
                        use_exact=use_exact, conv_step=conv_step, samples=samples)
            spin_matrices[idx] = SSS

            println("\n--- Computing dimer structure factor for J2=$(J2_values[idx]) ---")
            _, SD = plot_dimer_structure_factor(filenames[idx];
                        nq=nq, dimer_orientation=dimer_orientation,
                        max_separation=max_separation_dimer,
                        use_exact=use_exact, conv_step=conv_step, samples=samples)
            dimer_matrices[idx] = SD
        end
    end

    # --- Shared color ranges ---
    spin_min = minimum(minimum.(spin_matrices))
    spin_max = maximum(maximum.(spin_matrices))
    dimer_min = minimum(minimum.(dimer_matrices))
    dimer_max = maximum(maximum.(dimer_matrices))

    qvals = range(0.0, 2Float64(π), length=nq)

    # --- Build combined figure ---
    # Default: double-column APS width; height derived so each heatmap is square.
    # panel_width ≈ (total_width - colorbar_col) / n_columns
    # total_height ≈ n_rows × panel_width + top/bottom margins
    n = length(J2_values)
    _colorbar_w = 50
    _default_w  = first(PAPER_FIGSIZE_WIDE)   # 510 pt
    _panel_w    = (_default_w - _colorbar_w) ÷ n
    _default_h  = 2 * _panel_w + 60           # 2 rows + label margin
    _figsize    = isnothing(figsize) ? (_default_w, _default_h) : figsize
    fig = Figure(size=_figsize)

    local hm_spin, hm_dimer

    # Top row: spin structure factor S(q)
    for (j, J2) in enumerate(J2_values)
        ax = Axis(fig[1, j],
                  aspect=DataAspect(),
                  title="J₂ = $J2",
                  xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
                  yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))
        if j == 1
            ax.ylabel = "qᵧ"
        else
            ax.yticklabelsvisible = false
        end
        ax.xticklabelsvisible = false
        hm_spin = heatmap!(ax, qvals, qvals, spin_matrices[j],
                           colormap=:viridis, colorrange=(spin_min, spin_max))
    end
    Colorbar(fig[1, n + 1], hm_spin, label="S(q)")

    # Bottom row: dimer structure factor Sᴅ(q)
    for (j, J2) in enumerate(J2_values)
        ax = Axis(fig[2, j],
                  xlabel="qₓ",
                  aspect=DataAspect(),
                  xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
                  yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))
        if j == 1
            ax.ylabel = "qᵧ"
        else
            ax.yticklabelsvisible = false
        end
        hm_dimer = heatmap!(ax, qvals, qvals, dimer_matrices[j],
                            colormap=:viridis, colorrange=(dimer_min, dimer_max))
    end
    Colorbar(fig[2, n + 1], hm_dimer, label="Sᴅ(q)")

    # Row labels on the left
    Label(fig[1, 0], "S(q)", rotation=π/2, fontsize=16, tellheight=false)
    Label(fig[2, 0], "Sᴅ(q)", rotation=π/2, fontsize=16, tellheight=false)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, spin_matrices, dimer_matrices)
end

"""
    plot_plaquette_structure_factor(filename; nq=50, max_separation=20, use_exact=true,
                                    conv_step=1000, samples=100000, save_path=nothing)

Brillouin-zone heatmap of the plaquette static structure factor S_P(qx, qy).

# Arguments
- `filename`: Path to a saved optimization result JSON
- `nq`: Number of q-points along each axis (total grid: nq × nq)
- `max_separation`: Max column separation in the structure factor sum
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, SP)` where `SP` is the nq × nq matrix of S_P values
"""
function plot_plaquette_structure_factor(filename::String;
                                         nq::Int=50,
                                         max_separation::Int=20,
                                         use_exact::Bool=true,
                                         conv_step::Int=1000,
                                         samples::Int=100000,
                                         save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    qvals = range(-Float64(π), Float64(π), length=nq)
    SP = zeros(nq, nq)

    method_str = use_exact ? "exact" : "sampling"
    println("=== Plaquette Structure Factor S_P(q) [$method_str] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, nq=$nq, max_sep=$max_separation")

    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end

        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                SP[i, j] = plaquette_structure_factor(op, (qx, qy);
                                max_separation=max_separation)
            end
            print("\r  qx $i/$nq")
        end
        println()
    else
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                            samples=samples, measure_y=true)
        isnothing(resample_result) && error("Resampling failed for $filename")
        _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = Y_samples[conv_step+1:end]

        for (i, qx) in enumerate(qvals)
            for (j, qy) in enumerate(qvals)
                SP[i, j] = plaquette_structure_factor(X_vec, Z_vec, Y_vec, _row, (qx, qy);
                                max_separation=max_separation)
            end
            print("\r  qx $i/$nq")
        end
        println()
    end

    # --- Plot ---
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1],
              xlabel="qₓ",
              ylabel="qᵧ",
              title="Plaquette Structure Factor S_P(q)",
              aspect=DataAspect())

    hm = heatmap!(ax, qvals, qvals, SP, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_P(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, SP)
end

"""
    plot_dimer_bond_pattern(filename; max_cols=10, ref_pos=1, ref_col=1,
                            ref_orientation=:vertical, use_exact=true,
                            conv_step=1000, samples=100000, save_path=nothing)

Real-space bond pattern of connected dimer-dimer correlations on the cylinder lattice.

Picks a reference dimer at `(ref_pos, ref_col)` with the given orientation and colors
every bond (both vertical and horizontal) by its connected correlation C_D(ref, bond)
with the reference.

# Arguments
- `filename`: Path to a saved optimization result JSON
- `max_cols`: Number of columns to display
- `ref_pos`: Row position of the reference dimer (1-based)
- `ref_col`: Column of the reference dimer (1-based)
- `ref_orientation`: `:vertical` or `:horizontal`
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `save_path`: Optional path to save the figure

# Returns
- `(fig, correlation_data)` where `correlation_data` is a Dict with keys
  `:vertical => Matrix{Float64}(row, max_cols)` and
  `:horizontal => Matrix{Float64}(row, max_cols-1)`
"""
function plot_dimer_bond_pattern(filename::String;
                                 max_cols::Int=10,
                                 ref_pos::Int=1,
                                 ref_col::Int=1,
                                 ref_orientation::Symbol=:vertical,
                                 use_exact::Bool=true,
                                 conv_step::Int=1000,
                                 samples::Int=100000,
                                 save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    method_str = use_exact ? "exact" : "sampling"
    println("=== Dimer Bond Pattern [$method_str] (ref: $ref_orientation at pos=$ref_pos, col=$ref_col) ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, max_cols=$max_cols")

    max_sep = max_cols - 1

    if use_exact
        # --- Exact branch (unchanged) ---
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end
        # Compute correlations for ALL bonds (both orientations) using generalized helper
        vert_corr = zeros(Float64, _row, max_cols)
        for target_pos in 1:_row
            corrs = _dimer_general_correlation(op, 0:max_sep,
                        ref_orientation, ref_pos, :vertical, target_pos)
            for col in 1:max_cols
                sep = col - ref_col
                if sep >= 0 && haskey(corrs, sep)
                    vert_corr[target_pos, col] = real(corrs[sep])
                elseif sep < 0 && haskey(corrs, -sep)
                    vert_corr[target_pos, col] = real(corrs[-sep])
                end
            end
            print("\r  vertical pos $target_pos/$_row")
        end
        println()

        horiz_corr = zeros(Float64, _row, max_cols - 1)
        for target_pos in 1:_row
            corrs = _dimer_general_correlation(op, 0:max_sep,
                        ref_orientation, ref_pos, :horizontal, target_pos)
            for col in 1:(max_cols - 1)
                sep = col - ref_col
                if sep >= 0 && haskey(corrs, sep)
                    horiz_corr[target_pos, col] = real(corrs[sep])
                elseif sep < 0 && haskey(corrs, -sep)
                    horiz_corr[target_pos, col] = real(corrs[-sep])
                end
            end
            print("\r  horizontal pos $target_pos/$_row")
        end
        println()

    else
        # --- Sampling branch ---
        resample_result = resample_circuit(filename; conv_step=conv_step,
                                            samples=samples)
        isnothing(resample_result) && error("Resampling failed for $filename")
        if length(resample_result) == 6
            _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        else
            _rho, Z_samples, X_samples, _params, _gates = resample_result
            Y_samples = zeros(length(X_samples))
        end
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = length(resample_result) == 6 ? Y_samples[conv_step+1:end] : zeros(length(Z_vec))

        dimer_vals_v, dimer_vals_h = _build_all_dimer_values(X_vec, Z_vec, Y_vec, _row)
        ncols_v = size(dimer_vals_v, 2)
        ncols_h = size(dimer_vals_h, 2)

        # Reference bond dimer values (1D vector over columns)
        if ref_orientation == :vertical
            ref_dv = vec(dimer_vals_v[ref_pos, :])
            n_ref = ncols_v
        else
            ref_dv = vec(dimer_vals_h[ref_pos, :])
            n_ref = ncols_h
        end
        μ_ref = mean(ref_dv)

        # Compute connected correlations: C(Δcol) = ⟨D_ref(c) D_target(c+Δcol)⟩ - μ_ref * μ_target
        vert_corr = zeros(Float64, _row, max_cols)
        for target_pos in 1:_row
            target_dv = vec(dimer_vals_v[target_pos, :])
            μ_target = mean(target_dv)
            for col in 1:max_cols
                sep = col - ref_col
                # Use both directions for averaging (symmetry), avoid double-counting sep=0
                seps_to_try = sep == 0 ? [0] : [sep, -sep]
                total = 0.0
                count = 0
                for s in seps_to_try
                    s < 0 && continue
                    n_pairs = min(n_ref, ncols_v) - s
                    n_pairs < 1 && continue
                    for c in 1:n_pairs
                        total += ref_dv[c] * target_dv[c + s]
                    end
                    count += n_pairs
                end
                if count > 0
                    vert_corr[target_pos, col] = total / count - μ_ref * μ_target
                end
            end
        end

        horiz_corr = zeros(Float64, _row, max_cols - 1)
        for target_pos in 1:_row
            target_dv = vec(dimer_vals_h[target_pos, :])
            μ_target = mean(target_dv)
            for col in 1:(max_cols - 1)
                sep = col - ref_col
                seps_to_try = sep == 0 ? [0] : [sep, -sep]
                total = 0.0
                count = 0
                for s in seps_to_try
                    s < 0 && continue
                    n_pairs = min(n_ref, ncols_h) - s
                    n_pairs < 1 && continue
                    for c in 1:n_pairs
                        total += ref_dv[c] * target_dv[c + s]
                    end
                    count += n_pairs
                end
                if count > 0
                    horiz_corr[target_pos, col] = total / count - μ_ref * μ_target
                end
            end
        end
        println("  Sampling correlations computed.")
    end

    # For sampling: exclude ref bond from color scale (its variance is noise-dominated).
    # For exact: keep it (⟨D²⟩-⟨D⟩² is a physical quantity that sets the natural scale).
    if !use_exact
        if ref_orientation == :vertical
            vert_corr[ref_pos, ref_col] = NaN
        else
            if ref_col <= max_cols - 1
                horiz_corr[ref_pos, ref_col] = NaN
            end
        end
    end

    correlation_data = Dict(:vertical => vert_corr, :horizontal => horiz_corr)

    # --- Plot ---
    fig = Figure(size=(max(800, max_cols * 80), max(400, _row * 100 + 100)))

    model_str = get(input_args, :model, "")
    J2_str = haskey(input_args, :J2) ? ", J₂=$(input_args[:J2])" : ""
    title_str = "Dimer-Dimer Correlation [$method_str] (ref: $ref_orientation, pos=$ref_pos, col=$ref_col)$J2_str"

    ax = Axis(fig[1, 1],
              xlabel="Column",
              ylabel="Row",
              title=title_str,
              aspect=DataAspect(),
              yticks=1:_row,
              xticks=1:max_cols)

    # Determine color range (symmetric around zero)
    all_vals = filter(!isnan, vcat(vec(vert_corr), vec(horiz_corr)))
    if !isempty(all_vals)
        cmax = maximum(abs, all_vals)
        cmax = max(cmax, 1e-10)  # avoid zero range
    else
        cmax = 1.0
    end

    # Draw bonds
    # Vertical bonds: between (pos, col) and (pos%row+1, col)
    for col in 1:max_cols
        for pos in 1:_row
            pos2 = pos % _row + 1
            val = vert_corr[pos, col]
            y1, y2 = Float64(pos), Float64(pos2)
            # Handle wrap-around: if pos=row, pos2=1, draw to row+1 position
            if pos2 < pos
                # Draw two half-segments for the periodic wrap
                # Bottom half: pos -> row + 0.5 (visual)
                # Top half: 0.5 -> pos2
                if isnan(val)
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y1 + 0.5],
                                  color=:gray80, linewidth=1, linestyle=:dash)
                    linesegments!(ax, [Float64(col), Float64(col)], [y2 - 0.5, y2],
                                  color=:gray80, linewidth=1, linestyle=:dash)
                else
                    lw = 1.0 + 4.0 * abs(val) / cmax
                    c = val / cmax  # normalized to [-1, 1]
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y1 + 0.5],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                    linesegments!(ax, [Float64(col), Float64(col)], [y2 - 0.5, y2],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                end
            else
                if isnan(val)
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y2],
                                  color=:gray80, linewidth=1, linestyle=:dash)
                else
                    lw = 1.0 + 4.0 * abs(val) / cmax
                    c = val / cmax
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y2],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                end
            end
        end
    end

    # Horizontal bonds: between (pos, col) and (pos, col+1)
    for col in 1:(max_cols - 1)
        for pos in 1:_row
            val = horiz_corr[pos, col]
            if isnan(val)
                linesegments!(ax, [Float64(col), Float64(col + 1)], [Float64(pos), Float64(pos)],
                              color=:gray80, linewidth=1, linestyle=:dash)
            else
                lw = 1.0 + 4.0 * abs(val) / cmax
                c = val / cmax
                linesegments!(ax, [Float64(col), Float64(col + 1)], [Float64(pos), Float64(pos)],
                              color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                              linewidth=lw)
            end
        end
    end

    # Draw sites
    for col in 1:max_cols, pos in 1:_row
        scatter!(ax, [Float64(col)], [Float64(pos)], color=:gray40, markersize=6)
    end

    # Highlight reference bond
    if ref_orientation == :vertical
        rp2 = ref_pos % _row + 1
        if rp2 < ref_pos
            linesegments!(ax, [Float64(ref_col), Float64(ref_col)],
                          [Float64(ref_pos), Float64(ref_pos) + 0.5],
                          color=:black, linewidth=4)
            linesegments!(ax, [Float64(ref_col), Float64(ref_col)],
                          [Float64(rp2) - 0.5, Float64(rp2)],
                          color=:black, linewidth=4)
        else
            linesegments!(ax, [Float64(ref_col), Float64(ref_col)],
                          [Float64(ref_pos), Float64(rp2)],
                          color=:black, linewidth=4)
        end
    else
        linesegments!(ax, [Float64(ref_col), Float64(ref_col + 1)],
                      [Float64(ref_pos), Float64(ref_pos)],
                      color=:black, linewidth=4)
    end

    # Colorbar
    Colorbar(fig[1, 2], colormap=:RdBu, limits=(-cmax, cmax),
             label="C_D(ref, bond)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    return (fig, correlation_data)
end

"""
    plot_bond_energy_pattern(filename; max_cols=10, use_exact=true,
                             conv_step=1000, samples=100000, title="",
                             save_path=nothing)

Visualize bond energies ⟨S_i · S_j⟩ on every nearest-neighbour bond of the
cylinder lattice. Strong/weak bond alternation directly reveals VBS order.

# Arguments
- `filename`: Path to a saved optimization result JSON
- `max_cols`: Number of columns to display
- `use_exact`: If true, use exact transfer matrix; if false, use sampling
- `conv_step`: Thermalization steps for sampling (only when `use_exact=false`)
- `samples`: Number of measurement samples (only when `use_exact=false`)
- `title`: Optional figure title
- `save_path`: Optional path to save the figure

# Returns
- `(fig, bond_data)` where `bond_data` is a Dict with keys
  `:vertical => Matrix{Float64}(row, max_cols)` and
  `:horizontal => Matrix{Float64}(row, max_cols-1)`
"""
function plot_bond_energy_pattern(filename::String;
                                  max_cols::Int=10,
                                  use_exact::Bool=true,
                                  conv_step::Int=1000,
                                  samples::Int=100000,
                                  figsize=nothing,
                                  save_path=nothing)

    result, input_args = load_result(filename)
    params = result isa ExactOptimizationResult ? result.params : result.final_params
    _p = input_args[:p]
    _row = input_args[:row]
    _nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)
    is_2x2 = endswith(filename, "_2x2.json")

    method_str = use_exact ? "exact" : "sampling"
    println("=== Bond Energy Pattern [$method_str] ===")
    println("row=$_row, nqubits=$_nqubits, p=$_p, max_cols=$max_cols")

    # Unit-cell bond expectations: vert_uc[pos, col], horiz_uc[pos, col]
    if use_exact
        if is_2x2
            gates_odd, gates_even = build_unitary_gate_2x2(params, _p, _row, _nqubits)
            op = TransferOperator(gates_odd, gates_even, _row, _nqubits)
        else
            gates = build_unitary_gate(params, _p, _row, _nqubits; share_params=share_params)
            op = TransferOperator(gates, _row, _nqubits)
        end
        vert_uc, horiz_uc = all_bond_expectations(op)
        N_uc = length(op.columns)
        println("  Unit cell columns: $N_uc")
        println("  Vertical bond expectations:")
        for pos in 1:_row, c in 1:N_uc
            pos2 = pos % _row + 1
            println("    ($pos↔$pos2, col=$c): $(vert_uc[pos, c])")
        end
        if size(horiz_uc, 2) > 0
            println("  Horizontal bond expectations (intra-cell):")
            for pos in 1:_row, c in 1:size(horiz_uc, 2)
                println("    (pos=$pos, col=$c↔$(c+1)): $(horiz_uc[pos, c])")
            end
        end
        # Inter-period horizontal bond = horizontal bond spanning last col of period to first of next
        # This is the same as bond_expectation with orientation=:horizontal across period boundary
        # For 1x1 UC, ALL horizontal bonds are inter-period (computed via transfer matrix)
        # We compute it via: ⟨σ^α_{N,pos}(period k) σ^α_{1,pos}(period k+1)⟩ / 4
        # This requires the correlation function approach
        horiz_inter = zeros(Float64, _row)
        for pos in 1:_row
            corr = spin_spin_correlation(op, [1]; col1=N_uc, col2=1, pos1=pos, pos2=pos)
            horiz_inter[pos] = real(corr[1])
        end
        println("  Inter-period horizontal bond expectations:")
        for pos in 1:_row
            println("    (pos=$pos, across period): $(horiz_inter[pos])")
        end

    else
        # Sampling branch
        resample_result = resample_circuit(filename; conv_step=conv_step, samples=samples)
        isnothing(resample_result) && error("Resampling failed for $filename")
        if length(resample_result) == 6
            _rho, Z_samples, X_samples, Y_samples, _params, _gates = resample_result
        else
            _rho, Z_samples, X_samples, _params, _gates = resample_result
            Y_samples = zeros(length(X_samples))
        end
        Z_vec = Z_samples[conv_step+1:end]
        X_vec = X_samples[conv_step+1:end]
        Y_vec = length(resample_result) == 6 ? Y_samples[conv_step+1:end] : zeros(length(Z_vec))

        dimer_vals_v, dimer_vals_h = _build_all_dimer_values(X_vec, Z_vec, Y_vec, _row)

        # Average over columns to get unit-cell pattern
        N_uc = is_2x2 ? 2 : 1
        ncols_v = size(dimer_vals_v, 2)
        ncols_h = size(dimer_vals_h, 2)

        vert_uc = zeros(Float64, _row, N_uc)
        for pos in 1:_row, c in 1:N_uc
            cols = c:N_uc:ncols_v
            vert_uc[pos, c] = mean(dimer_vals_v[pos, cols])
        end

        # Horizontal bonds: for 1x1 UC, all horizontal bonds are equivalent
        # For 2x2 UC, separate intra-cell and inter-cell
        if N_uc == 1
            horiz_uc = zeros(Float64, _row, 0)
            horiz_inter = zeros(Float64, _row)
            for pos in 1:_row
                horiz_inter[pos] = mean(dimer_vals_h[pos, :])
            end
        else
            horiz_uc = zeros(Float64, _row, N_uc - 1)
            for pos in 1:_row
                # Intra-cell: odd-indexed horizontal bonds (col 1→2)
                intra_cols = 1:2:ncols_h
                horiz_uc[pos, 1] = mean(dimer_vals_h[pos, intra_cols])
            end
            horiz_inter = zeros(Float64, _row)
            for pos in 1:_row
                # Inter-cell: even-indexed horizontal bonds (col 2→3, i.e., period boundary)
                inter_cols = 2:2:ncols_h
                horiz_inter[pos] = mean(dimer_vals_h[pos, inter_cols])
            end
        end

        println("  Vertical bond expectations (column-averaged):")
        for pos in 1:_row, c in 1:N_uc
            pos2 = pos % _row + 1
            println("    ($pos↔$pos2, col=$c): $(vert_uc[pos, c])")
        end
        if size(horiz_uc, 2) > 0
            println("  Horizontal bond expectations (intra-cell, averaged):")
            for pos in 1:_row
                println("    (pos=$pos): $(horiz_uc[pos, 1])")
            end
        end
        println("  Inter-period horizontal bond expectations (averaged):")
        for pos in 1:_row
            println("    (pos=$pos): $(horiz_inter[pos])")
        end
    end

    # Tile unit-cell pattern over max_cols columns
    vert_tiled = zeros(Float64, _row, max_cols)
    for col in 1:max_cols, pos in 1:_row
        c_uc = ((col - 1) % N_uc) + 1
        vert_tiled[pos, col] = vert_uc[pos, c_uc]
    end

    horiz_tiled = zeros(Float64, _row, max_cols - 1)
    for col in 1:(max_cols - 1), pos in 1:_row
        c_uc = ((col - 1) % N_uc) + 1
        c_next_uc = (col % N_uc) + 1
        if c_next_uc > c_uc && size(horiz_uc, 2) >= c_uc
            # Intra-cell horizontal bond
            horiz_tiled[pos, col] = horiz_uc[pos, c_uc]
        else
            # Inter-period horizontal bond
            horiz_tiled[pos, col] = horiz_inter[pos]
        end
    end

    # --- Drawing ---
    all_vals = vcat(vec(vert_tiled), vec(horiz_tiled))
    cmax = isempty(all_vals) ? 1.0 : max(maximum(abs, all_vals), 1e-10)

    # Figure size: scale one lattice unit to ~35 pt, add colorbar column and margins
    _unit    = 35
    _cb_w    = 55
    _default_w = max_cols * _unit + _cb_w + 20
    _default_h = (_row + 1) * _unit + 20
    _figsize = isnothing(figsize) ? (_default_w, _default_h) : figsize

    fig = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax = Axis(fig[1, 1];
                  xlabel  = "Column",
                  ylabel  = "Row",
                  aspect  = DataAspect(),
                  xticklabelsize = PAPER_TICKLABELSIZE,
                  yticklabelsize = PAPER_TICKLABELSIZE)

        # Vertical bonds
        for col in 1:max_cols
            for pos in 1:_row
                pos2 = pos % _row + 1
                val  = vert_tiled[pos, col]
                y1, y2 = Float64(pos), Float64(pos2)
                lw = 0.8 + 3.0 * abs(val) / cmax
                c  = val / cmax
                if pos2 < pos
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y1 + 0.5],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                    linesegments!(ax, [Float64(col), Float64(col)], [y2 - 0.5, y2],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                else
                    linesegments!(ax, [Float64(col), Float64(col)], [y1, y2],
                                  color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                                  linewidth=lw)
                end
            end
        end

        # Horizontal bonds
        for col in 1:(max_cols - 1)
            for pos in 1:_row
                val = horiz_tiled[pos, col]
                lw  = 0.8 + 3.0 * abs(val) / cmax
                c   = val / cmax
                linesegments!(ax, [Float64(col), Float64(col + 1)],
                              [Float64(pos), Float64(pos)],
                              color=[c, c], colorrange=(-1, 1), colormap=:RdBu,
                              linewidth=lw)
            end
        end

        # Sites
        xs = [Float64(col) for col in 1:max_cols for _ in 1:_row]
        ys = [Float64(pos) for _   in 1:max_cols for pos in 1:_row]
        scatter!(ax, xs, ys; color=:gray30, markersize=5, strokewidth=0)

        Colorbar(fig[1, 2]; colormap=:RdBu, limits=(-cmax, cmax),
                 label="⟨𝐒ᵢ · 𝐒ⱼ⟩",
                 labelsize=PAPER_AXIS_LABELSIZE,
                 ticklabelsize=PAPER_TICKLABELSIZE,
                 width=12)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("Figure saved to: $save_path")
        end

        fig
    end

    bond_data = Dict(:vertical => vert_tiled, :horizontal => horiz_tiled)
    return (fig, bond_data)
end

# =============================================================================
# DMRG Structure Factor Heatmaps
# =============================================================================

"""
    plot_dmrg_spin_structure_factor(result; nq=50, bulk_fraction=0.5, save_path=nothing)

Brillouin-zone heatmap of spin structure factor S_SS(q) from DMRG ground state.

`result` is the NamedTuple returned by `dmrg_ground_state_2d`.
"""
function plot_dmrg_spin_structure_factor(result;
                                         nq::Int=50,
                                         max_separation::Int=10,
                                         J2::Float64=0.0,
                                         D::Int=2,
                                         save_path=nothing)
    qvals = range(0.0, 2Float64(π), length=nq)
    SSS = zeros(nq, nq)

    Lx = result.Lx
    Ly = result.Ly
    N = Lx * Ly

    # Use center 2*max_sep+1 columns as reference region
    n_bulk_cols = 2 * max_separation + 1
    center = div(Lx, 2)
    col_lo = center - max_separation
    col_hi = center + max_separation
    col_lo = max(1, col_lo)
    col_hi = min(Lx, col_hi)

    println("=== DMRG Spin Structure Factor (Lx=$Lx, Ly=$Ly, max_sep=$max_separation) ===")
    println("  Bulk columns: $col_lo to $col_hi ($(col_hi - col_lo + 1) cols)")

    # Precompute correlation matrix ONCE (the expensive part)
    println("  Computing correlation matrices...")
    SdotS = compute_SdotS_matrix(result)

    # site_to_2d: 1D index -> (col, row)
    coords = Vector{Tuple{Int,Int}}(undef, N)
    for s in 1:N
        coords[s] = (div(s - 1, Ly) + 1, mod(s - 1, Ly) + 1)
    end

    bulk_sites = [s for s in 1:N if col_lo <= coords[s][1] <= col_hi]

    # Precompute dx, dy arrays for bulk site pairs, filtered by max_separation
    pair_dx = Float64[]
    pair_dy = Float64[]
    pair_SS = Float64[]
    n_ref = 0  # count reference sites (for normalization)
    for (a, s1) in enumerate(bulk_sites)
        i1, j1 = coords[s1]
        n_ref += 1
        for (b, s2) in enumerate(bulk_sites)
            i2, j2 = coords[s2]
            dx = i2 - i1
            abs(dx) > max_separation && continue
            push!(pair_dx, dx)
            push!(pair_dy, j2 - j1)
            push!(pair_SS, SdotS[s1, s2])
        end
    end

    # Fourier transform: S(q) = (1/N_ref) Σ_{pairs} ⟨Si·Sj⟩ cos(q·Δr)
    println("  Computing Fourier transform on $nq×$nq grid...")
    for (i, qx) in enumerate(qvals)
        for (j, qy) in enumerate(qvals)
            SSS[i, j] = sum(pair_SS .* cos.(qx .* pair_dx .+ qy .* pair_dy)) / n_ref
        end
    end

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], xlabel="qₓ", ylabel="qᵧ",
              title="DMRG S_SS(q)  J₂=$J2, D=$D",
              aspect=DataAspect(),
              xticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]),
              yticks=([0, Float64(π), 2Float64(π)], ["0", "π", "2π"]))
    hm = heatmap!(ax, qvals, qvals, SSS, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_SS(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end
    return (fig, SSS)
end

"""
    plot_dmrg_dimer_structure_factor(result; nq=50, bulk_cols=20,
                                     dimer_orientation=:vertical, save_path=nothing)

Brillouin-zone heatmap of connected dimer structure factor S_D(q) from DMRG ground state.

Precomputes the ⟨D_b D_b'⟩ matrix once (expensive), then Fourier transforms for each q.
"""
function plot_dmrg_dimer_structure_factor(result;
                                          nq::Int=50,
                                          bulk_cols::Int=20,
                                          dimer_orientation::Symbol=:vertical,
                                          save_path=nothing)
    Lx = result.Lx
    Ly = result.Ly
    psi = result.psi
    sites = result.sites

    println("=== DMRG Dimer Structure Factor (Lx=$Lx, Ly=$Ly, $dimer_orientation) ===")

    # Precompute bonds and correlation data
    col_lo = max(1, div(Lx - bulk_cols, 2) + 1)
    col_hi = min(Lx, col_lo + bulk_cols - 1)

    # Use the DMRG correlation function to get precomputed data
    corr_data = compute_dimer_dimer_correlation_dmrg(result;
                    dimer_orientation=dimer_orientation, bulk_cols=bulk_cols)
    bonds = corr_data.bonds
    D_exp = corr_data.dimer_expectations
    DD = corr_data.DD_matrix
    n = length(bonds)

    # Precompute bond center coordinates
    bond_coords = Vector{Tuple{Float64,Float64}}(undef, n)
    for bi in 1:n
        _, _, col, row = bonds[bi]
        if dimer_orientation == :vertical
            bond_coords[bi] = (Float64(col), Float64(row) + 0.5)
        else
            bond_coords[bi] = (Float64(col) + 0.5, Float64(row))
        end
    end

    # Subtract global mean squared (not bond-specific product) to preserve VBS signal
    # On finite DMRG cylinders, ⟨D_b⟩ itself alternates due to pinned VBS order;
    # subtracting D_exp * D_exp' would remove that signal entirely.
    D_avg = Statistics.mean(D_exp)
    C_conn = DD .- D_avg^2

    println("Fourier transforming over $nq × $nq q-grid...")
    qvals = range(0.0, 2Float64(π), length=nq)
    SD = zeros(nq, nq)

    for (i, qx) in enumerate(qvals)
        for (j, qy) in enumerate(qvals)
            val = 0.0
            for bi in 1:n, bj in 1:n
                dx = bond_coords[bj][1] - bond_coords[bi][1]
                dy = bond_coords[bj][2] - bond_coords[bi][2]
                val += C_conn[bi, bj] * cos(qx * dx + qy * dy)
            end
            SD[i, j] = val / n
        end
        print("\r  qx $i/$nq")
    end
    println()

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], xlabel="qₓ", ylabel="qᵧ",
              title="DMRG Dimer Structure Factor S_D(q) [$dimer_orientation]",
              aspect=DataAspect(),
              xticks=([0, π, 2π], ["0", "π", "2π"]),
              yticks=([0, π, 2π], ["0", "π", "2π"]))
    hm = heatmap!(ax, qvals, qvals, SD, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_D(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end
    return (fig, SD)
end

"""
    plot_dmrg_plaquette_structure_factor(result; nq=50, bulk_cols=20, save_path=nothing)

Brillouin-zone heatmap of plaquette structure factor S_P(q) from DMRG ground state.
Uses disconnected approximation (2-point correlators only).
"""
function plot_dmrg_plaquette_structure_factor(result;
                                              nq::Int=50,
                                              bulk_cols::Int=20,
                                              save_path=nothing)
    qvals = range(-Float64(π), Float64(π), length=nq)
    SP = zeros(nq, nq)

    println("=== DMRG Plaquette Structure Factor (Lx=$(result.Lx), Ly=$(result.Ly)) ===")
    for (i, qx) in enumerate(qvals)
        for (j, qy) in enumerate(qvals)
            SP[i, j] = compute_plaquette_structure_factor_dmrg(result, (qx, qy);
                            bulk_cols=bulk_cols)
        end
        print("\r  qx $i/$nq")
    end
    println()

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], xlabel="qₓ", ylabel="qᵧ",
              title="DMRG Plaquette Structure Factor S_P(q) [disconnected]",
              aspect=DataAspect())
    hm = heatmap!(ax, qvals, qvals, SP, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="S_P(q)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end
    return (fig, SP)
end

"""
    plot_dmrg_dimer_bond_pattern(result; bulk_cols=20, ref_bond_idx=1,
                                 ref_orientation=:vertical, title="", save_path=nothing)

Real-space bond pattern of connected dimer-dimer correlations from DMRG ground state.
Shows both vertical and horizontal bonds on a single lattice, colored by connected
correlation C_D(ref, bond) with the reference bond.

- `ref_orientation`: orientation of the reference bond (`:vertical` or `:horizontal`)
- `ref_bond_idx`: index of the reference bond within bonds of `ref_orientation`
- `title`: custom figure title (auto-generated if empty)
"""
function plot_dmrg_dimer_bond_pattern(result;
                                      bulk_cols::Int=20,
                                      ref_bond_idx::Int=1,
                                      ref_orientation::Symbol=:vertical,
                                      title::String="",
                                      save_path=nothing)
    Lx = result.Lx
    Ly = result.Ly

    corr_data = compute_dimer_dimer_correlation_dmrg(result;
                    dimer_orientation=:both, bulk_cols=bulk_cols)
    bonds = corr_data.bonds
    orientations = corr_data.orientations
    D_exp = corr_data.dimer_expectations
    DD = corr_data.DD_matrix
    n = length(bonds)

    # Find the reference bond
    orient_indices = findall(o -> o == ref_orientation, orientations)
    ref = orient_indices[clamp(ref_bond_idx, 1, length(orient_indices))]

    # Connected correlations relative to reference bond
    C_ref = Float64[DD[ref, bj] - D_exp[ref] * D_exp[bj] for bj in 1:n]
    cmax = max(maximum(abs, C_ref), 1e-10)

    # Determine plot bounds
    col_min = minimum(b[3] for b in bonds)
    col_max = maximum(b[3] for b in bonds)
    if any(o == :horizontal for o in orientations)
        h_col_max = maximum(bonds[i][3] for i in 1:n if orientations[i] == :horizontal)
        col_max = max(col_max, h_col_max + 1)
    end

    fig = Figure(size=(max(800, (col_max - col_min + 1) * 60), max(400, Ly * 100 + 100)))
    ref_col, ref_row = bonds[ref][3], bonds[ref][4]
    plot_title = isempty(title) ?
        "DMRG Dimer Correlation (ref: $ref_orientation, pos=$ref_row, col=$ref_col)" :
        title
    ax = Axis(fig[1, 1],
              xlabel="Column", ylabel="Row",
              title=plot_title,
              aspect=DataAspect(),
              yticks=1:Ly)

    # Draw all bonds (both vertical and horizontal)
    for bi in 1:n
        _, _, col, row = bonds[bi]
        val = C_ref[bi]
        c_norm = val / cmax
        lw = 1.0 + 4.0 * abs(val) / cmax

        if orientations[bi] == :vertical
            row2 = row % Ly + 1
            if row2 < row
                linesegments!(ax, [Float64(col), Float64(col)],
                              [Float64(row), Float64(row) + 0.5],
                              color=[c_norm, c_norm], colorrange=(-1, 1),
                              colormap=:RdBu, linewidth=lw)
                linesegments!(ax, [Float64(col), Float64(col)],
                              [Float64(row2) - 0.5, Float64(row2)],
                              color=[c_norm, c_norm], colorrange=(-1, 1),
                              colormap=:RdBu, linewidth=lw)
            else
                linesegments!(ax, [Float64(col), Float64(col)],
                              [Float64(row), Float64(row2)],
                              color=[c_norm, c_norm], colorrange=(-1, 1),
                              colormap=:RdBu, linewidth=lw)
            end
        else
            linesegments!(ax, [Float64(col), Float64(col + 1)],
                          [Float64(row), Float64(row)],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
        end
    end

    # Highlight reference bond
    _, _, rc, rr = bonds[ref]
    if ref_orientation == :vertical
        rr2 = rr % Ly + 1
        if rr2 < rr
            linesegments!(ax, [Float64(rc), Float64(rc)],
                          [Float64(rr), Float64(rr) + 0.5],
                          color=:black, linewidth=4)
            linesegments!(ax, [Float64(rc), Float64(rc)],
                          [Float64(rr2) - 0.5, Float64(rr2)],
                          color=:black, linewidth=4)
        else
            linesegments!(ax, [Float64(rc), Float64(rc)],
                          [Float64(rr), Float64(rr2)],
                          color=:black, linewidth=4)
        end
    else
        linesegments!(ax, [Float64(rc), Float64(rc + 1)],
                      [Float64(rr), Float64(rr)],
                      color=:black, linewidth=4)
    end

    # Draw sites
    for col in col_min:col_max, row in 1:Ly
        scatter!(ax, [Float64(col)], [Float64(row)], color=:gray40, markersize=6)
    end

    Colorbar(fig[1, 2], colormap=:RdBu, limits=(-cmax, cmax),
             label="C_D(ref, bond)")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end
    return (fig, corr_data)
end

"""
    plot_dmrg_bond_energy_pattern(result; bulk_cols=20, title="", save_path=nothing)

Visualize bond energies ⟨S_i · S_j⟩ on every nearest-neighbour bond of the
DMRG ground state on an open cylinder. Strong/weak bond alternation directly
reveals VBS order.

# Arguments
- `result`: DMRG result (from `dmrg_ground_state_2d`)
- `bulk_cols`: Number of bulk columns to display (centered, avoiding boundaries)
- `title`: Custom figure title
- `save_path`: Optional path to save the figure

# Returns
- `(fig, bond_data)` where `bond_data` is a NamedTuple with fields
  `SdotS_matrix`, `bonds_v`, `bonds_h`, `D_v`, `D_h`
"""
function plot_dmrg_bond_energy_pattern(result;
                                       bulk_cols::Int=20,
                                       title::String="",
                                       save_path=nothing)
    Lx = result.Lx
    Ly = result.Ly

    # Get full S·S matrix
    SdotS = compute_SdotS_matrix(result)

    # Determine bulk region
    bulk_cols = min(bulk_cols, Lx)
    margin = div(Lx - bulk_cols, 2)
    col_lo = max(1, margin + 1)
    col_hi = min(Lx, col_lo + bulk_cols - 1)

    # Build bond lists and extract ⟨S_i · S_j⟩
    # Vertical bonds: (col, row) ↔ (col, row%Ly+1)
    bonds_v = Tuple{Int,Int,Int,Int}[]  # (si, sj, col, row)
    for col in col_lo:col_hi, row in 1:Ly
        row2 = row % Ly + 1
        si = (col - 1) * Ly + row
        sj = (col - 1) * Ly + row2
        push!(bonds_v, (si, sj, col, row))
    end

    # Horizontal bonds: (col, row) ↔ (col+1, row)
    bonds_h = Tuple{Int,Int,Int,Int}[]
    for col in col_lo:(col_hi - 1), row in 1:Ly
        si = (col - 1) * Ly + row
        sj = col * Ly + row
        push!(bonds_h, (si, sj, col, row))
    end

    D_v = Float64[SdotS[si, sj] for (si, sj, _, _) in bonds_v]
    D_h = Float64[SdotS[si, sj] for (si, sj, _, _) in bonds_h]

    all_vals = vcat(D_v, D_h)
    cmax = isempty(all_vals) ? 1.0 : max(maximum(abs, all_vals), 1e-10)

    println("=== DMRG Bond Energy Pattern ===")
    println("Lx=$Lx, Ly=$Ly, bulk cols=$col_lo:$col_hi")
    println("  Vertical bonds: $(length(D_v)), range [$(minimum(D_v)), $(maximum(D_v))]")
    println("  Horizontal bonds: $(length(D_h)), range [$(minimum(D_h)), $(maximum(D_h))]")
    println("  Color scale: ±$cmax")

    plot_title = isempty(title) ?
        "DMRG Bond Energy ⟨Sᵢ·Sⱼ⟩ (Lx=$Lx, Ly=$Ly)" : title

    fig = Figure(size=(max(800, (col_hi - col_lo + 2) * 60), max(400, Ly * 100 + 100)))
    ax = Axis(fig[1, 1], xlabel="Column", ylabel="Row",
              title=plot_title, aspect=DataAspect(), yticks=1:Ly)

    # Draw vertical bonds
    for (idx, (_, _, col, row)) in enumerate(bonds_v)
        row2 = row % Ly + 1
        val = D_v[idx]
        c_norm = val / cmax
        lw = 1.0 + 4.0 * abs(val) / cmax
        if row2 < row
            linesegments!(ax, [Float64(col), Float64(col)],
                          [Float64(row), Float64(row) + 0.5],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
            linesegments!(ax, [Float64(col), Float64(col)],
                          [Float64(row2) - 0.5, Float64(row2)],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
        else
            linesegments!(ax, [Float64(col), Float64(col)],
                          [Float64(row), Float64(row2)],
                          color=[c_norm, c_norm], colorrange=(-1, 1),
                          colormap=:RdBu, linewidth=lw)
        end
    end

    # Draw horizontal bonds
    for (idx, (_, _, col, row)) in enumerate(bonds_h)
        val = D_h[idx]
        c_norm = val / cmax
        lw = 1.0 + 4.0 * abs(val) / cmax
        linesegments!(ax, [Float64(col), Float64(col + 1)],
                      [Float64(row), Float64(row)],
                      color=[c_norm, c_norm], colorrange=(-1, 1),
                      colormap=:RdBu, linewidth=lw)
    end

    # Draw sites
    for col in col_lo:col_hi, row in 1:Ly
        scatter!(ax, [Float64(col)], [Float64(row)], color=:gray40, markersize=6)
    end

    Colorbar(fig[1, 2], colormap=:RdBu, limits=(-cmax, cmax),
             label="⟨Sᵢ · Sⱼ⟩")

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        println("Figure saved to: $save_path")
    end

    bond_data = (SdotS_matrix=SdotS, bonds_v=bonds_v, bonds_h=bonds_h,
                 D_v=D_v, D_h=D_h)
    return (fig, bond_data)
end

# ============================================================================
# plot_observable_convergence
# ============================================================================

"""
    plot_observable_convergence(filename::String; save_path)

Plot cumulative running mean of Z/X/Y basis measurements from a circuit
optimization result to visualize convergence to the fixed point.

Overlays exact fixed-point expectation values computed from the transfer
matrix as horizontal dashed reference lines. The model and unit cell type
are auto-detected from the saved input_args.

# Arguments
- `filename`: Path to a CircuitOptimizationResult JSON file
- `save_path::Union{String,Nothing}`: Optional path to save the figure

# Returns
- CairoMakie `Figure`

# Example
```julia
fig = plot_observable_convergence("result.json")
fig = plot_observable_convergence("result.json"; save_path="convergence.pdf")
```
"""
function plot_observable_convergence(filename::String;
        save_path::Union{String, Nothing} = nothing)

    result, input_args = load_result(filename)

    row = input_args[:row]
    nqubits = input_args[:nqubits]
    p = input_args[:p]
    conv_step = get(input_args, :conv_step, 1000)
    vq = (nqubits - 1) ÷ 2

    # Auto-detect model from saved input_args
    model_str = String(get(input_args, :model, "tfim"))
    model = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))

    Z_samples = result.final_Z_samples
    X_samples = result.final_X_samples
    Y_samples = result.final_Y_samples
    has_y = !isempty(Y_samples)

    # Cumulative running means
    running_Z = cumsum(Z_samples) ./ (1:length(Z_samples))
    running_X = cumsum(X_samples) ./ (1:length(X_samples))

    # Reconstruct gates and build TransferOperator (handles 1x1 and 2x2 unit cells)
    if default_unit_cell(model) == :two_by_two
        gates_odd, gates_even = build_unitary_gate_2x2(result.final_params, p, row, nqubits)
        op = TransferOperator([gates_odd, gates_even], row, vq)
    else
        gates = build_unitary_gate(result.final_params, p, row, nqubits)
        op = TransferOperator([gates], row, vq)
    end

    # Compute exact per-site reference values averaged over all (column, position)
    N = length(op.columns)
    exact_Z = sum(real(expect(op, :Z; col=c, position=pos))
                  for c in 1:N for pos in 1:row) / (N * row)
    exact_X = sum(real(expect(op, :X; col=c, position=pos))
                  for c in 1:N for pos in 1:row) / (N * row)

    # Exact energy reference (dispatch per model type)
    exact_E = if model isa TFIM
        e, _ = compute_exact_energy(model, op)
        real(e) / row
    elseif model isa HeisenbergJ1J2
        real(compute_exact_heisenberg_energy(op, model.J1, model.J2)) / row
    else
        error("compute_exact_energy not implemented for model type $(typeof(model))")
    end

    # Running mean energy on a downsampled grid (compute_energy_from_samples on prefix).
    # Start at 2*row so correlator pairs (horizontal bonds) are always well-defined.
    # Use minimum length across sample vectors to avoid out-of-bounds indexing.
    n_samples = has_y ? min(length(X_samples), length(Z_samples), length(Y_samples)) :
                        min(length(X_samples), length(Z_samples))
    min_k = max(2 * row, 1)
    eval_indices = unique(round.(Int, range(min_k, n_samples, length=min(500, n_samples - min_k + 1))))
    running_E = [compute_energy_from_samples(model,
                     X_samples[1:k], Z_samples[1:k],
                     has_y ? Y_samples[1:k] : Float64[], row)
                 for k in eval_indices]

    n_panels = has_y ? 4 : 3
    fig = Figure(size=(700, 300 * n_panels))

    # Z panel
    ax_z = Axis(fig[1, 1],
                xlabel="Measurement index", ylabel="Running mean ⟨Z⟩",
                title="Observable Convergence: $(basename(filename))",
                limits=(nothing, (-1, 1)))
    lines!(ax_z, 1:length(running_Z), running_Z,
           linewidth=1.5, color=:blue, label="⟨Z⟩ running mean")
    hlines!(ax_z, [exact_Z], linestyle=:dash, color=:red, linewidth=1.5,
            label="Exact ⟨Z⟩/site = $(round(exact_Z, digits=4))")
    vlines!(ax_z, [conv_step], linestyle=:dot, color=:gray, linewidth=1,
            label="conv_step=$conv_step")
    axislegend(ax_z, position=:rt)

    # X panel
    ax_x = Axis(fig[2, 1],
                xlabel="Measurement index", ylabel="Running mean ⟨X⟩",
                limits=(nothing, (-1, 1)))
    lines!(ax_x, 1:length(running_X), running_X,
           linewidth=1.5, color=:green, label="⟨X⟩ running mean")
    hlines!(ax_x, [exact_X], linestyle=:dash, color=:red, linewidth=1.5,
            label="Exact ⟨X⟩/site = $(round(exact_X, digits=4))")
    vlines!(ax_x, [conv_step], linestyle=:dot, color=:gray, linewidth=1,
            label="conv_step=$conv_step")
    axislegend(ax_x, position=:rt)

    # Y panel (if applicable)
    if has_y
        running_Y = cumsum(Y_samples) ./ (1:length(Y_samples))
        exact_Y = sum(real(expect(op, :Y; col=c, position=pos))
                      for c in 1:N for pos in 1:row) / (N * row)

        ax_y = Axis(fig[3, 1],
                    xlabel="Measurement index", ylabel="Running mean ⟨Y⟩",
                    limits=(nothing, (-1, 1)))
        lines!(ax_y, 1:length(running_Y), running_Y,
               linewidth=1.5, color=:orange, label="⟨Y⟩ running mean")
        hlines!(ax_y, [exact_Y], linestyle=:dash, color=:red, linewidth=1.5,
                label="Exact ⟨Y⟩/site = $(round(exact_Y, digits=4))")
        vlines!(ax_y, [conv_step], linestyle=:dot, color=:gray, linewidth=1,
                label="conv_step=$conv_step")
        axislegend(ax_y, position=:rt)
    end

    # Energy panel (always last row)
    energy_row = has_y ? 4 : 3
    ax_e = Axis(fig[energy_row, 1],
                xlabel="Measurement index", ylabel="Running mean Energy/site",
                title="", limits=(nothing, (-4.0, -1.0)))
    lines!(ax_e, eval_indices, running_E,
           linewidth=1.5, color=:purple, label="Energy running mean")
    hlines!(ax_e, [exact_E], linestyle=:dash, color=:red, linewidth=1.5,
            label="Exact E/site = $(round(exact_E, digits=4))")
    vlines!(ax_e, [conv_step], linestyle=:dot, color=:gray, linewidth=1,
            label="conv_step=$conv_step")
    axislegend(ax_e, position=:rt)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    return fig
end

"""
    plot_energy_convergence_vs_g(data_dir, g_values; J, row, p, nqubits, conv_step, ylims, save_path)

Plot the running-mean energy dynamics for multiple g values on a single figure.
Each g value is shown as a colored line (running mean from samples) with a matching
dashed horizontal line for the exact energy reference.

# Arguments
- `data_dir`: Directory containing result JSON files
- `g_values`: Vector of g values to overlay
- `J`: Coupling constant (default 1.0)
- `row`, `p`, `nqubits`: Circuit parameters (used to construct filenames)
- `conv_step`: Thermalization step marked by a vertical dotted line
- `ylims`: Y-axis limits, default `(-4.0, -1.0)`
- `save_path`: If provided, save figure to this path

# Example
```julia
fig = plot_energy_convergence_vs_g("project/results", [0.5, 1.0, 1.5, 2.0];
    J=1.0, row=3, p=3, nqubits=3, conv_step=100)
```
"""
function plot_energy_convergence_vs_g(data_dir::String, g_values::Vector{Float64};
        J=1.0, row::Int=3, p::Int=3, nqubits::Int=3,
        conv_step::Int=100,
        ylims=(-4.0, -1.0),
        save_path::Union{String,Nothing}=nothing)

    palette = [:steelblue, :firebrick, :seagreen, :darkorange,
               :purple, :saddlebrown, :hotpink, :teal, :gray]

    fig = Figure(size=PAPER_FIGSIZE_WIDE)
    ax = Axis(fig[1, 1],
              xlabel="Measurement index",
              ylabel="Running mean Energy/site",
              title="Energy Convergence: TFIM J=$J, row=$row",
              limits=(nothing, ylims))

    conv_drawn = false

    for (idx, g) in enumerate(g_values)
        filename = joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1.json")

        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping g=$g"
            continue
        end

        result, input_args = load_result(filename)
        model_str = String(get(input_args, :model, "tfim"))
        model = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))

        vq = (nqubits - 1) ÷ 2
        gates = build_unitary_gate(result.final_params, p, row, nqubits)
        op = TransferOperator([gates], row, vq)

        exact_E, _ = compute_exact_energy(model, op)
        exact_E = real(exact_E) / row

        X_samples = result.final_X_samples
        Z_samples = result.final_Z_samples
        n_samples = min(length(X_samples), length(Z_samples))
        min_k = max(2 * row, 1)
        eval_indices = unique(round.(Int, range(min_k, n_samples,
                                               length=min(500, n_samples - min_k + 1))))
        running_E = [compute_energy_from_samples(model,
                         X_samples[1:k], Z_samples[1:k], Float64[], row)
                     for k in eval_indices]

        color = palette[mod1(idx, length(palette))]
        lines!(ax, eval_indices, running_E, color=color, label="g=$g")
        hlines!(ax, [exact_E], linestyle=:dash, color=color,
                label="Exact g=$g = $(round(exact_E, digits=4))")

        if !conv_drawn
            vlines!(ax, [conv_step], linestyle=:dot, color=:gray,
                    label="conv_step=$conv_step")
            conv_drawn = true
        end
    end

    Legend(fig[1, 2], ax, merge=true)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    return fig
end

"""
    plot_energy_dynamics(filename; M, shots, conv_step, ylims, save_path)

Run M independent fresh circuit samples from the optimized parameters in `filename`
and plot the intra-run energy convergence with standard error across runs.

At each shot index k (1 to `shots`), each of the M independent runs contributes a
cumulative prefix-energy estimate. The mean and ±1 standard error band across M runs
are plotted, showing how fast a single run's estimate converges and the statistical
spread.

# Arguments
- `filename`: Path to a saved result JSON file (produced by `optimize_circuit`)
- `M`: Number of independent fresh circuit runs (default: 10_000)
- `shots`: Shots per run after thermalization (default: 1000)
- `conv_step`: Thermalization steps discarded at the start of each run (default: 100)
- `ylims`: Y-axis limits (default: `(-4.0, -1.0)`)
- `save_path`: If provided, save figure to this path

# Example
```julia
fig = plot_energy_dynamics("project/results/result.json"; M=1000, shots=500)
```
"""
function plot_energy_dynamics(filename::String;
        M::Int = 10_000,
        shots::Int = 1000,
        conv_step::Int = 100,
        ylims = (-4.0, -1.0),
        save_path::Union{String, Nothing} = nothing)

    result, input_args = load_result(filename)
    row     = input_args[:row]
    nqubits = input_args[:nqubits]
    p       = input_args[:p]
    vq      = (nqubits - 1) ÷ 2
    model_str = String(get(input_args, :model, "tfim"))
    model   = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))
    has_y   = needs_y_measurement(model)

    # Reconstruct gates and TransferOperator (handles 1x1 and 2x2 unit cells)
    two_by_two = default_unit_cell(model) == :two_by_two
    if two_by_two
        gates_odd, gates_even = build_unitary_gate_2x2(result.final_params, p, row, nqubits)
        op = TransferOperator([gates_odd, gates_even], row, vq)
    else
        gates = build_unitary_gate(result.final_params, p, row, nqubits)
        op = TransferOperator([gates], row, vq)
    end

    # Exact energy reference
    exact_E = if model isa TFIM
        e, _ = compute_exact_energy(model, op)
        real(e) / row
    elseif model isa HeisenbergJ1J2
        real(compute_exact_heisenberg_energy(op, model.J1, model.J2)) / row
    else
        error("Exact energy not implemented for model type $(typeof(model))")
    end

    # Downsampled evaluation indices within each run
    min_k = max(2 * row, 1)
    eval_indices = unique(round.(Int, range(min_k, shots,
                                            length=min(200, shots - min_k + 1))))
    n_eval = length(eval_indices)

    # M independent runs in parallel — energy_curves[m, i] = E at eval_indices[i] in run m
    energy_curves = Matrix{Float64}(undef, M, n_eval)
    Threads.@threads for m in 1:M
        ch = two_by_two ?
             sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                    conv_step=conv_step, samples=shots, model=model) :
             sample_quantum_channel(gates, row, nqubits;
                                    conv_step=conv_step, samples=shots, model=model)
        Z_s = ch[2][1:end]
        X_s = ch[3][1:end]
        Y_s = has_y ? ch[4][1:end] : Float64[]
        for (i, k) in enumerate(eval_indices)
            energy_curves[m, i] = compute_energy_from_samples(model,
                X_s[1:k], Z_s[1:k], has_y ? Y_s[1:k] : Float64[], row)
        end
    end

    # Mean and standard error across M runs at each eval index
    mean_E = vec(mean(energy_curves, dims=1))
    se_E   = vec(std(energy_curves,  dims=1)) ./ sqrt(M)

    fig = Figure(size=PAPER_FIGSIZE)
    ax = Axis(fig[1, 1],
              xlabel="Shot index (within run)",
              ylabel="Energy/site",
              title="Energy Dynamics: $(basename(filename))\nM=$M independent runs, $shots shots each",
              limits=(nothing, ylims))

    band!(ax, eval_indices, mean_E .- se_E, mean_E .+ se_E,
          color=(:steelblue, 0.3), label="±1 SE")
    lines!(ax, eval_indices, mean_E,
           color=:steelblue, label="Mean energy/site")
    hlines!(ax, [exact_E], linestyle=:dash, color=:firebrick,
            label="Exact E/site = $(round(exact_E, digits=4))")
    axislegend(ax, position=:rb)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    return fig
end

"""
    plot_energy_dynamics_vs_g(data_dir, g_values; J, row, p, nqubits, M, shots, conv_step, ylims, save_path)

Like `plot_energy_dynamics` but overlays results for multiple g values on the same figure.
Each g value gets its own color: a mean energy line and a ±1 SE band.

# Example
```julia
fig = plot_energy_dynamics_vs_g("project/results", [0.5, 1.0, 1.5, 2.0];
    J=1.0, row=3, p=3, nqubits=3, M=1000, shots=500)
```
"""
function plot_energy_dynamics_vs_g(data_dir::String, g_values::Vector{Float64};
        J=1.0, row::Int=3, p::Int=3, nqubits::Int=3,
        M::Int = 10_000,
        shots::Int = 1000,
        conv_step::Int = 100,
        ylims = (-5.0, -1.0),
        save_path::Union{String, Nothing} = nothing)

    palette = [:steelblue, :firebrick, :seagreen, :darkorange,
               :purple, :saddlebrown, :hotpink, :teal, :gray]

    fig = with_theme(paper_theme()) do
    fig = Figure(size=PAPER_FIGSIZE)
    ax = Axis(fig[1, 1];
              xlabel  = "Channel iteration",
              ylabel  = "E / site",
              limits  = (nothing, ylims))

    for (idx, g) in enumerate(g_values)
        filename = joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1.json")
        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping g=$g"
            continue
        end

        result, input_args = load_result(filename)
        model_str = String(get(input_args, :model, "tfim"))
        model     = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))
        has_y     = needs_y_measurement(model)

        two_by_two = default_unit_cell(model) == :two_by_two
        if two_by_two
            gates_odd, gates_even = build_unitary_gate_2x2(result.final_params, p, row, nqubits)
            op = TransferOperator([gates_odd, gates_even], row, (nqubits-1)÷2)
        else
            gates = build_unitary_gate(result.final_params, p, row, nqubits)
            op = TransferOperator([gates], row, (nqubits-1)÷2)
        end

        exact_E = if model isa TFIM
            e, _ = compute_exact_energy(model, op)
            real(e) / row
        elseif model isa HeisenbergJ1J2
            real(compute_exact_heisenberg_energy(op, model.J1, model.J2)) / row
        end

        min_k = max(2 * row, 1)
        eval_indices = unique(round.(Int, range(min_k, shots,
                                                length=min(200, shots - min_k + 1))))
        n_eval = length(eval_indices)

        energy_curves = Matrix{Float64}(undef, M, n_eval)
        Threads.@threads for m in 1:M
            ch = two_by_two ?
                 sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                        conv_step=conv_step, samples=shots, model=model) :
                 sample_quantum_channel(gates, row, nqubits;
                                        conv_step=conv_step, samples=shots, model=model)
            Z_s = ch[2][1:end]
            X_s = ch[3][1:end]
            Y_s = has_y ? ch[4][1:end] : Float64[]
            for (i, k) in enumerate(eval_indices)
                energy_curves[m, i] = compute_energy_from_samples(model,
                    X_s[1:k], Z_s[1:k], has_y ? Y_s[1:k] : Float64[], row)
            end
        end

        mean_E = vec(mean(energy_curves, dims=1))
        se_E   = vec(std(energy_curves,  dims=1)) ./ sqrt(M)
        color  = palette[mod1(idx, length(palette))]

        band!(ax, eval_indices, mean_E .- se_E, mean_E .+ se_E,
              color=(color, 0.2))
        lines!(ax, eval_indices, mean_E,
               color=color, label="g=$g")
        hlines!(ax, [exact_E], linestyle=:dash, color=color, label=nothing)
    end

    # Thermalization marker
    vlines!(ax, [100], linestyle=:dash, color=:black)
    text!(ax, 102, ylims[1] + 0.05 * (ylims[2] - ylims[1]),
          text="thermalization", color=:black)

    # Outside legend: tight row spacing so all entries fit within PAPER_FIGSIZE_WIDE height
    Legend(fig[1, 2], ax;
           merge        = true,
           labelsize    = PAPER_LEGEND_LABELSIZE,
           rowgap       = 0,
           patchsize    = (12, 8),
           padding      = (3, 3, 3, 3),
           framevisible = true,
           framewidth   = 0.5,
           valign       = :top)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    fig
    end  # with_theme

    return fig
end
