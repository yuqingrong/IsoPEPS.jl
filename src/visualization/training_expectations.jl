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
            (length(result.final_params) == gate_parameter_count(p, nqubits; unit_cell=:two_by_two))
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

