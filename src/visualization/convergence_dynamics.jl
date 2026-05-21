function _select_plot_params(result, parameter_source::Symbol, series_index::Integer; random_seed=nothing)
    if parameter_source === :optimized
        return result.final_params
    elseif parameter_source === :random
        n_params = length(result.final_params)
        rng = isnothing(random_seed) ? Random.default_rng() : Random.Xoshiro(Int(random_seed) + series_index - 1)
        return 2π .* rand(rng, n_params)
    else
        throw(ArgumentError("parameter_source must be :optimized or :random"))
    end
end

"""
    plot_energy_dynamics_vs_g(data_dir, g_values; J, row, p, nqubits, M, shots, conv_step, ylims, save_path)

Plot multi-g energy dynamics on a single figure.
Each g value gets its own color: a mean energy line and a ±1 SE band. Here one
channel iteration is one full column of `row` samples; the point at iteration `k`
uses only columns `k-1` and `k`.
For `nqubits >= 5`, the exact contraction reference line is omitted because the
transfer matrix is too large for this diagnostic plot.
Set `parameter_source=:random` to replace the optimized parameters loaded from
disk by random angles in `[0, 2π)`.

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
        parameter_source::Symbol = :optimized,
        random_seed = nothing,
        ylims = :auto,
        save_path::Union{String, Nothing} = nothing)

    shots >= 2 || throw(ArgumentError("shots must be at least 2 channel iterations"))
    effective_ylims = ylims === :auto ?
        (parameter_source === :random ? (-1.5, 1.0) : (-5.0, -3.0)) :
        ylims

    palette = [:steelblue, :firebrick, :seagreen, :darkorange,
               :purple, :saddlebrown, :hotpink, :teal, :gray]

    fig = with_theme(paper_theme()) do
    fig = Figure(size=PAPER_FIGSIZE)
    ax = Axis(fig[1, 1];
              xlabel  = "Channel iteration",
              ylabel  = "E / site",
              limits  = (nothing, effective_ylims))

    for (idx, g) in enumerate(g_values)
        filename = joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json")
        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping g=$g"
            continue
        end

        result, input_args = load_result(filename)
        model_str = String(get(input_args, :model, "tfim"))
        model     = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))
        has_y     = needs_y_measurement(model)
        params    = _select_plot_params(result, parameter_source, idx; random_seed=random_seed)

        two_by_two = default_unit_cell(model) == :two_by_two
        if two_by_two
            gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
        else
            gates = build_unitary_gate(params, p, row, nqubits)
        end

        exact_E = nothing
        if nqubits < 5
            op = if two_by_two
                TransferOperator([gates_odd, gates_even], row, (nqubits-1)÷2)
            else
                TransferOperator([gates], row, (nqubits-1)÷2)
            end

            exact_E = if model isa TFIM
                e, _ = compute_exact_energy(model, op)
                real(e) / row
            elseif model isa HeisenbergJ1J2
                real(compute_exact_heisenberg_energy(op, model.J1, model.J2)) / row
            end
        else
            @info "Skipping exact contraction reference for nqubits=$nqubits in plot_energy_dynamics_vs_g" g
        end

        eval_indices = unique(round.(Int, range(2, shots,
                                                length=min(200, shots - 1))))
        n_eval = length(eval_indices)
        conv_samples = conv_step * row
        run_samples = shots * row

        energy_curves = Matrix{Float64}(undef, M, n_eval)
        Threads.@threads for m in 1:M
            ch = two_by_two ?
                 sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                        conv_step=conv_samples, samples=run_samples, model=model) :
                 sample_quantum_channel(gates, row, nqubits;
                                        conv_step=conv_samples, samples=run_samples, model=model)
            sample_range = (conv_samples + 1):(conv_samples + run_samples)
            Z_s = ch[2][sample_range]
            X_s = ch[3][sample_range]
            Y_s = has_y ? ch[4][sample_range] : Float64[]
            for (i, k) in enumerate(eval_indices)
                cols = ((k - 2) * row + 1):(k * row)
                energy_curves[m, i] = compute_energy_from_samples(model,
                    X_s[cols], Z_s[cols], has_y ? Y_s[cols] : Float64[], row)
            end
        end

        mean_E = vec(mean(energy_curves, dims=1))
        se_E   = vec(std(energy_curves,  dims=1)) ./ sqrt(M)
        color  = palette[mod1(idx, length(palette))]

        band!(ax, eval_indices, mean_E .- se_E, mean_E .+ se_E,
              color=(color, 0.2))
        source_label = parameter_source === :random ? " random" : ""
        lines!(ax, eval_indices, mean_E,
               color=color, label="g=$g$source_label")
        if !isnothing(exact_E)
            hlines!(ax, [exact_E], linestyle=:dash, color=color, label=nothing)
        end
    end

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

function _channel_gate_blocks(gates, row::Int, nqubits::Int)
    remaining_qubits = (nqubits - 1) ÷ 2
    fixed_qubits     = (nqubits + 1) ÷ 2
    n_env            = remaining_qubits * (row + 1)
    total_qubits     = n_env + 1

    gate_blocks = Vector{Any}(undef, row)
    for j in 1:row
        qpos = tuple((1:fixed_qubits)...,
                     (fixed_qubits + (j-1)*remaining_qubits + 1:
                      fixed_qubits + j*remaining_qubits)...)
        gate_blocks[j] = put(total_qubits, qpos => matblock(gates[j]))
    end
    return gate_blocks, n_env, total_qubits
end

function _sample_basis_column_means(gates, row::Int, nqubits::Int, basis::Symbol;
                                    conv_step::Int, shots::Int,
                                    position::Union{Nothing,Int}=nothing)
    gate_blocks, n_env, total_qubits = _channel_gate_blocks(gates, row, nqubits)
    basis_block = basis === :X ? put(total_qubits, 1 => H) : nothing
    rho = zero_state(n_env)
    values = Float64[]
    sizehint!(values, shots)

    for col in 1:(conv_step + shots)
        col_values = Float64[]
        sizehint!(col_values, row)
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, gate_blocks[j])
            basis === :X && Yao.apply!(rho, basis_block)
            val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
            if isnothing(position) || position == j
                push!(col_values, val.buf)
            end
        end
        col > conv_step && push!(values, mean(col_values))
    end
    return values
end

function _sample_basis_column_means(gates_odd, gates_even, row::Int, nqubits::Int, basis::Symbol;
                                    conv_step::Int, shots::Int,
                                    position::Union{Nothing,Int}=nothing)
    gate_blocks_odd, n_env, total_qubits = _channel_gate_blocks(gates_odd, row, nqubits)
    gate_blocks_even, _, _ = _channel_gate_blocks(gates_even, row, nqubits)
    basis_block = basis === :X ? put(total_qubits, 1 => H) : nothing
    rho = zero_state(n_env)
    values = Float64[]
    sizehint!(values, shots)

    for col in 1:(conv_step + shots)
        gate_blocks = isodd(col) ? gate_blocks_odd : gate_blocks_even
        col_values = Float64[]
        sizehint!(col_values, row)
        for j in 1:row
            rho = join(rho, zero_state(1))
            Yao.apply!(rho, gate_blocks[j])
            basis === :X && Yao.apply!(rho, basis_block)
            val = 1 - 2*measure!(RemoveMeasured(), rho, 1)
            if isnothing(position) || position == j
                push!(col_values, val.buf)
            end
        end
        col > conv_step && push!(values, mean(col_values))
    end
    return values
end

"""
    plot_local_xz_dynamics_vs_g(data_dir, g_values; J, row, p, nqubits, M,
                                shots, conv_step, position, save_path)

Plot local X and Z observable dynamics for optimized TFIM circuit files. One
channel iteration is one full column. For each basis, every independent run
starts from a fresh boundary state, so the X and Z channel indices have the same
meaning. Set `parameter_source=:random` to replace the optimized parameters
loaded from disk by random angles in `[0, 2π)`.
"""
function plot_local_xz_dynamics_vs_g(data_dir::String, g_values::Vector{Float64};
        J=1.0, row::Int=3, p::Int=3, nqubits::Int=3,
        M::Int = 1000,
        shots::Int = 200,
        conv_step::Int = 0,
        position::Union{Nothing,Int}=nothing,
        parameter_source::Symbol = :optimized,
        random_seed = nothing,
        xlims = nothing,
        save_path::Union{String, Nothing} = nothing)

    shots >= 1 || throw(ArgumentError("shots must be at least 1 channel iteration"))
    if !isnothing(position) && !(1 <= position <= row)
        throw(ArgumentError("position must be between 1 and row"))
    end

    palette = [:steelblue, :firebrick, :seagreen, :darkorange,
               :purple, :saddlebrown, :hotpink, :teal, :gray]

    fig = with_theme(paper_theme()) do
    fig = Figure(size=PAPER_FIGSIZE_WIDE)
    ax_x = Axis(fig[1, 1];
                xlabel="Channel iteration",
                ylabel="⟨X⟩",
                limits=(xlims, (-1.05, 1.05)))
    ax_z = Axis(fig[2, 1];
                xlabel="Channel iteration",
                ylabel="⟨Z⟩",
                limits=(xlims, (-1.05, 1.05)))

    eval_indices = collect(1:shots)
    for (idx, g) in enumerate(g_values)
        filename = joinpath(data_dir, "circuit_tfim_J=$(J)_g=$(g)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json")
        if !isfile(filename)
            @warn "File not found: $(basename(filename)), skipping g=$g"
            continue
        end

        result, input_args = load_result(filename)
        model_str = String(get(input_args, :model, "tfim"))
        model     = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))
        params    = _select_plot_params(result, parameter_source, idx; random_seed=random_seed)
        two_by_two = default_unit_cell(model) == :two_by_two
        if two_by_two
            gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
        else
            gates = build_unitary_gate(params, p, row, nqubits)
        end

        x_curves = Matrix{Float64}(undef, M, shots)
        z_curves = Matrix{Float64}(undef, M, shots)
        Threads.@threads for m in 1:M
            if two_by_two
                x_curves[m, :] .= _sample_basis_column_means(gates_odd, gates_even, row, nqubits, :X;
                                                             conv_step=conv_step, shots=shots,
                                                             position=position)
                z_curves[m, :] .= _sample_basis_column_means(gates_odd, gates_even, row, nqubits, :Z;
                                                             conv_step=conv_step, shots=shots,
                                                             position=position)
            else
                x_curves[m, :] .= _sample_basis_column_means(gates, row, nqubits, :X;
                                                             conv_step=conv_step, shots=shots,
                                                             position=position)
                z_curves[m, :] .= _sample_basis_column_means(gates, row, nqubits, :Z;
                                                             conv_step=conv_step, shots=shots,
                                                             position=position)
            end
        end

        color = palette[mod1(idx, length(palette))]
        mean_x = vec(mean(x_curves, dims=1))
        se_x   = vec(std(x_curves, dims=1)) ./ sqrt(M)
        mean_z = vec(mean(z_curves, dims=1))
        se_z   = vec(std(z_curves, dims=1)) ./ sqrt(M)

        band!(ax_x, eval_indices, mean_x .- se_x, mean_x .+ se_x; color=(color, 0.2))
        source_label = parameter_source === :random ? " random" : ""
        lines!(ax_x, eval_indices, mean_x; color=color, label="g=$g$source_label")
        band!(ax_z, eval_indices, mean_z .- se_z, mean_z .+ se_z; color=(color, 0.2))
        lines!(ax_z, eval_indices, mean_z; color=color, label="g=$g")
    end

    Legend(fig[:, 2], ax_x;
           merge=true,
           labelsize=PAPER_LEGEND_LABELSIZE,
           rowgap=0,
           patchsize=(12, 8),
           padding=(3, 3, 3, 3),
           framevisible=true,
           framewidth=0.5,
           valign=:top)

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end

    fig
    end

    return fig
end
