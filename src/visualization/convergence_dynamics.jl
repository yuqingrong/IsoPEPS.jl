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
    compute_tfim_energy_dynamics_data(data_dir, scan_values; save_path=nothing, kwargs...)

Generate the sampled TFIM channel-energy dynamics used in the paper figure and
optionally save only its means and standard errors. Exact reference energies
remain in the independent processed TFIM energy scan.
"""
function compute_tfim_energy_dynamics_data(data_dir::String,
                                           scan_values::AbstractVector{<:Real};
                                           J::Real=1.0, row::Int=3, p::Int=3,
                                           nqubits::Int=3, M::Int=10_000,
                                           shots::Int=20, conv_step::Int=0,
                                           parameter_source::Symbol=:optimized,
                                           random_seed::Union{Int,Nothing}=nothing,
                                           save_path::Union{String,Nothing}=nothing)
    shots >= 2 || throw(ArgumentError("shots must be at least 2 channel iterations"))
    M >= 2 || throw(ArgumentError("M must be at least 2 to estimate a standard error"))
    values = Float64.(scan_values)
    eval_indices = unique(round.(Int, range(2, shots, length=min(200, shots - 1))))

    !isnothing(random_seed) && Random.seed!(random_seed)
    mean_energies = Vector{Vector{Float64}}()
    standard_errors = Vector{Vector{Float64}}()
    source_files = String[]

    for (idx, val) in enumerate(values)
        pattern = "circuit_tfim_J=$(J)_g=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)"
        matches = sort(filter(file -> startswith(file, pattern) && endswith(file, ".json"),
                              readdir(data_dir)))
        isempty(matches) && error("No file matching $(pattern)*.json")
        filename = joinpath(data_dir, first(matches))

        result, input_args = load_result(filename)
        model_str = String(get(input_args, :model, "tfim"))
        model_obj = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))
        model_obj isa TFIM || error("Processed energy dynamics currently supports TFIM only")
        params = _select_plot_params(result, parameter_source, idx; random_seed=random_seed)
        gates = build_unitary_gate(params, p, row, nqubits;
                                   share_params=get(input_args, :share_params, true),
                                   structure=get(input_args, :structure, nothing))

        energy_curves = Matrix{Float64}(undef, M, length(eval_indices))
        Threads.@threads for m in 1:M
            channel = sample_quantum_channel(gates, row, nqubits;
                                             conv_step=conv_step * row,
                                             samples=shots * row, model=model_obj)
            sample_range = (conv_step * row + 1):(conv_step * row + shots * row)
            Z_samples = channel[2][sample_range]
            X_samples = channel[3][sample_range]
            for (i, k) in enumerate(eval_indices)
                columns = ((k - 2) * row + 1):(k * row)
                energy_curves[m, i] = compute_energy_from_samples(
                    model_obj, X_samples[columns], Z_samples[columns], Float64[], row)
            end
        end

        push!(mean_energies, vec(mean(energy_curves, dims=1)))
        push!(standard_errors, vec(std(energy_curves, dims=1)) ./ sqrt(M))
        push!(source_files, _processed_data_source_path(filename, save_path))
    end

    data = (
        scan_values = values,
        iteration_indices = eval_indices,
        mean_energies = mean_energies,
        standard_errors = standard_errors,
        source_files = source_files,
        coupling_J = Float64(J),
        row = row,
        p = p,
        nqubits = nqubits,
        repeats = M,
        shots = shots,
        conv_step = conv_step,
        random_seed = random_seed,
        parameter_source = String(parameter_source),
    )
    if !isnothing(save_path)
        payload = Dict{String,Any}(
            "schema_version" => 1,
            "model" => "tfim",
            "unit_cell" => "abc",
            "quantity" => "channel_energy_dynamics",
            "scan_parameter" => "g",
            "scan_values" => data.scan_values,
            "iteration_indices" => data.iteration_indices,
            "mean_energies" => data.mean_energies,
            "standard_errors" => data.standard_errors,
            "source_files" => data.source_files,
            "coupling_J" => data.coupling_J,
            "row" => data.row,
            "p" => data.p,
            "nqubits" => data.nqubits,
            "repeats" => data.repeats,
            "shots" => data.shots,
            "conv_step" => data.conv_step,
            "random_seed" => data.random_seed,
            "parameter_source" => data.parameter_source,
        )
        _write_processed_tfim_data(save_path, payload)
        println("Processed TFIM energy-dynamics data saved to: $save_path")
    end
    return data
end

"""Load processed sampled TFIM channel-energy dynamics."""
function load_tfim_energy_dynamics_data(path::String)
    payload = _read_processed_tfim_data(path)
    Int(_json_get_any(payload, ("schema_version",); default=0)) == 1 || error(
        "Unsupported TFIM energy-dynamics schema in $path")
    String(_json_get_any(payload, ("model",); default="")) == "tfim" || error(
        "Processed data in $path is not TFIM energy dynamics")
    String(_json_get_any(payload, ("quantity",); default="")) == "channel_energy_dynamics" || error(
        "Processed data in $path is not channel-energy dynamics")

    scan_values = Float64.(collect(_json_get_any(payload, ("scan_values",))))
    iteration_indices = Int.(collect(_json_get_any(payload, ("iteration_indices",))))
    mean_energies = [Float64.(collect(series)) for series in _json_get_any(payload, ("mean_energies",))]
    standard_errors = [Float64.(collect(series)) for series in _json_get_any(payload, ("standard_errors",))]
    source_files = String.(collect(_json_get_any(payload, ("source_files",))))
    n_values = length(scan_values)
    length(mean_energies) == length(standard_errors) == length(source_files) == n_values || error(
        "TFIM energy-dynamics series have inconsistent lengths in $path")
    all(length(series) == length(iteration_indices) for series in mean_energies) || error(
        "TFIM energy-dynamics mean series have inconsistent iteration lengths in $path")
    all(length(series) == length(iteration_indices) for series in standard_errors) || error(
        "TFIM energy-dynamics error series have inconsistent iteration lengths in $path")

    seed_value = _json_get_any(payload, ("random_seed",); default=nothing)
    return (
        scan_values = scan_values,
        iteration_indices = iteration_indices,
        mean_energies = mean_energies,
        standard_errors = standard_errors,
        source_files = source_files,
        coupling_J = Float64(_json_get_any(payload, ("coupling_J",); default=1.0)),
        row = Int(_json_get_any(payload, ("row",); default=3)),
        p = Int(_json_get_any(payload, ("p",); default=3)),
        nqubits = Int(_json_get_any(payload, ("nqubits",); default=3)),
        repeats = Int(_json_get_any(payload, ("repeats",); default=0)),
        shots = Int(_json_get_any(payload, ("shots",); default=0)),
        conv_step = Int(_json_get_any(payload, ("conv_step",); default=0)),
        random_seed = isnothing(seed_value) ? nothing : Int(seed_value),
        parameter_source = String(_json_get_any(payload, ("parameter_source",); default="optimized")),
    )
end

"""
    plot_tfim_energy_dynamics_from_processed_data(dynamics_data_file, energy_data_file; save_path=nothing)

Render TFIM channel-energy dynamics from its saved sampling summary and the
separate processed exact-energy scan, without loading circuit inputs or
resampling the channel.
"""
function plot_tfim_energy_dynamics_from_processed_data(
        dynamics_data_file::String, energy_data_file::String;
        ylims=:auto, save_path::Union{String,Nothing}=nothing)
    dynamics_data = load_tfim_energy_dynamics_data(dynamics_data_file)
    energy_data = load_tfim_energy_vs_g_data(energy_data_file)
    exact_series = (scan_values=energy_data.scan_values, energies=energy_data.energies)
    exact_energies = [_lookup_series_energy(exact_series, value) for value in dynamics_data.scan_values]
    all(isfinite, exact_energies) || error(
        "Processed exact-energy scan is missing one or more TFIM dynamics g values")

    effective_ylims = if ylims !== :auto
        ylims
    elseif dynamics_data.parameter_source == "random"
        (-1.5, 1.0)
    else
        (-4.5, -1.5)
    end
    palette = [:steelblue, :firebrick, :seagreen, :darkorange,
               :purple, :saddlebrown, :hotpink, :teal, :gray]

    fig = with_theme(paper_theme()) do
        figure = Figure(size=PAPER_FIGSIZE)
        axis = Axis(figure[1, 1]; xlabel="Channel iteration", ylabel=ENERGY_PER_SITE_LABEL,
                    limits=(nothing, effective_ylims))
        for (idx, value) in enumerate(dynamics_data.scan_values)
            color = palette[mod1(idx, length(palette))]
            mean_energy = dynamics_data.mean_energies[idx]
            standard_error = dynamics_data.standard_errors[idx]
            band!(axis, dynamics_data.iteration_indices, mean_energy .- standard_error,
                  mean_energy .+ standard_error, color=(color, 0.2))
            lines!(axis, dynamics_data.iteration_indices, mean_energy;
                   color=color, label="g=$(value)")
            hlines!(axis, [exact_energies[idx]]; linestyle=:dash, color=color, label=nothing)
        end
        vlines!(axis, [10]; color=:black, linestyle=:dash, label="thermalization")
        Legend(figure[1, 2], axis;
               merge=true, labelsize=PAPER_LEGEND_LABELSIZE,
               rowgap=PAPER_LEGEND_ROWGAP, colgap=PAPER_LEGEND_COLGAP,
               patchsize=PAPER_LEGEND_PATCHSIZE,
               patchlabelgap=PAPER_LEGEND_PATCHLABELGAP,
               padding=(3, 3, 3, 3), framevisible=true, framewidth=0.5,
               valign=:top)
        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, figure)
            println("Energy-dynamics figure saved to: $save_path")
        end
        figure
    end
    return fig
end

"""
    plot_energy_dynamics_vs_g(data_dir, scan_values; model, scan_param, row, p, nqubits, M, shots, conv_step, ylims, save_path, fixed_params...)

Plot multi-value energy dynamics on a single figure.
Each scan value gets its own color: a mean energy line and a ±1 SE band. Here one
channel iteration is one full column of `row` samples; the point at iteration `k`
uses only columns `k-1` and `k`.
For `nqubits >= 5`, the exact contraction reference line is omitted because the
transfer matrix is too large for this diagnostic plot.
Set `parameter_source=:random` to replace the optimized parameters loaded from
disk by random angles in `[0, 2π)`.

# Examples
```julia
# TFIM — scan over g
fig = plot_energy_dynamics_vs_g("project/results", [0.5, 1.0, 1.5, 2.0];
    J=1.0, row=3, p=3, nqubits=3, M=1000, shots=500)

# Heisenberg J1-J2 — scan over J2
fig = plot_energy_dynamics_vs_g("project/results_heisenberg", [0.0, 0.25, 0.5];
    model="heisenberg_j1j2", scan_param="J2", J1=1.0,
    row=4, p=3, nqubits=5, M=1000, shots=200)
```
"""
function plot_energy_dynamics_vs_g(data_dir::String, scan_values::Vector{Float64};
        model::String      = "tfim",
        scan_param::String = "g",
        row::Int=3, p::Int=3, nqubits::Int=3,
        M::Int = 10_000,
        shots::Int = 1000,
        conv_step::Int = 100,
        parameter_source::Symbol = :optimized,
        random_seed = nothing,
        ylims = :auto,
        save_path::Union{String, Nothing} = nothing,
        fixed_params...)

    shots >= 2 || throw(ArgumentError("shots must be at least 2 channel iterations"))
    effective_ylims = if ylims !== :auto
        ylims
    elseif parameter_source === :random
        (-1.5, 1.0)
    elseif model == "tfim"
        (-4.5, -1.5)
    else
        nothing
    end

    palette = [:steelblue, :firebrick, :seagreen, :darkorange,
               :purple, :saddlebrown, :hotpink, :teal, :gray]

    fig = with_theme(paper_theme()) do
    fig = Figure(size=PAPER_FIGSIZE)
    ax = Axis(fig[1, 1];
              xlabel  = "Channel iteration",
              ylabel  = ENERGY_PER_SITE_LABEL,
              limits  = (nothing, effective_ylims))

    fixed_str   = join(["$(k)=$(v)" for (k, v) in sort(collect(fixed_params), by=first)], "_")
    name_prefix = isempty(fixed_str) ? "circuit_$(model)" : "circuit_$(model)_$(fixed_str)"

    for (idx, val) in enumerate(scan_values)
        pattern = "$(name_prefix)_$(scan_param)=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)"
        matches = filter(f -> startswith(f, pattern) && endswith(f, ".json"),
                         readdir(data_dir))
        if isempty(matches)
            @warn "No file matching $(pattern)*.json, skipping $(scan_param)=$(val)"
            continue
        end
        filename = joinpath(data_dir, first(matches))

        result, input_args = load_result(filename)
        model_str = String(get(input_args, :model, "tfim"))
        model_obj = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k, v) in input_args))
        has_y     = needs_y_measurement(model_obj)
        params    = _select_plot_params(result, parameter_source, idx; random_seed=random_seed)
        share_params = get(input_args, :share_params, true)
        structure = get(input_args, :structure, nothing)

        two_by_two = default_unit_cell(model_obj) == :two_by_two
        if two_by_two
            gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
        else
            gates = build_unitary_gate(params, p, row, nqubits;
                                       share_params=share_params,
                                       structure=structure)
        end

        exact_E = nothing
        if nqubits < 5
            op = if two_by_two
                TransferOperator([gates_odd, gates_even], row, (nqubits-1)÷2)
            else
                TransferOperator([gates], row, (nqubits-1)÷2)
            end

            exact_E = if model_obj isa TFIM
                e, _ = compute_exact_energy(model_obj, op)
                real(e) / row
            elseif model_obj isa HeisenbergJ1J2
                real(compute_exact_heisenberg_energy(op, model_obj.J1, model_obj.J2)) / row
            end
        else
            @info "Skipping exact contraction reference for nqubits=$nqubits in plot_energy_dynamics_vs_g" val
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
                                        conv_step=conv_samples, samples=run_samples, model=model_obj) :
                 sample_quantum_channel(gates, row, nqubits;
                                        conv_step=conv_samples, samples=run_samples, model=model_obj)
            sample_range = (conv_samples + 1):(conv_samples + run_samples)
            Z_s = ch[2][sample_range]
            X_s = ch[3][sample_range]
            Y_s = has_y ? ch[4][sample_range] : Float64[]
            for (i, k) in enumerate(eval_indices)
                cols = ((k - 2) * row + 1):(k * row)
                energy_curves[m, i] = compute_energy_from_samples(model_obj,
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
               color=color, label="$(scan_param)=$(val)$source_label")
        if !isnothing(exact_E)
            hlines!(ax, [exact_E], linestyle=:dash, color=color, label=nothing)
        end
    end

    vlines!(ax, [10];
            color=:black, linestyle=:dash, label="thermalization")

    # Outside legend: tight row spacing so all entries fit within PAPER_FIGSIZE_WIDE height
    Legend(fig[1, 2], ax;
           merge        = true,
           labelsize    = PAPER_LEGEND_LABELSIZE,
           rowgap       = PAPER_LEGEND_ROWGAP,
           colgap       = PAPER_LEGEND_COLGAP,
           patchsize    = PAPER_LEGEND_PATCHSIZE,
           patchlabelgap = PAPER_LEGEND_PATCHLABELGAP,
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
                ylabel=X_EXPECTATION_LABEL,
                limits=(xlims, (-1.05, 1.05)))
    ax_z = Axis(fig[2, 1];
                xlabel="Channel iteration",
                ylabel=Z_EXPECTATION_LABEL,
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
        share_params = get(input_args, :share_params, true)
        structure = get(input_args, :structure, nothing)
        two_by_two = default_unit_cell(model) == :two_by_two
        if two_by_two
            gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
        else
            gates = build_unitary_gate(params, p, row, nqubits;
                                       share_params=share_params,
                                       structure=structure)
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
        series_label = parameter_source === :random ?
            math_label("g=$g\\;\\mathrm{random}") :
            math_label("g=$g")
        lines!(ax_x, eval_indices, mean_x; color=color,
               label=series_label)
        band!(ax_z, eval_indices, mean_z .- se_z, mean_z .+ se_z; color=(color, 0.2))
        lines!(ax_z, eval_indices, mean_z; color=color, label=math_label("g=$g"))
    end

    Legend(fig[:, 2], ax_x;
           merge=true,
           labelsize=PAPER_LEGEND_LABELSIZE,
           rowgap=PAPER_LEGEND_ROWGAP,
           colgap=PAPER_LEGEND_COLGAP,
           patchsize=PAPER_LEGEND_PATCHSIZE,
           patchlabelgap=PAPER_LEGEND_PATCHLABELGAP,
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
