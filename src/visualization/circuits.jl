# ============================================================================
# Local circuit block diagram
# ============================================================================

function _plot_gate_box!(ax, x, y, label; width=0.58, height=0.46,
                         color=(:white, 1.0), strokecolor=:black,
                         fontsize=9)
    points = Point2f[
        (x - width/2, y - height/2),
        (x + width/2, y - height/2),
        (x + width/2, y + height/2),
        (x - width/2, y + height/2),
    ]
    poly!(ax, points; color=color, strokecolor=strokecolor, strokewidth=1.0)
    text!(ax, x, y; text=label, align=(:center, :center),
          fontsize=fontsize, color=:black)
end

function _plot_cnot!(ax, x, control_y, target_y; color=:black)
    lines!(ax, [x, x], [control_y, target_y]; color=color, linewidth=1.1)
    scatter!(ax, [x], [control_y]; marker=:circle, markersize=8,
             color=color, strokecolor=color)

    radius = 0.14
    theta = range(0, 2π; length=80)
    lines!(ax, x .+ radius .* cos.(theta), target_y .+ radius .* sin.(theta);
           color=color, linewidth=1.0)
    lines!(ax, [x - radius, x + radius], [target_y, target_y];
           color=color, linewidth=1.0)
    lines!(ax, [x, x], [target_y - radius, target_y + radius];
           color=color, linewidth=1.0)
end

function _circuit_plot_columns(p::Int, nqubits::Int; max_stride::Int=nqubits-1,
                               active_nqubits::Int=nqubits)
    ops = local_circuit_ops(p, nqubits;
                            max_stride=max_stride,
                            active_nqubits=active_nqubits)
    columns = Vector{Vector{Union{LocalCircuitOp,Nothing}}}()

    i = 1
    while i <= length(ops)
        op = ops[i]
        if op.kind == :cnot
            column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, nqubits)
            control, target = op.qubits
            column[control] = op
            column[target] = op
            push!(columns, column)
            i += 1
            continue
        end

        rx_column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, nqubits)
        rz_column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, nqubits)
        layer = op.layer
        while i <= length(ops) && ops[i].kind != :cnot && ops[i].layer == layer
            rot_op = ops[i]
            rot_op.kind == :rx && (rx_column[rot_op.qubits[1]] = rot_op)
            rot_op.kind == :rz && (rz_column[rot_op.qubits[1]] = rot_op)
            i += 1
        end
        push!(columns, rx_column)
        push!(columns, rz_column)
    end

    return columns
end

function _channel_gate_positions(row::Int, nqubits::Int)
    remaining_qubits = (nqubits - 1) ÷ 2
    fixed_qubits = (nqubits + 1) ÷ 2
    n_env = remaining_qubits * (row + 1)
    total_qubits = n_env + 1

    qpositions = Vector{Vector{Int}}(undef, row)
    for j in 1:row
        qpositions[j] = vcat(collect(1:fixed_qubits),
                             collect(fixed_qubits + (j-1)*remaining_qubits + 1:
                                     fixed_qubits + j*remaining_qubits))
    end
    return qpositions, total_qubits
end

function _embedded_circuit_columns(p::Int, nqubits::Int, total_qubits::Int,
                                   qpos::AbstractVector{Int};
                                   max_stride::Int=nqubits-1,
                                   active_nqubits::Int=nqubits,
                                   param_offset::Int=0)
    local_columns = _circuit_plot_columns(p, nqubits;
                                          max_stride=max_stride,
                                          active_nqubits=active_nqubits)
    columns = Vector{Vector{Union{LocalCircuitOp,Nothing}}}()

    for local_column in local_columns
        column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, total_qubits)
        for local_q in 1:nqubits
            op = local_column[local_q]
            isnothing(op) && continue
            if op.kind == :rx || op.kind == :rz
                global_q = qpos[op.qubits[1]]
                param_index = isnothing(op.param_index) ? nothing : param_offset + op.param_index
                column[global_q] = LocalCircuitOp(op.kind, (global_q,), op.layer, param_index)
            elseif op.kind == :cnot
                control, target = op.qubits
                global_control = qpos[control]
                global_target = qpos[target]
                global_op = LocalCircuitOp(:cnot, (global_control, global_target), op.layer, nothing)
                column[global_control] = global_op
                column[global_target] = global_op
            end
        end
        push!(columns, column)
    end

    return columns
end

function _unit_cell_label_and_offset(unit_cell::Symbol, cycle::Int, row_index::Int, chunk::Int)
    if unit_cell === :single
        return "U$row_index", 0
    elseif unit_cell === :two_by_two
        if isodd(cycle)
            isodd(row_index) && return "A", 0
            return "B", chunk
        else
            isodd(row_index) && return "C", 2chunk
            return "D", 3chunk
        end
    else
        throw(ArgumentError("unit_cell must be :single or :two_by_two"))
    end
end

function _plot_compact_channel_gate!(ax, x, ys, label, local_qubits;
                                     color=(:lavender, 0.9),
                                     linecolor=(:black, 0.45))
    ymin, ymax = extrema(ys)
    lines!(ax, [x, x], [ymin, ymax]; color=linecolor, linewidth=1.0)

    for (y, local_q) in zip(ys, local_qubits)
        _plot_gate_box!(ax, x, y, "$(label)\nq$local_q";
                        width=0.62, height=0.48,
                        color=color, strokecolor=:black,
                        fontsize=7)
    end
end

"""
    plot_circuit_block(p, nqubits; max_stride=nqubits-1, active_nqubits=nqubits,
                       show_params=true, save_path=nothing)

Draw the local circuit block generated by `build_unitary_gate`. The figure is
built from `local_circuit_ops`, so it shows the same `Rx`, `Rz`, and CNOT
ordering used by the gate constructor.
"""
function plot_circuit_block(p::Int, nqubits::Int; max_stride::Int=nqubits-1,
                            active_nqubits::Int=nqubits,
                            show_params::Bool=true,
                            save_path::Union{String,Nothing}=nothing)
    columns = _circuit_plot_columns(p, nqubits;
                                    max_stride=max_stride,
                                    active_nqubits=active_nqubits)
    ncols = length(columns)
    fig_width = max(520, 44 * ncols + 90)
    fig_height = max(220, 62 * nqubits + 60)
    fig = Figure(size=(fig_width, fig_height))
    ax = Axis(fig[1, 1];
              aspect=DataAspect(),
              xgridvisible=false,
              ygridvisible=false)
    hidedecorations!(ax)
    hidespines!(ax)

    xs = collect(1:ncols)
    y_for(q) = nqubits - q + 1
    for q in 1:nqubits
        y = y_for(q)
        lines!(ax, [0.4, ncols + 0.6], [y, y]; color=(:black, 0.55), linewidth=0.8)
        text!(ax, 0.12, y; text="q$q", align=(:right, :center),
              fontsize=10, color=:black)
    end

    for (colidx, column) in enumerate(columns)
        drawn_cnot = false
        for q in 1:nqubits
            op = column[q]
            isnothing(op) && continue
            y = y_for(q)
            if op.kind == :rx || op.kind == :rz
                gate = op.kind == :rx ? "Rx" : "Rz"
                label = show_params ? "$(gate)\nθ$(op.param_index)" : gate
                color = op.kind == :rx ? (:lightskyblue, 0.85) : (:peachpuff, 0.9)
                _plot_gate_box!(ax, xs[colidx], y, label; color=color)
            elseif op.kind == :cnot && !drawn_cnot
                control, target = op.qubits
                _plot_cnot!(ax, xs[colidx], y_for(control), y_for(target))
                drawn_cnot = true
            end
        end
    end

    xlims!(ax, -0.1, ncols + 0.8)
    ylims!(ax, 0.35, nqubits + 0.65)

    if !isnothing(save_path)
        save(save_path, fig)
    end
    return fig
end

"""
    plot_channel_circuit(row, p, nqubits; cycles=1, unit_cell=:single,
                         expanded=true, show_params=false, measure=true,
                         reset=true, initial_state=true, save_path=nothing)

Draw a finite unrolling of the quantum-channel circuit used by
`sample_quantum_channel`. `cycles` controls how many outer channel iterations
are shown; each cycle contains all `row` gate applications. With
`expanded=true`, every local `Rx`, `Rz`, and CNOT is embedded on the channel
wires. With `expanded=false`, each row gate is drawn only on the global wires
it actually touches, with labels `q1`, `q2`, ... for the local qubit positions
inside that block. When `measure=true`, the measured output wire is marked by
`M`; when `reset=true`, the following reset/preparation to `|0>` is also drawn.
When `initial_state=true`, every channel wire is initialized as `|0>` on the
left side of the diagram.
"""
function plot_channel_circuit(row::Int, p::Int, nqubits::Int;
                              cycles::Int=1,
                              unit_cell::Symbol=:single,
                              expanded::Bool=true,
                              max_stride::Int=nqubits-1,
                              active_nqubits::Int=nqubits,
                              show_params::Bool=false,
                              measure::Bool=true,
                              reset::Bool=true,
                              initial_state::Bool=true,
                              save_path::Union{String,Nothing}=nothing)
    cycles >= 1 || throw(ArgumentError("cycles must be at least 1"))
    qpositions, total_qubits = _channel_gate_positions(row, nqubits)
    chunk = _gate_param_count(p, nqubits;
                              max_stride=max_stride,
                              active_nqubits=active_nqubits)

    columns = Vector{Vector{Union{LocalCircuitOp,Nothing}}}()
    labels = String[]
    box_qubits = Vector{Vector{Int}}()
    box_local_qubits = Vector{Vector{Int}}()
    cycle_starts = Int[]

    for cycle in 1:cycles
        push!(cycle_starts, length(columns) + 1)
        for j in 1:row
            gate_label, param_offset = _unit_cell_label_and_offset(unit_cell, cycle, j, chunk)
            if expanded
                embedded = _embedded_circuit_columns(p, nqubits, total_qubits, qpositions[j];
                                                     max_stride=max_stride,
                                                     active_nqubits=active_nqubits,
                                                     param_offset=param_offset)
                append!(columns, embedded)
                append!(labels, fill("", length(embedded)))
                append!(box_qubits, fill(Int[], length(embedded)))
                append!(box_local_qubits, fill(Int[], length(embedded)))
            else
                column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, total_qubits)
                push!(columns, column)
                push!(labels, gate_label)
                push!(box_qubits, qpositions[j])
                push!(box_local_qubits, collect(1:length(qpositions[j])))
            end

            if measure
                column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, total_qubits)
                column[1] = LocalCircuitOp(:measure, (1,), cycle, nothing)
                push!(columns, column)
                push!(labels, "")
                push!(box_qubits, Int[])
                push!(box_local_qubits, Int[])

                if reset
                    column = Vector{Union{LocalCircuitOp,Nothing}}(nothing, total_qubits)
                    column[1] = LocalCircuitOp(:reset, (1,), cycle, nothing)
                    push!(columns, column)
                    push!(labels, "")
                    push!(box_qubits, Int[])
                    push!(box_local_qubits, Int[])
                end
            end
        end
    end

    ncols = length(columns)
    fig_width = max(620, min(2400, 38 * ncols + 110))
    fig_height = max(240, 42 * total_qubits + 70)
    fig = Figure(size=(fig_width, fig_height))
    ax = Axis(fig[1, 1];
              aspect=DataAspect(),
              xgridvisible=false,
              ygridvisible=false)
    hidedecorations!(ax)
    hidespines!(ax)

    y_for(q) = total_qubits - q + 1
    for q in 1:total_qubits
        y = y_for(q)
        lines!(ax, [0.4, ncols + 0.8], [y, y]; color=(:black, 0.45), linewidth=0.8)
        text!(ax, -0.12, y; text="q$q", align=(:right, :center), fontsize=9, color=:black)
        if initial_state
            text!(ax, 0.14, y; text="|0>", align=(:left, :center), fontsize=8, color=:black)
        end
    end

    for start in cycle_starts
        lines!(ax, [start - 0.5, start - 0.5], [0.35, total_qubits + 0.65];
               color=(:gray, 0.35), linewidth=0.8, linestyle=:dash)
    end

    for (colidx, column) in enumerate(columns)
        x = colidx
        if !expanded && !isempty(box_qubits[colidx])
            _plot_compact_channel_gate!(ax, x, y_for.(box_qubits[colidx]),
                                        labels[colidx], box_local_qubits[colidx])
            continue
        end

        drawn_cnot = false
        for q in 1:total_qubits
            op = column[q]
            isnothing(op) && continue
            y = y_for(q)
            if op.kind == :rx || op.kind == :rz
                gate = op.kind == :rx ? "Rx" : "Rz"
                label = show_params ? "$(gate)\nθ$(op.param_index)" : gate
                color = op.kind == :rx ? (:lightskyblue, 0.85) : (:peachpuff, 0.9)
                _plot_gate_box!(ax, x, y, label; color=color, fontsize=8)
            elseif op.kind == :cnot && !drawn_cnot
                control, target = op.qubits
                _plot_cnot!(ax, x, y_for(control), y_for(target))
                drawn_cnot = true
            elseif op.kind == :measure
                _plot_gate_box!(ax, x, y, "M"; width=0.42, height=0.36,
                                color=(:white, 1.0), fontsize=8)
            elseif op.kind == :reset
                _plot_gate_box!(ax, x, y, "|0>"; width=0.52, height=0.36,
                                color=(:honeydew, 1.0), fontsize=8)
            end
        end
    end

    xlims!(ax, -0.6, ncols + 0.9)
    ylims!(ax, 0.35, total_qubits + 0.65)

    if !isnothing(save_path)
        save(save_path, fig)
    end
    return fig
end

