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
        if !isnothing(ly_match)
            return "DMRG(W=$(ly_match.captures[1]))"
        end
        return "DMRG"
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
        if isnan(ref_energy) || ref_energy == 0.0
            push!(errors, NaN)
        else
            push!(errors, (energy - ref_energy) / abs(ref_energy))
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
                                save_path::Union{String,Nothing}=nothing,
                                save_path_error::Union{String,Nothing}=nothing)

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
             [(isnan(ref) || ref == 0.0) ? NaN : (energy - ref) / abs(ref) for (energy, ref) in zip(energies_exact, energies_ref)]
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

    _figsize = isnothing(figsize) ? PAPER_FIGSIZE : figsize

    colors = [:steelblue, :darkorange, :purple, :teal, :brown, :gray40, :dodgerblue4, :tomato3]
    markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :cross, :xcross, :star5]

    fig_energy = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax1 = Axis(fig[1, 1];
                   xlabel      = xlabel_str,
                   ylabel      = "E / site",
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

        add_paper_legend!(ax1; position=:lb)

        if !isnothing(save_path)
            mkpath(dirname(save_path))
            save(save_path, fig)
            println("\nEnergy figure saved to: $save_path")
        end

        fig
    end

    _error_path = if !isnothing(save_path_error)
        save_path_error
    elseif !isnothing(save_path)
        base, ext = splitext(save_path)
        "$(base)_error$(ext)"
    else
        nothing
    end

    fig_error = with_theme(paper_theme()) do
        fig = Figure(size=_figsize)

        ax2 = Axis(fig[1, 1];
                   xlabel      = xlabel_str,
                   ylabel      = "(E_IsoPEPS − E_DMRG) / |E_DMRG|",
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
            hlines!(ax2, [0.0]; color=:gray, linestyle=:dash, linewidth=0.5)
            add_paper_legend!(ax2; position=:rb)
        else
            text!(ax2, 0.5, 0.5; text="No reference data",
                  align=(:center, :center), space=:relative,
                  fontsize=PAPER_AXIS_LABELSIZE, color=:gray)
        end

        if !isnothing(_error_path)
            mkpath(dirname(_error_path))
            save(_error_path, fig)
            println("\nError figure saved to: $_error_path")
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

    return fig_energy, fig_error, data
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
            fit_params = corr_exp_fit(col_seps, corr_vals; include_zero=false, fit_range=exp_fit_range)
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
