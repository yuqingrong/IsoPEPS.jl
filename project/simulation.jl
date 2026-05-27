using IsoPEPS
using Optimization
using Random
using CairoMakie
set_theme!(IsoPEPS.paper_theme())
using Yao
using LinearAlgebra, OMEinsum
using JSON3


const PARAMS_PER_QUBIT_PER_LAYER = IsoPEPS.PARAMS_PER_QUBIT_PER_LAYER

"""
    _find_warm_start_params(output_dir, model, scan_param, scan_value, row, p, nqubits; fixed_params...)

Scan `output_dir` for existing result files matching the model/fixed params pattern,
find the one whose scan parameter value is closest to (but not equal to) `scan_value`,
and load its parameters.

Returns `(params, source_value)` or `(nothing, nothing)` if no file is found.
"""
function _find_warm_start_params(output_dir, model, scan_param, scan_value, row, p, nqubits; fixed_params...)
    !isdir(output_dir) && return nothing, nothing

    # Build prefix and suffix for filename matching
    # Filename format: circuit_{model}_{fixed_params}_{scan_param}={value}_row={row}_p={p}_nqubits={nqubits}.json
    fixed_str = join(["$(k)=$(v)" for (k, v) in sort(collect(fixed_params), by=first)], "_")
    prefix = isempty(fixed_str) ? "circuit_$(model)_$(scan_param)=" : "circuit_$(model)_$(fixed_str)_$(scan_param)="
    suffix = "_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json" # TODO: not always true

    best_params = nothing
    best_val = nothing
    best_dist = Inf

    for fname in readdir(output_dir)
        startswith(fname, prefix) || continue
        endswith(fname, suffix) || continue

        # Extract scan parameter value from the middle
        val_str = fname[length(prefix)+1 : end-length(suffix)]
        val_file = tryparse(Float64, val_str)
        val_file === nothing && continue
        val_file == scan_value && continue  # skip the same value

        dist = abs(val_file - scan_value)
        if dist < best_dist
            filepath = joinpath(output_dir, fname)
            try
                data = open(filepath, "r") do io
                    JSON3.read(io)
                end
                params_val = get(data, :params, get(data, "params", nothing))
                if params_val !== nothing && !isempty(params_val)
                    best_params = Float64.(collect(params_val))
                    best_val = val_file
                    best_dist = dist
                end
            catch e
                @warn "Failed to read $fname: $e"
            end
        end
    end

    return best_params, best_val
end

"""
    simulation(; model, scan_param, scan_values, row, p, nqubits, ...)

Run circuit optimization for multiple parameter values and save results to JSON files.

# Arguments
- `model::String`: Model type (`"tfim"` or `"heisenberg_j1j2"`)
- `scan_param::Symbol`: Parameter to scan over (e.g., `:g` for TFIM, `:J2` for Heisenberg)
- `scan_values::Vector{Float64}`: Values of the scan parameter
- `row::Int`: Number of rows in the PEPS
- `p::Int`: Circuit depth
- `nqubits::Int`: Number of qubits per row
- `maxiter`: Maximum iterations for optimization
- `seed`: Random seed for reproducibility
- `verbose`: Print progress information
- `output_dir`: Directory to save results (default: "data")
- `share_params`: Share parameters across circuit layers
- `model_params...`: Fixed model parameters (e.g., `J=1.0` for TFIM, `J1=1.0` for Heisenberg)

# Example
```julia
# TFIM: scan g with fixed J
simulation(model="tfim", scan_param=:g, scan_values=[1.0, 2.0, 3.0],
           J=1.0, row=3, p=2, nqubits=3, ...)

# Heisenberg J1-J2: scan J2 with fixed J1
simulation(model="heisenberg_j1j2", scan_param=:J2, scan_values=[0.0, 0.25, 0.5],
           J1=1.0, row=3, p=2, nqubits=3, ...)
```
"""
function simulation(; model::String="tfim", scan_param::Symbol, scan_values::Vector{Float64},
                    row::Int, p::Int, nqubits::Int,
                    maxiter::Int, seed::Int=123, verbose::Bool=true,
                    output_dir::String, share_params::Bool=true, conv_step::Int=100, samples::Int=10000,
                    n_runs::Int=44, abstol::Float64=0.01,
                    active_nqubits::Int=nqubits,
                    unit_cell::Symbol=:single,
                    model_params...)

    # Create output directory if it doesn't exist
    !isdir(output_dir) && mkpath(output_dir)

    # Separate fixed model params (everything except the scan parameter)
    fixed_params = Dict{Symbol,Any}(model_params)

    n = length(scan_values)
    results = Vector{CircuitOptimizationResult}(undef, n)

    verbose && println("Running $(n) simulations for model=$(model), scanning $(scan_param)...")
    verbose && println("Results will be saved to: $output_dir/")

    for i in 1:n
        val = scan_values[i]

        # Try to warm-start from the closest existing result file in output_dir
        # 1. First try same nqubits
        warm_params, warm_val = _find_warm_start_params(output_dir, model, scan_param, val, row, p, nqubits;
                                                         fixed_params...)
        warm_from_nqubits = nothing
        if nqubits > 3
            # For enlarged-D scans, start from an embedded lower-D state rather
            # than an existing full-D result so the new virtual legs begin at 0.
            warm_params = nothing
            warm_val = nothing
        end
        warm_params = nothing
        # 2. If no same-nqubits result, try smaller nqubits and embed
        if warm_params === nothing
            for nq in (nqubits-2):-2:3
                # Try same row first, then row+2 (for systems that grew in both dimensions)
                found = false
                for try_row in (row, row+2)
                    warm_params, warm_val = _find_warm_start_params(output_dir, model, scan_param, val, try_row, p, nq;
                                                                     fixed_params...)
                    if warm_params !== nothing
                        warm_params = embed_params(warm_params, p, nq, nqubits; unit_cell=unit_cell)
                        warm_from_nqubits = nq
                        found = true
                        break
                    end
                end
                found && break
            end
        end

        @show warm_val
        if warm_params !== nothing
            params = warm_params
            if warm_from_nqubits !== nothing
                verbose && println("Starting $(scan_param) = $(val), warm-started from nqubits=$(warm_from_nqubits) $(scan_param) = $(warm_val)")
            else
                verbose && println("Starting $(scan_param) = $(val), warm-started from saved $(scan_param) = $(warm_val)")
            end

        else
            Random.seed!(seed)
            n_params = gate_parameter_count(p, nqubits;
                                            unit_cell=unit_cell,
                                            row=row,
                                            share_params=share_params)
            params = rand(n_params)
            verbose && println("Starting $(scan_param) = $(val), random initialization (seed=$seed)")
        end

        # Build model kwargs: fixed params + scan param
        model_kw = merge(fixed_params, Dict{Symbol,Any}(scan_param => val))

        # Checkpoint file for crash recovery
        checkpoint_file = joinpath(output_dir, ".checkpoint_$(scan_param)=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")

        result = optimize_circuit(params, p, row, nqubits;
                                  model=model,
                                  maxiter=maxiter,
                                  share_params=share_params,
                                  conv_step=conv_step,
                                  samples=samples,
                                  n_runs=n_runs,
                                  abstol=abstol,
                                  active_nqubits=active_nqubits,
                                  unit_cell=unit_cell,
                                  checkpoint_file=checkpoint_file,
                                  model_kw...)

        results[i] = result

        # Save result to JSON
        fixed_str = join(["$(k)=$(v)" for (k, v) in sort(collect(fixed_params), by=first)], "_")
        name_prefix = isempty(fixed_str) ? "circuit_$(model)" : "circuit_$(model)_$(fixed_str)"
        filename = joinpath(output_dir, "$(name_prefix)_$(scan_param)=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits)_1x1_6w.json")
        input_args = Dict{Symbol,Any}(
            :model => model, :scan_param => scan_param, scan_param => val,
            :row => row, :p => p, :nqubits => nqubits,
            :maxiter => maxiter,
            :active_nqubits => active_nqubits,
            :share_params => share_params, :seed => seed,
            :warm_started_from => warm_val,
            :warm_started_from_nqubits => warm_from_nqubits
        )
        merge!(input_args, fixed_params)
        save_result(filename, result, input_args)

        verbose && println("Completed $(scan_param) = $(val), energy = $(result.final_cost), saved to $(basename(filename))")
    end
end

#=
# ── Example: TFIM ──
 simulation(;
     model="heisenberg_j1j2",
     scan_param=:J2,
     scan_values=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,4.25, 4.5, 4.75, 5.0],
     J1=1.0,
     row=3, p=3, nqubits=3,
     maxiter=500,
     seed=123,
     verbose=true,
     output_dir=joinpath(@__DIR__, "results"),
     share_params=true,
     conv_step=100,
     samples=40000,
     n_runs=1,
     abstol=1e-5,
     unit_cell=:two_by_two
 )
=#
# ── Example: Heisenberg J1-J2 ──
 simulation(;
     model="heisenberg_j1j2",
     scan_param=:J2,
     scan_values=[0.5],
     J1=1.0,
     row=4, p=3, nqubits=5,
     maxiter=500,
     seed=123,
     verbose=true,
     output_dir=joinpath(@__DIR__, "results_heisenberg"),
     share_params=true,
     conv_step=1000,
     samples=4000,
     n_runs=10,
     abstol=1e-5,
     unit_cell=:two_by_two
 )
#=
simulation(;
    model="tfim",
    scan_param=:g,
    scan_values=[1.0],
    J=1.0,
    row=3,
    p=5,
    nqubits=3,
    maxiter=500,
    seed=123,
    verbose=true,
    output_dir=joinpath(@__DIR__, "results"),
    share_params=true,
    conv_step=102,
    samples=6000,
    n_runs=10,
    abstol=1e-5)
=#