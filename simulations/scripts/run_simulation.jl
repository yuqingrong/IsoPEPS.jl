"""
Config-driven simulation runner for IsoPEPS.

Usage:
    julia --project=simulations simulations/scripts/run_simulation.jl configs/tfim_scan_g.toml
"""

using TOML
using IsoPEPS
using Random
using JSON3

function load_config(config_path::String)
    config = TOML.parsefile(config_path)
    return config
end

function construct_model_from_config(model_cfg::Dict)
    model_type = model_cfg["type"]
    if model_type == "TFIM"
        return TFIM(J=get(model_cfg, "J", 1.0), g=get(model_cfg, "g", 1.0))
    elseif model_type == "HeisenbergJ1J2"
        return HeisenbergJ1J2(J1=get(model_cfg, "J1", 1.0), J2=get(model_cfg, "J2", 0.0))
    else
        error("Unknown model type: $model_type")
    end
end

function run_simulation(config_path::String)
    config = load_config(config_path)

    # Parse sections
    model_cfg = config["model"]
    scan_cfg = config["scan"]
    circuit_cfg = config["circuit"]
    opt_cfg = config["optimization"]

    # Circuit parameters
    row = circuit_cfg["row"]
    p = circuit_cfg["p"]
    nqubits = circuit_cfg["nqubits"]
    unit_cell = Symbol(get(circuit_cfg, "unit_cell", "single"))
    share_params = get(circuit_cfg, "share_params", true)

    # Optimization parameters
    maxiter = opt_cfg["maxiter"]
    abstol = Float64(opt_cfg["abstol"])
    n_runs = get(opt_cfg, "n_runs", 44)
    samples = get(opt_cfg, "samples", 10000)
    conv_step = get(opt_cfg, "conv_step", 100)
    seed = get(opt_cfg, "seed", 123)

    # Scan parameters
    scan_param = Symbol(scan_cfg["parameter"])
    scan_values = Float64.(scan_cfg["values"])

    # Output directory
    base_model = construct_model_from_config(model_cfg)
    mname = model_name(base_model)
    output_dir = joinpath(@__DIR__, "..", "results", mname, "data")
    mkpath(output_dir)

    println("=" ^ 70)
    println("IsoPEPS Simulation Runner")
    println("Config: $config_path")
    println("Model: $(model_label(base_model))")
    println("Scanning $scan_param: $scan_values")
    println("Output: $output_dir")
    println("=" ^ 70)

    for val in scan_values
        # Construct model with current scan value
        model_params = copy(model_cfg)
        model_params[string(scan_param)] = val
        m = construct_model_from_config(model_params)

        # Initialize parameters
        Random.seed!(seed)
        n_params = IsoPEPS.gate_parameter_count(p, nqubits;
                                                unit_cell=unit_cell,
                                                row=row,
                                                share_params=share_params)
        params = rand(n_params)

        println("\nRunning $scan_param = $val ...")

        result = optimize_circuit(params, p, row, nqubits;
                                  model=m,
                                  maxiter=maxiter,
                                  share_params=share_params,
                                  conv_step=conv_step,
                                  samples=samples,
                                  n_runs=n_runs,
                                  abstol=abstol,
                                  unit_cell=unit_cell)

        # Save result
        filename = joinpath(output_dir, "circuit_$(mname)_$(scan_param)=$(val)_row=$(row)_p=$(p)_nqubits=$(nqubits).json")
        input_args = Dict{Symbol,Any}(
            :model => model_name(m),
            :scan_param => scan_param,
            scan_param => val,
            :row => row, :p => p, :nqubits => nqubits,
            :maxiter => maxiter, :seed => seed
        )
        save_result(filename, result, input_args)

        println("Completed $scan_param = $val, energy = $(result.final_cost)")
    end

    println("\nAll simulations complete.")
end

# Entry point
if !isempty(ARGS)
    run_simulation(ARGS[1])
else
    println("Usage: julia --project=simulations simulations/scripts/run_simulation.jl <config.toml>")
    println("Example: julia --project=simulations simulations/scripts/run_simulation.jl simulations/configs/tfim_scan_g.toml")
end


run_simulation("simulations/configs/heisenberg_j1j2_scan_J2.toml")
