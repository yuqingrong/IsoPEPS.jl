# Data I/O: save and load optimization results

# ============================================================================
# Data I/O with JSON
# ============================================================================

"""
    save_result(filename::String, result, input_args::Dict)

Save optimization result to JSON file with input arguments.

# Arguments
- `filename`: Path to save file
- `result`: Optimization result (CircuitOptimizationResult, ExactOptimizationResult, or ManifoldOptimizationResult)
- `input_args`: Dictionary of input arguments/metadata

# Example
```julia
result = optimize_circuit(...)
input_args = Dict(
    :g => 2.0, :J => 1.0, :row => 3,
    :initial_params => params,
    :maxiter => 5000
)
save_result("data/result.json", result, input_args)
```
"""
function save_result(filename::String, result::CircuitOptimizationResult, input_args::Dict)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)

    data = Dict{Symbol, Any}(
        :type => "CircuitOptimizationResult",
        :energy_history => result.energy_history,
        :params => result.final_params,
        :energy => result.final_cost,
        :converged => result.converged,
        :Z_samples => result.final_Z_samples,
        :X_samples => result.final_X_samples,
        :Y_samples => result.final_Y_samples,
        :input_args => input_args
    )

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    @info "Result saved to $filename"
end

function save_result(filename::String, result::ExactOptimizationResult, input_args::Dict)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)

    data = Dict{Symbol, Any}(
        :type => "ExactOptimizationResult",
        :energy_history => result.energy_history,
        :params => result.params,
        :energy => result.energy,
        :gap => result.gap,
        :eigenvalues => result.eigenvalues,
        :X_expectation => result.X_expectation,
        :ZZ_vertical => result.ZZ_vertical,
        :ZZ_horizontal => result.ZZ_horizontal,
        :converged => result.converged,
        :input_args => input_args
    )

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    @info "Result saved to $filename"
end

function save_result(filename::String, result::ManifoldOptimizationResult, input_args::Dict)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)

    data = Dict{Symbol, Any}(
        :type => "ManifoldOptimizationResult",
        :energy_history => result.energy_history,
        :gate => result.gate,
        :energy => result.energy,
        :gap_history => result.gap_history,
        :converged => result.converged,
        :input_args => input_args
    )

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    @info "Result saved to $filename"
end
"""
    save_results(filename::String; kwargs...)

Save arbitrary results to a JSON file. Accepts keyword arguments for flexibility.

# Example
```julia
save_results("results.json";
    g=2.0, row=3,
    energy_history=[1.0, 0.5, 0.3],
    correlation_matrix=corr_mat
)
```
"""
function save_results(filename::String; kwargs...)
    dir = dirname(filename)
    !isempty(dir) && !isdir(dir) && mkpath(dir)

    open(filename, "w") do io
        JSON3.pretty(io, Dict(kwargs))
    end
    @info "Results saved to $filename"
end

"""
    load_results(filename::String) -> Dict

Load results from a JSON file.
"""
function load_results(filename::String)
    open(filename, "r") do io
        JSON3.read(io, Dict)
    end
end

"""
    load_result(filename::String; result_type=:auto)

Load optimization result from JSON file.

# Arguments
- `filename`: Path to JSON file
- `result_type`: Result type to load (`:auto`, `:circuit`, `:exact`, or `:manifold`)

# Returns
Tuple of (result, input_args_dict)
"""
function load_result(filename::String; result_type::Symbol=:auto)
    data = open(filename, "r") do io
        JSON3.read(io, Dict)
    end

    # Determine type
    if result_type == :auto
        result_type_str = get(data, "type", get(data, :type, ""))
        result_type = Symbol(lowercase(replace(result_type_str, "OptimizationResult" => "")))
    end

    # Extract input_args and convert string keys to symbols
    input_args_raw = get(data, "input_args", get(data, :input_args, Dict()))
    input_args = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(input_args_raw))

    # Helper function to get data with both string and symbol key fallback
    function get_data(dict, key)
        get(dict, string(key), get(dict, Symbol(key), nothing))
    end

    # Helper to convert samples to matrix
    function samples_to_matrix(samples_data, input_args)
        if samples_data === nothing || isempty(samples_data)
            return Matrix{Float64}(undef, 0, 0)
        end

        # Check if data is nested array (new format) or flat vector (old format)
        if samples_data isa Vector && !isempty(samples_data) && samples_data[1] isa Vector
            # New format: nested array (array of arrays)
            # Each element is a row (chain) in the matrix
            return reduce(vcat, [Vector{Float64}(row)' for row in samples_data])
        else
            # Old format: flat vector - need to reshape
            samples_vec = Vector{Float64}(samples_data)
            n_parallel_runs = get(input_args, :n_parallel_runs, 1)

            # Infer actual samples per run from the data size
            actual_samples_per_run = div(length(samples_vec), n_parallel_runs)

            # Reshape: flat vector -> matrix (n_parallel_runs × actual_samples_per_run)
            # The samples are stored in row-major order
            return reshape(samples_vec, actual_samples_per_run, n_parallel_runs)'
        end
    end

    # Reconstruct result based on type
    if result_type == :circuit
        Z_samples_data = get(data, "Z_samples", get(data, :Z_samples, nothing))
        X_samples_data = get(data, "X_samples", get(data, :X_samples, nothing))
        Y_samples_data = get(data, "Y_samples", get(data, :Y_samples, nothing))

        energy_history = get_data(data, :energy_history)
        params = get_data(data, :params)
        energy = get_data(data, :energy)
        converged = get_data(data, :converged)

        # Convert samples to vectors (flatten if matrix)
        Z_samples = samples_to_matrix(Z_samples_data, input_args)
        X_samples = samples_to_matrix(X_samples_data, input_args)
        Z_samples_vec = Z_samples isa AbstractMatrix ? vec(collect(Z_samples)) : Vector{Float64}(collect(Z_samples))
        X_samples_vec = X_samples isa AbstractMatrix ? vec(collect(X_samples)) : Vector{Float64}(collect(X_samples))
        Y_samples_vec = Y_samples_data === nothing ? Float64[] : Vector{Float64}(collect(Y_samples_data))

        result = CircuitOptimizationResult(
            Vector{Float64}(energy_history === nothing ? Float64[] : energy_history),
            Vector{Matrix{ComplexF64}}[],  # Gates not saved to JSON
            Vector{Float64}(params === nothing ? Float64[] : params),
            Float64(energy === nothing ? 0.0 : energy),
            Z_samples_vec,
            X_samples_vec,
            Y_samples_vec,
            Bool(converged === nothing ? false : converged)
        )
    elseif result_type == :exact
        result = ExactOptimizationResult(
            Vector{Float64}(get_data(data, :energy_history)),
            Vector{Matrix{ComplexF64}}[],  # Gates not saved to JSON
            Vector{Float64}(get_data(data, :params)),
            Float64(get_data(data, :energy)),
            Float64(get_data(data, :gap)),
            Vector{Float64}(get_data(data, :eigenvalues)),
            Float64(get_data(data, :X_expectation)),
            Float64(get_data(data, :ZZ_vertical)),
            Float64(get_data(data, :ZZ_horizontal)),
            Bool(get_data(data, :converged))
        )
    elseif result_type == :manifold
        result = ManifoldOptimizationResult(
            Vector{Float64}(get_data(data, :energy_history)),
            Matrix{ComplexF64}(get_data(data, :gate)),
            Float64(get_data(data, :energy)),
            Vector{Float64}(get_data(data, :gap_history)),
            Bool(get_data(data, :converged))
        )
    else
        error("Unknown result type: $result_type")
    end

    return result, input_args
end
