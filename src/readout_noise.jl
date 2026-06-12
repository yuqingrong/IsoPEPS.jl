# =============================================================================
# Classical Readout Noise
# =============================================================================

"""
    apply_readout_noise(samples, p; rng=Random.default_rng())

Apply independent symmetric readout bit-flip errors to classical Pauli
measurement outcomes.

Each input outcome must be `+1` or `-1`. With probability `p`, an outcome is
reported with the opposite sign. The input is not modified.
"""
function apply_readout_noise(samples::AbstractArray{<:Real}, p::Real;
                             rng::AbstractRNG=Random.default_rng())
    probability = Float64(p)
    isfinite(probability) && 0.0 <= probability <= 1.0 ||
        throw(ArgumentError("p must be finite and between 0 and 1"))

    noisy = Float64.(samples)
    all(value -> value == -1.0 || value == 1.0, noisy) ||
        throw(ArgumentError("readout samples must contain only -1 and +1 outcomes"))

    probability == 0.0 && return noisy
    if probability == 1.0
        noisy .*= -1.0
        return noisy
    end

    for idx in eachindex(noisy)
        rand(rng) < probability && (noisy[idx] = -noisy[idx])
    end
    return noisy
end

function _readout_get(data, key::Symbol, default=nothing)
    haskey(data, key) && return data[key]
    string_key = String(key)
    haskey(data, string_key) && return data[string_key]
    return default
end

function _readout_sample_chains(raw_samples, row::Int;
                                burnin::Int=0,
                                n_chains_hint::Union{Nothing,Int}=nothing,
                                basis::String)
    raw_samples === nothing &&
        throw(ArgumentError("$basis samples are missing"))
    isempty(raw_samples) &&
        throw(ArgumentError("$basis samples are empty"))

    chains = if first(raw_samples) isa AbstractVector
        [Float64.(collect(chain)) for chain in raw_samples]
    else
        flat = Float64.(collect(raw_samples))
        if isnothing(n_chains_hint) || n_chains_hint == 1
            [flat]
        else
            n_chains_hint > 0 ||
                throw(ArgumentError("number of chains must be positive"))
            length(flat) % n_chains_hint == 0 ||
                throw(ArgumentError(
                    "$basis sample count $(length(flat)) is not divisible by " *
                    "n_chains=$n_chains_hint"))
            chain_length = length(flat) ÷ n_chains_hint
            [collect(@view flat[(idx-1)*chain_length+1:idx*chain_length])
             for idx in 1:n_chains_hint]
        end
    end

    processed = [_discard_burnin(chain, row, burnin) for chain in chains]
    for (idx, chain) in enumerate(processed)
        length(chain) >= 2 * row ||
            throw(ArgumentError(
                "$basis chain $idx must contain at least two complete columns"))
    end
    return processed
end

function _load_readout_samples(samples_file::String)
    isfile(samples_file) ||
        throw(ArgumentError("samples file does not exist: $samples_file"))

    data = load_results(samples_file)
    input_args = _readout_get(data, :input_args, nothing)
    metadata = isnothing(input_args) ? data : input_args

    model_str = String(_readout_get(
        metadata, :model, _readout_get(data, :model, "tfim")))
    row_data = _readout_get(metadata, :row, _readout_get(data, :row, nothing))
    isnothing(row_data) &&
        throw(ArgumentError("row metadata is missing from $samples_file"))
    row = Int(row_data)
    row > 0 || throw(ArgumentError("row must be positive"))

    # Finite-cylinder sample files store burn-in metadata with nested chains.
    # Optimization-result samples have already had burn-in removed.
    burnin = isnothing(input_args) ? Int(_readout_get(data, :conv_step, 0)) : 0
    n_chains_data = if isnothing(input_args)
        _readout_get(data, :n_chains, nothing)
    else
        _readout_get(
            input_args, :n_runs,
            _readout_get(input_args, :n_parallel_runs, nothing))
    end
    n_chains_hint = isnothing(n_chains_data) ? nothing : Int(n_chains_data)

    X_chains = _readout_sample_chains(
        _readout_get(data, :X_samples, nothing), row;
        burnin=burnin, n_chains_hint=n_chains_hint, basis="X")
    Z_chains = _readout_sample_chains(
        _readout_get(data, :Z_samples, nothing), row;
        burnin=burnin, n_chains_hint=n_chains_hint, basis="Z")

    model = if model_str == "tfim"
        TFIM(
            J=Float64(_readout_get(metadata, :J, _readout_get(data, :J, 1.0))),
            g=Float64(_readout_get(metadata, :g, _readout_get(data, :g, 1.0))),
        )
    elseif model_str == "heisenberg_j1j2"
        HeisenbergJ1J2(
            J1=Float64(_readout_get(
                metadata, :J1, _readout_get(data, :J1, 1.0))),
            J2=Float64(_readout_get(
                metadata, :J2, _readout_get(data, :J2, 0.0))),
        )
    else
        throw(ArgumentError("unsupported model in $samples_file: $model_str"))
    end

    Y_chains = if needs_y_measurement(model)
        _readout_sample_chains(
            _readout_get(data, :Y_samples, nothing), row;
            burnin=burnin, n_chains_hint=n_chains_hint, basis="Y")
    else
        Vector{Float64}[]
    end

    n_chains = length(X_chains)
    length(Z_chains) == n_chains ||
        throw(ArgumentError("X and Z samples contain different chain counts"))
    if needs_y_measurement(model)
        length(Y_chains) == n_chains ||
            throw(ArgumentError("X, Y, and Z samples contain different chain counts"))
    end

    for chain_idx in 1:n_chains
        length(X_chains[chain_idx]) == length(Z_chains[chain_idx]) ||
            throw(ArgumentError(
                "X and Z chain $chain_idx contain different sample counts"))
        if needs_y_measurement(model)
            length(Y_chains[chain_idx]) == length(X_chains[chain_idx]) ||
                throw(ArgumentError(
                    "X, Y, and Z chain $chain_idx contain different sample counts"))
        end
    end

    return (
        model=model,
        row=row,
        X_chains=X_chains,
        Z_chains=Z_chains,
        Y_chains=Y_chains,
    )
end

function _readout_mean_energy(model::AbstractModel,
                              X_chains::Vector{Vector{Float64}},
                              Z_chains::Vector{Vector{Float64}},
                              Y_chains::Vector{Vector{Float64}},
                              row::Int)
    energies = Vector{Float64}(undef, length(X_chains))
    for chain_idx in eachindex(X_chains)
        Y = needs_y_measurement(model) ? Y_chains[chain_idx] : Float64[]
        energies[chain_idx] = real(compute_energy_from_samples(
            model, X_chains[chain_idx], Z_chains[chain_idx], Y, row))
    end
    return mean(energies)
end

"""
    compute_readout_energy_scan(samples_file; kwargs...)

Apply classical readout bit-flip noise to saved measurement samples and
recompute the measured local energy density.

The saved quantum samples are never modified. For each probability and repeat,
the X, Z, and (when present) Y outcomes are independently flipped per site.
Energy is computed separately for each saved chain and then averaged, avoiding
spurious two-body terms across chain boundaries.

# Keyword arguments
- `p_values`: Readout flip probabilities.
- `repeats`: Independent readout-noise realizations per probability.
- `seed`: Seed for reproducible classical flips.
- `save_path`: Optional JSON output path.

# Returns
A named tuple containing the probability scan, repeated energies, summary
statistics, baseline energy, and sample metadata.
"""
function compute_readout_energy_scan(
        samples_file::String;
        p_values::AbstractVector{<:Real}=[0.0, 0.005, 0.01, 0.02, 0.05],
        repeats::Int=100,
        seed::Integer=1234,
        save_path::Union{Nothing,String}=nothing)
    isempty(p_values) &&
        throw(ArgumentError("p_values must not be empty"))
    repeats >= 2 ||
        throw(ArgumentError("repeats must be at least 2 to estimate uncertainty"))

    probabilities = Float64.(p_values)
    for probability in probabilities
        isfinite(probability) && 0.0 <= probability <= 1.0 ||
            throw(ArgumentError(
                "all readout probabilities must be finite and between 0 and 1"))
    end

    dataset = _load_readout_samples(samples_file)
    model = dataset.model
    row = dataset.row
    X_chains = dataset.X_chains
    Z_chains = dataset.Z_chains
    Y_chains = dataset.Y_chains

    baseline_energy = _readout_mean_energy(
        model, X_chains, Z_chains, Y_chains, row)
    rng = MersenneTwister(seed)
    energy_samples = Matrix{Float64}(undef, length(probabilities), repeats)

    for (p_idx, probability) in enumerate(probabilities)
        for repeat_idx in 1:repeats
            noisy_X = [apply_readout_noise(chain, probability; rng=rng)
                       for chain in X_chains]
            noisy_Z = [apply_readout_noise(chain, probability; rng=rng)
                       for chain in Z_chains]
            noisy_Y = needs_y_measurement(model) ?
                [apply_readout_noise(chain, probability; rng=rng)
                 for chain in Y_chains] :
                Vector{Float64}[]

            energy_samples[p_idx, repeat_idx] = _readout_mean_energy(
                model, noisy_X, noisy_Z, noisy_Y, row)
        end
    end

    energy_mean = vec(mean(energy_samples; dims=2))
    energy_std = vec(std(energy_samples; dims=2))
    energy_stderr = energy_std ./ sqrt(repeats)
    samples_per_chain = length.(Z_chains)

    result = (
        p_values=probabilities,
        energy_mean=energy_mean,
        energy_std=energy_std,
        energy_stderr=energy_stderr,
        energy_samples=energy_samples,
        baseline_energy=baseline_energy,
        repeats=repeats,
        seed=Int(seed),
        model=model_name(model),
        row=row,
        n_chains=length(Z_chains),
        samples_per_chain=samples_per_chain,
    )

    if !isnothing(save_path)
        saved_energy_samples = [
            collect(@view energy_samples[p_idx, :])
            for p_idx in axes(energy_samples, 1)
        ]
        save_results(save_path;
                     p_values=probabilities,
                     energy_mean=energy_mean,
                     energy_std=energy_std,
                     energy_stderr=energy_stderr,
                     energy_samples=saved_energy_samples,
                     baseline_energy=baseline_energy,
                     repeats=repeats,
                     seed=Int(seed),
                     model=model_name(model),
                     row=row,
                     n_chains=length(Z_chains),
                     samples_per_chain=samples_per_chain,
                     source_file=samples_file)
    end

    return result
end
