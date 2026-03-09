# =============================================================================
# Observable Expectation Values
# =============================================================================
# Functions for computing expectation values from the transfer matrix fixed point

"""
    correlation_function(gates, row, virtual_qubits, observable, separations; 
                         position::Int=1, connected::Bool=false, optimizer=GreedyMethod())

Compute two-point correlation function ⟨O_i O_{i+r}⟩ using transfer matrix formalism.

The correlation is computed as:
    ⟨O_i O_{i+r}⟩ = l† · E_O · E^(r-1) · E_O · r / (l† · r)

where E is the transfer matrix, E_O has observable O inserted, and l, r are
the left and right fixed points.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `observable`: Observable operator, either `:X`, `:Z`, or a 2×2 matrix
- `separations`: Separations to compute (Int, range, or vector)
- `position`: Row position (1 to row) where to insert the observable (default: 1)
- `connected`: If true, return connected correlation ⟨O_i O_{i+r}⟩ - ⟨O⟩² (default: false)
- `optimizer`: Contraction optimizer (default: GreedyMethod())

# Returns
- `correlations`: Dictionary mapping separation r => correlation value

# Example
```julia
gates = build_unitary_gate(params, p, row, nqubits)
# Compute ⟨Z_i Z_{i+r}⟩ for r = 1 to 10
corr = correlation_function(gates, row, virtual_qubits, :Z, 1:10)
# Connected correlation
corr_c = correlation_function(gates, row, virtual_qubits, :Z, 1:10; connected=true)
```
"""
function correlation_function(gates, row, virtual_qubits, observable::Union{Symbol,AbstractMatrix}, 
                              separations; position::Int=1, connected::Bool=false, 
                              optimizer=GreedyMethod())
    # Get the operator matrix
    O = if observable isa Symbol
        observable == :X ? Matrix(X) : (observable == :Z ? Matrix(Z) : error("Unknown observable: $observable"))
    else
        observable
    end
    
    # Handle single separation
    seps = separations isa Integer ? [separations] : collect(separations)
    if isempty(seps)
        return Dict{Int, ComplexF64}()
    end
    
    # Get transfer matrix and E_O
    E = get_transfer_matrix(gates, row, virtual_qubits)
    E_O = get_transfer_matrix_with_operator(gates, row, virtual_qubits, O; position=position, optimizer=optimizer)
    
    # Compute right fixed point: E * r = λ * r
    # Using eigen(E') to get right eigenvectors of E
    eig_right = eigen(E)
    sorted_idx_r = sortperm(abs.(eig_right.values), rev=true)
    r_vec = eig_right.vectors[:, sorted_idx_r[1]]  # Dominant right eigenvector
    
    # Compute left fixed point: l† * E = λ * l†
    # Equivalently: E' * l = λ* * l, so l is right eigenvector of E'
    eig_left = eigen(E')
    sorted_idx_l = sortperm(abs.(eig_left.values), rev=true)
    l_vec = eig_left.vectors[:, sorted_idx_l[1]]  # Dominant left eigenvector
    
    # Biorthogonal normalization factor
    norm_factor = dot(l_vec, r_vec)
    
    # Efficient computation for multiple separations
    # Sort separations to iterate in order
    sorted_seps = sort(seps)
    max_sep = maximum(sorted_seps)
    
    # Initialize: current = E_O * r (right boundary with second operator applied)
    current = E_O * r_vec
    
    # l† * E_O for left boundary with first operator
    l_E_O = E_O' * l_vec
    
    # Compute correlations iteratively
    correlations = Dict{Int, ComplexF64}()
    prev_sep = 1
    
    for sep in sorted_seps
        # Apply E^(sep - prev_sep) times to advance from previous separation
        for _ in 1:(sep - prev_sep)
            current = E * current
        end
        prev_sep = sep
        
        # Compute correlation: l† · E_O · E^(r-1) · E_O · r = (E_O' * l)† · current
        corr_value = dot(l_E_O, current) / norm_factor
        correlations[sep] = corr_value
    end
    
    # Subtract ⟨O⟩² for connected correlation
    if connected
        # Compute ⟨O⟩ = l† · E_O · r / (l† · r)  (separation r=0)
        O_expectation = dot(l_vec, E_O * r_vec) / norm_factor
        O_squared = O_expectation^2
        
        for sep in keys(correlations)
            correlations[sep] -= O_squared
        end
    end
    
    return correlations
end


"""
    expect(gates, row, virtual_qubits, observable; position=1, optimizer=GreedyMethod())
    expect(gates, row, virtual_qubits, operators::Dict; optimizer=GreedyMethod())

Compute expectation value ⟨O⟩ or ⟨O₁ O₂ ...⟩ using transfer matrix formalism.

The expectation is computed as:
    ⟨O⟩ = l† · E_O · r / (l† · r)

where E_O is the transfer matrix with observable(s) inserted, and l, r are
the left and right fixed points.

# Arguments
- `gates`: Vector of gate matrices
- `row`: Number of rows
- `virtual_qubits`: Number of virtual qubits per bond
- `observable`: Observable operator, either `:X`, `:Z`, or a 2×2 matrix (single operator case)
- `operators`: Dict mapping position (1 to row) => operator (multiple operators case)
- `position`: Row position (1 to row) where to insert the observable (default: 1)
- `optimizer`: Contraction optimizer (default: GreedyMethod())

# Returns
- `expectation`: Complex expectation value

# Example
```julia
gates = build_unitary_gate(params, p, row, nqubits)
# Single operator
Z_expect = expect(gates, row, virtual_qubits, :Z)
X_expect = expect(gates, row, virtual_qubits, :X; position=2)
# Two operators at positions 1 and 2 (vertical ZZ within column)
ZZ_vert = expect(gates, row, virtual_qubits, Dict(1 => :Z, 2 => :Z))
# All Z operators
Z_all = expect(gates, row, virtual_qubits, Dict(i => :Z for i in 1:row))
```
"""
function expect(gates, row, virtual_qubits, observable::Union{Symbol,AbstractMatrix}; 
                position::Int=1, optimizer=GreedyMethod())
    # Single operator case - convert to Dict and call the multi-operator version
    operators = Dict(position => observable)
    return expect(gates, row, virtual_qubits, operators; optimizer=optimizer)
end

function expect(gates, row, virtual_qubits, operators::Dict{Int,<:Union{Symbol,AbstractMatrix}}; 
                optimizer=GreedyMethod())
    # Convert symbols to matrices
    op_matrices = Dict{Int, Matrix{ComplexF64}}()
    for (pos, op) in operators
        O = if op isa Symbol
            op == :X ? Matrix(X) : (op == :Z ? Matrix(Z) : error("Unknown observable: $op"))
        else
            Matrix{ComplexF64}(op)
        end
        op_matrices[pos] = O
    end
    
    # Get transfer matrix and E_O with multiple operators
    E = get_transfer_matrix(gates, row, virtual_qubits)
    E_O = get_transfer_matrix_with_operator(gates, row, virtual_qubits, op_matrices; optimizer=optimizer)
    
    # Compute right fixed point: E * r = λ * r
    eig_right = eigen(E)
    sorted_idx_r = sortperm(abs.(eig_right.values), rev=true)
    r_vec = eig_right.vectors[:, sorted_idx_r[1]]  # Dominant right eigenvector
    
    # Compute left fixed point: l† * E = λ * l†
    # Equivalently: E' * l = λ* * l, so l is right eigenvector of E'
    eig_left = eigen(E')
    sorted_idx_l = sortperm(abs.(eig_left.values), rev=true)
    l_vec = eig_left.vectors[:, sorted_idx_l[1]]  # Dominant left eigenvector
    
    # Biorthogonal normalization factor
    norm_factor = dot(l_vec, r_vec)
    
    # Compute ⟨O₁ O₂ ...⟩ = l† · E_O · r / (l† · r)
    O_expectation = dot(l_vec, E_O * r_vec) / norm_factor
    
    return O_expectation
end

"""
    compute_acf(data; max_lag::Int=100, n_bootstrap::Int=100, normalize::Bool=true)

Compute autocorrelation function with error estimates.

Accepts either:
- `Matrix{Float64}`: Each row is an independent chain (preferred for parallel sampling)
- `Vector{Float64}`: Single time series (uses bootstrap for errors)

When given a matrix (multiple chains), uses ALL pairs from ALL chains to compute ACF.
For example, with 2 chains of length 5, lag=1 uses pairs:
  chain1: (1,2), (2,3), (3,4), (4,5)
  chain2: (1,2), (2,3), (3,4), (4,5)
Total: 8 pairs for better statistics.

Error bars from standard error across chains (each chain contributes one ACF estimate).

# Arguments
- `data`: Time series data (Vector or Matrix)
- `max_lag`: Maximum lag to compute (default: 100)
- `row`: Subsample step - take every `row`-th sample from each chain (default: 1, no subsampling)
- `n_bootstrap`: Number of bootstrap samples for error estimation (only used for Vector input)
- `normalize`: Whether to normalize by variance (default: true)

# Returns
- `lags`: Lag values (0 to max_lag-1)
- `acf`: Normalized autocorrelation at each lag (connected correlation / variance)
- `acf_err`: Standard error of normalized ACF
- `corr`: Full correlation ⟨X_i X_{i+r}⟩ at each lag
- `corr_err`: Standard error of full correlation
- `corr_connected`: Connected correlation ⟨X_i X_{i+r}⟩ - ⟨X⟩² at each lag
- `corr_connected_err`: Standard error of connected correlation
"""
function compute_acf(data::Matrix{Float64}; max_lag::Int=100, row::Int=1, n_bootstrap::Int=100, normalize::Bool=true)
    n_chains, n_samples_raw = size(data)

    # Subsample: take every row-th sample from each chain, averaging over all starting offsets
    # This makes lag in subsampled data = horizontal separation in PEPS
    n_samples = div(n_samples_raw, row)

    # Limit max_lag to avoid unreliable estimates at large lags
    # The ACF estimator has significant negative bias when lag > n_samples/10
    # Standard practice in time series analysis is to limit to N/10
    max_lag_limit = div(n_samples, 10)
    if max_lag > max_lag_limit
        @warn "Requested max_lag=$max_lag is too large for n_samples=$n_samples (after subsampling with row=$row). Using max_lag=$max_lag_limit (n_samples/10) to avoid biased estimates."
        max_lag = max_lag_limit
    end

    # We'll average over all starting offsets (1 to row) for each chain
    # Total number of subsampled chains: n_chains * row
    n_total_subchains = n_chains * row

    acf_per_subchain = zeros(n_total_subchains, max_lag)           # Normalized ACF
    corr_per_subchain = zeros(n_total_subchains, max_lag)          # Full correlation ⟨X_i X_{i+r}⟩
    corr_connected_per_subchain = zeros(n_total_subchains, max_lag) # Connected correlation

    # Standard errors within each subchain (from variance of products)
    corr_stderr_per_subchain = zeros(n_total_subchains, max_lag)
    corr_connected_stderr_per_subchain = zeros(n_total_subchains, max_lag)

    # Compute correlations for each chain and each starting offset
    subchain_idx = 1
    for i in 1:n_chains
        for offset in 1:row
            # Subsample: take every row-th sample starting from offset
            chain = data[i, offset:row:end]
            n_chain = length(chain)
            μ_chain = mean(chain)
            chain_centered = chain .- μ_chain
            var_chain = mean(chain_centered.^2)

            for k in 1:max_lag
                lag = k - 1
                n_pairs = n_chain - lag
                if n_pairs < 1
                    continue
                end

                # Full correlation: ⟨X_i X_{i+r}⟩
                products_full = [chain[j] * chain[j + lag] for j in 1:n_pairs]
                full_corr = mean(products_full)
                corr_per_subchain[subchain_idx, k] = full_corr
                # Standard error: std(products) / sqrt(n_pairs)
                corr_stderr_per_subchain[subchain_idx, k] = std(products_full) / sqrt(n_pairs)

                # Connected correlation: ⟨X_i X_{i+r}⟩ - ⟨X⟩² = ⟨(X_i - μ)(X_{i+r} - μ)⟩
                products_connected = [chain_centered[j] * chain_centered[j + lag] for j in 1:n_pairs]
                connected_corr = mean(products_connected)
                corr_connected_per_subchain[subchain_idx, k] = connected_corr
                # Standard error: std(products) / sqrt(n_pairs)
                corr_connected_stderr_per_subchain[subchain_idx, k] = std(products_connected) / sqrt(n_pairs)

                # Normalized ACF: connected / variance
                acf_per_subchain[subchain_idx, k] = connected_corr / var_chain
            end

            subchain_idx += 1
        end
    end

    # Average across all subchains (chains × offsets)
    acf = vec(mean(acf_per_subchain, dims=1))
    corr = vec(mean(corr_per_subchain, dims=1))
    corr_connected = vec(mean(corr_connected_per_subchain, dims=1))

    # SE for ACF: between-subchain std / sqrt(n), valid regardless of chain count
    acf_err = vec(std(acf_per_subchain, dims=1) / sqrt(n_total_subchains))

    # SE for corr / corr_connected: use pooled within-subchain SE.
    #
    # The between-subchain SE (std across n_total_subchains estimates) is only
    # reliable when n_total_subchains is large.  In the common case of a single
    # input chain (n_chains=1), n_total_subchains = row (typically 1–5), giving
    # only ~2 degrees of freedom — the estimate is too noisy to decrease with
    # more samples and will dominate a max().
    #
    # The pooled within-subchain SE pools all products from every subchain and
    # uses std(products) / sqrt(total_products), which scales as 1/sqrt(samples)
    # and is stable even for small n_total_subchains.
    #
    # Note: this assumes approximate independence of consecutive products within
    # a subchain.  For a well-mixing quantum channel this is acceptable.
    corr_err = vec(mean(corr_stderr_per_subchain, dims=1)) / sqrt(n_total_subchains)
    corr_connected_err = vec(mean(corr_connected_stderr_per_subchain, dims=1)) / sqrt(n_total_subchains)

    return 0:(max_lag-1), acf, acf_err, corr, corr_err, corr_connected, corr_connected_err
end

# Vector method: convert to single-row matrix
function compute_acf(data::Vector{Float64}; max_lag::Int=100, row::Int=1, n_bootstrap::Int=100, normalize::Bool=true)
    return compute_acf(reshape(data, 1, :); max_lag=max_lag, row=row, n_bootstrap=n_bootstrap, normalize=normalize)
end