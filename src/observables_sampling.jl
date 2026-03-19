# =============================================================================
# Sampling-Based Observable Calculations
# =============================================================================
# Functions that compute expectation values and energies from measurement samples
# (as opposed to exact tensor network contraction in observables_exact.jl)

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

"""
    compute_tfim_energy(X_samples, Z_samples, g, J, row)

Compute TFIM energy from measurement samples.

# Arguments
- `X_samples`: Vector of X measurement outcomes
- `Z_samples`: Vector of Z measurement outcomes
- `g`: Transverse field strength
- `J`: Coupling strength (default: 1.0)
- `row`: Number of rows

# Returns
- Energy estimate: E = -g⟨X⟩ - J⟨ZZ⟩

# Description
Computes the transverse field Ising model energy:
H = -g ∑ᵢ Xᵢ - J ∑⟨ij⟩ ZᵢZⱼ

Sample layout for row=4:
  Z[1]  Z[5]  Z[9]   ...   ← row 1
  Z[2]  Z[6]  Z[10]  ...   ← row 2
  Z[3]  Z[7]  Z[11]  ...   ← row 3
  Z[4]  Z[8]  Z[12]  ...   ← row 4
   ↑     ↑     ↑
  col1  col2  col3

- Vertical bonds: Z[i]*Z[i+1] only when i % row != 0 (not at last row of column)
- Horizontal bonds: Z[i]*Z[i+row] (same row, adjacent columns)
"""
function compute_tfim_energy(X_samples, Z_samples, g, J, row)
    X_mean = mean(X_samples)
    N = length(Z_samples)

    if row == 1
        # Row=1: only horizontal bonds (no vertical neighbors)
        ZZ_horiz = mean(Z_samples[i] * Z_samples[i+1] for i in 1:N-1)
        ZZ_mean = ZZ_horiz
    else
        # Vertical bonds: Z[i]*Z[i+1] only when NOT at the last row of a column
        # Skip when i % row == 0 (e.g., Z[4]*Z[5] would be diagonal, not vertical)
        ZZ_vert_pairs = [Z_samples[i] * Z_samples[i+1]
                         for i in 1:N-1 if i % row != 0]
        ZZ_vert = mean(ZZ_vert_pairs)

        # Horizontal bonds: Z[i]*Z[i+row] (same row, adjacent columns)
        ZZ_horiz = mean(Z_samples[i] * Z_samples[i+row] for i in 1:N-row)
        # Both contribute to energy
        ZZ_mean = ZZ_vert + ZZ_horiz
    end

    return -g * X_mean - J * ZZ_mean
end

"""
    compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)

Compute Heisenberg J1-J2 energy from X, Y, Z measurement samples.

    S_i · S_j = (X_i X_j + Y_i Y_j + Z_i Z_j) / 4

# Arguments
- `X_samples`: Vector of X measurement outcomes
- `Z_samples`: Vector of Z measurement outcomes
- `Y_samples`: Vector of Y measurement outcomes
- `J1`: Nearest-neighbor coupling
- `J2`: Next-nearest-neighbor (diagonal) coupling
- `row`: Number of rows

# Returns
- Energy estimate per column
"""
function compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)
    all_samples = (Z_samples, X_samples, Y_samples)

    # Helper: compute vertical and horizontal correlations for one set of samples
    function _correlations(S, row)
        N = length(S)
        if row == 1
            vert = 0.0
            horiz = mean(S[i] * S[i+1] for i in 1:N-1)
        else
            vert_pairs = [S[i] * S[i+1] for i in 1:N-1 if i % row != 0]
            vert = mean(vert_pairs)
            horiz = mean(S[i] * S[i+row] for i in 1:N-row)
        end
        return vert, horiz
    end

    # Sum XX + YY + ZZ for NN bonds
    SS_vert = 0.0
    SS_horiz = 0.0
    for S in all_samples
        v, h = _correlations(S, row)
        SS_vert += v
        SS_horiz += h
    end

    energy = J1 * (row == 1 ? SS_horiz : SS_vert + SS_horiz) / 4.0

    # J2: diagonal NNN bonds
    if J2 != 0.0 && row > 1
        SS_diag = 0.0
        for S in all_samples
            N = length(S)
            # Diagonal: (pos,col)->(pos+1,col+1)
            diag1 = [S[i] * S[i+row+1] for i in 1:N-row-1 if i % row != 0]
            # Anti-diagonal: (pos,col)->(pos-1,col+1)
            diag2 = [S[i] * S[i+row-1] for i in 1:N-row+1 if (i-1) % row != 0]
            SS_diag += mean(diag1) + mean(diag2)
        end
        energy += J2 * SS_diag / 4.0
    end
    return energy
end
