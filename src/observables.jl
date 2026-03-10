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

# =============================================================================
# Observable Expectation Values (legacy interface using transfer matrix)
# =============================================================================
# These functions use the `expect` and `correlation_function` infrastructure
# internally. The `rho` argument is accepted for API compatibility but unused.

"""
    compute_X_expectation(rho, gates, row, virtual_qubits)

Compute total ⟨X⟩ = Σ_pos ⟨X_pos⟩ summed over all row positions.
"""
function compute_X_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    total = 0.0
    for pos in 1:row
        total += real(expect(gates, row, virtual_qubits, :X; position=pos, optimizer=optimizer))
    end
    return total
end

"""
    compute_Z_expectation(rho, gates, row, virtual_qubits)

Compute total ⟨Z⟩ = Σ_pos ⟨Z_pos⟩ summed over all row positions.
"""
function compute_Z_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    total = 0.0
    for pos in 1:row
        total += real(expect(gates, row, virtual_qubits, :Z; position=pos, optimizer=optimizer))
    end
    return total
end

"""
    compute_single_expectation(rho, gates, row, virtual_qubits, observable; position=1)

Compute single-site expectation ⟨O_pos⟩.
"""
function compute_single_expectation(rho, gates, row, virtual_qubits, observable;
                                     position::Int=1, optimizer=GreedyMethod())
    return expect(gates, row, virtual_qubits, observable; position=position, optimizer=optimizer)
end

"""
    compute_ZZ_expectation(rho, gates, row, virtual_qubits)

Compute vertical and horizontal ZZ correlations.

Returns `(ZZ_vert, ZZ_horiz)`:
- `ZZ_vert`: Σ ⟨Z_i Z_{i+1}⟩ over vertical bonds in one column (periodic boundary)
- `ZZ_horiz`: Σ ⟨Z_pos(col) Z_pos(col+1)⟩ over horizontal bonds between adjacent columns
"""
function compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer=GreedyMethod())
    # Vertical ZZ: sum over all vertical bonds (periodic)
    ZZ_vert = 0.0
    if row > 1
        for i in 1:(row-1)
            ZZ_vert += real(expect(gates, row, virtual_qubits, Dict(i => :Z, i+1 => :Z); optimizer=optimizer))
        end
        # Periodic boundary: bond between row and 1
        ZZ_vert += real(expect(gates, row, virtual_qubits, Dict(row => :Z, 1 => :Z); optimizer=optimizer))
    end

    # Horizontal ZZ: correlation at separation 1
    ZZ_horiz = 0.0
    for pos in 1:row
        corr = correlation_function(gates, row, virtual_qubits, :Z, [1]; position=pos, optimizer=optimizer)
        ZZ_horiz += real(corr[1])
    end

    return ZZ_vert, ZZ_horiz
end

"""
    compute_exact_energy(gates, row, virtual_qubits, J, g)

Compute exact TFIM energy per column: E = -g Σ⟨X⟩ - J Σ⟨ZZ⟩.
"""
function compute_exact_energy(gates, row, virtual_qubits, J, g; optimizer=GreedyMethod())
    X_total = compute_X_expectation(nothing, gates, row, virtual_qubits; optimizer=optimizer)
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(nothing, gates, row, virtual_qubits; optimizer=optimizer)
    return -g * X_total - J * (row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
end

# =============================================================================
# Inter-column Correlation (different row positions on adjacent columns)
# =============================================================================

"""
    intercolumn_correlation(gates, row, virtual_qubits, O1, pos1, O2, pos2;
                            l_vec=nothing, r_vec=nothing, norm_factor=nothing,
                            E_O_cache=nothing, optimizer=GreedyMethod())

Compute ⟨O1_{pos1}(col) O2_{pos2}(col+1)⟩ between adjacent columns.

Optionally pass precomputed `l_vec`, `r_vec`, `norm_factor`, and `E_O_cache`
(Dict mapping (operator_index, position) => E_O matrix) to avoid redundant work.
"""
function intercolumn_correlation(gates, row, virtual_qubits,
                                  O1::AbstractMatrix, pos1::Int,
                                  O2::AbstractMatrix, pos2::Int;
                                  l_vec=nothing, r_vec=nothing, norm_factor=nothing,
                                  E_O_cache=nothing, optimizer=GreedyMethod())
    # Get or compute E_O matrices
    if E_O_cache !== nothing && haskey(E_O_cache, (O1, pos1)) && haskey(E_O_cache, (O2, pos2))
        E_O1 = E_O_cache[(O1, pos1)]
        E_O2 = E_O_cache[(O2, pos2)]
    else
        E_O1 = get_transfer_matrix_with_operator(gates, row, virtual_qubits, O1; position=pos1, optimizer=optimizer)
        E_O2 = get_transfer_matrix_with_operator(gates, row, virtual_qubits, O2; position=pos2, optimizer=optimizer)
    end

    # Compute fixed points if not provided
    if l_vec === nothing || r_vec === nothing || norm_factor === nothing
        E = get_transfer_matrix(gates, row, virtual_qubits)
        eig_r = eigen(E)
        idx_r = sortperm(abs.(eig_r.values), rev=true)
        r_vec = eig_r.vectors[:, idx_r[1]]
        eig_l = eigen(E')
        idx_l = sortperm(abs.(eig_l.values), rev=true)
        l_vec = eig_l.vectors[:, idx_l[1]]
        norm_factor = dot(l_vec, r_vec)
    end

    # ⟨O1_{pos1}(col) O2_{pos2}(col+1)⟩ = l† · E_{O1} · E_{O2} · r / norm
    return dot(l_vec, E_O1 * E_O2 * r_vec) / norm_factor
end

# =============================================================================
# Heisenberg J1-J2 Energy (exact tensor contraction)
# =============================================================================

"""
    compute_exact_heisenberg_energy(gates, row, virtual_qubits, J1, J2; optimizer=GreedyMethod())

Compute Heisenberg J1-J2 energy per column using exact tensor contraction.

    H = J1 Σ_{⟨i,j⟩} S_i · S_j + J2 Σ_{⟨⟨i,j⟩⟩} S_i · S_j

where S_i · S_j = (X_i X_j + Y_i Y_j + Z_i Z_j) / 4 for spin-1/2.

Bonds:
- J1 vertical: (pos, col)-(pos+1, col) within column (periodic in vertical direction)
- J1 horizontal: (pos, col)-(pos, col+1) between adjacent columns
- J2 diagonal: (pos, col)-(pos±1, col+1) between adjacent columns
"""
function compute_exact_heisenberg_energy(gates, row, virtual_qubits, J1::Float64, J2::Float64;
                                          optimizer=GreedyMethod())
    σx = Matrix{ComplexF64}(Matrix(X))
    σy = ComplexF64[0 -im; im 0]
    σz = Matrix{ComplexF64}(Matrix(Z))
    paulis = [σx, σy, σz]

    # Precompute transfer matrix fixed points
    E = get_transfer_matrix(gates, row, virtual_qubits)
    eig_r = eigen(E)
    idx_r = sortperm(abs.(eig_r.values), rev=true)
    r_vec = eig_r.vectors[:, idx_r[1]]
    eig_l = eigen(E')
    idx_l = sortperm(abs.(eig_l.values), rev=true)
    l_vec = eig_l.vectors[:, idx_l[1]]
    nf = dot(l_vec, r_vec)

    # Precompute E_O for each (pauli, position)
    E_O_cache = Dict{Tuple{Matrix{ComplexF64},Int}, Matrix{ComplexF64}}()
    for σ in paulis
        for pos in 1:row
            E_O_cache[(σ, pos)] = get_transfer_matrix_with_operator(
                gates, row, virtual_qubits, σ; position=pos, optimizer=optimizer)
        end
    end

    energy = 0.0

    # --- J1: Nearest-neighbor vertical bonds (within column, periodic) ---
    if row > 1
        for i in 1:row
            j = i % row + 1  # next position with periodic wrap
            for σ in paulis
                E_OO = get_transfer_matrix_with_operator(
                    gates, row, virtual_qubits, Dict(i => σ, j => σ); optimizer=optimizer)
                val = dot(l_vec, E_OO * r_vec) / nf
                energy += J1 * real(val) / 4.0
            end
        end
    end

    # --- J1: Nearest-neighbor horizontal bonds (same position, adjacent columns) ---
    for σ in paulis
        for pos in 1:row
            E_O = E_O_cache[(σ, pos)]
            # ⟨σ_pos(col) σ_pos(col+1)⟩ = l† · E_O² · r / norm
            val = dot(l_vec, E_O * E_O * r_vec) / nf
            energy += J1 * real(val) / 4.0
        end
    end

    # --- J2: Next-nearest-neighbor diagonal bonds ---
    if J2 != 0.0
        for i in 1:row
            j_up = i % row + 1                    # pos+1 with wrap
            j_down = (i - 2 + row) % row + 1      # pos-1 with wrap

            for σ in paulis
                E_O_i = E_O_cache[(σ, i)]
                E_O_up = E_O_cache[(σ, j_up)]
                E_O_down = E_O_cache[(σ, j_down)]

                # Diagonal: (i, col) -> (i+1, col+1)
                val1 = dot(l_vec, E_O_i * E_O_up * r_vec) / nf
                energy += J2 * real(val1) / 4.0

                # Anti-diagonal: (i, col) -> (i-1, col+1)
                val2 = dot(l_vec, E_O_i * E_O_down * r_vec) / nf
                energy += J2 * real(val2) / 4.0
            end
        end
    end

    return energy
end