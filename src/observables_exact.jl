# =============================================================================
# Exact Observable Expectation Values (Transfer Matrix Contraction)
# =============================================================================
# Unified API for computing expectation values, correlations, and energies
# using the TransferOperator (N-column unit cell) with backward-compatible
# wrappers for the legacy 1×1 and 2×2 interfaces.

# =============================================================================
# Section 1: Internal Helpers (private)
# =============================================================================

"""Convert a symbol (`:X`, `:Y`, `:Z`) or matrix to `Matrix{ComplexF64}`."""
_resolve_op(obs::Symbol) = if obs == :X
    Matrix{ComplexF64}(Matrix(X))
elseif obs == :Y
    ComplexF64[0 -im; im 0]
elseif obs == :Z
    Matrix{ComplexF64}(Matrix(Z))
else
    error("Unknown observable: $obs")
end
_resolve_op(obs::AbstractMatrix) = Matrix{ComplexF64}(obs)

"""
    _fixed_points(T) → (l_vec, r_vec, nf, λ)

Dominant left/right eigenvectors of `T` with biorthogonal norm `nf = l†r`
and dominant eigenvalue `λ`.
"""
function _fixed_points(T::AbstractMatrix)
    eig_r = eigen(T)
    idx_r = sortperm(abs.(eig_r.values), rev=true)
    r_vec = eig_r.vectors[:, idx_r[1]]
    λ     = eig_r.values[idx_r[1]]

    eig_l = eigen(T')
    idx_l = sortperm(abs.(eig_l.values), rev=true)
    l_vec = eig_l.vectors[:, idx_l[1]]

    nf = dot(l_vec, r_vec)
    return l_vec, r_vec, nf, λ
end

"""Per-column transfer matrices (transposed convention, consistent with `get_transfer_matrix`)."""
_column_transfer_matrices(op::TransferOperator) =
    [get_transfer_matrix(g, op.row, op.virtual_qubits) for g in op.columns]

"""
    _precompute_shifted_vectors(T_cols, l_vec, r_vec)

Return `(l_pre, r_suf)` where
  `l_pre[c] = (T₁⋯T_{c-1})† l`  and  `r_suf[c] = T_{c+1}⋯T_N r`.
"""
function _precompute_shifted_vectors(T_cols, l_vec, r_vec)
    N = length(T_cols)
    l_pre = Vector{Vector{ComplexF64}}(undef, N)
    r_suf = Vector{Vector{ComplexF64}}(undef, N)

    l_pre[1] = l_vec
    for c in 2:N
        l_pre[c] = T_cols[c-1]' * l_pre[c-1]
    end

    r_suf[N] = r_vec
    for c in (N-1):-1:1
        r_suf[c] = T_cols[c+1] * r_suf[c+1]
    end

    return l_pre, r_suf
end

# =============================================================================
# Section 2: Core expect API
# =============================================================================

"""
    expect(op::TransferOperator, obs; col=1, position=1, optimizer=GreedyMethod())

Single-site expectation ⟨O_{col,pos}⟩ for an N-column unit cell.

`obs` can be `:X`, `:Y`, `:Z`, or a 2×2 matrix.
"""
function expect(op::TransferOperator, obs;
                col::Int=1, position::Int=1, optimizer=GreedyMethod())
    O = _resolve_op(obs)
    T_combined = reduce(*, _column_transfer_matrices(op))
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)
    T̃ = get_transfer_matrix_with_operator(op, Dict((col, position) => O);
                                          optimizer=optimizer)
    return dot(l_vec, T̃ * r_vec) / nf
end

"""
    expect(op::TransferOperator, sites::Dict{Tuple{Int,Int}}; optimizer=GreedyMethod())

Multi-site expectation ⟨O₁ O₂ ⋯⟩ at arbitrary `(col, position)` sites
within one period.
"""
function expect(op::TransferOperator, sites::Dict{Tuple{Int,Int},<:Any};
                optimizer=GreedyMethod())
    ops = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}(
        k => _resolve_op(v) for (k, v) in sites)
    T_combined = reduce(*, _column_transfer_matrices(op))
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)
    T̃ = get_transfer_matrix_with_operator(op, ops; optimizer=optimizer)
    return dot(l_vec, T̃ * r_vec) / nf
end

# --- Backward-compat wrappers (1×1 UC) ---

"""
    expect(gates, row, virtual_qubits, observable; position=1, optimizer=GreedyMethod())

Single-operator expectation for a 1×1 unit cell (legacy interface).
"""
function expect(gates, row, virtual_qubits, observable::Union{Symbol,AbstractMatrix};
                position::Int=1, optimizer=GreedyMethod())
    return expect(gates, row, virtual_qubits,
                  Dict(position => observable); optimizer=optimizer)
end

"""
    expect(gates, row, virtual_qubits, operators::Dict{Int}; optimizer=GreedyMethod())

Multi-operator expectation within a single column (legacy interface).
"""
function expect(gates, row, virtual_qubits,
                operators::Dict{Int,<:Union{Symbol,AbstractMatrix}};
                optimizer=GreedyMethod())
    sites = Dict{Tuple{Int,Int}, Any}((1, pos) => op for (pos, op) in operators)
    return expect(TransferOperator([gates], row, virtual_qubits), sites;
                  optimizer=optimizer)
end

# =============================================================================
# Section 3: Correlation Functions
# =============================================================================

"""
    correlation_function(op::TransferOperator, observable, separations;
                         col=1, position=1, connected=false, optimizer=GreedyMethod())

Two-point correlation ⟨O(col,pos,period 0) O(col,pos,period r)⟩ for an
N-column unit cell.  Separations `r` are in units of full unit-cell periods.

    ⟨O₀ O_r⟩ = l† T̃ T^{r-1} T̃ r / (l†r)

where T̃ is the per-period transfer matrix with the observable inserted at
`(col, position)`, and T = T_combined.
"""
function correlation_function(op::TransferOperator,
                              observable::Union{Symbol,AbstractMatrix},
                              separations;
                              col::Int=1, position::Int=1,
                              connected::Bool=false,
                              optimizer=GreedyMethod())
    O    = _resolve_op(observable)
    seps = separations isa Integer ? [separations] : collect(separations)
    isempty(seps) && return Dict{Int, ComplexF64}()

    T_combined = reduce(*, _column_transfer_matrices(op))
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)
    E_O = get_transfer_matrix_with_operator(
        op, Dict((col, position) => O); optimizer=optimizer)

    sorted_seps = sort(seps)
    current = E_O * r_vec
    l_E_O   = E_O' * l_vec

    correlations = Dict{Int, ComplexF64}()
    prev_sep = 1
    for sep in sorted_seps
        for _ in 1:(sep - prev_sep)
            current = T_combined * current
        end
        prev_sep = sep
        correlations[sep] = dot(l_E_O, current) / nf
    end

    if connected
        O_sq = (dot(l_vec, E_O * r_vec) / nf)^2
        for k in keys(correlations)
            correlations[k] -= O_sq
        end
    end
    return correlations
end

"""
    correlation_function(gates, row, virtual_qubits, observable, separations;
                         position=1, connected=false, optimizer=GreedyMethod())

Two-point correlation for a 1×1 unit cell (legacy interface).
Separations are in units of single columns.
"""
function correlation_function(gates, row, virtual_qubits,
                              observable::Union{Symbol,AbstractMatrix},
                              separations;
                              position::Int=1, connected::Bool=false,
                              optimizer=GreedyMethod())
    op = TransferOperator([gates], row, virtual_qubits)
    return correlation_function(op, observable, separations;
                                col=1, position=position,
                                connected=connected, optimizer=optimizer)
end

# =============================================================================
# Section 4: TFIM Energy
# =============================================================================

"""
    compute_exact_energy(m::TFIM, op::TransferOperator; optimizer=GreedyMethod())

Exact TFIM energy per column for any N-column unit cell.

Returns `(energy, X_total, ZZ_vert, ZZ_horiz)` — all quantities are
per-column averages.
"""
function compute_exact_energy(m::TFIM, op::TransferOperator;
                              optimizer=GreedyMethod())
    N   = length(op.columns)
    row = op.row
    vq  = op.virtual_qubits
    σx  = _resolve_op(:X)
    σz  = _resolve_op(:Z)

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)
    l_pre, r_suf = _precompute_shifted_vectors(T_cols, l_vec, r_vec)

    # Precompute single-operator transfer matrices
    E_O_X = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    E_O_Z = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    for c in 1:N, pos in 1:row
        E_O_X[(c, pos)] = get_transfer_matrix_with_operator(
            op.columns[c], row, vq, σx; position=pos, optimizer=optimizer)
        E_O_Z[(c, pos)] = get_transfer_matrix_with_operator(
            op.columns[c], row, vq, σz; position=pos, optimizer=optimizer)
    end

    # ⟨X⟩ summed over all sites
    X_total = 0.0
    for c in 1:N, pos in 1:row
        X_total += real(dot(l_pre[c], E_O_X[(c, pos)] * r_suf[c]) / nf)
    end

    # Vertical ZZ (within each column, periodic boundary)
    ZZ_vert = 0.0
    if row > 1
        for c in 1:N, i in 1:row
            j = i % row + 1
            E_OO = get_transfer_matrix_with_operator(
                op.columns[c], row, vq, Dict(i => σz, j => σz);
                optimizer=optimizer)
            ZZ_vert += real(dot(l_pre[c], E_OO * r_suf[c]) / nf)
        end
    end

    # Horizontal ZZ — within-period (c → c+1, c < N)
    ZZ_horiz = 0.0
    for c in 1:(N-1), pos in 1:row
        val = dot(l_pre[c],
                  E_O_Z[(c, pos)] * E_O_Z[(c+1, pos)] * r_suf[c+1]) / nf
        ZZ_horiz += real(val)
    end

    # Horizontal ZZ — cross-period (N → 1′)
    l_shifted = l_pre[N]
    r_shifted = r_suf[1]
    for pos in 1:row
        val = dot(l_shifted,
                  E_O_Z[(N, pos)] * E_O_Z[(1, pos)] * r_shifted) / nf
        ZZ_horiz += real(val)
    end

    # Per-column averages
    X_total  /= N
    ZZ_vert  /= N
    ZZ_horiz /= N

    energy = -m.g * X_total - m.J * (row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
    return energy, X_total, ZZ_vert, ZZ_horiz
end

# --- Legacy wrappers ---

"""
    compute_exact_energy(gates, row, virtual_qubits, J, g; optimizer)

TFIM energy per column for a 1×1 unit cell (legacy interface).
"""
function compute_exact_energy(gates, row, virtual_qubits, J, g;
                              optimizer=GreedyMethod())
    X_total = compute_X_expectation(nothing, gates, row, virtual_qubits;
                                    optimizer=optimizer)
    ZZ_vert, ZZ_horiz = compute_ZZ_expectation(nothing, gates, row,
                                                virtual_qubits;
                                                optimizer=optimizer)
    return -g * X_total - J * (row == 1 ? ZZ_horiz : ZZ_vert + ZZ_horiz)
end

function _compute_pauli_total(gates, row, virtual_qubits, obs::Symbol; optimizer=GreedyMethod())
    total = 0.0
    for pos in 1:row
        total += real(expect(gates, row, virtual_qubits, obs;
                             position=pos, optimizer=optimizer))
    end
    return total
end

"""
    compute_X_expectation(rho, gates, row, virtual_qubits; optimizer)

Total ⟨X⟩ = Σ_pos ⟨X_pos⟩ summed over all row positions.
`rho` is accepted for API compatibility but unused.
"""
compute_X_expectation(rho, gates, row, virtual_qubits;
                      optimizer=GreedyMethod()) =
    _compute_pauli_total(gates, row, virtual_qubits, :X; optimizer=optimizer)

"""
    compute_Z_expectation(rho, gates, row, virtual_qubits; optimizer)

Total ⟨Z⟩ = Σ_pos ⟨Z_pos⟩ summed over all row positions.
`rho` is accepted for API compatibility but unused.
"""
compute_Z_expectation(rho, gates, row, virtual_qubits;
                      optimizer=GreedyMethod()) =
    _compute_pauli_total(gates, row, virtual_qubits, :Z; optimizer=optimizer)

"""
    compute_single_expectation(rho, gates, row, virtual_qubits, observable;
                                position=1, optimizer)

Single-site expectation ⟨O_pos⟩.
`rho` is accepted for API compatibility but unused.
"""
compute_single_expectation(rho, gates, row, virtual_qubits, observable;
                           position::Int=1, optimizer=GreedyMethod()) =
    expect(gates, row, virtual_qubits, observable;
           position=position, optimizer=optimizer)

"""
    compute_ZZ_expectation(rho, gates, row, virtual_qubits; optimizer)

Vertical and horizontal ZZ correlations for a single column.

Returns `(ZZ_vert, ZZ_horiz)`:
- `ZZ_vert`:  Σ ⟨Z_i Z_{i+1}⟩ over vertical bonds (periodic boundary)
- `ZZ_horiz`: Σ ⟨Z_pos(col) Z_pos(col+1)⟩ over horizontal bonds
"""
function compute_ZZ_expectation(rho, gates, row, virtual_qubits;
                                optimizer=GreedyMethod())
    ZZ_vert = 0.0
    if row > 1
        for i in 1:(row-1)
            ZZ_vert += real(expect(gates, row, virtual_qubits,
                                  Dict(i => :Z, i+1 => :Z);
                                  optimizer=optimizer))
        end
        ZZ_vert += real(expect(gates, row, virtual_qubits,
                               Dict(row => :Z, 1 => :Z);
                               optimizer=optimizer))
    end

    ZZ_horiz = 0.0
    for pos in 1:row
        corr = correlation_function(gates, row, virtual_qubits, :Z, [1];
                                    position=pos, optimizer=optimizer)
        ZZ_horiz += real(corr[1])
    end
    return ZZ_vert, ZZ_horiz
end

# =============================================================================
# Section 5: Heisenberg J1-J2 Energy
# =============================================================================

"""
    compute_exact_heisenberg_energy(op::TransferOperator, J1, J2;
                                     optimizer=GreedyMethod())

Heisenberg J1-J2 energy per column for any N-column unit cell.

    H = J1 Σ_{⟨i,j⟩} S_i·S_j  +  J2 Σ_{⟨⟨i,j⟩⟩} S_i·S_j

with S_i·S_j = (σx σx + σy σy + σz σz)/4.

Bonds:
- J1 vertical:   (pos, col)-(pos±1, col)   within column (periodic)
- J1 horizontal:  (pos, col)-(pos, col+1)   between adjacent columns
- J2 diagonal:    (pos, col)-(pos±1, col+1) between adjacent columns

Returns energy per column (total energy / N_columns).
"""
function compute_exact_heisenberg_energy(op::TransferOperator,
                                          J1::Float64, J2::Float64;
                                          optimizer=GreedyMethod())
    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    N   = length(op.columns)
    row = op.row
    vq  = op.virtual_qubits

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)
    l_pre, r_suf = _precompute_shifted_vectors(T_cols, l_vec, r_vec)

    # Precompute single-operator E_O for each (column, pauli_idx, position)
    E_O = Dict{Tuple{Int,Int,Int}, Matrix{ComplexF64}}()
    for c in 1:N, (si, σ) in enumerate(paulis), pos in 1:row
        E_O[(c, si, pos)] = get_transfer_matrix_with_operator(
            op.columns[c], row, vq, σ; position=pos, optimizer=optimizer)
    end

    energy = 0.0

    # === J1: Vertical bonds (within column, periodic) ===
    if row > 1
        for c in 1:N, i in 1:row
            j = i % row + 1
            for σ in paulis
                E_OO = get_transfer_matrix_with_operator(
                    op.columns[c], row, vq, Dict(i => σ, j => σ);
                    optimizer=optimizer)
                val = dot(l_pre[c], E_OO * r_suf[c]) / nf
                energy += J1 * real(val) / 4.0
            end
        end
    end

    # === J1: Horizontal bonds (same position, adjacent columns) ===
    # Within-period (c → c+1, c < N)
    for c in 1:(N-1), (si, _) in enumerate(paulis), pos in 1:row
        val = dot(l_pre[c],
                  E_O[(c, si, pos)] * E_O[(c+1, si, pos)] * r_suf[c+1]) / nf
        energy += J1 * real(val) / 4.0
    end

    # Cross-period (N → 1′)
    l_shifted = l_pre[N]
    r_shifted = r_suf[1]
    for (si, _) in enumerate(paulis), pos in 1:row
        val = dot(l_shifted,
                  E_O[(N, si, pos)] * E_O[(1, si, pos)] * r_shifted) / nf
        energy += J1 * real(val) / 4.0
    end

    # === J2: Diagonal NNN bonds ===
    if J2 != 0.0
        for i in 1:row
            j_up   = i % row + 1
            j_down = (i - 2 + row) % row + 1

            for (si, _) in enumerate(paulis)
                # Within-period (c → c+1)
                for c in 1:(N-1), j in (j_up, j_down)
                    val = dot(l_pre[c],
                              E_O[(c, si, i)] * E_O[(c+1, si, j)] *
                              r_suf[c+1]) / nf
                    energy += J2 * real(val) / 4.0
                end

                # Cross-period (N → 1′)
                for j in (j_up, j_down)
                    val = dot(l_shifted,
                              E_O[(N, si, i)] * E_O[(1, si, j)] *
                              r_shifted) / nf
                    energy += J2 * real(val) / 4.0
                end
            end
        end
    end

    return energy / N
end

# --- Legacy wrappers ---

"""
    compute_exact_heisenberg_energy(gates, row, virtual_qubits, J1, J2; optimizer)

Heisenberg J1-J2 energy per column for a 1×1 unit cell (legacy interface).
"""
function compute_exact_heisenberg_energy(gates, row, virtual_qubits,
                                          J1::Float64, J2::Float64;
                                          optimizer=GreedyMethod())
    op = TransferOperator([gates], row, virtual_qubits)
    return compute_exact_heisenberg_energy(op, J1, J2; optimizer=optimizer)
end

"""
    compute_exact_heisenberg_energy_2x2(gates_odd, gates_even, row,
                                         virtual_qubits, J1, J2; optimizer)

Heisenberg J1-J2 energy per column for a 2×2 unit cell (legacy interface).
"""
function compute_exact_heisenberg_energy_2x2(gates_odd, gates_even, row,
                                              virtual_qubits,
                                              J1::Float64, J2::Float64;
                                              optimizer=GreedyMethod())
    op = TransferOperator([gates_odd, gates_even], row, virtual_qubits)
    return compute_exact_heisenberg_energy(op, J1, J2; optimizer=optimizer)
end

# =============================================================================
# Section 6: Inter-column Correlation (unchanged)
# =============================================================================

"""
    intercolumn_correlation(gates, row, virtual_qubits, O1, pos1, O2, pos2;
                            l_vec=nothing, r_vec=nothing, norm_factor=nothing,
                            E_O_cache=nothing, optimizer=GreedyMethod())

Compute ⟨O1_{pos1}(col) O2_{pos2}(col+1)⟩ between adjacent columns.

Optionally pass precomputed `l_vec`, `r_vec`, `norm_factor`, and `E_O_cache`
(Dict mapping (operator, position) => E_O matrix) to avoid redundant work.
"""
function intercolumn_correlation(gates, row, virtual_qubits,
                                  O1::AbstractMatrix, pos1::Int,
                                  O2::AbstractMatrix, pos2::Int;
                                  l_vec=nothing, r_vec=nothing,
                                  norm_factor=nothing,
                                  E_O_cache=nothing,
                                  optimizer=GreedyMethod())
    if E_O_cache !== nothing &&
       haskey(E_O_cache, (O1, pos1)) && haskey(E_O_cache, (O2, pos2))
        E_O1 = E_O_cache[(O1, pos1)]
        E_O2 = E_O_cache[(O2, pos2)]
    else
        E_O1 = get_transfer_matrix_with_operator(
            gates, row, virtual_qubits, O1; position=pos1,
            optimizer=optimizer)
        E_O2 = get_transfer_matrix_with_operator(
            gates, row, virtual_qubits, O2; position=pos2,
            optimizer=optimizer)
    end

    if l_vec === nothing || r_vec === nothing || norm_factor === nothing
        E = get_transfer_matrix(gates, row, virtual_qubits)
        l_vec, r_vec, norm_factor, _ = _fixed_points(E)
    end

    return dot(l_vec, E_O1 * E_O2 * r_vec) / norm_factor
end
