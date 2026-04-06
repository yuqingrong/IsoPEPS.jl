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
# Section 5b: Spin-Spin, Dimer-Dimer, Plaquette-Plaquette Correlations
# =============================================================================

"""
    spin_spin_correlation(op::TransferOperator, separations;
                          col1::Int=1, col2::Int=1,
                          pos1::Int=1, pos2::Int=1,
                          connected::Bool=false,
                          optimizer=GreedyMethod())

Exact spin-spin correlation ⟨S_{col1,pos1}(0) · S_{col2,pos2}(r)⟩ = Σ_α ⟨σ^α σ^α⟩/4
for an N-column unit cell. Separations are in units of full unit-cell periods.

# Arguments
- `op`: TransferOperator wrapping the unit cell
- `separations`: Period separations (integer or collection)
- `col1`, `col2`: Column indices (1:N_cols) within the unit cell for each operator
- `pos1`, `pos2`: Row positions of the two spins
- `connected`: If true, subtract ⟨S⟩·⟨S⟩

# Returns
- `Dict{Int, ComplexF64}` mapping separation → ⟨S·S⟩
"""
function spin_spin_correlation(op::TransferOperator, separations;
                               col1::Int=1, col2::Int=1,
                               pos1::Int=1, pos2::Int=1,
                               connected::Bool=false,
                               optimizer=GreedyMethod())
    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    seps = separations isa Integer ? [separations] : collect(separations)
    isempty(seps) && return Dict{Int, ComplexF64}()

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)

    # Build per-Pauli period transfer matrices with operator at (col, pos)
    T_σ1 = Vector{Matrix{ComplexF64}}(undef, 3)
    T_σ2 = Vector{Matrix{ComplexF64}}(undef, 3)
    for (si, σ) in enumerate(paulis)
        T_σ1[si] = get_transfer_matrix_with_operator(
            op, Dict((col1, pos1) => σ); optimizer=optimizer)
        T_σ2[si] = get_transfer_matrix_with_operator(
            op, Dict((col2, pos2) => σ); optimizer=optimizer)
    end

    correlations = Dict{Int, ComplexF64}()
    for sep in sort(seps)
        val = zero(ComplexF64)
        for si in 1:3
            if sep == 0 && col1 == col2 && pos1 == pos2
                # σ_a² = I → ⟨I⟩ = 1 (trace of identity in fixed-point basis)
                val += dot(l_vec, T_combined * r_vec) / nf
            elseif sep == 0
                # Same period: insert both operators in their respective columns
                T_both = get_transfer_matrix_with_operator(
                    op, Dict((col1, pos1) => paulis[si], (col2, pos2) => paulis[si]);
                    optimizer=optimizer)
                val += dot(l_vec, T_both * r_vec) / nf
            else
                # l† T̃_1 T^{sep-1} T̃_2 r / nf
                current = T_σ2[si] * r_vec
                for _ in 1:(sep - 1)
                    current = T_combined * current
                end
                val += dot(T_σ1[si]' * l_vec, current) / nf
            end
        end
        correlations[sep] = val / 4.0
    end

    if connected
        μ = zero(ComplexF64)
        for si in 1:3
            μ1 = dot(l_vec, T_σ1[si] * r_vec) / nf
            μ2 = dot(l_vec, T_σ2[si] * r_vec) / nf
            μ += μ1 * μ2
        end
        μ /= 4.0
        for k in keys(correlations)
            correlations[k] -= μ
        end
    end
    return correlations
end

"""
    dimer_dimer_correlation(op::TransferOperator, separations;
                            dimer_orientation::Symbol=:vertical,
                            pos::Int=1, connected::Bool=true,
                            optimizer=GreedyMethod())

Exact dimer-dimer correlation on a cylinder. The dimer operator is the bond energy
D_ij = S_i · S_j = Σ_α σ^α_i σ^α_j / 4.

For `:vertical` dimers: bond between (pos, col) and (pos', col) where pos' = pos%row+1.
For `:horizontal` dimers: bond between (pos, col) and (pos, col+1).

Separations are in units of full unit-cell periods.

# Returns
- `Dict{Int, ComplexF64}` mapping separation → dimer-dimer correlation
"""
function dimer_dimer_correlation(op::TransferOperator, separations;
                                 dimer_orientation::Symbol=:vertical,
                                 pos::Int=1,
                                 col::Int=1,
                                 connected::Bool=true,
                                 optimizer=GreedyMethod())
    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    seps = separations isa Integer ? [separations] : collect(separations)
    isempty(seps) && return Dict{Int, ComplexF64}()

    N   = length(op.columns)
    row = op.row
    vq  = op.virtual_qubits
    pos2 = pos % row + 1  # vertical partner (periodic)

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)

    if dimer_orientation == :vertical
        # Vertical dimer D = Σ_α σ^α_{pos} σ^α_{pos2} / 4 within column `col`
        T_D_col = zeros(ComplexF64, size(T_cols[1]))
        for σ in paulis
            E_OO = get_transfer_matrix_with_operator(
                op.columns[col], row, vq, Dict(pos => σ, pos2 => σ);
                optimizer=optimizer)
            T_D_col .+= E_OO
        end
        T_D_col ./= 4.0

        T_before = col > 1 ? reduce(*, T_cols[1:col-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_after  = col < N ? reduce(*, T_cols[col+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_D_period = T_before * T_D_col * T_after

    elseif dimer_orientation == :horizontal
        # Horizontal dimer spans 2 columns: D_h = Σ_α σ^α_{pos,col} σ^α_{pos,col+1} / 4
        col2 = col < N ? col + 1 : 1  # wrap within unit cell
        T_D_2col = zeros(ComplexF64, size(T_cols[1]))
        for (si, σ) in enumerate(paulis)
            E1 = get_transfer_matrix_with_operator(
                op.columns[col], row, vq, σ; position=pos, optimizer=optimizer)
            E2 = get_transfer_matrix_with_operator(
                op.columns[col2], row, vq, σ; position=pos, optimizer=optimizer)
            T_D_2col .+= E1 * E2
        end
        T_D_2col ./= 4.0

        T_before = col > 1 ? reduce(*, T_cols[1:col-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_after  = col2 < N ? reduce(*, T_cols[col2+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_D_period = T_before * T_D_2col * T_after
    else
        error("dimer_orientation must be :vertical or :horizontal, got $dimer_orientation")
    end

    # Compute correlations: l† T_D T^{r-1} T_D r / nf
    l_TD = T_D_period' * l_vec
    correlations = Dict{Int, ComplexF64}()
    for sep in sort(seps)
        if sep == 0 && dimer_orientation == :vertical
            # ⟨D²⟩ at same column: 4-operator insertion
            # D² = (Σ_α σ^α_{pos}σ^α_{pos2})² / 16
            # = Σ_{a,b} (σ^a σ^b)_{pos} · (σ^a σ^b)_{pos2} / 16
            val = zero(ComplexF64)
            for σa in paulis, σb in paulis
                O_pos = σa * σb
                O_pos2 = σa * σb
                E_OO = get_transfer_matrix_with_operator(
                    op.columns[col], row, vq, Dict(pos => O_pos, pos2 => O_pos2);
                    optimizer=optimizer)
                T_after_local = col < N ? reduce(*, T_cols[col+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                T_before_local = col > 1 ? reduce(*, T_cols[1:col-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
                val += dot(l_vec, T_before_local * E_OO * T_after_local * r_vec) / nf
            end
            correlations[sep] = val / 16.0
        else
            current = T_D_period * r_vec
            for _ in 1:max(sep - 1, 0)
                current = T_combined * current
            end
            correlations[sep] = dot(l_TD, current) / nf
        end
    end

    if connected
        μ_D = dot(l_vec, T_D_period * r_vec) / nf
        for k in keys(correlations)
            correlations[k] -= μ_D^2
        end
    end
    return correlations
end

"""
    bond_expectation(op::TransferOperator; orientation=:vertical, pos=1,
                     col=1, optimizer=GreedyMethod())

Compute the bond energy ⟨S_i · S_j⟩ = ⟨Σ_α σ^α_i σ^α_j⟩/4 for a single
nearest-neighbour bond within one unit-cell period.

# Arguments
- `orientation`: `:vertical` (same column, rows `pos` and `pos%row+1`) or
  `:horizontal` (same row `pos`, columns `col` and `col+1`)
- `pos`: Row position (1-based)
- `col`: Column within the unit cell (1-based, relevant for vertical bonds in 2×2 cells)

# Returns
- `Float64` bond energy
"""
function bond_expectation(op::TransferOperator;
                          orientation::Symbol=:vertical,
                          pos::Int=1, col::Int=1,
                          optimizer=GreedyMethod())
    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    row = op.row
    val = zero(ComplexF64)
    if orientation == :vertical
        pos2 = pos % row + 1
        for σ in paulis
            val += expect(op, Dict((col, pos) => σ, (col, pos2) => σ); optimizer=optimizer)
        end
    elseif orientation == :horizontal
        N = length(op.columns)
        col2 = col + 1
        col2 <= N || error("Horizontal bond at col=$col requires col+1=$col2 ≤ N=$N")
        for σ in paulis
            val += expect(op, Dict((col, pos) => σ, (col2, pos) => σ); optimizer=optimizer)
        end
    else
        error("orientation must be :vertical or :horizontal, got $orientation")
    end
    return real(val) / 4.0
end

"""
    all_bond_expectations(op::TransferOperator; optimizer=GreedyMethod())

Compute ⟨S_i · S_j⟩ for every distinct nearest-neighbour bond in the unit cell.

# Returns
- `(vert, horiz)` where:
  - `vert[pos, col]`: vertical bond energy at row `pos`, unit-cell column `col`
  - `horiz[pos, col]`: horizontal bond energy at row `pos`, between columns `col` and `col+1`
  - For a 1×1 unit cell (N=1): `vert` is `row × 1`, `horiz` is `row × 0`
  - For a 2×2 unit cell (N=2): `vert` is `row × 2`, `horiz` is `row × 1`
"""
function all_bond_expectations(op::TransferOperator; optimizer=GreedyMethod())
    row = op.row
    N = length(op.columns)
    vert = zeros(Float64, row, N)
    for c in 1:N, pos in 1:row
        vert[pos, c] = bond_expectation(op; orientation=:vertical, pos=pos, col=c,
                                        optimizer=optimizer)
    end
    horiz = zeros(Float64, row, max(N - 1, 0))
    for c in 1:(N - 1), pos in 1:row
        horiz[pos, c] = bond_expectation(op; orientation=:horizontal, pos=pos, col=c,
                                         optimizer=optimizer)
    end
    return (vert, horiz)
end

"""
    plaquette_plaquette_correlation(op::TransferOperator, separations;
                                    pos::Int=1, connected::Bool=true,
                                    optimizer=GreedyMethod())

Exact plaquette-plaquette correlation on a cylinder.

The plaquette operator on the 2×2 square with corners
(pos,c), (pos',c), (pos',c+1), (pos,c+1) where pos' = pos%row+1 is:

    Q_□ = S_{pos,c}·S_{pos',c} + S_{pos',c}·S_{pos',c+1}
        + S_{pos',c+1}·S_{pos,c+1} + S_{pos,c+1}·S_{pos,c}

Each bond is S_i·S_j = Σ_α σ^α_i σ^α_j / 4. The plaquette spans columns (1, 2)
of the unit cell. Separations are in units of full unit-cell periods.

# Returns
- `Dict{Int, ComplexF64}` mapping separation → plaquette-plaquette correlation
"""
function plaquette_plaquette_correlation(op::TransferOperator, separations;
                                          pos::Int=1,
                                          connected::Bool=true,
                                          optimizer=GreedyMethod())
    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    seps = separations isa Integer ? [separations] : collect(separations)
    isempty(seps) && return Dict{Int, ComplexF64}()

    N   = length(op.columns)
    row = op.row
    vq  = op.virtual_qubits
    pos2 = pos % row + 1  # periodic wrap

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)

    # Precompute single-operator E_O for columns 1,2 at pos and pos2
    E_O = Dict{Tuple{Int,Int,Int}, Matrix{ComplexF64}}()
    for c in 1:min(N, 2), (si, σ) in enumerate(paulis), p in (pos, pos2)
        E_O[(c, si, p)] = get_transfer_matrix_with_operator(
            op.columns[c], row, vq, σ; position=p, optimizer=optimizer)
    end

    # Q_□ spans columns (1, 2) with 4 bonds:
    #   Bond 1 (left vert):  (pos,1)-(pos',1)   → two-op in col 1
    #   Bond 2 (bottom):     (pos',1)-(pos',2)   → single-op in col 1 × single-op in col 2
    #   Bond 3 (right vert): (pos,2)-(pos',2)    → two-op in col 2
    #   Bond 4 (top):        (pos,1)-(pos,2)     → single-op in col 1 × single-op in col 2

    T_tail = N > 2 ? reduce(*, T_cols[3:end]) : Matrix{ComplexF64}(I, size(T_cols[1]))

    T_Q_period = zeros(ComplexF64, size(T_cols[1]))
    for (si, σ) in enumerate(paulis)
        # Bond 1: vertical in col 1
        E_vert1 = get_transfer_matrix_with_operator(
            op.columns[1], row, vq, Dict(pos => σ, pos2 => σ); optimizer=optimizer)
        T_Q_period .+= E_vert1 * T_cols[2] * T_tail

        # Bond 2: horizontal at pos2 across (col1, col2)
        T_Q_period .+= E_O[(1, si, pos2)] * E_O[(2, si, pos2)] * T_tail

        # Bond 3: vertical in col 2
        E_vert2 = get_transfer_matrix_with_operator(
            op.columns[2], row, vq, Dict(pos => σ, pos2 => σ); optimizer=optimizer)
        T_Q_period .+= T_cols[1] * E_vert2 * T_tail

        # Bond 4: horizontal at pos across (col1, col2)
        T_Q_period .+= E_O[(1, si, pos)] * E_O[(2, si, pos)] * T_tail
    end
    T_Q_period ./= 4.0

    # Compute correlations: l† T_Q T^{r-1} T_Q r / nf
    l_TQ = T_Q_period' * l_vec
    correlations = Dict{Int, ComplexF64}()
    for sep in sort(seps)
        current = T_Q_period * r_vec
        for _ in 1:max(sep - 1, 0)
            current = T_combined * current
        end
        correlations[sep] = dot(l_TQ, current) / nf
    end

    if connected
        μ_Q = dot(l_vec, T_Q_period * r_vec) / nf
        for k in keys(correlations)
            correlations[k] -= μ_Q^2
        end
    end
    return correlations
end

# =============================================================================
# Section 5c: Structure Factors (exact, via transfer matrix correlations)
# =============================================================================

"""
    spin_spin_structure_factor(op::TransferOperator, q::Tuple{Real,Real};
                               max_separation::Int=20, optimizer=GreedyMethod())

Exact spin-spin static structure factor on a cylinder:

    S_SS(q) = (1/N) Σ_{i,j} ⟨Sᵢ · Sⱼ⟩ e^{iq·(rᵢ - rⱼ)}

Sums over all column pairs (c1, c2) within the unit cell, all row-position
pairs (pos1, pos2), and period separations Δp_sep. Physical x-distance is
`Δp_sep * N_cols + (c2 - c1)` where N_cols = length(op.columns).

Common choices: q = (π,π) Néel, q = (π,0) stripe.
"""
function spin_spin_structure_factor(op::TransferOperator, q::Tuple{Real,Real};
                                    max_separation::Int=20,
                                    optimizer=GreedyMethod())
    qx, qy = Float64(q[1]), Float64(q[2])
    row = op.row
    N_cols = length(op.columns)

    S = 0.0
    for c1 in 1:N_cols, c2 in 1:N_cols
        Δc_intra = c2 - c1   # intra-period column offset
        for pos1 in 1:row, pos2 in 1:row
            Δp = pos2 - pos1  # row displacement
            corrs = spin_spin_correlation(op, 0:max_separation;
                                          col1=c1, col2=c2,
                                          pos1=pos1, pos2=pos2, connected=false,
                                          optimizer=optimizer)
            # Period separation Δ = 0
            x_phys = Δc_intra
            S += cos(qx * x_phys + qy * Δp) * real(corrs[0])
            # Period separation Δ > 0 (±Δ combined via cosine)
            for Δ in 1:max_separation
                haskey(corrs, Δ) || continue
                x_phys_fwd = Δ * N_cols + Δc_intra   # forward
                x_phys_bwd = -Δ * N_cols + Δc_intra   # backward
                S += cos(qx * x_phys_fwd + qy * Δp) * real(corrs[Δ])
                S += cos(qx * x_phys_bwd + qy * Δp) * real(corrs[Δ])
            end
        end
    end

    # Standard convention: S(q) = (1/N_uc) Σ_{i∈uc} Σ_j ⟨Si Sj⟩ e^{iq·(ri-rj)}
    N_uc = row * N_cols
    return S / N_uc
end

"""
    magnetic_order_squared(op::TransferOperator, q::Tuple{Real,Real};
                           max_separation::Int=20, optimizer=GreedyMethod())

Exact magnetic order parameter squared M²(q) = S_SS(q) / N_eff via transfer matrix.

Returns M²(q) = (1/N²) Σ_{i,j} ⟨Sᵢ·Sⱼ⟩ e^{iq·(rᵢ-rⱼ)}, where N_eff = N_uc × (2·max_sep + 1).

Common choices: q = (π,π) Néel, q = (π,0) stripe.
"""
function magnetic_order_squared(op::TransferOperator, q::Tuple{Real,Real};
                                max_separation::Int=20,
                                optimizer=GreedyMethod())
    S = spin_spin_structure_factor(op, q; max_separation=max_separation, optimizer=optimizer)
    N_uc = op.row * length(op.columns)
    N_eff = N_uc * (2 * max_separation + 1)
    return S / N_eff
end

"""
    dimer_structure_factor(op::TransferOperator, q::Tuple{Real,Real};
                           dimer_orientation::Symbol=:vertical,
                           max_separation::Int=20, optimizer=GreedyMethod())

Exact dimer static structure factor (connected) on a cylinder:

    S_D(q) = (1/N_d) Σ_{b,b'} [⟨D_b D_{b'}⟩ - ⟨D_b⟩⟨D_{b'}⟩] e^{iq·(r_b - r_{b'})}

Sums over all row positions for the dimer and column separations.
"""
function dimer_structure_factor(op::TransferOperator, q::Tuple{Real,Real};
                                 dimer_orientation::Symbol=:vertical,
                                 max_separation::Int=20,
                                 optimizer=GreedyMethod())
    qx, qy = Float64(q[1]), Float64(q[2])
    row = op.row

    # Compute correlations for every row-position pair
    # For vertical dimers at positions pos1, pos2 (each anchored at their row position)
    SD = 0.0
    for pos1 in 1:row, pos2 in 1:row
        Δp = pos2 - pos1
        corrs = dimer_dimer_correlation(op, 0:max_separation;
                                         dimer_orientation=dimer_orientation,
                                         pos=pos1, connected=false,
                                         optimizer=optimizer)
        # We need ⟨D_{pos1}(0) D_{pos2}(r)⟩ but our dimer_dimer_correlation
        # uses the same pos for both dimers. For the structure factor we need
        # cross-position correlations. For pos1 ≠ pos2, compute separately.
        if pos1 == pos2
            # Same dimer position: use directly
            μ_D = real(corrs[0])  # will subtract disconnected below
            # Actually, compute ⟨D⟩ for this position
            corrs_conn = dimer_dimer_correlation(op, 0:max_separation;
                                                  dimer_orientation=dimer_orientation,
                                                  pos=pos1, connected=true,
                                                  optimizer=optimizer)
            SD += cos(qy * Δp) * real(corrs_conn[0])
            for Δc in 1:max_separation
                haskey(corrs_conn, Δc) || continue
                SD += 2.0 * cos(qx * Δc + qy * Δp) * real(corrs_conn[Δc])
            end
        else
            # Cross-position dimer correlations: ⟨D_{pos1}(0) D_{pos2}(r)⟩
            # Build the two dimer transfer matrices at different positions
            corrs_cross = _dimer_cross_correlation(op, 0:max_separation,
                                                    dimer_orientation, pos1, pos2;
                                                    optimizer=optimizer)
            for (Δc, val) in corrs_cross
                if Δc == 0
                    SD += cos(qy * Δp) * real(val)
                else
                    SD += 2.0 * cos(qx * Δc + qy * Δp) * real(val)
                end
            end
        end
    end

    N_d = row * (2 * max_separation + 1)
    return SD / N_d
end

"""Cross-position dimer-dimer correlation (internal helper).
`col1`/`col2` specify which column within the unit cell each dimer is in."""
function _dimer_cross_correlation(op::TransferOperator, separations,
                                   dimer_orientation::Symbol,
                                   pos1::Int, pos2::Int;
                                   col1::Int=1, col2::Int=1,
                                   optimizer=GreedyMethod())
    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    seps = separations isa Integer ? [separations] : collect(separations)

    N   = length(op.columns)
    row = op.row
    vq  = op.virtual_qubits

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)

    if dimer_orientation == :vertical
        pos1b = pos1 % row + 1
        pos2b = pos2 % row + 1

        # T_D1: dimer at (col1, pos1)
        T_D1_col = zeros(ComplexF64, size(T_cols[1]))
        for σ in paulis
            E = get_transfer_matrix_with_operator(
                op.columns[col1], row, vq, Dict(pos1 => σ, pos1b => σ); optimizer=optimizer)
            T_D1_col .+= E
        end
        T_D1_col ./= 4.0

        # T_D2: dimer at (col2, pos2)
        T_D2_col = zeros(ComplexF64, size(T_cols[1]))
        for σ in paulis
            E = get_transfer_matrix_with_operator(
                op.columns[col2], row, vq, Dict(pos2 => σ, pos2b => σ); optimizer=optimizer)
            T_D2_col .+= E
        end
        T_D2_col ./= 4.0

        # Build full-period transfer matrices
        T_before1 = col1 > 1 ? reduce(*, T_cols[1:col1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_after1  = col1 < N ? reduce(*, T_cols[col1+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_D1 = T_before1 * T_D1_col * T_after1

        T_before2 = col2 > 1 ? reduce(*, T_cols[1:col2-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_after2  = col2 < N ? reduce(*, T_cols[col2+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_D2 = T_before2 * T_D2_col * T_after2

    else  # :horizontal
        T_D1_2col = zeros(ComplexF64, size(T_cols[1]))
        T_D2_2col = zeros(ComplexF64, size(T_cols[1]))
        for (si, σ) in enumerate(paulis)
            E1_p1 = get_transfer_matrix_with_operator(op.columns[col1], row, vq, σ; position=pos1, optimizer=optimizer)
            col1b = col1 < N ? col1 + 1 : 1
            E2_p1 = get_transfer_matrix_with_operator(op.columns[col1b], row, vq, σ; position=pos1, optimizer=optimizer)
            T_D1_2col .+= E1_p1 * E2_p1
            E1_p2 = get_transfer_matrix_with_operator(op.columns[col2], row, vq, σ; position=pos2, optimizer=optimizer)
            col2b = col2 < N ? col2 + 1 : 1
            E2_p2 = get_transfer_matrix_with_operator(op.columns[col2b], row, vq, σ; position=pos2, optimizer=optimizer)
            T_D2_2col .+= E1_p2 * E2_p2
        end
        T_D1_2col ./= 4.0
        T_D2_2col ./= 4.0
        T_before1 = col1 > 1 ? reduce(*, T_cols[1:col1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        col1b = col1 < N ? col1 + 1 : 1
        T_after1  = col1b < N ? reduce(*, T_cols[col1b+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_D1 = T_before1 * T_D1_2col * T_after1
        T_before2 = col2 > 1 ? reduce(*, T_cols[1:col2-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        col2b = col2 < N ? col2 + 1 : 1
        T_after2  = col2b < N ? reduce(*, T_cols[col2b+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
        T_D2 = T_before2 * T_D2_2col * T_after2
    end

    # ⟨D_{pos1}(0) D_{pos2}(r)⟩ - ⟨D_{pos1}⟩⟨D_{pos2}⟩
    μ1 = dot(l_vec, T_D1 * r_vec) / nf
    μ2 = dot(l_vec, T_D2 * r_vec) / nf
    l_TD1 = T_D1' * l_vec

    correlations = Dict{Int, ComplexF64}()

    # sep=0: both dimers in the same column — requires 4-operator insertion
    if 0 in seps
        if dimer_orientation == :vertical
            # D_{pos1} D_{pos2} = (1/16) Σ_{α,β} σ^α_{pos1} σ^α_{pos1b} σ^β_{pos2} σ^β_{pos2b}
            val0 = zero(ComplexF64)
            for σa in paulis, σb in paulis
                site_ops = Dict{Int, Matrix{ComplexF64}}()
                for (site, op_mat) in [(pos1, σa), (pos1b, σa), (pos2, σb), (pos2b, σb)]
                    site_ops[site] = haskey(site_ops, site) ? site_ops[site] * op_mat : op_mat
                end
                E = get_transfer_matrix_with_operator(
                    op.columns[col1], row, vq, site_ops; optimizer=optimizer)
                val0 += dot(l_vec, T_before1 * E * T_after1 * r_vec) / nf
            end
            correlations[0] = val0 / 16.0 - μ1 * μ2
        else  # :horizontal — both horizontal dimers in same column pair
            val0 = zero(ComplexF64)
            col1b_h = col1 < N ? col1 + 1 : 1
            T_before_h = col1 > 1 ? reduce(*, T_cols[1:col1-1]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            T_after_h = col1b_h < N ? reduce(*, T_cols[col1b_h+1:N]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            for σa in paulis, σb in paulis
                E_col1 = get_transfer_matrix_with_operator(
                    op.columns[col1], row, vq, Dict(pos1 => σa, pos2 => σb); optimizer=optimizer)
                E_col2 = get_transfer_matrix_with_operator(
                    op.columns[col1b_h], row, vq, Dict(pos1 => σa, pos2 => σb); optimizer=optimizer)
                val0 += dot(l_vec, T_before_h * E_col1 * E_col2 * T_after_h * r_vec) / nf
            end
            correlations[0] = val0 / 16.0 - μ1 * μ2
        end
    end

    # sep ≥ 1: use full-period transfer matrices (existing formula, correct)
    for sep in sort(filter(s -> s >= 1, seps))
        current = T_D2 * r_vec
        for _ in 1:(sep - 1)
            current = T_combined * current
        end
        correlations[sep] = dot(l_TD1, current) / nf - μ1 * μ2
    end
    return correlations
end

"""Generalized dimer-dimer correlation allowing different orientations for ref and target."""
function _dimer_general_correlation(op::TransferOperator, separations,
                                     orient1::Symbol, pos1::Int,
                                     orient2::Symbol, pos2::Int;
                                     optimizer=GreedyMethod())
    # Delegate to existing same-orientation helpers when possible
    if orient1 == orient2
        if pos1 == pos2
            return dimer_dimer_correlation(op, separations;
                        dimer_orientation=orient1, pos=pos1, connected=true, optimizer=optimizer)
        else
            return _dimer_cross_correlation(op, separations,
                        orient1, pos1, pos2; optimizer=optimizer)
        end
    end

    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    seps = separations isa Integer ? [separations] : collect(separations)
    N   = length(op.columns)
    row = op.row
    vq  = op.virtual_qubits

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)

    # Build T_D for a dimer with given orientation and position
    function _build_T_D(orient, pos)
        if orient == :vertical
            pos2 = pos % row + 1
            T_D_col = zeros(ComplexF64, size(T_cols[1]))
            for σ in paulis
                E = get_transfer_matrix_with_operator(
                    op.columns[1], row, vq, Dict(pos => σ, pos2 => σ); optimizer=optimizer)
                T_D_col .+= E
            end
            T_D_col ./= 4.0
            T_tail = N > 1 ? reduce(*, T_cols[2:end]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            return T_D_col * T_tail
        else  # :horizontal
            T_D_2col = zeros(ComplexF64, size(T_cols[1]))
            for σ in paulis
                E1 = get_transfer_matrix_with_operator(
                    op.columns[1], row, vq, σ; position=pos, optimizer=optimizer)
                E2 = get_transfer_matrix_with_operator(
                    op.columns[min(2, N)], row, vq, σ; position=pos, optimizer=optimizer)
                T_D_2col .+= E1 * E2
            end
            T_D_2col ./= 4.0
            T_tail = N > 2 ? reduce(*, T_cols[3:end]) : Matrix{ComplexF64}(I, size(T_cols[1]))
            return T_D_2col * T_tail
        end
    end

    T_D1 = _build_T_D(orient1, pos1)
    T_D2 = _build_T_D(orient2, pos2)

    μ1 = dot(l_vec, T_D1 * r_vec) / nf
    μ2 = dot(l_vec, T_D2 * r_vec) / nf
    l_TD1 = T_D1' * l_vec

    correlations = Dict{Int, ComplexF64}()
    for sep in sort(seps)
        current = T_D2 * r_vec
        for _ in 1:max(sep - 1, 0)
            current = T_combined * current
        end
        correlations[sep] = dot(l_TD1, current) / nf - μ1 * μ2
    end
    return correlations
end

"""
    plaquette_structure_factor(op::TransferOperator, q::Tuple{Real,Real};
                               max_separation::Int=20, optimizer=GreedyMethod())

Exact plaquette static structure factor (connected) on a cylinder:

    S_P(q) = (1/N_p) Σ_{□,□'} [⟨Q_□ Q_{□'}⟩ - ⟨Q_□⟩⟨Q_{□'}⟩] e^{iq·(r_□ - r_{□'})}

Sums over all row positions for the plaquette top-left corner and column separations.
"""
function plaquette_structure_factor(op::TransferOperator, q::Tuple{Real,Real};
                                     max_separation::Int=20,
                                     optimizer=GreedyMethod())
    qx, qy = Float64(q[1]), Float64(q[2])
    row = op.row

    # Build T_Q_period for each row position
    T_Q = Dict{Int, Matrix{ComplexF64}}()
    μ_Q = Dict{Int, ComplexF64}()

    paulis = [_resolve_op(:X), _resolve_op(:Y), _resolve_op(:Z)]
    N   = length(op.columns)
    vq  = op.virtual_qubits

    T_cols     = _column_transfer_matrices(op)
    T_combined = reduce(*, T_cols)
    l_vec, r_vec, nf, _ = _fixed_points(T_combined)
    T_tail = N > 2 ? reduce(*, T_cols[3:end]) : Matrix{ComplexF64}(I, size(T_cols[1]))

    for pos in 1:row
        pos2 = pos % row + 1
        T_Q_period = zeros(ComplexF64, size(T_cols[1]))

        # Precompute single-op E_O for this position pair
        E_O_local = Dict{Tuple{Int,Int,Int}, Matrix{ComplexF64}}()
        for c in 1:min(N, 2), (si, σ) in enumerate(paulis), p in (pos, pos2)
            E_O_local[(c, si, p)] = get_transfer_matrix_with_operator(
                op.columns[c], row, vq, σ; position=p, optimizer=optimizer)
        end

        for (si, σ) in enumerate(paulis)
            E_vert1 = get_transfer_matrix_with_operator(
                op.columns[1], row, vq, Dict(pos => σ, pos2 => σ); optimizer=optimizer)
            T_Q_period .+= E_vert1 * T_cols[2] * T_tail
            T_Q_period .+= E_O_local[(1, si, pos2)] * E_O_local[(2, si, pos2)] * T_tail
            E_vert2 = get_transfer_matrix_with_operator(
                op.columns[2], row, vq, Dict(pos => σ, pos2 => σ); optimizer=optimizer)
            T_Q_period .+= T_cols[1] * E_vert2 * T_tail
            T_Q_period .+= E_O_local[(1, si, pos)] * E_O_local[(2, si, pos)] * T_tail
        end
        T_Q_period ./= 4.0

        T_Q[pos] = T_Q_period
        μ_Q[pos] = dot(l_vec, T_Q_period * r_vec) / nf
    end

    # Compute structure factor
    SP = 0.0
    for p1 in 1:row, p2 in 1:row
        Δp = p2 - p1
        l_TQ1 = T_Q[p1]' * l_vec

        for Δc in 0:max_separation
            current = T_Q[p2] * r_vec
            for _ in 1:max(Δc - 1, 0)
                current = T_combined * current
            end
            corr_conn = real(dot(l_TQ1, current) / nf - μ_Q[p1] * μ_Q[p2])
            if Δc == 0
                SP += cos(qy * Δp) * corr_conn
            else
                SP += 2.0 * cos(qx * Δc + qy * Δp) * corr_conn
            end
        end
    end

    N_p = row * (2 * max_separation + 1)
    return SP / N_p
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
