# =============================================================================
# Exact Observable Expectation Values (Transfer Matrix Contraction)
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

# =============================================================================
# Heisenberg J1-J2 Energy for 2×2 Unit Cell (exact tensor contraction)
# =============================================================================

"""
    compute_exact_heisenberg_energy_2x2(gates_odd, gates_even, row, virtual_qubits, J1, J2;
                                         optimizer=GreedyMethod())

Compute Heisenberg J1-J2 energy per column for a 2-column unit cell.

Odd columns use `gates_odd`, even columns use `gates_even`.
The combined transfer matrix is T = T_odd * T_even with fixed points l, r.

Energy contributions:
- Within-period bonds (odd→even column)
- Cross-period bonds (even→next odd column, using dominant eigenvalue λ)

Returns energy per column (total / 2).
"""
function compute_exact_heisenberg_energy_2x2(gates_odd, gates_even, row, virtual_qubits,
                                              J1::Float64, J2::Float64;
                                              optimizer=GreedyMethod())
    σx = Matrix{ComplexF64}(Matrix(X))
    σy = ComplexF64[0 -im; im 0]
    σz = Matrix{ComplexF64}(Matrix(Z))
    paulis = [σx, σy, σz]

    # Build individual and combined transfer matrices
    T_odd  = get_transfer_matrix(gates_odd, row, virtual_qubits)
    T_even = get_transfer_matrix(gates_even, row, virtual_qubits)
    T_combined = T_odd * T_even

    # Fixed points of T_combined
    eig_r = eigen(T_combined)
    idx_r = sortperm(abs.(eig_r.values), rev=true)
    r_vec = eig_r.vectors[:, idx_r[1]]
    λ = eig_r.values[idx_r[1]]

    eig_l = eigen(T_combined')
    idx_l = sortperm(abs.(eig_l.values), rev=true)
    l_vec = eig_l.vectors[:, idx_l[1]]
    nf = dot(l_vec, r_vec)

    # Precompute E_O for each (pauli, position) for both gate sets
    E_O_odd  = Dict{Tuple{Matrix{ComplexF64},Int}, Matrix{ComplexF64}}()
    E_O_even = Dict{Tuple{Matrix{ComplexF64},Int}, Matrix{ComplexF64}}()
    for σ in paulis, pos in 1:row
        E_O_odd[(σ, pos)]  = get_transfer_matrix_with_operator(
            gates_odd, row, virtual_qubits, σ; position=pos, optimizer=optimizer)
        E_O_even[(σ, pos)] = get_transfer_matrix_with_operator(
            gates_even, row, virtual_qubits, σ; position=pos, optimizer=optimizer)
    end

    # Precompute intermediate vectors for cross-period bonds
    r_mid = T_even * r_vec   # T_even · r
    l_mid = T_odd' * l_vec   # T_odd† · l

    energy = 0.0

    # =====================================================================
    # J1: Vertical bonds (within column, periodic)
    # =====================================================================
    if row > 1
        for col_type in (:odd, :even)
            gates_col = col_type == :odd ? gates_odd : gates_even
            T_next = col_type == :odd ? T_even : T_odd
            for i in 1:row
                j = i % row + 1
                for σ in paulis
                    E_OO = get_transfer_matrix_with_operator(
                        gates_col, row, virtual_qubits, Dict(i => σ, j => σ);
                        optimizer=optimizer)
                    if col_type == :odd
                        # l† · E_OO_odd · T_even · r / nf
                        val = dot(l_vec, E_OO * T_even * r_vec) / nf
                    else
                        # l† · T_odd · E_OO_even · r / nf
                        val = dot(l_vec, T_odd * E_OO * r_vec) / nf
                    end
                    energy += J1 * real(val) / 4.0
                end
            end
        end
    end

    # =====================================================================
    # J1: Horizontal bonds (same position, adjacent columns)
    # =====================================================================
    for σ in paulis, pos in 1:row
        # Within-period: odd→even (same period)
        # l† · E_O_odd(σ,pos) · E_O_even(σ,pos) · r / nf
        val_wp = dot(l_vec, E_O_odd[(σ, pos)] * E_O_even[(σ, pos)] * r_vec) / nf
        energy += J1 * real(val_wp) / 4.0

        # Cross-period: even→next odd
        # l_mid† · E_O_even(σ,pos) · E_O_odd(σ,pos) · r_mid / (λ · nf)
        val_cp = dot(l_mid, E_O_even[(σ, pos)] * E_O_odd[(σ, pos)] * r_mid) / (λ * nf)
        energy += J1 * real(val_cp) / 4.0
    end

    # =====================================================================
    # J2: Diagonal NNN bonds
    # =====================================================================
    if J2 != 0.0
        for i in 1:row
            j_up   = i % row + 1
            j_down = (i - 2 + row) % row + 1

            for σ in paulis
                # Within-period: odd(i)→even(i±1)
                for j in (j_up, j_down)
                    val = dot(l_vec, E_O_odd[(σ, i)] * E_O_even[(σ, j)] * r_vec) / nf
                    energy += J2 * real(val) / 4.0
                end

                # Cross-period: even(i)→next odd(i±1)
                for j in (j_up, j_down)
                    val = dot(l_mid, E_O_even[(σ, i)] * E_O_odd[(σ, j)] * r_mid) / (λ * nf)
                    energy += J2 * real(val) / 4.0
                end
            end
        end
    end

    # Energy per column = total / 2 (2-column unit cell)
    return energy / 2.0
end
