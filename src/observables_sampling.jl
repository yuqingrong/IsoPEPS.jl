# =============================================================================
# Sampling-Based Observable Calculations
# =============================================================================
# Functions that compute expectation values and energies from measurement samples
# (as opposed to exact tensor network contraction in observables_exact.jl)

# =============================================================================
# Section 1: Internal Helpers (private)
# =============================================================================

"""Number of columns in sample vector."""
_n_cols(samples, row) = div(length(samples), row)

# =============================================================================
# Section 2: Core expect API for samples
# =============================================================================

"""
    expect(samples::Vector{Float64}, row::Int)
    expect(samples::Vector{Float64}, row::Int; position::Int)

Single-site expectation from samples:
- without `position`: averaged over all sampled sites
- with `position`: averaged over columns at fixed row position
"""
function expect(samples::Vector{Float64}, row::Int; position::Union{Nothing,Int}=nothing)
    if isnothing(position)
        return mean(samples)
    end
    return mean(samples[position:row:end])
end

"""
    expect(samples::Vector{Float64}, row::Int, pos1::Int, pos2::Int; col_separation::Int=0)

Two-site expectation ⟨O_{pos1,c} O_{pos2,c+sep}⟩ averaged over all valid columns.
"""
function expect(samples::Vector{Float64}, row::Int, pos1::Int, pos2::Int;
                col_separation::Int=0)
    ncols = _n_cols(samples, row)
    total = 0.0
    count = 0
    for c in 1:(ncols - col_separation)
        i = row * (c - 1) + pos1
        j = row * (c - 1 + col_separation) + pos2
        total += samples[i] * samples[j]
        count += 1
    end
    return total / count
end

# =============================================================================
# Section 3: Correlation Functions from samples
# =============================================================================

"""
    correlation_function(samples::Vector{Float64}, row::Int, separations;
                         position::Int=1, connected::Bool=false)

Return a dictionary mapping separation `r` to
`⟨O_{position,c} O_{position,c+r}⟩`.
"""
function correlation_function(samples::Vector{Float64}, row::Int, separations;
                              position::Int=1, connected::Bool=false)
    seps = separations isa Integer ? [separations] : collect(separations)
    correlations = Dict{Int, Float64}()
    for sep in seps
        correlations[sep] = expect(samples, row, position, position;
                                   col_separation=sep)
    end

    if connected
        μ = expect(samples, row; position=position)
        for k in keys(correlations)
            correlations[k] -= μ^2
        end
    end
    return correlations
end

# =============================================================================
# Section 3b: Structure Factor (magnetic order parameter squared)
# =============================================================================

"""
    structure_factor(samples::Vector{Float64}, row::Int, q::Tuple{Real,Real};
                     max_separation::Int=20)

Magnetic order parameter squared (static structure factor per site) on a
cylinder geometry from measurement samples:

    M²(q) = (1/N²) Σ_{i,j} ⟨Oᵢ Oⱼ⟩ e^{iq·(rᵢ - rⱼ)}

where N = row × n_cols is the total number of sites.  On the cylinder with
translation invariance along the column direction this reduces to:

    M²(q) = (1/N) Σ_{p1,p2=1}^{row} [ cos(qy·Δp)·C(p1,p2,0)
              + 2 Σ_{Δc=1}^{max} cos(qx·Δc + qy·Δp)·C(p1,p2,Δc) ]

where Δp = p2 - p1, C(p1,p2,Δc) = ⟨O_{p1,c} O_{p2,c+Δc}⟩, and
N = row × (2·max_separation + 1) is the effective number of sites summed over.

# Arguments
- `samples`: Measurement outcome vector (layout: column-major, row sites per column)
- `row`: Number of rows (circumference of the cylinder)
- `q`: Momentum vector (qx, qy)
- `max_separation`: Maximum column separation for the sum (default: 20)

# Returns
- Real-valued M²(q) (the 1/N² normalised structure factor)
"""
function structure_factor(samples::Vector{Float64}, row::Int, q::Tuple{Real,Real};
                          max_separation::Int=20)
    qx, qy = Float64(q[1]), Float64(q[2])
    ncols = _n_cols(samples, row)
    max_sep = min(max_separation, ncols - 1)

    S = 0.0

    # Δc = 0 terms
    for pos1 in 1:row, pos2 in 1:row
        Δpos = pos2 - pos1
        corr = expect(samples, row, pos1, pos2; col_separation=0)
        S += cos(qy * Δpos) * corr
    end

    # Δc > 0 terms (both +Δc and -Δc combined via cosine)
    for Δc in 1:max_sep
        for pos1 in 1:row, pos2 in 1:row
            Δpos = pos2 - pos1
            corr = expect(samples, row, pos1, pos2; col_separation=Δc)
            S += 2.0 * cos(qx * Δc + qy * Δpos) * corr
        end
    end

    # N = row × (2·max_sep + 1) effective sites in the summation window
    N = row*row * (2 * max_sep + 1)
    return S /N
end

"""
    magnetic_order_squared(X_samples, Z_samples, Y_samples, row, q; max_separation=20)

Full-spin magnetic order parameter squared M²(q) = (1/4)[S_X(q) + S_Y(q) + S_Z(q)].

Computes M²(q) = (1/N²) Σ_{i,j} ⟨Sᵢ·Sⱼ⟩ e^{iq·(rᵢ-rⱼ)} where Sᵅ = σᵅ/2
are spin-1/2 operators. The factor 1/4 converts from Pauli (σ = ±1) to
spin-1/2 (S = ±1/2) convention.

Common choices:
- q = (π, π): Néel antiferromagnetic order
- q = (π, 0): Stripe antiferromagnetic order
"""
function magnetic_order_squared(X_samples::Vector{Float64},
                                Z_samples::Vector{Float64},
                                Y_samples::Vector{Float64},
                                row::Int, q::Tuple{Real,Real};
                                max_separation::Int=20)
    return (structure_factor(X_samples, row, q; max_separation=max_separation) +
            structure_factor(Y_samples, row, q; max_separation=max_separation) +
            structure_factor(Z_samples, row, q; max_separation=max_separation)) / 4
end

# =============================================================================
# Section 4: Energy (model-dispatched wrappers + existing implementations)
# =============================================================================

compute_energy(m::TFIM, X_samples, Z_samples, row) =
    compute_tfim_energy(X_samples, Z_samples, m.g, m.J, row)

compute_energy(m::HeisenbergJ1J2, X_samples, Z_samples, Y_samples, row) =
    compute_heisenberg_energy(X_samples, Z_samples, Y_samples, m.J1, m.J2, row)

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
Computes the transverse field Ising model energy on a cylinder
(periodic in y-direction):
H = -g ∑ᵢ Xᵢ - J ∑⟨ij⟩ ZᵢZⱼ

Sample layout for row=4:
  Z[1]  Z[5]  Z[9]   ...   ← row 1
  Z[2]  Z[6]  Z[10]  ...   ← row 2
  Z[3]  Z[7]  Z[11]  ...   ← row 3
  Z[4]  Z[8]  Z[12]  ...   ← row 4 (wraps to row 1)
   ↑     ↑     ↑
  col1  col2  col3

- Vertical bonds: Z[i]*Z[i+1] within column + periodic wrap Z[row,col]↔Z[1,col]
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
        n_cols = div(N, row)
        # Open vertical bonds within each column: (pos, pos+1)
        ZZ_vert_pairs = [Z_samples[i] * Z_samples[i+1]
                         for i in 1:N-1 if i % row != 0]
        # Periodic wrap bonds: (row, col) <-> (1, col)
        wrap_pairs = [Z_samples[c*row] * Z_samples[(c-1)*row+1] for c in 1:n_cols]
        ZZ_vert = mean(vcat(ZZ_vert_pairs, wrap_pairs))

        # Horizontal bonds: Z[i]*Z[i+row] (same row, adjacent columns)
        ZZ_horiz = mean(Z_samples[i] * Z_samples[i+row] for i in 1:N-row)
        ZZ_mean = ZZ_vert + ZZ_horiz
    end

    return -g * X_mean - J * ZZ_mean
end

"""
    compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)

Compute Heisenberg J1-J2 energy from X, Y, Z measurement samples on a cylinder
(periodic in y-direction).

    S_i · S_j = (X_i X_j + Y_i Y_j + Z_i Z_j) / 4

# Arguments
- `X_samples`: Vector of X measurement outcomes
- `Z_samples`: Vector of Z measurement outcomes
- `Y_samples`: Vector of Y measurement outcomes
- `J1`: Nearest-neighbor coupling
- `J2`: Next-nearest-neighbor (diagonal) coupling
- `row`: Number of rows (cylinder circumference)

# Returns
- Energy estimate per column
"""
function compute_heisenberg_energy(X_samples, Z_samples, Y_samples, J1, J2, row)
    all_samples = (Z_samples, X_samples, Y_samples)

    # Helper: compute vertical and horizontal correlations for one set of samples
    # Vertical bonds are periodic (cylinder): includes wrap from row `row` to row 1
    function _correlations(S, row)
        N = length(S)
        n_cols = div(N, row)
        if row == 1
            vert = 0.0
            horiz = mean(S[i] * S[i+1] for i in 1:N-1)
        else
            # Open vertical bonds within each column: (pos, pos+1)
            vert_pairs = [S[i] * S[i+1] for i in 1:N-1 if i % row != 0]
            # Periodic wrap bonds: (row, col) <-> (1, col)
            wrap_pairs = [S[c*row] * S[(c-1)*row+1] for c in 1:n_cols]
            vert = mean(vcat(vert_pairs, wrap_pairs))
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

    # J2: diagonal NNN bonds (periodic in y)
    if J2 != 0.0 && row > 1
        SS_diag = 0.0
        for S in all_samples
            N = length(S)
            n_cols = div(N, row)
            # Open diagonal: (pos,col)->(pos+1,col+1), skip column boundaries
            diag_up = [S[i] * S[i+row+1] for i in 1:N-row-1 if i % row != 0]
            # Open anti-diagonal: (pos,col)->(pos-1,col+1), skip column boundaries
            diag_down = [S[i] * S[i+row-1] for i in 1:N-row+1 if (i-1) % row != 0]
            # Periodic wrap diagonals:
            # (row, col) -> (1, col+1): index c*row -> c*row+1
            wrap_up = [S[c*row] * S[c*row+1] for c in 1:n_cols-1]
            # (1, col) -> (row, col+1): index (c-1)*row+1 -> (c+1)*row
            wrap_down = [S[(c-1)*row+1] * S[(c+1)*row] for c in 1:n_cols-1]
            SS_diag += mean(vcat(diag_up, wrap_up)) + mean(vcat(diag_down, wrap_down))
        end
        energy += J2 * SS_diag / 4.0
    end
    return energy
end

# =============================================================================
# Section 4b: Spin-Spin, Dimer-Dimer, Plaquette-Plaquette Correlations
# =============================================================================

"""
    spin_spin_correlation(X_samples, Z_samples, Y_samples, row, separations;
                          pos1::Int=1, pos2::Int=1, connected::Bool=false)

Spin-spin correlation ⟨S_{pos1,c} · S_{pos2,c+r}⟩ = (⟨XX⟩ + ⟨YY⟩ + ⟨ZZ⟩)/4
from measurement samples on a cylinder.

# Arguments
- `X_samples`, `Z_samples`, `Y_samples`: Measurement outcome vectors
- `row`: Number of rows (cylinder circumference)
- `separations`: Column separations (integer or collection)
- `pos1`, `pos2`: Row positions of the two spins
- `connected`: If true, subtract ⟨S⟩·⟨S⟩ (always zero by symmetry for Heisenberg)

# Returns
- `Dict{Int, Float64}` mapping separation → ⟨S·S⟩
"""
function spin_spin_correlation(X_samples::Vector{Float64},
                               Z_samples::Vector{Float64},
                               Y_samples::Vector{Float64},
                               row::Int, separations;
                               pos1::Int=1, pos2::Int=1,
                               connected::Bool=false)
    seps = separations isa Integer ? [separations] : collect(separations)
    correlations = Dict{Int, Float64}()
    for sep in seps
        SS = 0.0
        for S in (X_samples, Y_samples, Z_samples)
            SS += expect(S, row, pos1, pos2; col_separation=sep)
        end
        correlations[sep] = SS / 4.0
    end
    if connected
        # ⟨S⟩ = 0 by symmetry, but compute explicitly for generality
        μ = 0.0
        for S in (X_samples, Y_samples, Z_samples)
            μ += mean(S[pos1:row:end]) * mean(S[pos2:row:end])
        end
        μ /= 4.0
        for k in keys(correlations)
            correlations[k] -= μ
        end
    end
    return correlations
end

"""
    dimer_dimer_correlation(X_samples, Z_samples, Y_samples, row, separations;
                            dimer_orientation::Symbol=:vertical, pos::Int=1,
                            connected::Bool=true)

Dimer-dimer correlation from measurement samples on a cylinder.

The dimer operator is the bond energy D_ij = S_i · S_j = (X_iX_j + Y_iY_j + Z_iZ_j)/4.

For `:vertical` dimers: D connects (pos, col) and (pos', col) where pos' = pos%row+1 (periodic).
For `:horizontal` dimers: D connects (pos, col) and (pos, col+1).

Returns connected correlator ⟨D(0)D(r)⟩ - ⟨D⟩² when `connected=true`.

# Arguments
- `X_samples`, `Z_samples`, `Y_samples`: Measurement outcome vectors
- `row`: Number of rows (cylinder circumference)
- `separations`: Column separations (integer or collection)
- `dimer_orientation`: `:vertical` or `:horizontal`
- `pos`: Row position of the reference dimer
- `connected`: Subtract ⟨D⟩² (default: true)

# Returns
- `Dict{Int, Float64}` mapping separation → dimer-dimer correlation
"""
function dimer_dimer_correlation(X_samples::Vector{Float64},
                                 Z_samples::Vector{Float64},
                                 Y_samples::Vector{Float64},
                                 row::Int, separations;
                                 dimer_orientation::Symbol=:vertical,
                                 pos::Int=1,
                                 connected::Bool=true)
    seps = separations isa Integer ? [separations] : collect(separations)
    all_samples = (X_samples, Y_samples, Z_samples)
    ncols = _n_cols(Z_samples, row)

    # Build per-column dimer values: D_c = Σ_α S_α[site1,c]*S_α[site2,c] / 4
    if dimer_orientation == :vertical
        pos2 = pos % row + 1  # periodic wrap
        dimer_vals = zeros(ncols)
        for S in all_samples
            for c in 1:ncols
                i1 = row * (c - 1) + pos
                i2 = row * (c - 1) + pos2
                dimer_vals[c] += S[i1] * S[i2] / 4.0
            end
        end
    elseif dimer_orientation == :horizontal
        # Horizontal dimer spans (pos, c) and (pos, c+1), so we have ncols-1 dimers
        dimer_vals = zeros(ncols - 1)
        for S in all_samples
            for c in 1:(ncols - 1)
                i1 = row * (c - 1) + pos
                i2 = row * c + pos  # same row, next column
                dimer_vals[c] += S[i1] * S[i2] / 4.0
            end
        end
    else
        error("dimer_orientation must be :vertical or :horizontal, got $dimer_orientation")
    end

    n_dimers = length(dimer_vals)
    correlations = Dict{Int, Float64}()
    for sep in seps
        total = 0.0
        count = 0
        for c in 1:(n_dimers - sep)
            total += dimer_vals[c] * dimer_vals[c + sep]
            count += 1
        end
        correlations[sep] = total / count
    end

    if connected
        μ_D = mean(dimer_vals)
        for k in keys(correlations)
            correlations[k] -= μ_D^2
        end
    end
    return correlations
end

"""
    plaquette_plaquette_correlation(X_samples, Z_samples, Y_samples, row, separations;
                                    pos::Int=1, connected::Bool=true)

Plaquette-plaquette correlation from measurement samples on a cylinder.

The plaquette operator on a 2×2 square with corners (pos,c), (pos',c), (pos',c+1), (pos,c+1)
where pos' = pos%row+1 is:

    Q_□ = S_{pos,c}·S_{pos',c} + S_{pos',c}·S_{pos',c+1}
        + S_{pos',c+1}·S_{pos,c+1} + S_{pos,c+1}·S_{pos,c}

This is a sum of 4 bond operators (left, bottom, right, top of the plaquette).
Each plaquette is centered between columns c and c+1.

# Arguments
- `X_samples`, `Z_samples`, `Y_samples`: Measurement outcome vectors
- `row`: Number of rows (cylinder circumference)
- `separations`: Column separations between plaquette centers (integer or collection)
- `pos`: Row position of the top-left corner
- `connected`: Subtract ⟨Q⟩² (default: true)

# Returns
- `Dict{Int, Float64}` mapping separation → plaquette-plaquette correlation
"""
function plaquette_plaquette_correlation(X_samples::Vector{Float64},
                                         Z_samples::Vector{Float64},
                                         Y_samples::Vector{Float64},
                                         row::Int, separations;
                                         pos::Int=1,
                                         connected::Bool=true)
    seps = separations isa Integer ? [separations] : collect(separations)
    all_samples = (X_samples, Y_samples, Z_samples)
    ncols = _n_cols(Z_samples, row)
    pos2 = pos % row + 1  # periodic wrap in y

    # Four corners of plaquette between columns c and c+1:
    # TL = (pos, c), BL = (pos', c), BR = (pos', c+1), TR = (pos, c+1)
    # Q = S_TL·S_BL + S_BL·S_BR + S_BR·S_TR + S_TR·S_TL
    # Each bond = Σ_α S_α[i]*S_α[j]/4

    n_plaq = ncols - 1  # plaquettes between consecutive columns
    plaq_vals = zeros(n_plaq)
    for S in all_samples
        for c in 1:n_plaq
            tl = row * (c - 1) + pos
            bl = row * (c - 1) + pos2
            br = row * c + pos2
            tr = row * c + pos
            # Four bonds of the plaquette
            plaq_vals[c] += (S[tl]*S[bl] + S[bl]*S[br] + S[br]*S[tr] + S[tr]*S[tl]) / 4.0
        end
    end

    correlations = Dict{Int, Float64}()
    for sep in seps
        total = 0.0
        count = 0
        for c in 1:(n_plaq - sep)
            total += plaq_vals[c] * plaq_vals[c + sep]
            count += 1
        end
        correlations[sep] = total / count
    end

    if connected
        μ_Q = mean(plaq_vals)
        for k in keys(correlations)
            correlations[k] -= μ_Q^2
        end
    end
    return correlations
end

# =============================================================================
# Section 4c: Structure Factors (Fourier transforms of correlations)
# =============================================================================

"""
    spin_spin_structure_factor(X_samples, Z_samples, Y_samples, row, q;
                               max_separation::Int=20)

Spin-spin static structure factor on a cylinder:

    S_SS(q) = (1/N) Σ_{i,j} ⟨S_i · S_j⟩ e^{iq·(r_i - r_j)}

where S_i · S_j = (X_iX_j + Y_iY_j + Z_iZ_j)/4. This equals `magnetic_order_squared`
and is provided as an alias with a consistent naming convention.

Common choices:
- q = (π, π): Néel order parameter
- q = (π, 0) or (0, π): Stripe order parameter
"""
function spin_spin_structure_factor(X_samples::Vector{Float64},
                                    Z_samples::Vector{Float64},
                                    Y_samples::Vector{Float64},
                                    row::Int, q::Tuple{Real,Real};
                                    max_separation::Int=20)
    return magnetic_order_squared(X_samples, Z_samples, Y_samples, row, q;
                                  max_separation=max_separation)
end

"""
    dimer_structure_factor(X_samples, Z_samples, Y_samples, row, q;
                           dimer_orientation::Symbol=:vertical,
                           max_separation::Int=20)

Dimer static structure factor on a cylinder:

    S_D(q) = (1/N_d) Σ_{b,b'} [⟨D_b D_{b'}⟩ - ⟨D_b⟩⟨D_{b'}⟩] e^{iq·(r_b - r_{b'})}

where D_b = S_i · S_j is the dimer (bond) operator, and the sum runs over all
pairs of dimers with the same orientation. The positions r_b are the bond centers.

For vertical dimers, r_b = (c, pos + 0.5) so Δr = (Δc, Δpos).
For horizontal dimers, r_b = (c + 0.5, pos) so Δr = (Δc, Δpos).

# Arguments
- `X_samples`, `Z_samples`, `Y_samples`: Measurement outcome vectors
- `row`: Cylinder circumference
- `q`: Momentum vector (qx, qy)
- `dimer_orientation`: `:vertical` or `:horizontal`
- `max_separation`: Maximum column separation

# Returns
- Real-valued structure factor S_D(q)
"""
function dimer_structure_factor(X_samples::Vector{Float64},
                                Z_samples::Vector{Float64},
                                Y_samples::Vector{Float64},
                                row::Int, q::Tuple{Real,Real};
                                dimer_orientation::Symbol=:vertical,
                                max_separation::Int=20)
    qx, qy = Float64(q[1]), Float64(q[2])
    all_samples = (X_samples, Y_samples, Z_samples)
    ncols = _n_cols(Z_samples, row)

    # Build dimer values for ALL row positions: dimer_vals[pos, col]
    if dimer_orientation == :vertical
        # Vertical dimer at (pos, col) connects pos and pos%row+1
        dimer_vals = zeros(row, ncols)
        for S in all_samples
            for c in 1:ncols, pos in 1:row
                pos2 = pos % row + 1
                i1 = row * (c - 1) + pos
                i2 = row * (c - 1) + pos2
                dimer_vals[pos, c] += S[i1] * S[i2] / 4.0
            end
        end
        n_pos = row
        n_cols_d = ncols
    elseif dimer_orientation == :horizontal
        # Horizontal dimer at (pos, col) connects (pos,col) and (pos,col+1)
        dimer_vals = zeros(row, ncols - 1)
        for S in all_samples
            for c in 1:(ncols - 1), pos in 1:row
                i1 = row * (c - 1) + pos
                i2 = row * c + pos
                dimer_vals[pos, c] += S[i1] * S[i2] / 4.0
            end
        end
        n_pos = row
        n_cols_d = ncols - 1
    else
        error("dimer_orientation must be :vertical or :horizontal, got $dimer_orientation")
    end

    max_sep = min(max_separation, n_cols_d - 1)

    # Fourier transform: S(q) = (1/N_d) Σ_{pos1,pos2,Δc} C(pos1,pos2,Δc) cos(qx Δc + qy Δpos)
    # where C = ⟨D_{pos1,c} D_{pos2,c+Δc}⟩ - ⟨D_{pos1}⟩⟨D_{pos2}⟩
    μ = vec(mean(dimer_vals, dims=2))  # mean dimer value per row position

    SD = 0.0
    # Δc = 0 terms
    for p1 in 1:n_pos, p2 in 1:n_pos
        Δp = p2 - p1
        corr = mean(dimer_vals[p1, c] * dimer_vals[p2, c] for c in 1:n_cols_d)
        SD += cos(qy * Δp) * (corr - μ[p1] * μ[p2])
    end

    # Δc > 0 terms
    for Δc in 1:max_sep
        for p1 in 1:n_pos, p2 in 1:n_pos
            Δp = p2 - p1
            corr = mean(dimer_vals[p1, c] * dimer_vals[p2, c + Δc] for c in 1:(n_cols_d - Δc))
            SD += 2.0 * cos(qx * Δc + qy * Δp) * (corr - μ[p1] * μ[p2])
        end
    end

    N_d = n_pos * (2 * max_sep + 1)
    return SD / N_d
end

"""
    plaquette_structure_factor(X_samples, Z_samples, Y_samples, row, q;
                               max_separation::Int=20)

Plaquette static structure factor on a cylinder:

    S_P(q) = (1/N_p) Σ_{□,□'} [⟨Q_□ Q_{□'}⟩ - ⟨Q_□⟩⟨Q_{□'}⟩] e^{iq·(r_□ - r_{□'})}

where Q_□ = S_1·S_2 + S_2·S_3 + S_3·S_4 + S_4·S_1 is the plaquette operator
on a 2×2 square, and positions are indexed by (pos, col) of the top-left corner.

# Arguments
- `X_samples`, `Z_samples`, `Y_samples`: Measurement outcome vectors
- `row`: Cylinder circumference
- `q`: Momentum vector (qx, qy)
- `max_separation`: Maximum column separation

# Returns
- Real-valued structure factor S_P(q)
"""
function plaquette_structure_factor(X_samples::Vector{Float64},
                                    Z_samples::Vector{Float64},
                                    Y_samples::Vector{Float64},
                                    row::Int, q::Tuple{Real,Real};
                                    max_separation::Int=20)
    qx, qy = Float64(q[1]), Float64(q[2])
    all_samples = (X_samples, Y_samples, Z_samples)
    ncols = _n_cols(Z_samples, row)
    n_plaq_cols = ncols - 1  # plaquettes between consecutive columns

    # Build plaquette values for ALL row positions: plaq_vals[pos, col]
    # Plaquette at (pos, c) has corners: (pos,c), (pos2,c), (pos2,c+1), (pos,c+1)
    plaq_vals = zeros(row, n_plaq_cols)
    for S in all_samples
        for c in 1:n_plaq_cols, pos in 1:row
            pos2 = pos % row + 1
            tl = row * (c - 1) + pos
            bl = row * (c - 1) + pos2
            br = row * c + pos2
            tr = row * c + pos
            plaq_vals[pos, c] += (S[tl]*S[bl] + S[bl]*S[br] + S[br]*S[tr] + S[tr]*S[tl]) / 4.0
        end
    end

    max_sep = min(max_separation, n_plaq_cols - 1)
    μ = vec(mean(plaq_vals, dims=2))  # mean per row position

    SP = 0.0
    # Δc = 0 terms
    for p1 in 1:row, p2 in 1:row
        Δp = p2 - p1
        corr = mean(plaq_vals[p1, c] * plaq_vals[p2, c] for c in 1:n_plaq_cols)
        SP += cos(qy * Δp) * (corr - μ[p1] * μ[p2])
    end

    # Δc > 0 terms
    for Δc in 1:max_sep
        for p1 in 1:row, p2 in 1:row
            Δp = p2 - p1
            corr = mean(plaq_vals[p1, c] * plaq_vals[p2, c + Δc] for c in 1:(n_plaq_cols - Δc))
            SP += 2.0 * cos(qx * Δc + qy * Δp) * (corr - μ[p1] * μ[p2])
        end
    end

    N_p = row * (2 * max_sep + 1)
    return SP / N_p
end

# =============================================================================
# Section 5: ACF (unchanged)
# =============================================================================

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
# Section 6: Circuit Resampling and Gate Reconstruction
# =============================================================================

"""
    resample_circuit(filename::String; conv_step=1000, samples=100000, measure_first=nothing)

Extract final parameters from a saved result and re-run the circuit to generate new samples.

# Arguments
- `filename`: Path to JSON result file containing CircuitOptimizationResult
- `conv_step`: Number of convergence steps before sampling (default: 1000)
- `samples`: Number of samples to collect (default: 100000)
- `measure_first`: Which observable to measure first, :X or :Z (default: use value from saved result)

# Returns
- Tuple of (rho, Z_samples, X_samples, params, gates) where:
  - `rho`: Final quantum state
  - `Z_samples`: Vector of Z measurement outcomes
  - `X_samples`: Vector of X measurement outcomes
  - `params`: Parameters used (from the saved result)
  - `gates`: Gates reconstructed from parameters

# Example
```julia
rho, Z_samples, X_samples, params, gates = resample_circuit("results/circuit_J=1.0_g=2.0_row=6.json"; samples=50000)
```
"""
function resample_circuit(filename::String; conv_step=100, samples=1000000, measure_first=nothing, measure_y=false)
    result, input_args = load_result(filename)

    if !(result isa CircuitOptimizationResult)
        @warn "Result is not CircuitOptimizationResult, cannot resample"
        return nothing
    end

    # Extract parameters from result
    params = result.final_params

    # Extract circuit configuration from input_args
    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]
    share_params = get(input_args, :share_params, true)

    # Use measure_first from result if not specified
    if isnothing(measure_first)
        measure_first = Symbol(get(input_args, :measure_first, "Z"))
    end

    println("=== Resampling Circuit ===")
    println("File: ", basename(filename))
    println("Parameters: $(length(params)) params")
    println("Configuration: p=$p, row=$row, nqubits=$nqubits")
    println("Share params: $share_params")
    println("Measure first: $measure_first")
    println("Conv steps: $conv_step, Samples: $samples")

    # Detect unit cell type: 2x2 if param count matches 4*3*nqubits*p
    model_str = get(input_args, :model, "tfim")
    m = _construct_model(model_str, Dict{Symbol,Any}(k => v for (k,v) in input_args if k in (:J, :g, :J1, :J2)))
    is_two_by_two = (model_str == "heisenberg_j1j2") && (length(params) == 4 * PARAMS_PER_QUBIT_PER_LAYER * nqubits * p)

    # Reconstruct gates from parameters
    if is_two_by_two
        gates_odd, gates_even = build_unitary_gate_2x2(params, p, row, nqubits)
        println("Unit cell: 2x2 (gates_odd + gates_even)")
    else
        gates = build_unitary_gate(params, p, row, nqubits; share_params=share_params)
    end

    # Run the quantum channel to generate new samples
    println("\nGenerating new samples...")
    need_y = needs_y_measurement(m)
    if is_two_by_two
        channel_result = sample_quantum_channel(gates_odd, gates_even, row, nqubits;
                                                conv_step=conv_step,
                                                samples=samples,
                                                model=m)
    else
        channel_result = sample_quantum_channel(gates, row, nqubits;
                                                conv_step=conv_step,
                                                samples=samples,
                                                model=m)
    end

    if need_y
        rho, Z_samples, X_samples, Y_samples = channel_result
        println("Generated $(length(Z_samples)) Z, $(length(X_samples)) X, $(length(Y_samples)) Y samples")
        return rho, Z_samples, X_samples, Y_samples, params, (is_two_by_two ? (gates_odd, gates_even) : gates)
    else
        rho, Z_samples, X_samples = channel_result
        println("Generated $(length(Z_samples)) Z samples and $(length(X_samples)) X samples")
        return rho, Z_samples, X_samples, params, (is_two_by_two ? (gates_odd, gates_even) : gates)
    end
end

"""
    reconstruct_gates(filename::String; share_params=true, plot=true, save_plot=false)

Reconstruct gates from optimization result stored in JSON file and analyze transfer spectrum.

# Arguments
- `filename`: Path to JSON result file
- `share_params`: Share parameters across circuit layers (default: true)
- `plot`: Display eigenvalue spectrum plot (default: true)
- `save_plot`: Save plot to PDF file (default: false)

# Returns
- Tuple of (gates, rho, gap, eigenvalues)

# Example
```julia
# With visualization (default)
gates, rho, gap, eigenvalues = reconstruct_gates("result.json")

# Without visualization
gates, rho, gap, eigenvalues = reconstruct_gates("result.json"; plot=false)

# Save the plot
gates, rho, gap, eigenvalues = reconstruct_gates("result.json"; save_plot=true)
```
"""
function reconstruct_gates(filename::String; share_params=true, plot=true, save_plot=true, use_iterative=:auto, matrix_free=:auto)
    result, input_args = load_result(filename)

    p = input_args[:p]
    row = input_args[:row]
    nqubits = input_args[:nqubits]

    gates = build_unitary_gate(result.final_params, p, row, nqubits; share_params=share_params)

    # Compute transfer spectrum
    rho, gap, eigenvalues, eigenvalues_raw = compute_transfer_spectrum(gates, row, nqubits; use_iterative=use_iterative, matrix_free=matrix_free)

    println("=== Gate Analysis for $(basename(filename)) ===")
    println("Spectral gap: ", gap)
    println("Largest eigenvalue: ", maximum(abs.(eigenvalues)))
    println("Second largest eigenvalue: ", eigenvalues[2])
    println("Correlation length ξ: ", round(1/gap, digits=2))

    # Count eigenvalues near 1
    n_near_one = sum(eigenvalues .> 0.99)
    println("Eigenvalues > 0.99: $n_near_one / $(length(eigenvalues))")

    # Status indicator
    if gap > 0.1
        println("Status: ✓ Good spectral gap")
    elseif gap > 0.01
        println("Status: ⚠ Poor spectral gap")
    else
        println("Status: ✗ Very small spectral gap - optimization issue likely")
    end

    # Plot eigenvalue spectrum if requested
    if plot
        save_path = save_plot ? replace(filename, ".json" => "_eigenvalues.pdf") : nothing
        fig = plot_eigenvalue_spectrum(eigenvalues_raw;
                                        title=basename(filename),
                                        save_path=save_path,
                                        show_gap=true)
        display(fig)

        if save_plot
            println("Plot saved to: $save_path")
        end
    end

    return gates, rho, gap, eigenvalues
end
