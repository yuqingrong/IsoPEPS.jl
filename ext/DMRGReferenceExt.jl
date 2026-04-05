module DMRGReferenceExt

using IsoPEPS
using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using Serialization

"""
    column_major_2d_to_1d(Lx::Int, Ly::Int)

Map 2D lattice coordinates to 1D chain using column-major ordering.
Returns a dictionary mapping (i,j) -> site_index.

This ordering is optimal for DMRG on cylinders: the short direction (Ly)
is the inner loop, so vertical bonds are nearest-neighbor in the 1D chain
(distance 1) and horizontal bonds are short-range (distance Ly).

# Example for 3x3 lattice (Lx=3 columns, Ly=3 rows):
```
1  4  7
↓  ↓  ↓
2  5  8
↓  ↓  ↓
3  6  9
```
"""
function IsoPEPS.column_major_2d_to_1d(Lx::Int, Ly::Int)
    mapping = Dict{Tuple{Int,Int}, Int}()
    site = 1

    for i in 1:Lx         # columns (long direction) outer
        for j in 1:Ly      # rows (short direction) inner
            mapping[(i, j)] = site
            site += 1
        end
    end

    return mapping
end

"""
    build_2d_tfim_hamiltonian(Lx::Int, Ly::Int, J::Float64, g::Float64)

Build 2D Transverse Field Ising Model Hamiltonian mapped to 1D chain.

H = -J Σ_{⟨i,j⟩_2D} Z_i Z_j - g Σ_i X_i
"""
function IsoPEPS.build_2d_tfim_hamiltonian(Lx::Int, Ly::Int, J::Float64, g::Float64)
    N = Lx * Ly
    sites = siteinds("S=1", N)

    coord_to_site = IsoPEPS.column_major_2d_to_1d(Lx, Ly)

    os = OpSum()

    # Transverse field term: -g Σ_i X_i
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site = coord_to_site[(i, j)]
            os += -g, "Sx", site
        end
    end

    # Ising coupling: -J Σ_{⟨i,j⟩} Z_i Z_j
    # Horizontal bonds
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i+1, j)]
            os += -J, "Sz", site1, "Sz", site2
        end
    end

    # Vertical bonds
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i, j+1)]
            os += -J, "Sz", site1, "Sz", site2
        end
    end

    H = MPO(os, sites)
    return H, sites
end

"""
    build_2d_heisenberg_j1j2_hamiltonian(Lx::Int, Ly::Int, J1::Float64, J2::Float64)

Build 2D Heisenberg J1-J2 Hamiltonian on a cylinder (periodic in y, open in x).

H = J1 Σ_{⟨i,j⟩} S_i · S_j + J2 Σ_{⟨⟨i,j⟩⟩} S_i · S_j
"""
function IsoPEPS.build_2d_heisenberg_j1j2_hamiltonian(Lx::Int, Ly::Int, J1::Float64, J2::Float64)
    N = Lx * Ly
    sites = siteinds("S=1/2", N)

    coord_to_site = IsoPEPS.column_major_2d_to_1d(Lx, Ly)

    os = OpSum()

    # J1: Nearest-neighbor Heisenberg coupling
    # Horizontal bonds (open in x)
    for j in 1:Ly
        for i in 1:(Lx-1)
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i+1, j)]
            os += J1 * 0.5, "S+", site1, "S-", site2
            os += J1 * 0.5, "S-", site1, "S+", site2
            os += J1,       "Sz", site1, "Sz", site2
        end
    end

    # Vertical bonds (periodic in y)
    for j in 1:Ly
        j_next = mod1(j + 1, Ly)
        for i in 1:Lx
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i, j_next)]
            os += J1 * 0.5, "S+", site1, "S-", site2
            os += J1 * 0.5, "S-", site1, "S+", site2
            os += J1,       "Sz", site1, "Sz", site2
        end
    end

    # J2: Next-nearest-neighbor (diagonal) coupling (periodic in y, open in x)
    if J2 != 0.0
        for j in 1:Ly
            j_next = mod1(j + 1, Ly)
            for i in 1:(Lx-1)
                # Diagonal (i,j) -> (i+1,j_next)
                site1 = coord_to_site[(i, j)]
                site2 = coord_to_site[(i+1, j_next)]
                os += J2 * 0.5, "S+", site1, "S-", site2
                os += J2 * 0.5, "S-", site1, "S+", site2
                os += J2,       "Sz", site1, "Sz", site2

                # Anti-diagonal (i+1,j) -> (i,j_next)
                site1 = coord_to_site[(i+1, j)]
                site2 = coord_to_site[(i, j_next)]
                os += J2 * 0.5, "S+", site1, "S-", site2
                os += J2 * 0.5, "S-", site1, "S+", site2
                os += J2,       "Sz", site1, "Sz", site2
            end
        end
    end

    H = MPO(os, sites)
    return H, sites
end

"""
    build_hamiltonian(model::String, Lx::Int, Ly::Int; kwargs...)

Build a 2D Hamiltonian for the specified model.

Supported models: `"tfim"`, `"heisenberg_j1j2"`.
"""
function IsoPEPS.build_hamiltonian(model::String, Lx::Int, Ly::Int; kwargs...)
    kw = Dict(kwargs)
    if model == "tfim"
        J = get(kw, :J, 1.0)
        g = get(kw, :g, 1.0)
        return IsoPEPS.build_2d_tfim_hamiltonian(Lx, Ly, Float64(J), Float64(g))
    elseif model == "heisenberg_j1j2"
        J1 = get(kw, :J1, 1.0)
        J2 = get(kw, :J2, 0.0)
        return IsoPEPS.build_2d_heisenberg_j1j2_hamiltonian(Lx, Ly, Float64(J1), Float64(J2))
    else
        error("Unknown model: \"$model\". Supported models: \"tfim\", \"heisenberg_j1j2\"")
    end
end

"""
    dmrg_ground_state_2d(Lx, Ly; model, maxdim, cutoff, nsweeps, noise, model_params...)

Compute 2D ground state using DMRG for a chosen model.
"""
function IsoPEPS.dmrg_ground_state_2d(Lx::Int, Ly::Int; model::String="tfim",
                              maxdim::Int=100, cutoff::Float64=1e-10,
                              nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps),
                              psi0::Union{MPS,Nothing}=nothing,
                              model_params...)

    println("Building 2D $model Hamiltonian (Lx=$Lx, Ly=$Ly, $(join(["$k=$v" for (k,v) in model_params], ", ")))")
    H, sites = IsoPEPS.build_hamiltonian(model, Lx, Ly; model_params...)

    N = Lx * Ly
    if isnothing(psi0)
        psi0 = randomMPS(sites, linkdims=10)
    else
        for j in 1:N
            old_s = siteind(psi0, j)
            new_s = sites[j]
            if old_s != new_s
                psi0[j] = replaceind(psi0[j], old_s, new_s)
            end
        end
    end

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim)
    setcutoff!(sweeps, cutoff)
    setnoise!(sweeps, noise...)

    println("Running DMRG...")
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=1)

    energy_per_site = energy / N

    println("Ground state energy: $energy")
    println("Energy per site: $energy_per_site")

    return (energy=energy, energy_per_site=energy_per_site,
            psi=psi, sites=sites, H=H, Lx=Lx, Ly=Ly, model=model)
end

# Backward-compatible convenience method for TFIM
function IsoPEPS.dmrg_ground_state_2d(Lx::Int, Ly::Int, J::Float64, g::Float64;
                              maxdim::Int=100, cutoff::Float64=1e-10,
                              nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps))
    return IsoPEPS.dmrg_ground_state_2d(Lx, Ly; model="tfim", J=J, g=g,
                                maxdim=maxdim, cutoff=cutoff,
                                nsweeps=nsweeps, noise=noise)
end

"""
    compute_magnetization(result)

Compute average magnetization ⟨Sx⟩ and ⟨Sz⟩.
"""
function IsoPEPS.compute_magnetization(result)
    psi = result.psi
    sites = result.sites
    N = length(sites)

    Sx_avg = sum(ITensorMPS.expect(psi, "Sx")) / N
    Sz_avg = sum(ITensorMPS.expect(psi, "Sz")) / N

    return (Sx=Sx_avg, Sz=Sz_avg)
end

"""
    compute_correlation_length_dmrg(result; max_distance::Int=60)

Estimate correlation length from connected correlation function decay.
Uses C_connected(r) = ⟨Sz(i) Sz(i+r)⟩ - ⟨Sz(i)⟩⟨Sz(i+r)⟩
"""
function IsoPEPS.compute_correlation_length_dmrg(result; max_distance::Int=60)
    psi = result.psi
    sites = result.sites
    N = length(sites)

    max_dist = min(max_distance, N ÷ 6)
    n_avg = N - 3 * max_dist

    Sz_expect = ITensorMPS.expect(psi, "Sz")
    C = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)

    C_connected = zeros(N, N)
    for i in 1:N
        for j in 1:N
            C_connected[i,j] = real(C[i,j]) - Sz_expect[i] * Sz_expect[j]
        end
    end

    correlations = zeros(max_dist)
    for i in 1:n_avg
        for r in 1:max_dist
            correlations[r] += C_connected[i, i + 3*r]
        end
    end
    correlations ./= n_avg

    distances = collect(1:max_dist)

    valid_idx = abs.(correlations) .> 1e-12
    if sum(valid_idx) < 2
        return (ξ=1e6, correlations=correlations, distances=distances)
    end

    log_corr = log.(abs.(correlations[valid_idx]))
    valid_distances = distances[valid_idx]

    A = hcat(ones(length(valid_distances)), -valid_distances)
    coeffs = A \ log_corr
    ξ = 1.0 / coeffs[2]

    ξ = abs(ξ)
    ξ = clamp(ξ, 0.01, 1e6)

    return (ξ=ξ, correlations=correlations, distances=distances)
end

"""
    site_to_2d(Lx, Ly)

Inverse of column_major_2d_to_1d: maps 1D site index to (column, row) coordinates.
"""
function site_to_2d(Lx::Int, Ly::Int)
    coords = Vector{Tuple{Int,Int}}(undef, Lx * Ly)
    for s in 1:(Lx * Ly)
        i = div(s - 1, Ly) + 1   # column (1-indexed)
        j = mod(s - 1, Ly) + 1   # row (1-indexed)
        coords[s] = (i, j)
    end
    return coords
end

"""
    compute_spin_spin_correlation_dmrg(result; connected=false)

Compute ⟨S_i · S_j⟩ = ⟨Sx_i Sx_j⟩ + ⟨Sy_i Sy_j⟩ + ⟨Sz_i Sz_j⟩ for the DMRG ground state.

Returns a NamedTuple with:
- `distances`: sorted unique 2D distances
- `correlations`: distance-averaged ⟨S_i · S_j⟩ values
- `correlation_matrix_full`: full N×N correlation matrix
"""
function IsoPEPS.compute_spin_spin_correlation_dmrg(result; connected::Bool=false)
    psi = result.psi
    Lx = result.Lx
    Ly = result.Ly
    N = Lx * Ly

    # ⟨Sx_i Sx_j⟩ + ⟨Sy_i Sy_j⟩ = (⟨S+_i S-_j⟩ + ⟨S-_i S+_j⟩) / 2
    # ishermitian=false avoids an internal Float64 adapt that fails with complex MPS tensors
    Cpm = correlation_matrix(psi, "S+", "S-"; ishermitian=false)
    Cmp = correlation_matrix(psi, "S-", "S+"; ishermitian=false)
    Czz = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)

    SdotS = real.(Cpm .+ Cmp) ./ 2 .+ real.(Czz)

    if connected
        Sx_exp = ITensorMPS.expect(psi, "Sx")
        Sy_exp = ITensorMPS.expect(psi, "Sy")
        Sz_exp = ITensorMPS.expect(psi, "Sz")
        for i in 1:N, j in 1:N
            SdotS[i, j] -= Sx_exp[i] * Sx_exp[j] + Sy_exp[i] * Sy_exp[j] + Sz_exp[i] * Sz_exp[j]
        end
    end

    # Map to 2D distances and average
    coords = site_to_2d(Lx, Ly)
    dist_corr = Dict{Float64, Vector{Float64}}()

    for s1 in 1:N, s2 in (s1+1):N
        i1, j1 = coords[s1]
        i2, j2 = coords[s2]
        dx = i2 - i1
        dy_raw = abs(j2 - j1)
        dy = min(dy_raw, Ly - dy_raw)  # periodic y
        d = sqrt(dx^2 + dy^2)
        d_key = round(d, digits=6)
        if !haskey(dist_corr, d_key)
            dist_corr[d_key] = Float64[]
        end
        push!(dist_corr[d_key], SdotS[s1, s2])
    end

    sorted_dists = sort(collect(keys(dist_corr)))
    avg_corr = [Statistics.mean(dist_corr[d]) for d in sorted_dists]

    return (distances=sorted_dists, correlations=avg_corr, correlation_matrix_full=SdotS)
end

"""
    compute_SdotS_matrix(result)

Precompute the full ⟨S_i · S_j⟩ correlation matrix from a DMRG result.
Returns a real N×N matrix where N = Lx * Ly.
"""
function IsoPEPS.compute_SdotS_matrix(result)
    psi = result.psi
    Cpm = correlation_matrix(psi, "S+", "S-"; ishermitian=false)
    Cmp = correlation_matrix(psi, "S-", "S+"; ishermitian=false)
    Czz = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)
    return real.(Cpm .+ Cmp) ./ 2 .+ real.(Czz)
end

"""
    compute_structure_factor_dmrg(result, q::Tuple{Real,Real}; bulk_fraction=0.5)

Compute the spin structure factor S(q) = (1/N_bulk) Σ_{i,j ∈ bulk} ⟨S_i · S_j⟩ exp(iq·(r_i - r_j)).

Only sites in the middle `bulk_fraction` of the cylinder (in x) are used,
avoiding open-boundary artifacts.  Set `bulk_fraction=1.0` to recover the
full-system sum.
"""
function IsoPEPS.compute_structure_factor_dmrg(result, q::Tuple{Real,Real};
                                               bulk_fraction::Float64=0.5)
    psi = result.psi
    Lx = result.Lx
    Ly = result.Ly
    N = Lx * Ly

    # Bulk column range (middle portion)
    margin = round(Int, Lx * (1 - bulk_fraction) / 2)
    col_lo = max(1, margin + 1)
    col_hi = min(Lx, Lx - margin)

    # ishermitian=false avoids an internal Float64 adapt that fails with complex MPS tensors
    Cpm = correlation_matrix(psi, "S+", "S-"; ishermitian=false)
    Cmp = correlation_matrix(psi, "S-", "S+"; ishermitian=false)
    Czz = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)
    SdotS = real.(Cpm .+ Cmp) ./ 2 .+ real.(Czz)

    coords = site_to_2d(Lx, Ly)
    qx, qy = q

    # Build set of bulk 1D site indices
    bulk_sites = Int[]
    for s in 1:N
        col, _ = coords[s]
        if col_lo <= col <= col_hi
            push!(bulk_sites, s)
        end
    end
    N_bulk = length(bulk_sites)

    Sq = 0.0
    for s1 in bulk_sites, s2 in bulk_sites
        i1, j1 = coords[s1]
        i2, j2 = coords[s2]
        dx = i2 - i1
        dy = j2 - j1  # literal displacement; fine for commensurate q on periodic y
        phase = cos(qx * dx + qy * dy)
        Sq += SdotS[s1, s2] * phase
    end

    return Sq / N_bulk
end

"""
    compute_M2_dmrg(result, q::Tuple{Real,Real}; bulk_fraction=0.5, max_separation=20)

Compute the squared magnetic order parameter M²(q) = (1/N_pairs) Σ_{i,j} ⟨S_i·S_j⟩ exp(iq·(r_i-r_j)).

Only includes pairs with |Δx| ≤ `max_separation` columns, matching the
truncation used by the IsoPEPS exact/sampling structure factor.
Only the middle `bulk_fraction` of the cylinder is used (default 0.5).

Common q values:
- (π, π): Neel antiferromagnetic order
- (π, 0) or (0, π): Stripe order
- (0, 0): Ferromagnetic order
"""
function IsoPEPS.compute_M2_dmrg(result, q::Tuple{Real,Real};
                                 bulk_fraction::Float64=0.5,
                                 max_separation::Int=20)
    psi = result.psi
    Lx = result.Lx
    Ly = result.Ly
    N = Lx * Ly

    margin = round(Int, Lx * (1 - bulk_fraction) / 2)
    col_lo = max(1, margin + 1)
    col_hi = min(Lx, Lx - margin)

    Cpm = correlation_matrix(psi, "S+", "S-"; ishermitian=false)
    Cmp = correlation_matrix(psi, "S-", "S+"; ishermitian=false)
    Czz = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)
    SdotS = real.(Cpm .+ Cmp) ./ 2 .+ real.(Czz)

    coords = site_to_2d(Lx, Ly)
    qx, qy = q

    bulk_sites = Int[]
    for s in 1:N
        col, _ = coords[s]
        if col_lo <= col <= col_hi
            push!(bulk_sites, s)
        end
    end

    Sq = 0.0
    n_pairs = 0
    for s1 in bulk_sites, s2 in bulk_sites
        i1, j1 = coords[s1]
        i2, j2 = coords[s2]
        dx = i2 - i1
        abs(dx) > max_separation && continue
        dy = j2 - j1
        Sq += SdotS[s1, s2] * cos(qx * dx + qy * dy)
        n_pairs += 1
    end

    return Sq / n_pairs
end

"""
    save_dmrg_state(result, filename::String; model_params...)

Save DMRG ground state to a .jls file (Julia serialization) with JSON metadata sidecar.
"""
function IsoPEPS.save_dmrg_state(result, filename::String; model_params...)
    mkpath(dirname(abspath(filename)))

    # Save MPS using Julia serialization
    open(filename, "w") do io
        serialize(io, result.psi)
    end

    # Save metadata as JSON sidecar
    meta = Dict{String,Any}(
        "energy" => result.energy,
        "energy_per_site" => result.energy_per_site,
        "Lx" => result.Lx,
        "Ly" => result.Ly,
        "model" => result.model,
    )
    for (k, v) in model_params
        meta[string(k)] = v
    end

    meta_file = filename * ".json"
    open(meta_file, "w") do io
        print(io, "{\n")
        sorted_keys = sort(collect(keys(meta)))
        for (idx, k) in enumerate(sorted_keys)
            v = meta[k]
            if v isa AbstractString
                print(io, "  \"$k\": \"$v\"")
            else
                print(io, "  \"$k\": $v")
            end
            idx < length(sorted_keys) && print(io, ",")
            print(io, "\n")
        end
        print(io, "}")
    end

    println("DMRG state saved to $filename (+ $meta_file)")
    return filename
end

"""
    load_dmrg_state(filename::String)

Load DMRG ground state from a .jls file with JSON metadata sidecar.

Returns a NamedTuple compatible with `dmrg_ground_state_2d` output.
The Hamiltonian H is rebuilt from the saved model parameters.
"""
function IsoPEPS.load_dmrg_state(filename::String)
    # Load MPS
    psi = open(filename, "r") do io
        deserialize(io)
    end
    sites = siteinds(psi)

    # Load metadata from JSON sidecar
    meta_file = filename * ".json"
    meta_str = read(meta_file, String)
    meta = Dict{String,Any}()
    for m in eachmatch(r"\"(\w+)\":\s*(?:\"([^\"]*)\"|([^\s,}]+))", meta_str)
        key = m.captures[1]
        val = m.captures[2] !== nothing ? m.captures[2] : m.captures[3]
        if val !== nothing
            parsed = tryparse(Float64, val)
            if parsed !== nothing
                int_parsed = tryparse(Int, val)
                meta[key] = int_parsed !== nothing ? int_parsed : parsed
            else
                meta[key] = val
            end
        end
    end

    Lx = meta["Lx"]
    Ly = meta["Ly"]
    model = get(meta, "model", "unknown")
    energy = get(meta, "energy", NaN)
    energy_per_site = get(meta, "energy_per_site", NaN)

    # Rebuild Hamiltonian
    model_kw = Dict{Symbol,Any}()
    for (k, v) in meta
        k in ("energy", "energy_per_site", "Lx", "Ly", "model") && continue
        model_kw[Symbol(k)] = v
    end

    H = nothing
    try
        H, _ = IsoPEPS.build_hamiltonian(model, Lx, Ly; model_kw...)
    catch e
        @warn "Could not rebuild Hamiltonian: $e"
    end

    println("DMRG state loaded from $filename")

    return (energy=energy, energy_per_site=energy_per_site,
            psi=psi, sites=sites, H=H, Lx=Lx, Ly=Ly, model=model)
end

# =============================================================================
# Helpers for dimer/plaquette structure factors
# =============================================================================

"""
    _bulk_col_range(Lx, bulk_cols)

Return the column range (lo, hi) for the middle `bulk_cols` columns of a cylinder
with `Lx` columns. Clamps to [1, Lx].
"""
function _bulk_col_range(Lx::Int, bulk_cols::Int)
    bulk_cols = min(bulk_cols, Lx)
    margin = div(Lx - bulk_cols, 2)
    col_lo = max(1, margin + 1)
    col_hi = min(Lx, col_lo + bulk_cols - 1)
    return col_lo, col_hi
end

"""
    _bond_list(Lx, Ly, orientation, col_lo, col_hi)

Return a vector of `(site_i, site_j, col, row)` tuples for all bonds of the
given orientation within columns `col_lo:col_hi`.

Bond center coordinates: vertical at (col, row+0.5), horizontal at (col+0.5, row).
"""
function _bond_list(Lx::Int, Ly::Int, orientation::Symbol, col_lo::Int, col_hi::Int)
    bonds = Tuple{Int,Int,Int,Int}[]  # (site_i, site_j, col, row)
    if orientation == :vertical
        for col in col_lo:col_hi, row in 1:Ly
            row2 = row % Ly + 1
            si = (col - 1) * Ly + row
            sj = (col - 1) * Ly + row2
            push!(bonds, (si, sj, col, row))
        end
    elseif orientation == :horizontal
        for col in col_lo:(col_hi - 1), row in 1:Ly
            si = (col - 1) * Ly + row
            sj = col * Ly + row
            push!(bonds, (si, sj, col, row))
        end
    else
        error("orientation must be :vertical or :horizontal, got $orientation")
    end
    return bonds
end

"""
    _four_point_inner(psi, sites, op_pairs)

Compute ⟨ψ| O₁(s₁) O₂(s₂) O₃(s₃) O₄(s₄) |ψ⟩ using ITensors inner().

`op_pairs` is a vector of (operator_name::String, site_index::Int) tuples,
sorted by site index (ascending). All site indices must be distinct.
"""
function _four_point_inner(psi, sites, op_pairs::Vector{Tuple{String,Int}})
    # Sort by site index
    sorted_ops = sort(op_pairs, by=x -> x[2])

    # Build the operator product as an MPO-like object
    psi_copy = copy(psi)
    for (op_name, si) in sorted_ops
        O = ITensors.op(op_name, sites[si])
        psi_copy[si] = noprime(O * psi_copy[si])
    end
    return real(inner(psi, psi_copy))
end

"""
    _compute_dimer_expectations(psi, sites, Lx, Ly, bonds)

Compute ⟨D_b⟩ = Σ_α ⟨σ^α_i σ^α_j⟩ / 4 for each bond using correlation_matrix.
Returns a vector of Float64 dimer expectations, one per bond.
"""
function _compute_dimer_expectations(psi, sites, Lx::Int, Ly::Int,
                                      bonds::Vector{Tuple{Int,Int,Int,Int}})
    # Get full correlation matrices
    Cpm = correlation_matrix(psi, "S+", "S-"; ishermitian=false)
    Cmp = correlation_matrix(psi, "S-", "S+"; ishermitian=false)
    Czz = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)
    # S·S = (S+S- + S-S+)/2 + SzSz
    SdotS = real.(Cpm .+ Cmp) ./ 2 .+ real.(Czz)

    dimer_exp = Float64[]
    for (si, sj, _, _) in bonds
        push!(dimer_exp, SdotS[si, sj])
    end
    return dimer_exp
end

"""
    _compute_dimer_dimer_matrix(psi, sites, bonds)

Compute the full ⟨D_b D_b'⟩ matrix for all bond pairs using 4-point MPS contractions.
Returns a symmetric Matrix{Float64} of size (n_bonds, n_bonds).

D_b D_b' = (Σ_α S^α_i S^α_j)(Σ_β S^β_k S^β_l)  [ITensors S ops = σ/2]
"""
function _compute_dimer_dimer_matrix(psi, sites,
                                      bonds::Vector{Tuple{Int,Int,Int,Int}})
    n = length(bonds)
    DD = zeros(Float64, n, n)
    pauli_ops = ["Sx", "Sy", "Sz"]
    total_pairs = div(n * (n + 1), 2)
    count = 0

    for bi in 1:n, bj in bi:n
        si1, sj1, _, _ = bonds[bi]
        si2, sj2, _, _ = bonds[bj]

        val = 0.0
        for α in pauli_ops, β in pauli_ops
            # Sites involved: si1, sj1, si2, sj2
            # They may overlap (e.g., same bond or adjacent bonds sharing a site)
            all_sites = sort(unique([si1, sj1, si2, sj2]))

            if length(all_sites) == 4
                # All 4 sites distinct: standard 4-point
                ops = [(α, si1), (α, sj1), (β, si2), (β, sj2)]
                val += _four_point_inner(psi, sites, ops)
            elseif length(all_sites) == 3
                # One site shared: 3-point with combined operator on shared site
                # Build operator assignments per site
                site_ops = Dict{Int, Vector{String}}()
                for (op_name, s) in [(α, si1), (α, sj1), (β, si2), (β, sj2)]
                    if !haskey(site_ops, s)
                        site_ops[s] = String[]
                    end
                    push!(site_ops[s], op_name)
                end
                # For the shared site, multiply operators
                psi_copy = copy(psi)
                for s in sort(collect(keys(site_ops)))
                    ops_at_s = site_ops[s]
                    if length(ops_at_s) == 1
                        O = ITensors.op(ops_at_s[1], sites[s])
                    else
                        O = ITensors.op(ops_at_s[1], sites[s]) * ITensors.op(ops_at_s[2], sites[s])
                        O = mapprime(O, 2 => 1)
                    end
                    psi_copy[s] = noprime(O * psi_copy[s])
                end
                val += real(inner(psi, psi_copy))
            elseif length(all_sites) == 2
                # Same bond or reversed: 2-point with squared operators
                site_ops = Dict{Int, Vector{String}}()
                for (op_name, s) in [(α, si1), (α, sj1), (β, si2), (β, sj2)]
                    if !haskey(site_ops, s)
                        site_ops[s] = String[]
                    end
                    push!(site_ops[s], op_name)
                end
                psi_copy = copy(psi)
                for s in sort(collect(keys(site_ops)))
                    ops_at_s = site_ops[s]
                    if length(ops_at_s) == 1
                        O = ITensors.op(ops_at_s[1], sites[s])
                    else
                        O = ITensors.op(ops_at_s[1], sites[s]) * ITensors.op(ops_at_s[2], sites[s])
                        O = mapprime(O, 2 => 1)
                    end
                    psi_copy[s] = noprime(O * psi_copy[s])
                end
                val += real(inner(psi, psi_copy))
            end
        end
        DD[bi, bj] = val
        DD[bj, bi] = DD[bi, bj]

        count += 1
        if count % 1000 == 0
            print("\r  bond pairs: $count/$total_pairs")
        end
    end
    if total_pairs >= 1000
        println()
    end
    return DD
end

# =============================================================================
# Dimer structure factor (DMRG)
# =============================================================================

"""
    compute_dimer_structure_factor_dmrg(result, q; bulk_cols=20, dimer_orientation=:vertical)

Connected dimer structure factor from DMRG ground state using 4-point MPS contractions.

S_D(q) = (1/N_b) Σ_{b,b'} [⟨D_b D_b'⟩ - ⟨D_b⟩⟨D_b'⟩] cos(q·(r_b - r_b'))

Only bonds in `bulk_cols` columns centered in the cylinder are included.
"""
function IsoPEPS.compute_dimer_structure_factor_dmrg(result, q::Tuple{Real,Real};
                                                      bulk_cols::Int=20,
                                                      dimer_orientation::Symbol=:vertical)
    psi = result.psi
    Lx = result.Lx
    Ly = result.Ly
    sites = result.sites

    col_lo, col_hi = _bulk_col_range(Lx, bulk_cols)
    bonds = _bond_list(Lx, Ly, dimer_orientation, col_lo, col_hi)
    n = length(bonds)

    println("Computing dimer structure factor: $(length(bonds)) $dimer_orientation bonds in cols $col_lo:$col_hi")

    # Dimer expectations ⟨D_b⟩
    D_exp = _compute_dimer_expectations(psi, sites, Lx, Ly, bonds)

    # Dimer-dimer matrix ⟨D_b D_b'⟩
    DD = _compute_dimer_dimer_matrix(psi, sites, bonds)

    # Connected correlations and Fourier transform
    qx, qy = Float64(q[1]), Float64(q[2])
    SD = 0.0
    for bi in 1:n, bj in 1:n
        _, _, col_i, row_i = bonds[bi]
        _, _, col_j, row_j = bonds[bj]
        if dimer_orientation == :vertical
            dx = col_j - col_i
            dy = (row_j + 0.5) - (row_i + 0.5)  # bond center offset
        else
            dx = (col_j + 0.5) - (col_i + 0.5)
            dy = row_j - row_i
        end
        C_conn = DD[bi, bj] - D_exp[bi] * D_exp[bj]
        SD += C_conn * cos(qx * dx + qy * dy)
    end
    return SD / n
end

# =============================================================================
# Plaquette structure factor (DMRG) — disconnected approximation
# =============================================================================

"""
    compute_plaquette_structure_factor_dmrg(result, q; bulk_cols=20)

Disconnected plaquette structure factor from DMRG ground state.

Q_□ = Σ_{bonds in □} S_i · S_j / 4 (sum of 4 bond operators around a plaquette).

Uses 2-point correlators only (disconnected approximation):
S_P(q) = (1/N_p) Σ_{□,□'} [⟨Q_□⟩⟨Q_□'⟩ - μ_Q²] cos(q·Δr)
"""
function IsoPEPS.compute_plaquette_structure_factor_dmrg(result, q::Tuple{Real,Real};
                                                          bulk_cols::Int=20)
    psi = result.psi
    Lx = result.Lx
    Ly = result.Ly
    sites = result.sites

    col_lo, col_hi = _bulk_col_range(Lx, bulk_cols)

    # Get S·S correlation matrix
    Cpm = correlation_matrix(psi, "S+", "S-"; ishermitian=false)
    Cmp = correlation_matrix(psi, "S-", "S+"; ishermitian=false)
    Czz = correlation_matrix(psi, "Sz", "Sz"; ishermitian=false)
    SdotS = real.(Cpm .+ Cmp) ./ 2 .+ real.(Czz)

    # Compute plaquette expectation for each plaquette in bulk
    # Plaquette at (col, row) has corners: (col,row), (col,row2), (col+1,row2), (col+1,row)
    plaquettes = Tuple{Float64,Int,Int}[]  # (Q_value, col, row)
    for col in col_lo:(col_hi - 1), row in 1:Ly
        row2 = row % Ly + 1
        tl = (col - 1) * Ly + row      # top-left
        bl = (col - 1) * Ly + row2     # bottom-left
        tr = col * Ly + row             # top-right
        br = col * Ly + row2            # bottom-right
        Q = (SdotS[tl, bl] + SdotS[bl, br] + SdotS[br, tr] + SdotS[tr, tl]) / 4.0
        push!(plaquettes, (Q, col, row))
    end

    n = length(plaquettes)
    if n == 0
        return 0.0
    end

    μ_Q = Statistics.mean(p[1] for p in plaquettes)
    qx, qy = Float64(q[1]), Float64(q[2])

    SP = 0.0
    for pi in 1:n, pj in 1:n
        Q_i, col_i, row_i = plaquettes[pi]
        Q_j, col_j, row_j = plaquettes[pj]
        dx = (col_j + 0.5) - (col_i + 0.5)
        dy = (row_j + 0.5) - (row_i + 0.5)
        SP += (Q_i * Q_j - μ_Q^2) * cos(qx * dx + qy * dy)
    end
    return SP / n
end

# =============================================================================
# Dimer-dimer correlation (DMRG)
# =============================================================================

"""
    compute_dimer_dimer_correlation_dmrg(result; dimer_orientation=:vertical, bulk_cols=20)

Compute dimer-dimer correlations from DMRG ground state.

`dimer_orientation` can be `:vertical`, `:horizontal`, or `:both`.

Returns a NamedTuple:
- `distances`: sorted unique 2D distances between bond centers
- `correlations`: distance-averaged connected dimer-dimer correlations
- `dimer_expectations`: ⟨D_b⟩ for each bond
- `bonds`: list of (site_i, site_j, col, row) bond tuples
- `DD_matrix`: full ⟨D_b D_b'⟩ matrix
- `orientations`: Vector{Symbol} indicating each bond's orientation (only present when `dimer_orientation=:both`)
"""
function IsoPEPS.compute_dimer_dimer_correlation_dmrg(result;
                                                       dimer_orientation::Symbol=:vertical,
                                                       bulk_cols::Int=20)
    psi = result.psi
    Lx = result.Lx
    Ly = result.Ly
    sites = result.sites

    col_lo, col_hi = _bulk_col_range(Lx, bulk_cols)

    if dimer_orientation == :both
        bonds_v = _bond_list(Lx, Ly, :vertical, col_lo, col_hi)
        bonds_h = _bond_list(Lx, Ly, :horizontal, col_lo, col_hi)
        bonds = vcat(bonds_v, bonds_h)
        orientations = vcat(fill(:vertical, length(bonds_v)),
                            fill(:horizontal, length(bonds_h)))
    else
        bonds = _bond_list(Lx, Ly, dimer_orientation, col_lo, col_hi)
        orientations = fill(dimer_orientation, length(bonds))
    end
    n = length(bonds)

    println("Computing dimer-dimer correlations: $n bonds ($dimer_orientation) in cols $col_lo:$col_hi")

    D_exp = _compute_dimer_expectations(psi, sites, Lx, Ly, bonds)
    DD = _compute_dimer_dimer_matrix(psi, sites, bonds)

    # Map to distances and average
    dist_corr = Dict{Float64, Vector{Float64}}()
    for bi in 1:n, bj in (bi + 1):n
        _, _, col_i, row_i = bonds[bi]
        _, _, col_j, row_j = bonds[bj]
        dx = Float64(col_j - col_i)
        dy = Float64(row_j - row_i)
        # Periodic y distance
        dy_abs = abs(dy)
        dy_min = min(dy_abs, Ly - dy_abs)
        d = sqrt(dx^2 + dy_min^2)
        d_key = round(d, digits=6)

        C_conn = DD[bi, bj] - D_exp[bi] * D_exp[bj]
        if !haskey(dist_corr, d_key)
            dist_corr[d_key] = Float64[]
        end
        push!(dist_corr[d_key], C_conn)
    end

    sorted_dists = sort(collect(keys(dist_corr)))
    avg_corr = [Statistics.mean(dist_corr[d]) for d in sorted_dists]

    return (distances=sorted_dists, correlations=avg_corr,
            dimer_expectations=D_exp, bonds=bonds, DD_matrix=DD,
            orientations=orientations)
end

end # module DMRGReferenceExt
