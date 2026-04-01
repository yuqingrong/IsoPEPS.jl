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
    compute_M2_dmrg(result, q::Tuple{Real,Real}; bulk_fraction=0.5)

Compute the squared magnetic order parameter M²(q) = S(q)/N_bulk
= (1/N_bulk²) Σ_{i,j ∈ bulk} ⟨S_i·S_j⟩ exp(iq·(r_i-r_j)).

Only the middle `bulk_fraction` of the cylinder is used (default 0.5).

Common q values:
- (π, π): Neel antiferromagnetic order
- (π, 0) or (0, π): Stripe order
- (0, 0): Ferromagnetic order
"""
function IsoPEPS.compute_M2_dmrg(result, q::Tuple{Real,Real};
                                 bulk_fraction::Float64=0.5)
    Lx = result.Lx
    Ly = result.Ly
    margin = round(Int, Lx * (1 - bulk_fraction) / 2)
    col_lo = max(1, margin + 1)
    col_hi = min(Lx, Lx - margin)
    N_bulk = (col_hi - col_lo + 1) * Ly
    Sq = IsoPEPS.compute_structure_factor_dmrg(result, q; bulk_fraction=bulk_fraction)
    return Sq / N_bulk
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

end # module DMRGReferenceExt
