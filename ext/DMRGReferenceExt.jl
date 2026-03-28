module DMRGReferenceExt

using IsoPEPS
using ITensors
using ITensorMPS
using LinearAlgebra

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
                              model_params...)

    println("Building 2D $model Hamiltonian (Lx=$Lx, Ly=$Ly, $(join(["$k=$v" for (k,v) in model_params], ", ")))")
    H, sites = IsoPEPS.build_hamiltonian(model, Lx, Ly; model_params...)

    N = Lx * Ly
    psi0 = randomMPS(sites, linkdims=10)

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
    C = correlation_matrix(psi, "Sz", "Sz")

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

end # module DMRGReferenceExt
