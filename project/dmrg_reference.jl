using ITensors
using ITensorMPS
using JSON3
using CairoMakie

"""
    snake_order_2d_to_1d(Lx::Int, Ly::Int)

Map 2D lattice coordinates to 1D chain using snake ordering.
Returns a dictionary mapping (i,j) -> site_index.

# Example for 3x3 lattice:
```
1 → 2 → 3
    ↓
6 ← 5 ← 4
↓
7 → 8 → 9
```
"""
function snake_order_2d_to_1d(Lx::Int, Ly::Int)
    mapping = Dict{Tuple{Int,Int}, Int}()
    site = 1

    for j in 1:Ly
        if j % 2 == 1  # Odd rows: left to right
            for i in 1:Lx
                mapping[(i, j)] = site
                site += 1
            end
        else  # Even rows: right to left
            for i in Lx:-1:1
                mapping[(i, j)] = site
                site += 1
            end
        end
    end

    return mapping
end

"""
    build_2d_tfim_hamiltonian(Lx::Int, Ly::Int, J::Float64, g::Float64)

Build 2D Transverse Field Ising Model Hamiltonian mapped to 1D chain.

H = -J Σ_{⟨i,j⟩_2D} Z_i Z_j - g Σ_i X_i

# Arguments
- `Lx`: Lattice size in x direction
- `Ly`: Lattice size in y direction
- `J`: Ising coupling strength for 2D nearest neighbors
- `g`: Transverse field strength

# Returns
- `H`: MPO Hamiltonian
- `sites`: Site indices
"""
function build_2d_tfim_hamiltonian(Lx::Int, Ly::Int, J::Float64, g::Float64)
    N = Lx * Ly
    sites = siteinds("S=1", N)

    # Get 2D to 1D mapping
    coord_to_site = snake_order_2d_to_1d(Lx, Ly)

    # Build Hamiltonian using OpSum
    os = OpSum()

    # Transverse field term: -g Σ_i X_i
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site = coord_to_site[(i, j)]
            os += -g, "Sx", site
        end
    end

    # Ising coupling: -J Σ_{⟨i,j⟩} Z_i Z_j
    # Horizontal bonds (2D lattice)
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i+1, j)]
            os += -J, "Sz", site1, "Sz", site2
        end
    end

    # Vertical bonds (2D lattice)
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

Build 2D Heisenberg J1-J2 Hamiltonian mapped to 1D chain.

H = J1 Σ_{⟨i,j⟩} S_i · S_j + J2 Σ_{⟨⟨i,j⟩⟩} S_i · S_j

where ⟨i,j⟩ are nearest neighbors and ⟨⟨i,j⟩⟩ are next-nearest neighbors (diagonals).

# Arguments
- `Lx`: Lattice size in x direction
- `Ly`: Lattice size in y direction
- `J1`: Nearest-neighbor coupling strength
- `J2`: Next-nearest-neighbor (diagonal) coupling strength

# Returns
- `H`: MPO Hamiltonian
- `sites`: Site indices
"""
function build_2d_heisenberg_j1j2_hamiltonian(Lx::Int, Ly::Int, J1::Float64, J2::Float64)
    N = Lx * Ly
    sites = siteinds("S=1/2", N)

    # Get 2D to 1D mapping
    coord_to_site = snake_order_2d_to_1d(Lx, Ly)

    os = OpSum()

    # J1: Nearest-neighbor Heisenberg coupling S_i · S_j
    # Horizontal bonds
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i+1, j)]
            os += J1 * 0.5, "S+", site1, "S-", site2
            os += J1 * 0.5, "S-", site1, "S+", site2
            os += J1,       "Sz", site1, "Sz", site2
        end
    end

    # Vertical bonds
    for j in 1:(Ly-1)
        for i in 1:(Lx-1)
            site1 = coord_to_site[(i, j)]
            site2 = coord_to_site[(i, j+1)]
            os += J1 * 0.5, "S+", site1, "S-", site2
            os += J1 * 0.5, "S-", site1, "S+", site2
            os += J1,       "Sz", site1, "Sz", site2
        end
    end

    # J2: Next-nearest-neighbor (diagonal) Heisenberg coupling
    if J2 != 0.0
        for j in 1:(Ly-1)
            for i in 1:(Lx-1)
                # Diagonal (i,j) -> (i+1,j+1)
                site1 = coord_to_site[(i, j)]
                site2 = coord_to_site[(i+1, j+1)]
                os += J2 * 0.5, "S+", site1, "S-", site2
                os += J2 * 0.5, "S-", site1, "S+", site2
                os += J2,       "Sz", site1, "Sz", site2

                # Anti-diagonal (i+1,j) -> (i,j+1)
                site1 = coord_to_site[(i+1, j)]
                site2 = coord_to_site[(i, j+1)]
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

# Supported models
- `"tfim"`: Transverse Field Ising Model. kwargs: `J` (coupling), `g` (transverse field)
- `"heisenberg_j1j2"`: Heisenberg J1-J2 model. kwargs: `J1` (NN coupling), `J2` (NNN coupling)

# Returns
- `H`: MPO Hamiltonian
- `sites`: Site indices
"""
function build_hamiltonian(model::String, Lx::Int, Ly::Int; kwargs...)
    kw = Dict(kwargs)
    if model == "tfim"
        J = get(kw, :J, 1.0)
        g = get(kw, :g, 1.0)
        return build_2d_tfim_hamiltonian(Lx, Ly, Float64(J), Float64(g))
    elseif model == "heisenberg_j1j2"
        J1 = get(kw, :J1, 1.0)
        J2 = get(kw, :J2, 0.0)
        return build_2d_heisenberg_j1j2_hamiltonian(Lx, Ly, Float64(J1), Float64(J2))
    else
        error("Unknown model: \"$model\". Supported models: \"tfim\", \"heisenberg_j1j2\"")
    end
end

"""
    dmrg_ground_state_2d(Lx::Int, Ly::Int; model::String="tfim",
                         maxdim::Int=100, cutoff::Float64=1e-10,
                         nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps),
                         model_params...)

Compute 2D ground state using DMRG for a chosen model.

# Arguments
- `Lx`, `Ly`: Lattice dimensions
- `model`: Model name, one of `"tfim"` or `"heisenberg_j1j2"`
- `maxdim`: Maximum bond dimension
- `cutoff`: Truncation cutoff
- `nsweeps`: Number of DMRG sweeps
- `noise`: Noise schedule for DMRG
- `model_params...`: Model-specific parameters forwarded to `build_hamiltonian`:
  - TFIM: `J` (coupling, default 1.0), `g` (transverse field, default 1.0)
  - Heisenberg J1-J2: `J1` (NN coupling, default 1.0), `J2` (NNN coupling, default 0.0)

# Returns
Named tuple with:
- `energy`: Ground state energy
- `energy_per_site`: Energy per site
- `psi`: Ground state MPS
- `sites`: Site indices
- `H`: Hamiltonian MPO
"""
function dmrg_ground_state_2d(Lx::Int, Ly::Int; model::String="tfim",
                              maxdim::Int=100, cutoff::Float64=1e-10,
                              nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps),
                              model_params...)

    println("Building 2D $model Hamiltonian (Lx=$Lx, Ly=$Ly, $(join(["$k=$v" for (k,v) in model_params], ", ")))")
    H, sites = build_hamiltonian(model, Lx, Ly; model_params...)

    # Initialize random MPS
    N = Lx * Ly
    psi0 = randomMPS(sites, linkdims=10)

    # DMRG parameters
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim)
    setcutoff!(sweeps, cutoff)
    setnoise!(sweeps, noise...)

    println("Running DMRG...")
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=1)

    # Normalize energy per site
    energy_per_site = energy / ((Ly-1)*(Lx-1))

    println("Ground state energy: $energy")
    println("Energy per site: $energy_per_site")

    return (energy=energy, energy_per_site=energy_per_site,
            psi=psi, sites=sites, H=H, Lx=Lx, Ly=Ly, model=model)
end

# Backward-compatible convenience method for TFIM
function dmrg_ground_state_2d(Lx::Int, Ly::Int, J::Float64, g::Float64;
                              maxdim::Int=100, cutoff::Float64=1e-10,
                              nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps))
    return dmrg_ground_state_2d(Lx, Ly; model="tfim", J=J, g=g,
                                maxdim=maxdim, cutoff=cutoff,
                                nsweeps=nsweeps, noise=noise)
end

"""
    compute_magnetization(result)

Compute average magnetization ⟨Sx⟩ and ⟨Sz⟩.
"""
function compute_magnetization(result)
    psi = result.psi
    sites = result.sites
    N = length(sites)

    Sx_avg = sum(ITensorMPS.expect(psi, "Sx")) / N
    Sz_avg = sum(ITensorMPS.expect(psi, "Sz")) / N

    return (Sx=Sx_avg, Sz=Sz_avg)
end

"""
    compute_correlation_length_dmrg(result; max_distance::Int=20)

Estimate correlation length from connected correlation function decay.
Uses C_connected(r) = ⟨Sz(i) Sz(i+r)⟩ - ⟨Sz(i)⟩⟨Sz(i+r)⟩
"""
function compute_correlation_length_dmrg(result; max_distance::Int=60)
    psi = result.psi
    sites = result.sites
    N = length(sites)

    # stride = Ly = 3: sample along x-direction; ensure i + 3*max_dist <= N
    max_dist = min(max_distance, N ÷ 6)
    n_avg = N - 3 * max_dist

    # Compute magnetization expectation values
    Sz_expect = ITensorMPS.expect(psi, "Sz")

    # Compute full correlation matrix once
    C = correlation_matrix(psi, "Sz", "Sz")

    # Compute connected correlation function: C_conn(i,j) = C(i,j) - ⟨Sz(i)⟩⟨Sz(j)⟩
    C_connected = zeros(N, N)
    for i in 1:N
        for j in 1:N
            C_connected[i,j] = real(C[i,j]) - Sz_expect[i] * Sz_expect[j]
        end
    end

    # Average connected correlations with stride 3 (along x-direction)
    correlations = zeros(max_dist)

    for i in 1:n_avg
        for r in 1:max_dist
            correlations[r] += C_connected[i, i + 3*r] # TODO: modify to Ly
        end
    end
    correlations ./= n_avg

    # Fit exponential decay: C_connected(r) ~ exp(-r/ξ)
    # Take log and fit linear: log|C_connected(r)| = a - r/ξ
    distances = collect(1:max_dist)

    # Filter out non-positive correlations for log
    valid_idx = abs.(correlations) .> 1e-12
    if sum(valid_idx) < 2
        return (ξ=1e6, correlations=correlations, distances=distances)
    end

    log_corr = log.(abs.(correlations[valid_idx]))
    valid_distances = distances[valid_idx]

    # Linear fit: log|C(r)| = a - r/ξ
    A = hcat(ones(length(valid_distances)), -valid_distances)
    coeffs = A \ log_corr
    ξ = 1.0 / coeffs[2]

    # Ensure positive correlation length and clamp to reasonable range
    ξ = abs(ξ)
    ξ = clamp(ξ, 0.01, 1e6)

    return (ξ=ξ, correlations=correlations, distances=distances)
end

"""
    run_dmrg_scan(; model="tfim", Lx=4, Ly=4,
                   scan_param=:g, scan_values=0.0:0.5:4.0,
                   maxdim=100, cutoff=1e-10, nsweeps=10,
                   output_file="dmrg_results.json", model_params...)

Run DMRG ground state scan over a parameter range for a given model.

# Arguments
- `model`: `"tfim"` or `"heisenberg_j1j2"`
- `scan_param`: Symbol of the parameter to scan (e.g. `:g` for TFIM, `:J2` for J1-J2)
- `scan_values`: Range of values for the scanned parameter
- `model_params...`: Base model parameters (the scanned param is overridden each step)

# Examples
```julia
run_dmrg_scan(model="tfim", scan_param=:g, scan_values=0.0:0.5:4.0, J=1.0)
run_dmrg_scan(model="heisenberg_j1j2", scan_param=:J2, scan_values=0.0:0.1:1.0, J1=1.0)
```
"""
function run_dmrg_scan(; model::String="tfim", Lx::Int=4, Ly::Int=4,
                        scan_param::Symbol=:g, scan_values=0.0:0.5:4.0,
                        maxdim::Int=100, cutoff::Float64=1e-10,
                        nsweeps::Int=10,
                        output_file::String="dmrg_results.json",
                        model_params...)

    base_params = Dict{Symbol,Any}(model_params)

    results = Dict(
        "parameters" => Dict(
            "model" => model,
            "Lx" => Lx,
            "Ly" => Ly,
            "scan_param" => string(scan_param),
            "base_params" => Dict(string(k) => v for (k,v) in base_params),
            "maxdim" => maxdim,
            "cutoff" => cutoff,
            "nsweeps" => nsweeps
        ),
        "scan_values" => collect(scan_values),
        "energies" => Union{Float64,Nothing}[],
        "energies_per_site" => Union{Float64,Nothing}[],
        "Sx_avg" => Union{Float64,Nothing}[],
        "Sz_avg" => Union{Float64,Nothing}[],
        "correlation_lengths" => Union{Float64,Nothing}[]
    )

    mkpath(dirname(output_file))

    println("=" ^ 60)
    println("DMRG 2D $model Scan")
    println("Lx=$Lx, Ly=$Ly, maxdim=$maxdim")
    println("Scanning $scan_param: ", collect(scan_values))
    println("Base params: ", base_params)
    println("=" ^ 60)

    for (i, val) in enumerate(scan_values)
        println("\n[$i/$(length(scan_values))] Running $scan_param = $val ...")

        # Merge base params with the current scan value
        params = copy(base_params)
        params[scan_param] = val

        try
            result = dmrg_ground_state_2d(Lx, Ly; model=model,
                                          maxdim=maxdim, cutoff=cutoff, nsweeps=nsweeps,
                                          params...)

            mag = compute_magnetization(result)
            corr = compute_correlation_length_dmrg(result)

            push!(results["energies"], result.energy)
            push!(results["energies_per_site"], result.energy_per_site)
            push!(results["Sx_avg"], mag.Sx)
            push!(results["Sz_avg"], mag.Sz)
            push!(results["correlation_lengths"], corr.ξ)

            println("  Energy per site: $(result.energy_per_site)")
            println("  ⟨Sx⟩: $(mag.Sx), ⟨Sz⟩: $(mag.Sz)")
            println("  Correlation length: $(corr.ξ)")

        catch e
            println("  ERROR: $e")
            push!(results["energies"], nothing)
            push!(results["energies_per_site"], nothing)
            push!(results["Sx_avg"], nothing)
            push!(results["Sz_avg"], nothing)
            push!(results["correlation_lengths"], nothing)
        end

        # Save intermediate results
        open(output_file, "w") do io
            JSON3.pretty(io, results)
        end
        println("  Results saved to $output_file")
    end

    println("\n" * "=" ^ 60)
    println("Scan complete!")
    println("=" ^ 60)

    return results
end

"""
    plot_dmrg_results(json_file::String; save_path=nothing)

Plot DMRG results: energy, magnetization, and correlation length vs g.
"""
function plot_dmrg_results(json_file::String; save_path=nothing)
    data = JSON3.read(read(json_file, String))

    scan_values = collect(data.scan_values)
    energies = collect(data.energies_per_site)
    Sx_avg = collect(data.Sx_avg)
    Sz_avg = collect(data.Sz_avg)
    corr_lengths = collect(data.correlation_lengths)

    Lx = data.parameters.Lx
    Ly = data.parameters.Ly
    scan_param = string(data.parameters.scan_param)
    model = string(data.parameters.model)

    fig = Figure(size=(1400, 400))

    # Energy plot
    ax1 = Axis(fig[1, 1],
               xlabel=scan_param,
               ylabel="Energy per site",
               title="$model DMRG (Lx=$Lx, Ly=$Ly)")

    scatterlines!(ax1, scan_values, energies,
                  color=:steelblue, marker=:circle, markersize=8,
                  linewidth=2)

    # Magnetization plot
    ax2 = Axis(fig[1, 2],
               xlabel=scan_param,
               ylabel="Magnetization",
               title="Magnetization")

    scatterlines!(ax2, scan_values, Sx_avg,
                  color=:red, marker=:circle, markersize=8,
                  linewidth=2, label="⟨Sx⟩")
    scatterlines!(ax2, scan_values, abs.(Sz_avg),
                  color=:blue, marker=:circle, markersize=8,
                  linewidth=2, label="|⟨Sz⟩|")

    axislegend(ax2, position=:rt)

    # Correlation length plot
    ax3 = Axis(fig[1, 3],
               xlabel=scan_param,
               ylabel="Correlation length ξ",
               title="Correlation Length")

    scatterlines!(ax3, scan_values[3:end], corr_lengths[3:end],
                  color=:green, marker=:circle, markersize=8,
                  linewidth=2)

    # Add critical point line if scanning across phase transition
    if minimum(scan_values) < 3.04 < maximum(scan_values)
        vlines!(ax3, [3.04], color=:red, linestyle=:dash,
                linewidth=1.5, label="g_c ≈ 3.04")
        axislegend(ax3, position=:lt)
    end

    if !isnothing(save_path)
        save(save_path, fig)
        println("Figure saved to $save_path")
    end

    return fig
end

"""
    plot_correlation_decay(result; max_distance::Int=50, save_path=nothing)

Plot the connected correlation function and its exponential fit.

# Arguments
- `result`: DMRG result from dmrg_ground_state_2d
- `max_distance`: Maximum distance to plot
- `save_path`: Optional path to save the figure

# Returns
- `fig`: Makie Figure object
"""
function plot_correlation_decay(result; max_distance::Int=60, save_path=nothing)
    # Compute correlation length and get correlation data
    corr_result = compute_correlation_length_dmrg(result; max_distance=max_distance)

    ξ = corr_result.ξ
    correlations = corr_result.correlations
    distances = corr_result.distances

    # Filter valid points for fitting
    valid_idx = abs.(correlations) .> 1e-12
    valid_corr = abs.(correlations[valid_idx])
    valid_dist = distances[valid_idx]

    # Compute fit line
    if length(valid_dist) >= 2
        log_corr = log.(valid_corr)
        A = hcat(ones(length(valid_dist)), -valid_dist)
        coeffs = A \ log_corr
        a, slope = coeffs[1], coeffs[2]

        # Generate fit curve
        fit_distances = range(minimum(valid_dist), maximum(valid_dist), length=100)
        fit_curve = exp.(a .- fit_distances ./ ξ)
    else
        fit_distances = distances
        fit_curve = zeros(length(distances))
    end

    # Create figure with two subplots
    fig = Figure(size=(1000, 400))

    # Linear scale plot
    ax1 = Axis(fig[1, 1],
               xlabel="Distance r",
               ylabel="|C_connected(r)|",
               title="Connected Correlation Function")

    scatter!(ax1, distances, abs.(correlations),
             color=:blue, markersize=8, label="Data")
    lines!(ax1, fit_distances, fit_curve,
           color=:red, linewidth=2, linestyle=:dash,
           label="Fit: exp(-r/ξ), ξ=$(round(ξ, digits=2))")

    axislegend(ax1, position=:rt)

    # Log scale plot
    ax2 = Axis(fig[1, 2],
               xlabel="Distance r",
               ylabel="log|C_connected(r)|",
               title="Log Scale (Linear Fit)")

    scatter!(ax2, valid_dist, log.(valid_corr),
             color=:blue, markersize=8, label="log|Data|")

    if length(valid_dist) >= 2
        lines!(ax2, fit_distances, a .- fit_distances ./ ξ,
               color=:red, linewidth=2, linestyle=:dash,
               label="Linear fit: a - r/ξ")
    end

    axislegend(ax2, position=:rt)

    if !isnothing(save_path)
        save(save_path, fig)
        println("Figure saved to $save_path")
    end

    return fig
end



# ==================== Example usage ====================
# Choose which model to run: "tfim" or "heisenberg_j1j2"
model_choice = "heisenberg_j1j2"  # <-- change this to switch model

Lx = 100; Ly = 3;

if model_choice == "tfim"
    # --- Transverse Field Ising Model ---
    # Scan over transverse field g with fixed J
    results = run_dmrg_scan(;
        model = "tfim",
        Lx = Lx, Ly = Ly,
        scan_param = :g,
        scan_values = 0.0:0.25:4.0,
        J = 1.0,
        maxdim = 2, cutoff = 1e-10, nsweeps = 10,
        output_file = joinpath(@__DIR__, "results", "dmrg_tfim_$(Lx)x$(Ly).json")
    )

    fig = plot_dmrg_results(joinpath(@__DIR__, "results", "dmrg_tfim_$(Lx)x$(Ly).json"); save_path=joinpath(@__DIR__, "results", "figures", "dmrg_tfim_$(Lx)x$(Ly).pdf"))
    display(fig)

elseif model_choice == "heisenberg_j1j2"
    # --- Heisenberg J1-J2 Model ---
    # Scan over frustration J2 with fixed J1
    results = run_dmrg_scan(;
        model = "heisenberg_j1j2",
        Lx = Lx, Ly = Ly,
        scan_param = :J2,
        scan_values = 0.0:0.1:1.0,
        J1 = 1.0,
        maxdim = 2, cutoff = 1e-10, nsweeps = 10,
        output_file = joinpath(@__DIR__, "results", "dmrg_j1j2_$(Lx)x$(Ly).json")
    )

    fig = plot_dmrg_results(joinpath(@__DIR__, "results", "dmrg_j1j2_$(Lx)x$(Ly).json"); save_path=joinpath(@__DIR__, "results", "figures", "dmrg_j1j2_$(Lx)x$(Ly).pdf"))
    display(fig)
end

