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
    build_2d_tfim_hamiltonian(Lx::Int, Ly::Int, J::Float64, g::Float64; J_1d::Float64=0.0)

Build 2D Transverse Field Ising Model Hamiltonian mapped to 1D chain.

H = -J Σ_{⟨i,j⟩_2D} Z_i Z_j - J_1d Σ_{⟨i,i+1⟩_1D} Z_i Z_{i+1} - g Σ_i X_i

# Arguments
- `Lx`: Lattice size in x direction
- `Ly`: Lattice size in y direction
- `J`: Ising coupling strength for 2D nearest neighbors
- `g`: Transverse field strength
- `J_1d`: Ising coupling strength for 1D chain nearest neighbors (default: 0.0)

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
    dmrg_ground_state_2d(Lx::Int, Ly::Int, J::Float64, g::Float64;
                         J_1d::Float64=0.0, maxdim::Int=100, cutoff::Float64=1e-10,
                         nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps))

Compute 2D TFIM ground state using DMRG.

# Arguments
- `Lx`, `Ly`: Lattice dimensions
- `J`: Ising coupling for 2D lattice
- `g`: Transverse field
- `J_1d`: Ising coupling for 1D chain nearest neighbors (default: 0.0)
- `maxdim`: Maximum bond dimension
- `cutoff`: Truncation cutoff
- `nsweeps`: Number of DMRG sweeps
- `noise`: Noise schedule for DMRG

# Returns
Named tuple with:
- `energy`: Ground state energy
- `psi`: Ground state MPS
- `sites`: Site indices
- `H`: Hamiltonian MPO
"""
function dmrg_ground_state_2d(Lx::Int, Ly::Int, J::Float64, g::Float64;
                              maxdim::Int=100, cutoff::Float64=1e-10,
                              nsweeps::Int=10, noise::Vector{Float64}=zeros(nsweeps))

    println("Building 2D TFIM Hamiltonian (Lx=$Lx, Ly=$Ly, J=$J, g=$g")
    H, sites = build_2d_tfim_hamiltonian(Lx, Ly, J, g)

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
            psi=psi, sites=sites, H=H, Lx=Lx, Ly=Ly)
end

"""
    compute_magnetization(result)

Compute average magnetization ⟨Sx⟩ and ⟨Sz⟩.
"""
function compute_magnetization(result)
    psi = result.psi
    sites = result.sites
    N = length(sites)

    Sx_avg = sum(expect(psi, "Sx")) / N
    Sz_avg = sum(expect(psi, "Sz")) / N

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
    Sz_expect = expect(psi, "Sz")

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
    run_dmrg_scan(; Lx=4, Ly=4, J=1.0, g_values=0.0:0.5:4.0,
                   maxdim=100, cutoff=1e-10, nsweeps=10,
                   output_file="dmrg_results.json")

Run DMRG for multiple g values and save results.
"""
function run_dmrg_scan(; Lx::Int=4, Ly::Int=4, J::Float64=1.0,
                        g_values=0.0:0.5:4.0,
                        maxdim::Int=100, cutoff::Float64=1e-10,
                        nsweeps::Int=10,
                        output_file::String="dmrg_results.json")

    results = Dict(
        "parameters" => Dict(
            "Lx" => Lx,
            "Ly" => Ly,
            "J" => J,
            "maxdim" => maxdim,
            "cutoff" => cutoff,
            "nsweeps" => nsweeps
        ),
        "g_values" => collect(g_values),
        "energies" => Union{Float64,Nothing}[],
        "energies_per_site" => Union{Float64,Nothing}[],
        "Sx_avg" => Union{Float64,Nothing}[],
        "Sz_avg" => Union{Float64,Nothing}[],
        "correlation_lengths" => Union{Float64,Nothing}[]
    )

    mkpath(dirname(output_file))

    println("=" ^ 60)
    println("DMRG 2D TFIM Scan")
    println("Lx=$Lx, Ly=$Ly, J=$J, maxdim=$maxdim")
    println("g values: ", collect(g_values))
    println("=" ^ 60)

    for (i, g) in enumerate(g_values)
        println("\n[$i/$(length(g_values))] Running g = $g ...")

        try
            result = dmrg_ground_state_2d(Lx, Ly, J, g;
                                          maxdim=maxdim, cutoff=cutoff, nsweeps=nsweeps)

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

    g_values = collect(data.g_values)
    energies = collect(data.energies_per_site)
    Sx_avg = collect(data.Sx_avg)
    Sz_avg = collect(data.Sz_avg)
    corr_lengths = collect(data.correlation_lengths)

    Lx = data.parameters.Lx
    Ly = data.parameters.Ly

    fig = Figure(size=(1400, 400))

    # Energy plot
    ax1 = Axis(fig[1, 1],
               xlabel="g (transverse field)",
               ylabel="Energy per site",
               title="2D TFIM DMRG (Lx=$Lx, Ly=$Ly)")

    scatterlines!(ax1, g_values, energies,
                  color=:steelblue, marker=:circle, markersize=8,
                  linewidth=2)

    # Magnetization plot
    ax2 = Axis(fig[1, 2],
               xlabel="g (transverse field)",
               ylabel="Magnetization",
               title="Magnetization")

    scatterlines!(ax2, g_values, Sx_avg,
                  color=:red, marker=:circle, markersize=8,
                  linewidth=2, label="⟨Sx⟩")
    scatterlines!(ax2, g_values, abs.(Sz_avg),
                  color=:blue, marker=:circle, markersize=8,
                  linewidth=2, label="|⟨Sz⟩|")

    axislegend(ax2, position=:rt)

    # Correlation length plot
    ax3 = Axis(fig[1, 3],
               xlabel="g (transverse field)",
               ylabel="Correlation length ξ",
               title="Correlation Length")

    scatterlines!(ax3, g_values[3:end], corr_lengths[3:end],
                  color=:green, marker=:circle, markersize=8,
                  linewidth=2)

    # Add critical point line if scanning across phase transition
    if minimum(g_values) < 3.04 < maximum(g_values)
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

# Example usage
Lx=100; Ly=3;
    results = run_dmrg_scan(;
        Lx = Lx,
        Ly = Ly,
        J = 1.0,
        g_values = 0.0:0.25:4.0,
        maxdim = 2,
        cutoff = 1e-10,
        nsweeps = 10,
        output_file = joinpath(@__DIR__, "results", "dmrg_results_$Lx$Ly.json")
    )

    fig = plot_dmrg_results(joinpath(@__DIR__, "results", "dmrg_results_$Lx$Ly.json");
                            save_path=joinpath(@__DIR__, "results", "figures", "dmrg_results.pdf"))

    result_corr = dmrg_ground_state_2d(Lx, Ly, 1.0, 1.5;
                                       maxdim=2, cutoff=1e-10, nsweeps=10)
    fig2 = plot_correlation_decay(result_corr)
    display(fig2)

