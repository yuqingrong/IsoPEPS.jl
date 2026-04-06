using IsoPEPS
using ITensors
using ITensorMPS  # triggers DMRGReferenceExt
using JSON3
using CairoMakie
using Statistics

# Core DMRG functions (dmrg_ground_state_2d, build_hamiltonian, etc.)
# are now provided by IsoPEPS via the DMRGReferenceExt extension.

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
    compute_bulk_energy_density(Ly, Lx1, E1, Lx2, E2)

Extract bulk energy density per site from two DMRG runs on open cylinders
with the same width `Ly` but different lengths `Lx1` and `Lx2`.

    E(Lx) = e_bulk * Ly * Lx + E_left + E_right

Boundary terms cancel:

    e_bulk = (E2 - E1) / (Ly * (Lx2 - Lx1))
"""
function compute_bulk_energy_density(Ly::Int, Lx1::Int, E1::Float64, Lx2::Int, E2::Float64)
    Lx1 != Lx2 || error("Lx1 and Lx2 must differ")
    return (E2 - E1) / (Ly * (Lx2 - Lx1))
end

"""
    plot_dmrg_results(json_file::String; save_path=nothing)

Plot DMRG results: energy, magnetization, and correlation length vs scan parameter.
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

    ax1 = Axis(fig[1, 1],
               xlabel=scan_param,
               ylabel="Energy per site",
               title="$model DMRG (Lx=$Lx, Ly=$Ly)")
    scatterlines!(ax1, scan_values, energies,
                  color=:steelblue, marker=:circle, markersize=8, linewidth=2)

    ax2 = Axis(fig[1, 2],
               xlabel=scan_param,
               ylabel="Magnetization",
               title="Magnetization")
    scatterlines!(ax2, scan_values, Sx_avg,
                  color=:red, marker=:circle, markersize=8, linewidth=2, label="⟨Sx⟩")
    scatterlines!(ax2, scan_values, abs.(Sz_avg),
                  color=:blue, marker=:circle, markersize=8, linewidth=2, label="|⟨Sz⟩|")
    axislegend(ax2, position=:rt)

    ax3 = Axis(fig[1, 3],
               xlabel=scan_param,
               ylabel="Correlation length ξ",
               title="Correlation Length")
    scatterlines!(ax3, scan_values[3:end], corr_lengths[3:end],
                  color=:green, marker=:circle, markersize=8, linewidth=2)

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
    plot_correlation_decay(result; max_distance::Int=60, save_path=nothing)

Plot the connected correlation function and its exponential fit.
"""
function plot_correlation_decay(result; max_distance::Int=60, save_path=nothing)
    corr_result = compute_correlation_length_dmrg(result; max_distance=max_distance)

    ξ = corr_result.ξ
    correlations = corr_result.correlations
    distances = corr_result.distances

    valid_idx = abs.(correlations) .> 1e-12
    valid_corr = abs.(correlations[valid_idx])
    valid_dist = distances[valid_idx]

    if length(valid_dist) >= 2
        log_corr = log.(valid_corr)
        A = hcat(ones(length(valid_dist)), -valid_dist)
        coeffs = A \ log_corr
        a, slope = coeffs[1], coeffs[2]
        fit_distances = range(minimum(valid_dist), maximum(valid_dist), length=100)
        fit_curve = exp.(a .- fit_distances ./ ξ)
    else
        fit_distances = distances
        fit_curve = zeros(length(distances))
    end

    fig = Figure(size=(1000, 400))

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


run_dmrg_scan(
    model      = "heisenberg_j1j2",
    Lx         = 100,
    Ly         = 4,
    scan_param = :J2,
    scan_values = 0.0:0.1:1.0,    # or 0.0:0.1:1.0 for a full scan
    maxdim     = 2,
    cutoff     = 1e-10,
    nsweeps    = 20,
    output_file = "project/results/dmrg_j1j2_100x4_D=2.json",
    J1         = 1.0
)

# ==================== Example usage ====================
Ly = 4
D = 2
model_choice = "heisenberg_j1j2"
Lx1 = 100
Lx2 = 200
J2_values = collect(0.0:0.1:1.0)

if model_choice == "heisenberg_j1j2"
    scan_results = Dict(
        "parameters" => Dict(
            "model"   => model_choice,
            "Ly"      => Ly,
            "Lx1"     => Lx1,
            "Lx2"     => Lx2,
            "D"       => D,
            "J1"      => 1.0,
            "cutoff1" => 1e-10,
            "cutoff2" => 1e-10,
            "nsweeps" => 20
        ),
        "J2_values"            => J2_values,
        "Lx1_energies"         => Float64[],
        "Lx1_energies_per_site"=> Float64[],
        "Lx2_energies"         => Float64[],
        "Lx2_energies_per_site"=> Float64[],
        "e_bulk_values"        => Float64[],
        "M2_neel_Lx1"         => Float64[],
        "M2_neel_Lx2"         => Float64[],
        "M2_stripe_Lx1"       => Float64[],
        "M2_stripe_Lx2"       => Float64[],
        "M2_0pi_Lx1"          => Float64[],
        "M2_0pi_Lx2"          => Float64[]
    )
    output_file = "dmrg_bulk_$(model_choice)_Ly$(Ly)_D$(D)_J2scan.json"

    println("=" ^ 60)
    println("J2 scan: $(J2_values)")
    println("=" ^ 60)

    prev_psi1 = nothing  # warm-start from previous J2
    prev_psi2 = nothing

    for (i, J2) in enumerate(J2_values)
        println("\n[$i/$(length(J2_values))] J2 = $J2")

        result1 = dmrg_ground_state_2d(Lx1, Ly;
            model="heisenberg_j1j2", J1=1.0, J2=J2,
            maxdim=D, cutoff=1e-10, nsweeps=20,
            noise=[1e-1, 1e-2, 1e-3, 1e-4, fill(0.0, 16)...],
            psi0=prev_psi1)

        result2 = dmrg_ground_state_2d(Lx2, Ly;
            model="heisenberg_j1j2", J1=1.0, J2=J2,
            maxdim=D, cutoff=1e-10, nsweeps=20,
            noise=[1e-1, 1e-2, 1e-3, 1e-4, fill(0.0, 16)...],
            psi0=prev_psi2)

        prev_psi1 = result1.psi
        prev_psi2 = result2.psi

        e_bulk = compute_bulk_energy_density(Ly, Lx1, result1.energy, Lx2, result2.energy)

        # Compute M² order parameters
        M2_neel1 = compute_M2_dmrg(result1, (pi, pi))
        M2_neel2 = compute_M2_dmrg(result2, (pi, pi))
        M2_stripe1 = compute_M2_dmrg(result1, (pi, 0.0))
        M2_stripe2 = compute_M2_dmrg(result2, (pi, 0.0))
        M2_0pi1 = compute_M2_dmrg(result1, (0.0, pi))
        M2_0pi2 = compute_M2_dmrg(result2, (0.0, pi))

        println("  e_bulk = $e_bulk")
        println("  E(Lx=$Lx1) = $(result1.energy), E/N = $(result1.energy_per_site)")
        println("  E(Lx=$Lx2) = $(result2.energy), E/N = $(result2.energy_per_site)")
        println("  M²(π,π) Lx1=$(M2_neel1), Lx2=$(M2_neel2)")
        println("  M²(π,0) Lx1=$(M2_stripe1), Lx2=$(M2_stripe2)")
        println("  M²(0,π) Lx1=$(M2_0pi1), Lx2=$(M2_0pi2)")

        push!(scan_results["Lx1_energies"],          result1.energy)
        push!(scan_results["Lx1_energies_per_site"], result1.energy_per_site)
        push!(scan_results["Lx2_energies"],          result2.energy)
        push!(scan_results["Lx2_energies_per_site"], result2.energy_per_site)
        push!(scan_results["e_bulk_values"],         e_bulk)
        push!(scan_results["M2_neel_Lx1"],          M2_neel1)
        push!(scan_results["M2_neel_Lx2"],          M2_neel2)
        push!(scan_results["M2_stripe_Lx1"],        M2_stripe1)
        push!(scan_results["M2_stripe_Lx2"],        M2_stripe2)
        push!(scan_results["M2_0pi_Lx1"],           M2_0pi1)
        push!(scan_results["M2_0pi_Lx2"],           M2_0pi2)

        # Save DMRG states for later analysis
        state_dir = "states"
        mkpath(state_dir)
        save_dmrg_state(result1, joinpath(state_dir, "dmrg_$(model_choice)_Ly$(Ly)_Lx$(Lx1)_D$(D)_J2$(J2).jls"); J1=1.0, J2=J2)
        save_dmrg_state(result2, joinpath(state_dir, "dmrg_$(model_choice)_Ly$(Ly)_Lx$(Lx2)_D$(D)_J2$(J2).jls"); J1=1.0, J2=J2)

        open(output_file, "w") do io
            JSON3.pretty(io, scan_results)
        end
        println("  Saved to $output_file")
    end

    println("\n" * "=" ^ 60)
    println("Scan complete! Results saved to $output_file")
    println("=" ^ 60)
end
# ==================== Plot M² vs J2 ====================
data = JSON3.read(read(output_file, String))
J2_vals = Float64.(data.J2_values)

fig = Figure(size=(900, 500))
ax = Axis(fig[1, 1],
            xlabel="J₂ / J₁",
            ylabel="M²(q)",
            title="DMRG M²(q) vs J₂  (Ly=$Ly, D=$D)")

# Use longer system (Lx2) for better bulk approximation
    scatterlines!(ax, J2_vals, Float64.(data.M2_neel_Lx2),
                  label="M²(π,π) Néel  Lx=$Lx2", color=:blue,
                  marker=:circle, markersize=10, linewidth=2)
    scatterlines!(ax, J2_vals, Float64.(data.M2_stripe_Lx2),
                  label="M²(π,0) Stripe  Lx=$Lx2", color=:red,
                  marker=:diamond, markersize=10, linewidth=2)
    scatterlines!(ax, J2_vals, Float64.(data.M2_0pi_Lx2),
                  label="M²(0,π) Stripe  Lx=$Lx2", color=:green,
                  marker=:rect, markersize=10, linewidth=2)

    # Also show shorter system for comparison
    scatterlines!(ax, J2_vals, Float64.(data.M2_neel_Lx1),
                  label="M²(π,π) Néel  Lx=$Lx1", color=:blue,
                  linestyle=:dash, marker=:utriangle, markersize=8, linewidth=1.5)
    scatterlines!(ax, J2_vals, Float64.(data.M2_stripe_Lx1),
                  label="M²(π,0) Stripe  Lx=$Lx1", color=:red,
                  linestyle=:dash, marker=:utriangle, markersize=8, linewidth=1.5)
    scatterlines!(ax, J2_vals, Float64.(data.M2_0pi_Lx1),
                  label="M²(0,π) Stripe  Lx=$Lx1", color=:green,
                  linestyle=:dash, marker=:utriangle, markersize=8, linewidth=1.5)

    axislegend(ax, position=:rt)

    fig_path = "dmrg_M2_vs_J2_Ly$(Ly)_D$(D).pdf"
    save(fig_path, fig)
    println("M² plot saved to $fig_path")


# 1. Run DMRG to get the ground state                     
result = dmrg_ground_state_2d(200, 4;
    model="heisenberg_j1j2",
    J1=1.0, J2=0.5,
    nsweeps=20, maxdim=2
)

# 2. Plot the spin structure factor heatmap
fig, SSS = plot_dmrg_spin_structure_factor(result;
nq=50,              # resolution in BZ
max_separation=10,  # use middle 50% of cylinder to avoid edges
save_path="project/results/figures/spin_SF_heatmap.pdf"
)


# real space dimer-dimer bond pattern (vertical ref + horizontal ref side by side)
fig, corr_data = plot_dmrg_dimer_bond_pattern(result;
bulk_cols=20,
ref_bond_idx=1,
title="DMRG Dimer Correlation (J₂=0.5, D=2)",
save_path="project/results/figures/dmrg_dimer_bond.pdf"
)

# Bond energy pattern: ⟨S_i · S_j⟩ on each bond (reveals VBS strong/weak bonds)

fig, bond_data = plot_dmrg_bond_energy_pattern(result;
bulk_cols=20,
title="DMRG Bond Energy ⟨Sᵢ·Sⱼ⟩ (J₂=0.5, D=2)",
save_path="project/results/figures/dmrg_bond_energy.pdf"
)

# dimer structure factor
fig, SD = plot_dmrg_dimer_structure_factor(result;
nq=50,              # resolution in BZ
bulk_cols=20,       # use middle 20 columns to avoid edges
save_path="project/results/figures/dimer_SF_heatmap.pdf"
)