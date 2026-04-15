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

"""
    run_dmrg_bulk_scan(; model="heisenberg_j1j2", Ly=4, Lx1=100, Lx2=200, D=2,
                         scan_param=:J2, scan_values=0.0:0.1:1.0,
                         cutoff=1e-10, nsweeps=20,
                         noise=[1e-1, 1e-2, 1e-3, 1e-4, fill(0.0, 16)...],
                         output_file=nothing, state_dir="states",
                         model_params...)

Run DMRG parameter scan with two system lengths to extract bulk energy density
via finite-size subtraction.  Model-specific observables are computed automatically:

- `"heisenberg_j1j2"`: M² order parameters (Néel, stripe, (0,π))
- `"tfim"`:            ⟨Sx⟩ and ⟨Sz⟩ magnetizations

# Arguments
- `model`: `"tfim"` or `"heisenberg_j1j2"`
- `scan_param`: Symbol of the parameter to scan (e.g. `:J2` for J1-J2, `:g` for TFIM)
- `scan_values`: Range of values for the scanned parameter
- `model_params...`: Fixed model parameters (the scanned param is overridden each step)

# Examples
```julia
run_dmrg_bulk_scan(model="heisenberg_j1j2", scan_param=:J2, scan_values=0.0:0.1:1.0, J1=1.0)
run_dmrg_bulk_scan(model="tfim", scan_param=:g, scan_values=0.0:0.5:4.0, J=1.0)
```
"""
function run_dmrg_bulk_scan(; model::String="heisenberg_j1j2",
                              Ly::Int=4, Lx1::Int=100, Lx2::Int=200, D::Int=2,
                              scan_param::Symbol=:J2, scan_values=0.0:0.1:1.0,
                              cutoff::Float64=1e-10, nsweeps::Int=20,
                              noise=[1e-1, 1e-2, 1e-3, 1e-4, fill(0.0, 16)...],
                              output_file::Union{String,Nothing}=nothing,
                              state_dir::String="states",
                              model_params...)

    scan_values = collect(scan_values)
    base_params = Dict{Symbol,Any}(model_params)

    if isnothing(output_file)
        output_file = "dmrg_bulk_$(model)_Ly$(Ly)_D$(D)_$(scan_param)scan.json"
    end

    is_j1j2 = (model == "heisenberg_j1j2")

    scan_results = Dict{String,Any}(
        "parameters" => Dict(
            "model"        => model,
            "Ly"           => Ly,
            "Lx1"          => Lx1,
            "Lx2"          => Lx2,
            "D"            => D,
            "scan_param"   => string(scan_param),
            "base_params"  => Dict(string(k) => v for (k,v) in base_params),
            "cutoff"       => cutoff,
            "nsweeps"      => nsweeps
        ),
        "scan_values"          => scan_values,
        "Lx1_energies"         => Float64[],
        "Lx1_energies_per_site"=> Float64[],
        "Lx2_energies"         => Float64[],
        "Lx2_energies_per_site"=> Float64[],
        "e_bulk_values"        => Float64[]
    )

    if is_j1j2
        scan_results["M2_neel_Lx1"]    = Float64[]
        scan_results["M2_neel_Lx2"]    = Float64[]
        scan_results["M2_stripe_Lx1"]  = Float64[]
        scan_results["M2_stripe_Lx2"]  = Float64[]
        scan_results["M2_0pi_Lx1"]     = Float64[]
        scan_results["M2_0pi_Lx2"]     = Float64[]
    else
        scan_results["Sx_Lx1"] = Float64[]
        scan_results["Sx_Lx2"] = Float64[]
        scan_results["Sz_Lx1"] = Float64[]
        scan_results["Sz_Lx2"] = Float64[]
    end

    println("=" ^ 60)
    println("Model: $model — scanning $scan_param: $(scan_values)")
    println("Base params: $base_params")
    println("=" ^ 60)

    prev_psi1 = nothing
    prev_psi2 = nothing

    for (i, val) in enumerate(scan_values)
        println("\n[$i/$(length(scan_values))] $scan_param = $val")

        params = copy(base_params)
        params[scan_param] = val

        result1 = dmrg_ground_state_2d(Lx1, Ly;
            model=model, maxdim=D, cutoff=cutoff, nsweeps=nsweeps,
            noise=noise, psi0=prev_psi1, params...)

        result2 = dmrg_ground_state_2d(Lx2, Ly;
            model=model, maxdim=D, cutoff=cutoff, nsweeps=nsweeps,
            noise=noise, psi0=prev_psi2, params...)

        prev_psi1 = result1.psi
        prev_psi2 = result2.psi

        e_bulk = compute_bulk_energy_density(Ly, Lx1, result1.energy, Lx2, result2.energy)

        println("  e_bulk = $e_bulk")
        println("  E(Lx=$Lx1) = $(result1.energy), E/N = $(result1.energy_per_site)")
        println("  E(Lx=$Lx2) = $(result2.energy), E/N = $(result2.energy_per_site)")

        push!(scan_results["Lx1_energies"],          result1.energy)
        push!(scan_results["Lx1_energies_per_site"], result1.energy_per_site)
        push!(scan_results["Lx2_energies"],          result2.energy)
        push!(scan_results["Lx2_energies_per_site"], result2.energy_per_site)
        push!(scan_results["e_bulk_values"],         e_bulk)

        if is_j1j2
            M2_neel1   = compute_M2_dmrg(result1, (pi, pi))
            M2_neel2   = compute_M2_dmrg(result2, (pi, pi))
            M2_stripe1 = compute_M2_dmrg(result1, (pi, 0.0))
            M2_stripe2 = compute_M2_dmrg(result2, (pi, 0.0))
            M2_0pi1    = compute_M2_dmrg(result1, (0.0, pi))
            M2_0pi2    = compute_M2_dmrg(result2, (0.0, pi))

            println("  M²(π,π) Lx1=$(M2_neel1), Lx2=$(M2_neel2)")
            println("  M²(π,0) Lx1=$(M2_stripe1), Lx2=$(M2_stripe2)")
            println("  M²(0,π) Lx1=$(M2_0pi1), Lx2=$(M2_0pi2)")

            push!(scan_results["M2_neel_Lx1"],   M2_neel1)
            push!(scan_results["M2_neel_Lx2"],   M2_neel2)
            push!(scan_results["M2_stripe_Lx1"], M2_stripe1)
            push!(scan_results["M2_stripe_Lx2"], M2_stripe2)
            push!(scan_results["M2_0pi_Lx1"],    M2_0pi1)
            push!(scan_results["M2_0pi_Lx2"],    M2_0pi2)
        else
            mag1 = compute_magnetization(result1)
            mag2 = compute_magnetization(result2)

            println("  ⟨Sx⟩ Lx1=$(mag1.Sx), Lx2=$(mag2.Sx)")
            println("  ⟨Sz⟩ Lx1=$(mag1.Sz), Lx2=$(mag2.Sz)")

            push!(scan_results["Sx_Lx1"], mag1.Sx)
            push!(scan_results["Sx_Lx2"], mag2.Sx)
            push!(scan_results["Sz_Lx1"], mag1.Sz)
            push!(scan_results["Sz_Lx2"], mag2.Sz)
        end

        mkpath(state_dir)
        param_str = join(["$(k)$(v)" for (k,v) in params], "_")
        save_dmrg_state(result1, joinpath(state_dir, "dmrg_$(model)_Ly$(Ly)_Lx$(Lx1)_D$(D)_$(param_str).jls"); params...)
        save_dmrg_state(result2, joinpath(state_dir, "dmrg_$(model)_Ly$(Ly)_Lx$(Lx2)_D$(D)_$(param_str).jls"); params...)

        open(output_file, "w") do io
            JSON3.pretty(io, scan_results)
        end
        println("  Saved to $output_file")
    end

    println("\n" * "=" ^ 60)
    println("Scan complete! Results saved to $output_file")
    println("=" ^ 60)

    return scan_results
end

"""
    check_bulk_convergence(; Ly=4, Lx_values=[20,40,60,80,100,150,200],
                             model="heisenberg_j1j2", maxdim=2,
                             cutoff=1e-10, nsweeps=20,
                             noise=[1e-1,1e-2,1e-3,1e-4,fill(0.0,16)...],
                             output_file=nothing, model_params...)

Run DMRG at multiple cylinder lengths to check convergence of bulk energy density.

The model `E(Lx) = e_bulk * Ly * Lx + E_boundary` implies `E/N = e_bulk + E_boundary/(Ly*Lx)`.
Plotting E/N vs 1/Lx should yield a straight line whose intercept is the converged `e_bulk`.
Pairwise `e_bulk` from consecutive (Lx_i, Lx_{i+1}) pairs should stabilize as Lx grows.

# Returns
Dict with keys:
- `"Lx_values"`, `"energies"`, `"energies_per_site"`
- `"pairwise_Lx_mid"`, `"pairwise_e_bulk"` — from consecutive pairs
- `"fit_intercept"` (extrapolated e_bulk), `"fit_slope"` (E_boundary/Ly coefficient)
"""
function check_bulk_convergence(; Ly::Int=4,
                                  Lx_values::Vector{Int}=[20, 40, 60, 80, 100, 150, 200],
                                  model::String="heisenberg_j1j2",
                                  maxdim::Int=2,
                                  cutoff::Float64=1e-10,
                                  nsweeps::Int=20,
                                  noise=[1e-1, 1e-2, 1e-3, 1e-4, fill(0.0, 16)...],
                                  output_file::Union{String,Nothing}=nothing,
                                  model_params...)

    Lx_sorted = sort(Lx_values)

    if isnothing(output_file)
        output_file = "dmrg_convergence_$(model)_Ly$(Ly)_D$(maxdim).json"
    end

    results = Dict(
        "parameters" => Dict(
            "model"   => model,
            "Ly"      => Ly,
            "maxdim"  => maxdim,
            "cutoff"  => cutoff,
            "nsweeps" => nsweeps,
            "model_params" => Dict(string(k) => v for (k,v) in model_params)
        ),
        "Lx_values"         => Lx_sorted,
        "energies"          => Float64[],
        "energies_per_site" => Float64[],
        "pairwise_Lx_mid"   => Float64[],
        "pairwise_e_bulk"   => Float64[],
        "fit_intercept"     => nothing,
        "fit_slope"         => nothing
    )

    println("=" ^ 60)
    println("Bulk Energy Convergence Check")
    println("Model: $model, Ly=$Ly, maxdim=$maxdim")
    println("Lx values: $Lx_sorted")
    println("Params: ", Dict(model_params))
    println("=" ^ 60)

    for (i, Lx) in enumerate(Lx_sorted)
        println("\n[$i/$(length(Lx_sorted))] Lx = $Lx ...")

        result = dmrg_ground_state_2d(Lx, Ly;
            model=model, maxdim=maxdim, cutoff=cutoff, nsweeps=nsweeps,
            noise=Float64.(noise),
            model_params...)

        push!(results["energies"], result.energy)
        push!(results["energies_per_site"], result.energy_per_site)

        println("  E = $(result.energy),  E/N = $(result.energy_per_site)")

        # Pairwise e_bulk from consecutive pairs
        if i >= 2
            Lx_prev = Lx_sorted[i-1]
            E_prev = results["energies"][i-1]
            e_bulk = compute_bulk_energy_density(Ly, Lx_prev, E_prev, Lx, result.energy)
            Lx_mid = (Lx_prev + Lx) / 2.0
            push!(results["pairwise_Lx_mid"], Lx_mid)
            push!(results["pairwise_e_bulk"], e_bulk)
            println("  Pairwise e_bulk($Lx_prev,$Lx) = $e_bulk")
        end

        # Linear fit E/N vs 1/Lx with all data so far
        if i >= 2
            inv_Lx = 1.0 ./ Lx_sorted[1:i]
            E_per_site = results["energies_per_site"]
            A = hcat(ones(i), inv_Lx)
            coeffs = A \ E_per_site
            results["fit_intercept"] = coeffs[1]
            results["fit_slope"] = coeffs[2]
            println("  Linear fit: e_bulk(extrapolated) = $(coeffs[1]),  slope = $(coeffs[2])")
        end

        # Incremental save
        open(output_file, "w") do io
            JSON3.pretty(io, results)
        end
    end

    println("\n" * "=" ^ 60)
    println("Convergence check complete!")
    println("Extrapolated e_bulk = $(results["fit_intercept"])")
    if length(results["pairwise_e_bulk"]) >= 2
        last_two = results["pairwise_e_bulk"][end-1:end]
        println("Last two pairwise e_bulk: $(last_two)  (Δ = $(abs(last_two[2]-last_two[1])))")
    end
    println("Results saved to $output_file")
    println("=" ^ 60)

    return results
end

"""
    plot_bulk_convergence(result_or_file; save_path=nothing)

Plot bulk energy convergence:
- Left: E/N vs 1/Lx with linear fit and extrapolated e_bulk
- Right: pairwise e_bulk vs 1/Lx_mid showing stabilization
"""
function plot_bulk_convergence(result_or_file; save_path::Union{String,Nothing}=nothing)
    if result_or_file isa String
        data = JSON3.read(read(result_or_file, String))
        Lx_vals = Int.(data.Lx_values)
        E_per_site = Float64.(data.energies_per_site)
        pw_Lx_mid = Float64.(data.pairwise_Lx_mid)
        pw_e_bulk = Float64.(data.pairwise_e_bulk)
        e_bulk_fit = Float64(data.fit_intercept)
        slope = Float64(data.fit_slope)
        Ly = Int(data.parameters.Ly)
        model = string(data.parameters.model)
        maxdim = Int(data.parameters.maxdim)
    else
        data = result_or_file
        Lx_vals = data["Lx_values"]
        E_per_site = data["energies_per_site"]
        pw_Lx_mid = data["pairwise_Lx_mid"]
        pw_e_bulk = data["pairwise_e_bulk"]
        e_bulk_fit = data["fit_intercept"]
        slope = data["fit_slope"]
        Ly = data["parameters"]["Ly"]
        model = data["parameters"]["model"]
        maxdim = data["parameters"]["maxdim"]
    end

    inv_Lx = 1.0 ./ Lx_vals
    inv_Lx_mid = 1.0 ./ pw_Lx_mid

    fig = Figure(size=(1200, 500))

    # Left panel: E/N vs 1/Lx
    ax1 = Axis(fig[1, 1],
               xlabel="1/Lx",
               ylabel="E / N",
               title="$model (Ly=$Ly, D=$maxdim): E/N vs 1/Lx")

    scatterlines!(ax1, inv_Lx, E_per_site,
                  color=:steelblue, marker=:circle, markersize=10, linewidth=2,
                  label="DMRG data")

    # Fit line
    fit_x = range(0, maximum(inv_Lx) * 1.1, length=100)
    fit_y = e_bulk_fit .+ slope .* fit_x
    lines!(ax1, fit_x, fit_y,
           color=:red, linewidth=2, linestyle=:dash,
           label="Fit: e∞ = $(round(e_bulk_fit, digits=6))")
    hlines!(ax1, [e_bulk_fit], color=:gray, linestyle=:dot, linewidth=1)

    axislegend(ax1, position=:rt)

    # Right panel: pairwise e_bulk vs 1/Lx_mid
    ax2 = Axis(fig[1, 2],
               xlabel="1/Lx_mid",
               ylabel="Pairwise e_bulk",
               title="Pairwise Bulk Energy Density")

    scatterlines!(ax2, inv_Lx_mid, pw_e_bulk,
                  color=:darkgreen, marker=:diamond, markersize=10, linewidth=2,
                  label="(Lxᵢ, Lxᵢ₊₁) pairs")
    hlines!(ax2, [e_bulk_fit], color=:red, linestyle=:dash, linewidth=1.5,
            label="Fit e∞ = $(round(e_bulk_fit, digits=6))")

    axislegend(ax2, position=:rt)

    if isnothing(save_path)
        save_path = "bulk_convergence_$(model)_Ly$(Ly)_D$(maxdim).pdf"
    end
    save(save_path, fig)
    println("Convergence plot saved to $save_path")

    return fig
end

"""
    plot_M2_vs_J2(json_file; Lx1=100, Lx2=200, save_path=nothing)

Plot M²(q) order parameters vs J₂ from a bulk scan JSON file.
"""
function plot_M2_vs_J2(json_file::String; Lx1::Int=100, Lx2::Int=200,
                       save_path::Union{String,Nothing}=nothing)
    data = JSON3.read(read(json_file, String))
    J2_vals = Float64.(haskey(data, :scan_values) ? data.scan_values : data.J2_values)
    Ly = data.parameters.Ly
    D = data.parameters.D

    fig = Figure(size=(900, 500))
    ax = Axis(fig[1, 1],
              xlabel="J₂ / J₁",
              ylabel="M²(q)",
              title="DMRG M²(q) vs J₂  (Ly=$Ly, D=$D)")

    scatterlines!(ax, J2_vals, Float64.(data.M2_neel_Lx2),
                  label="M²(π,π) Néel  Lx=$Lx2", color=:blue,
                  marker=:circle, markersize=10, linewidth=2)
    scatterlines!(ax, J2_vals, Float64.(data.M2_stripe_Lx2),
                  label="M²(π,0) Stripe  Lx=$Lx2", color=:red,
                  marker=:diamond, markersize=10, linewidth=2)
    scatterlines!(ax, J2_vals, Float64.(data.M2_0pi_Lx2),
                  label="M²(0,π) Stripe  Lx=$Lx2", color=:green,
                  marker=:rect, markersize=10, linewidth=2)

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

    if isnothing(save_path)
        save_path = "dmrg_M2_vs_J2_Ly$(Ly)_D$(D).pdf"
    end
    save(save_path, fig)
    println("M² plot saved to $save_path")

    return fig
end


conv = check_bulk_convergence(
      Ly=3,
      Lx_values=[100, 150, 180, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1500,2000,2500,3000,3500,4000],
      model="tfim",
      maxdim=2,
      J1=1.0, g=0.0
  )

fig = plot_bulk_convergence(conv; save_path="project/results/figures/bulk_convergence.pdf")
display(fig)


# ==================== Example usage ====================
scan_results = run_dmrg_bulk_scan(
    model="heisenberg_j1j2",
    Ly=4, Lx1=1000, Lx2=1200, D=2,
    scan_param=:J2, scan_values=0.0:0.1:1.0,
    J1=1.0
)

# TFIM
run_dmrg_bulk_scan(
    model="tfim",
    Ly=4, Lx1=1000, Lx2=1500, D=2,
    scan_param=:g, scan_values=0.0:0.5:4.0,
    J=1.0
)

output_file = "dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json"
plot_M2_vs_J2(output_file; Lx1=100, Lx2=200)

# Check convergence of bulk energy at J2=0.5
conv = check_bulk_convergence(
    Ly=4, Lx_values=[100, 150, 180, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1500,2000,2500,3000],
    model="heisenberg_j1j2", maxdim=2,
    J1=1.0, J2=0.0
)
plot_bulk_convergence(conv; save_path="project/results/figures/bulk_convergence.pdf")


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