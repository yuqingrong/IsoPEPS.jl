using IsoPEPS
using ITensors
using ITensorMPS  # triggers DMRGReferenceExt
using JSON3
using CairoMakie

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


# ==================== Example usage ====================
model_choice = "heisenberg_j1j2"

Lx = 100; Ly = 4; D = 2

if model_choice == "tfim"
    results = run_dmrg_scan(;
        model = "tfim",
        Lx = Lx, Ly = Ly,
        scan_param = :g,
        scan_values = 0.0:0.25:4.0,
        J = 1.0,
        maxdim = D, cutoff = 1e-10, nsweeps = 10,
        output_file = joinpath(@__DIR__, "results", "dmrg_tfim_$(Lx)x$(Ly)_D=$(D).json")
    )
    fig = plot_dmrg_results(joinpath(@__DIR__, "results", "dmrg_tfim_$(Lx)x$(Ly)_D=$(D).json");
        save_path=joinpath(@__DIR__, "results", "figures", "dmrg_tfim_$(Lx)x$(Ly)_D=$(D).pdf"))
    display(fig)

elseif model_choice == "heisenberg_j1j2"
    results = run_dmrg_scan(;
        model = "heisenberg_j1j2",
        Lx = Lx, Ly = Ly,
        scan_param = :J2,
        scan_values = 0.0:0.1:1.0,
        J1 = 1.0,
        maxdim = D, cutoff = 1e-10, nsweeps = 10,
        output_file = joinpath(@__DIR__, "results", "dmrg_j1j2_$(Lx)x$(Ly)_D=$(D).json")
    )
    fig = plot_dmrg_results(joinpath(@__DIR__, "results", "dmrg_j1j2_$(Lx)x$(Ly)_D=$(D).json");
        save_path=joinpath(@__DIR__, "results", "figures", "dmrg_j1j2_$(Lx)x$(Ly)_D=$(D).pdf"))
    display(fig)
end
