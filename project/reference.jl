using IsoPEPS
using PEPSKit
using JSON3
using CairoMakie
set_theme!(IsoPEPS.paper_theme())

"""
    run_pepskit_scan(; d=2, D=2, J=1.0, g_values=0.0:0.25:4.0, χ=10, 
                      ctmrg_tol=1e-10, grad_tol=1e-6, maxiter=1000,
                      output_file="pepskit_results.json")

Run pepskit_ground_state for a range of g values and save results to JSON.

# Arguments
- `d`: Physical dimension (default: 2)
- `D`: PEPS bond dimension (default: 2)
- `J`: Coupling strength (default: 1.0)
- `g_values`: Range of transverse field values (default: 0.0:0.25:4.0)
- `χ`: Environment bond dimension for CTMRG (default: 20)
- `ctmrg_tol`: CTMRG convergence tolerance (default: 1e-10)
- `grad_tol`: Gradient tolerance (default: 1e-6)
- `maxiter`: Maximum iterations (default: 1000)
- `output_file`: Path to save JSON results (default: "pepskit_results.json")

# Returns
Dictionary with results for each g value
"""
function run_pepskit_scan(; d::Int=2, D::Int=2, J::Float64=1.0,
                           g_values=0.0:0.25:4.0,
                           χ::Int=10, ctmrg_tol::Float64=1e-8,
                           grad_tol::Float64=1e-4, maxiter::Int=1000,
                           ctmrg_maxiter::Int=400, reuse_env::Bool=true,
                           robust_svd::Bool=false,
                           warm_start::Bool=true,
                           n_starts::Int=2,
                           output_file::String="pepskit_results.json")

    results = Dict(
        "parameters" => Dict(
            "d" => d,
            "D" => D,
            "J" => J,
            "χ" => χ,
            "ctmrg_tol" => ctmrg_tol,
            "grad_tol" => grad_tol,
            "maxiter" => maxiter,
            "ctmrg_maxiter" => ctmrg_maxiter,
            "reuse_env" => reuse_env,
            "robust_svd" => robust_svd,
            "warm_start" => warm_start,
            "n_starts" => n_starts
        ),
        "g_values" => collect(g_values),
        "energies" => Any[],
        "correlation_lengths" => Any[]
    )

    # Ensure output directory exists
    mkpath(dirname(output_file))

    println("=" ^ 60)
    println("PEPSKit Ground State Scan")
    println("d=$d, D=$D, J=$J, χ=$χ")
    println("g values: ", collect(g_values))
    println("=" ^ 60)

    prev_peps = nothing
    prev_env  = nothing

    # Helper: run pepskit_ground_state from a given init, returning result or nothing.
    function _try_run(g_val, peps_init, env_init; force_robust::Bool=false)
        try
            return pepskit_ground_state(d, D, J, g_val; χ=χ, ctmrg_tol=ctmrg_tol,
                                        grad_tol=grad_tol, maxiter=maxiter,
                                        ctmrg_maxiter=ctmrg_maxiter,
                                        reuse_env=reuse_env,
                                        robust_svd=(robust_svd || force_robust),
                                        peps_init=peps_init, env_init=env_init)
        catch e
            println("    init failed: $e")
            return nothing
        end
    end

    for (i, g) in enumerate(g_values)
        println("\n[$i/$(length(g_values))] Running g = $g ...")

        # Build candidate inits: warm-start first (if available), then fresh random seeds.
        # Keep the lowest-energy successful result.
        candidates = Any[]
        if warm_start && prev_peps !== nothing
            push!(candidates, (prev_peps, prev_env))
        end
        for _ in 1:max(1, n_starts - length(candidates))
            push!(candidates, (nothing, nothing))  # fresh random init inside pepskit_ground_state
        end

        best = nothing
        for (k, (p0, e0)) in enumerate(candidates)
            println("  start $k/$(length(candidates)) (warm=$(p0 !== nothing)) ...")
            res = _try_run(g, p0, e0)
            if res === nothing
                continue
            end
            if best === nothing || real(res.energy) < real(best.energy)
                best = res
            end
        end

        # Last-chance retry with robust SVD if every candidate crashed.
        if best === nothing
            println("  all starts failed; retrying with force_robust=true ...")
            best = _try_run(g, nothing, nothing; force_robust=true)
        end

        if best !== nothing
            energy = real(best.energy)
            ξ = best.correlation_length
            push!(results["energies"], energy)
            push!(results["correlation_lengths"], ξ)
            prev_peps = best.peps
            prev_env  = best.env
            println("  Energy: $energy")
            println("  Correlation length: $ξ")
        else
            println("  FAILED at g=$g; keeping previous warm-start for next point.")
            push!(results["energies"], nothing)
            push!(results["correlation_lengths"], nothing)
            # Intentionally do NOT zero prev_peps/prev_env so the chain survives.
        end

        # Save intermediate results
        open(output_file, "w") do io
            JSON3.pretty(io, results)
        end
        println("  Results saved to $output_file")
    end
    
    println("\n" * "=" ^ 60)
    println("Scan complete! Results saved to $output_file")
    println("=" ^ 60)
    
    return results
end

"""
    plot_pepskit_energy(json_file::String; save_path=nothing)

Plot ground state energy vs transverse field g from pepskit results.

# Arguments
- `json_file`: Path to the pepskit results JSON file
- `save_path`: Optional path to save the figure

# Returns
- `fig`: Makie Figure object
"""
function plot_pepskit_energy(json_file::String; save_path=nothing)
    # Load JSON data
    data = JSON3.read(read(json_file, String))
    
    g_values = collect(data.g_values)
    energies = collect(data.energies)
    D = data.parameters.D

    mask = [e !== nothing && !isnothing(e) for e in energies]
    gs = Float64[g_values[i] for i in eachindex(mask) if mask[i]]
    es = Float64[Float64(energies[i]) for i in eachindex(mask) if mask[i]]

    # Create figure
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              xlabel="g (transverse field)",
              ylabel="Energy",
              title="TFIM Ground State Energy (PEPSKit D=$D)")

    scatterlines!(ax, gs, es,
                  color=:steelblue, marker=:circle, markersize=8,
                  linewidth=2, label="E(g)")
    
    # Save if requested
    if !isnothing(save_path)
        save(save_path, fig)
        println("Figure saved to $save_path")
    end
    
    return fig
end

"""
    plot_corr_PEPSKit(filename::String; save_path=nothing)

Plot correlation length vs transverse field g from PEPSKit results.

# Arguments
- `filename`: Path to JSON file with PEPSKit results
- `save_path`: Optional path to save the figure

# Returns
- `fig`: Makie Figure object
"""
function plot_corr_PEPSKit(filename::String="project/results/pepskit_results_D=2.json"; 
                           save_path::Union{String,Nothing}=nothing)
    # Load data
    data = JSON3.read(read(filename, String))
    
    g_values = collect(data[:g_values])
    correlation_lengths = collect(data[:correlation_lengths])
    energies = collect(data[:energies])

    # Drop points where either ξ or E failed (JSON null → Nothing).
    mask = [ξ !== nothing && !isnothing(ξ) for ξ in correlation_lengths]
    gs = Float64[g_values[i] for i in eachindex(mask) if mask[i]]
    ξs = Float64[Float64(correlation_lengths[i]) for i in eachindex(mask) if mask[i]]

    # Create figure
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              xlabel="Transverse field g",
              ylabel="Correlation length ξ",
              title="PEPSKit: Correlation Length vs g (D=$(data[:parameters][:D]))")

    # Plot correlation length (null entries filtered out so the line is continuous)
    lines!(ax, gs, ξs, color=:blue, linewidth=2, label="PEPSKit")
    scatter!(ax, gs, ξs, color=:blue, markersize=8)
    
    # Add vertical line at critical point g ≈ 3.04
    vlines!(ax, [3.04], color=:red, linestyle=:dash, linewidth=1.5, label="g_c ≈ 3.04")
    
    # Legend
    axislegend(ax, position=:rt)
    
    if !isnothing(save_path)
        save(save_path, fig)
        @info "Figure saved to $save_path"
    end
    
    return fig
end

# Run the scan

"""
    run_pepskit_scan_bidirectional(; g_values, output_file, kwargs...)

Run `run_pepskit_scan` forward and backward over `g_values` and merge by lowest
energy at each g. Writes three JSON files: `<output_file>` (merged),
`<output_file>.fwd.json`, `<output_file>.bwd.json`.
"""
function run_pepskit_scan_bidirectional(; g_values, output_file::String, kwargs...)
    gs = g_values
    fwd_file = replace(output_file, r"\.json$" => ".fwd.json")
    bwd_file = replace(output_file, r"\.json$" => ".bwd.json")

    fwd = run_pepskit_scan(; g_values=gs,                    output_file=fwd_file, kwargs...)
    bwd = run_pepskit_scan(; g_values=reverse(gs),           output_file=bwd_file, kwargs...)

    # Merge: at each g, pick the result with lower energy (ignoring nothings).
    fwd_map = Dict(zip(fwd["g_values"], zip(fwd["energies"], fwd["correlation_lengths"])))
    bwd_map = Dict(zip(bwd["g_values"], zip(bwd["energies"], bwd["correlation_lengths"])))

    energies = Any[]
    ξs = Any[]
    for g in gs
        ef, ξf = get(fwd_map, g, (nothing, nothing))
        eb, ξb = get(bwd_map, g, (nothing, nothing))
        pick_fwd = ef !== nothing && (eb === nothing || ef <= eb)
        if pick_fwd
            push!(energies, ef); push!(ξs, ξf)
        elseif eb !== nothing
            push!(energies, eb); push!(ξs, ξb)
        else
            push!(energies, nothing); push!(ξs, nothing)
        end
    end

    merged = Dict(
        "parameters" => fwd["parameters"],
        "g_values" => gs,
        "energies" => energies,
        "correlation_lengths" => ξs,
    )
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        JSON3.pretty(io, merged)
    end
    println("Merged bidirectional scan written to $output_file")
    return merged
end

# Piecewise-refined g grid: coarse far from g_c ≈ 3.04, fine across the transition.
_g_grid = vcat(collect(4.0:-0.25:3.75), collect(3.5:-0.05:2.6),collect(2.5:-0.25:0.5))
                            
                        

output_file = joinpath(@__DIR__, "results", "pepskit_results_D=2_g=4.25_4.5_4.75_5.0.json")

results = run_pepskit_scan_bidirectional(
        d = 2,
        D = 2,
        J = 1.0,
        g_values = [4.25, 4.5, 4.75, 5.0],
        χ = 10,
        ctmrg_tol = 1e-8,
        grad_tol = 1e-4,
        maxiter = 50,
        ctmrg_maxiter = 800,
        robust_svd = true,
        warm_start = true,
        n_starts = 1,
        output_file = output_file,
    )

plot_pepskit_energy(output_file; save_path=joinpath(@__DIR__, "results", "figures", "pepskit_energy_smooth_D=4.pdf"))

fig = plot_corr_PEPSKit(output_file;
                        save_path=joinpath(@__DIR__, "results", "figures", "pepskit_correlation_length_smooth.pdf"))
display(fig)