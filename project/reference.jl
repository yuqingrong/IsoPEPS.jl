using IsoPEPS
using PEPSKit
using JSON3
using CairoMakie

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
                           χ::Int=10, ctmrg_tol::Float64=1e-10, 
                           grad_tol::Float64=1e-6, maxiter::Int=1000,
                           output_file::String="pepskit_results.json")
    
    results = Dict(
        "parameters" => Dict(
            "d" => d,
            "D" => D,
            "J" => J,
            "χ" => χ,
            "ctmrg_tol" => ctmrg_tol,
            "grad_tol" => grad_tol,
            "maxiter" => maxiter
        ),
        "g_values" => collect(g_values),
        "energies" => Float64[],
        "correlation_lengths" => Float64[]
    )
    
    # Ensure output directory exists
    mkpath(dirname(output_file))
    
    println("=" ^ 60)
    println("PEPSKit Ground State Scan")
    println("d=$d, D=$D, J=$J, χ=$χ")
    println("g values: ", collect(g_values))
    println("=" ^ 60)
    
    for (i, g) in enumerate(g_values)
        println("\n[$i/$(length(g_values))] Running g = $g ...")
        
        try
            result = pepskit_ground_state(d, D, J, g; χ=χ, ctmrg_tol=ctmrg_tol, 
                                          grad_tol=grad_tol, maxiter=maxiter)
            
            energy = real(result.energy)
            ξ = result.correlation_length
            
            push!(results["energies"], energy)
            push!(results["correlation_lengths"], ξ)
            
            println("  Energy: $energy")
            println("  Correlation length: $ξ")
            
        catch e
            println("  ERROR: $e")
            push!(results["energies"], NaN)
            push!(results["correlation_lengths"], NaN)
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
    
    # Create figure
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              xlabel="g (transverse field)",
              ylabel="Energy",
              title="TFIM Ground State Energy (PEPSKit D=$D)")
    
    # Plot energy vs g
    scatterlines!(ax, g_values, energies, 
                  color=:steelblue, marker=:circle, markersize=8,
                  linewidth=2, label="E(g)")
    
    # Save if requested
    if !isnothing(save_path)
        save(save_path, fig)
        println("Figure saved to $save_path")
    end
    
    return fig
end

# Run the scan
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_pepskit_scan(
        d = 2,
        D = 4,
        J = 1.0,
        g_values = 0.0:0.25:4.0,
        χ = 20,
        ctmrg_tol = 1e-10,
        grad_tol = 1e-6,
        maxiter = 1000,
        output_file = joinpath(@__DIR__, "results", "pepskit_results_D=4_χ=20.json")
    )
end

plot_pepskit_energy("project/results/pepskit_results_D=2.json")