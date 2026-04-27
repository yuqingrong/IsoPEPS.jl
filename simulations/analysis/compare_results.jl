"""
Compare results across models and methods.

Usage:
    julia --project=simulations simulations/analysis/compare_results.jl
"""

using IsoPEPS
using JSON3
using CairoMakie
set_theme!(IsoPEPS.paper_theme())

"""
    compare_energies(result_files::Vector{String}; labels=nothing, save_path=nothing)

Compare energy results from multiple simulation runs.
"""
function compare_energies(result_files::Vector{String};
                          labels::Union{Vector{String},Nothing}=nothing,
                          save_path::Union{String,Nothing}=nothing)
    fig = Figure(size=IsoPEPS.PAPER_FIGSIZE)
    ax = Axis(fig[1, 1], xlabel="Scan parameter", ylabel="Energy")

    for (i, f) in enumerate(result_files)
        data = open(f, "r") do io
            JSON3.read(io, Dict)
        end
        label = isnothing(labels) ? basename(f) : labels[i]
        energy = get(data, "energy", get(data, :energy, NaN))
        println("$label: E = $energy")
    end

    if !isnothing(save_path)
        save(save_path, fig)
    end
    return fig
end
