
using Colors, Plots, Statistics

function draw_figure()
    p_values = [2, 4, 6]
    colors = [RGBA(1,0,0,0.5), RGBA(0,1,0,0.7), RGBA(1,0.5,0,0.5), RGBA(0,0,1,0.5)]
    markers = [:+, :x, :star,:diamond]

    fig = Plots.plot(xlabel="g", ylabel="ε - ε_exact", title="Energy Error vs Transverse Field", 
                     ylims=(1e-15, 10), xlims=(-0.25, 2), 
                     yscale=:log10, 
                     yticks=[1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1],
                     xticks=[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    
    for (p_idx, p_val) in enumerate(p_values)
        filename = "data/energy_results_p=$(p_val).dat"
        
        if isfile(filename)
            g_values = Float64[]
            errors = Float64[]
            
            open(filename, "r") do file
                for line in eachline(file)
                    if !isempty(strip(line)) && !startswith(line, "#")
                        parts = split(line) 
                        if length(parts) >= 3
                            try
                                g = parse(Float64, parts[1])
                                error = parse(Float64, parts[3]) 
                                push!(g_values, g)
                                push!(errors, max(error, 1e-15))
                            catch e
                                println("Warning: Could not parse line: $line")
                            end
                        end
                    end
                end
            end
            
            if !isempty(g_values)
                Plots.plot!(fig, g_values, errors,
                    label="p=$p_val",
                    color=colors[p_idx],
                    marker=markers[p_idx],
                    markersize=4,
                    linewidth=2)
                println("Added data for p=$p_val with $(length(g_values)) points")
            else
                println("Warning: No data found in $filename")
            end
        else
            println("Warning: File $filename not found")
        end
    end
    
    mps_filename = "data/exact_energies_mps.dat"
    if isfile(mps_filename)
        g_values_mps = Float64[]
        errors_mps = Float64[]
        
        open(mps_filename, "r") do file
            for line in eachline(file)
                if !isempty(strip(line)) && !startswith(line, "#")
                    parts = split(line)  
                    if length(parts) >= 3
                        try
                            g = parse(Float64, parts[1])
                            error = parse(Float64, parts[3])  
                            push!(g_values_mps, g)
                            push!(errors_mps, max(error, 1e-15))
                        catch e
                            println("Warning: Could not parse MPS line: $line")
                        end
                    end
                end
            end
        end
        
        if !isempty(g_values_mps)
            Plots.plot!(fig, g_values_mps, errors_mps,
                label="iMPS",
                color=colors[4],
                marker=markers[4],
                markersize=4,
                linewidth=2,
                linestyle=:dash)
            println("Added MPS vs Integration comparison")
        end
    end
    
    Plots.savefig(fig, "data/energy_error_plot.png")
    println("Plot saved as data/energy_error_plot.png")
    return fig
end

draw_figure()
