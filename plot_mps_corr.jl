using Plots
using Statistics

# Data from mps_corr_1d.dat - Section 1 (single values)
g_values_single = [0.0, 0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
corr_values = [1.6807228510032075, 0.09938670503395748, 0.2573062539192835, 
               0.2943180399170259, 0.4032480892655934, 0.604021257475444, 
               0.6600294372998066, 0.5780850766348179, 0.5258118967403669, 
               0.488680392165385]

# Data from mps_corr_1d.dat - Section 2 (multiple gap values)
gap_data = Dict(
    0.0 => [0.13056698538248107, 2.1967886858247745, 0.8295259618607611, 
            0.5331832542885059, 0.29068188885632923, 0.36671446750162595, 
            0.6817940573561174, 2.0836568508625226, 0.9695605202970027],
    0.25 => [1.7921195052424679, 2.3381582525694347, 4.836380591824487, 
             1.95886744120102, 1.3448302312439586, 3.5173976440790247, 
             0.023709189981265442, 2.21939230734811, 2.0842568945917637],
    0.5 => [3.2089781711818683, 1.5894289983905805, 3.5311199546498457, 
            3.397902224257587],
    0.75 => [4.434121677836256, 2.4798670107995866]
)

g_values_gaps = sort(collect(keys(gap_data)))

# === Plot 1: Correlation values from section 1 ===
p1 = plot(
    title = "Correlation Length vs g",
    xlabel = "g",
    ylabel = "Correlation Length",
    legend = false,
    size = (800, 600),
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10
)

plot!(p1, g_values_single, corr_values,
      marker = :circle,
      markersize = 8,
      linewidth = 2,
      color = :blue)

# === Plot 2: Scatter plot of all gap values ===
p2 = plot(
    title = "Spectral Gaps Distribution",
    xlabel = "g",
    ylabel = "Gap Value",
    legend = false,
    size = (800, 600),
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10
)

# Plot all gap values as scatter points
for g in g_values_gaps
    gaps = gap_data[g]
    g_positions = fill(g, length(gaps))
    scatter!(p2, g_positions, gaps,
             marker = :circle,
             markersize = 8,
             alpha = 0.7,
             color = :red)
end

# === Plot 3: Gap values with statistics (mean and std) ===
p3 = plot(
    title = "Spectral Gaps with Mean ± Std",
    xlabel = "g",
    ylabel = "Gap Value",
    legend = :topright,
    size = (800, 600),
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10
)

# Plot individual points
for g in g_values_gaps
    gaps = gap_data[g]
    g_positions = fill(g, length(gaps))
    scatter!(p3, g_positions, gaps,
             marker = :circle,
             markersize = 6,
             alpha = 0.5,
             color = :lightblue,
             label = (g == g_values_gaps[1] ? "Individual gaps" : ""))
end

# Calculate and plot mean values
means = [mean(gap_data[g]) for g in g_values_gaps]
stds = [std(gap_data[g]) for g in g_values_gaps]

plot!(p3, g_values_gaps, means,
      ribbon = stds,
      marker = :diamond,
      markersize = 10,
      linewidth = 3,
      color = :darkblue,
      fillalpha = 0.3,
      label = "Mean ± Std")

# === Plot 4: Sorted gap values (showing gap index) ===
p4 = plot(
    title = "Sorted Spectral Gaps",
    xlabel = "g",
    ylabel = "Gap Value",
    legend = :topright,
    size = (800, 600),
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10
)

# Sort gaps for each g and plot them by index
colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :cyan]
for g in g_values_gaps
    sorted_gaps = sort(gap_data[g], rev=true)  # Sort from largest to smallest
    for (idx, gap) in enumerate(sorted_gaps)
        color_idx = min(idx, length(colors))
        scatter!(p4, [g], [gap],
                marker = :circle,
                markersize = 8,
                color = colors[color_idx],
                label = (g == g_values_gaps[1] ? "Gap $idx" : ""),
                alpha = 0.7)
    end
end

# === Combine all plots ===
combined = plot(p1, p2, p3, p4, 
                layout = (2, 2), 
                size = (1600, 1200),
                margin = 5Plots.mm)

# Save plots
savefig(combined, "/home/yuqingrong/IsoPEPS.jl/mps_corr_analysis.pdf")
savefig(p1, "/home/yuqingrong/IsoPEPS.jl/correlation_length.pdf")
savefig(p2, "/home/yuqingrong/IsoPEPS.jl/spectral_gaps_scatter.pdf")
savefig(p3, "/home/yuqingrong/IsoPEPS.jl/spectral_gaps_statistics.pdf")
savefig(p4, "/home/yuqingrong/IsoPEPS.jl/spectral_gaps_sorted.pdf")

println("Plots saved:")
println("  - mps_corr_analysis.pdf (all 4 plots combined)")
println("  - correlation_length.pdf (plot 1)")
println("  - spectral_gaps_scatter.pdf (plot 2)")
println("  - spectral_gaps_statistics.pdf (plot 3)")
println("  - spectral_gaps_sorted.pdf (plot 4)")

display(combined)

