using Plots

# Data from iter_steps.dat
g_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
error_accuracies = [1e-2, 1e-4, 1e-6, 1e-8]
error_labels = ["1e-2", "1e-4", "1e-6", "1e-8"]

# Iteration steps for each g value and error accuracy
iter_steps = [
    [569, 862, 1208, 1568],      # g=0.0
    [645, 9128, 14817, 81747],   # g=0.25
    [579, 7663, 9162, 28406],    # g=0.5
    [2526, 5995, 7030, 6474],    # g=0.75
    [2723, 4262, 3194, 4288],    # g=1.0
    [1199, 3397, 3198, 3158],    # g=1.25
    [1579, 2546, 3098, 2906],    # g=1.5
    [1349, 1563, 2539, 2936],    # g=1.75
    [1107, 2240, 3064, 2425],    # g=2.0
]

# Plot 1: Iteration steps vs error accuracy for each g
p1 = plot(
    title = "Iteration Steps vs Error Accuracy",
    xlabel = "Error Accuracy",
    ylabel = "Iteration Steps",
    xscale = :log10,
    yscale = :log10,
    legend = :outertopright,
    size = (800, 600),
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    xflip = true,  # Higher accuracy (smaller error) on the right
    legendfontsize = 8,
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10
)

# Plot each g value
for (i, g) in enumerate(g_values)
    plot!(p1, error_accuracies, iter_steps[i], 
          marker = :circle, 
          markersize = 6,
          linewidth = 2,
          label = "g=$(g)")
end

# Plot 2: Iteration steps vs g for each error accuracy
p2 = plot(
    title = "Iteration Steps vs g Value",
    xlabel = "g",
    ylabel = "Iteration Steps",
    yscale = :log10,
    legend = :topright,
    size = (800, 600),
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    legendfontsize = 10,
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10
)

# Plot each error accuracy
for (j, acc) in enumerate(error_accuracies)
    steps_for_this_acc = [iter_steps[i][j] for i in 1:length(g_values)]
    plot!(p2, g_values, steps_for_this_acc,
          marker = :circle,
          markersize = 6,
          linewidth = 2,
          label = error_labels[j])
end

# Combine plots
combined_plot = plot(p1, p2, layout = (1, 2), size = (1600, 600))

# Save plots as PDF
savefig(combined_plot, "/home/yuqingrong/IsoPEPS.jl/iteration_steps_analysis.pdf")
savefig(p1, "/home/yuqingrong/IsoPEPS.jl/iteration_steps_vs_error.pdf")
savefig(p2, "/home/yuqingrong/IsoPEPS.jl/iteration_steps_vs_g.pdf")

println("Plots saved:")
println("  - iteration_steps_analysis.pdf (combined)")
println("  - iteration_steps_vs_error.pdf (left plot)")
println("  - iteration_steps_vs_g.pdf (right plot)")

display(combined_plot)
