using CairoMakie, isoPEPS

function plot_convergence(x, y)
    fig = Figure()
    ax2 = Axis(fig[1, 1], title = "Krylovkit and Yao", xlabel = "iter", ylabel = "δE/L")
    lines!(ax2, x, y, color = :blue,label="δE/L")
    axislegend(ax2)
    display(fig)
    save("KrylovKitYao.png", fig)
    fig
end