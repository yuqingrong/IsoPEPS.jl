using IsoPEPS
using CairoMakie
using Graphs, GraphPlot, Compose


function plot_peps(Lx::Int, Ly::Int)
    g_nodirection = Graphs.grid([Lx, Ly])
    draw(PNG("g_nodirection.png", 16cm, 16cm), gplot(g_nodirection))
    save("example/figures/g_nodirection.png", gplot(g_nodirection))
end










plot_peps(4,4)








