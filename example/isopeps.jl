# 4*4 square lattice Ising model

using IsoPEPS
using CairoMakie
using Graphs, GraphPlot, Compose
using Manifolds
using Yao
using YaoPlots


function plot_isopeps(Lx::Int, Ly::Int)
    g_direction = SimpleDiGraph(Lx*Ly)
    g2 = Graphs.grid([Lx, Ly])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g_direction, i, j)
    end
    draw(PNG("g_direction.png", 16cm, 16cm), gplot(g_direction))
    save("example/figures/g_direction.png", gplot(g_direction))
end


function plot_cir(pepsu, g)
    circ = get_circuit(pepsu, g)
    viz = vizcircuit(circ)
    draw(PNG("example/figures/circuit.png", 12cm, 8cm), viz)
    save("example/figures/circuit.png", viz)
end



function energy_vs_iteration(peps::PEPS, M::ProductManifold, matrix_dims::Vector, g, J::Float64)  # different h
    h_values = [0.2, 0.4, 0.6]
    steps = collect(1:10:100)
    
    # Collect energy values for each h
    energy_curves = Dict{Float64, Vector{Float64}}()
    iteration_list = Dict{Float64, Vector{Int}}()
    
    for h in h_values
        # Create a fresh PEPS for each h value
        fresh_peps, fresh_matrix_dims = isometric_peps(Float64, g, 2, 2, IsoPEPS.TreeSA(), IsoPEPS.MergeGreedy())
        fresh_M = ProductManifold([Manifolds.Stiefel(n, p) for (n, p) in fresh_matrix_dims]...)
        
        current_iterations = Int[]
        current_energies = Float64[]
        
       
        result, energy, optimized_peps, record = isopeps_optimize_ising(
            fresh_peps, fresh_M, fresh_matrix_dims, g, J, h, 
            IsoPEPS.GreedyMethod(), IsoPEPS.MergeGreedy(), 100
        )
        
       
        iteration_list[h] = [ r[1] for r in record[1:10:end] ]
        energy_curves[h] = [ r[2] for r in record[1:10:end] ]
    end
    
    # Create plot
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
              xlabel = "Iteration", 
              ylabel = "Energy", 
              title = "Energy vs Iteration for Different h Values")
    
    # Plot each curve
    for h in h_values
        lines!(ax, iteration_list[h], energy_curves[h], 
               label = "h = $h", 
               linewidth = 2)
    end
    
    axislegend(ax, position = :rt)
    save("example/figures/energy_vs_iteration.png", fig)
    
    return fig
end


function energy_accuracy_vs_D(Lx::Int, Ly::Int, g::SimpleDiGraph, J::Float64, h::Float64)  # different D
    D_values = [2, 3]
    steps = collect(1:10:100)
    energy_differences = Dict{Float64, Vector{Float64}}()
    iteration_list = Dict{Float64, Vector{Int}}()

    hami = ising_hamiltonian_2d(Lx, Ly, J, h)
    eigenval, eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(hami), 1, :SR; ishermitian=true)
    energy_exact = eigenval[1]

    for D in D_values 
        
        
        peps, matrix_dims = isometric_peps(Float64, g, D, 2, TreeSA(), MergeGreedy())
        M = ProductManifold([Manifolds.Stiefel(n, p) for (n, p) in matrix_dims]...)   
        result, energy, optimized_peps, record = isopeps_optimize_ising(peps, M, matrix_dims, g, J, h, IsoPEPS.GreedyMethod(), IsoPEPS.MergeGreedy(), 50)
            
        @show record
        iteration_list[D] = [ r[1] for r in record[1:5:end] ]
        energy_differences[D] = [ abs(r[2] - energy_exact) for r in record[1:5:end] ]
      
    end
    
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
              xlabel = "Bond Dimension D", 
              ylabel = "|Energy - Exact Energy|", 
              title = "Energy Accuracy vs iterations ($(Lx)×$(Ly) lattice)",
              yscale = log10)
    

    for D in D_values
        lines!(ax, iteration_list[D], energy_differences[D], 
               label = "D = $D", 
               linewidth = 2 )
    end

    axislegend(ax, position = :rt)
    save("example/figures/energy_accuracy_vs_D.png", fig)
    
    return fig
end


function correlation()

end


Lx = Ly = 4
D = 2
g = SimpleDiGraph(Lx*Ly)
g2 = Graphs.grid([Lx, Ly])
edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
for (i,j) in edge_pairs
    add_edge!(g, i, j)
end
peps, matrix_dims = isometric_peps(Float64, g, D, 2, TreeSA(), MergeGreedy())
    
M = ProductManifold([Manifolds.Stiefel(n, p) for (n, p) in matrix_dims]...)

J,h = 1.0, 0.2

hami = ising_hamiltonian_2d(4, 4, J, h)
eigenval, eigenvec = IsoPEPS.eigsolve(IsoPEPS.mat(hami), 1, :SR; ishermitian=true)
energy_exact = eigenval[1]

plot_isopeps(4, 4)

#energy vs iteration plot
energy_vs_iteration(peps, M, matrix_dims, g, J)


#energy accuracy vs D plot
energy_accuracy_vs_D(Lx, Ly, g, J, h)



# plot quantum circuit
pepsu = isometric_peps_to_unitary(peps, g)
plot_cir(pepsu, g)