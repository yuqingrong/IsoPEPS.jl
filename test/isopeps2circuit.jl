using IsoPEPS
using Test
using Graphs, GraphPlot, Compose
import Yao

@testset "isopeps2circuit" begin
    g = SimpleDiGraph(16)
    g2 = grid([4,4])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    #draw(PNG("g.png", 16cm, 16cm), gplot(g))
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    peps, pepsu = peps2ugate(peps, g)
    
    @test isapprox(pepsu.vertex_tensors[1]'*pepsu.vertex_tensors[1], Matrix(I, 8, 8),atol=1e-10)  
    @test isapprox(pepsu.vertex_tensors[1]*pepsu.vertex_tensors[1]', Matrix(I, 8, 8),atol=1e-10)  
    @test isapprox((reshape(peps.vertex_tensors[2], 8, 2)'*reshape(peps.vertex_tensors[2], 8, 2)), Matrix(I, 2, 2), atol=1e-10)


    circ = get_circuit(pepsu, g)
    @test collect_blocks(IsoPEPS.Measure, circ)|>length == 16
  
    reg = Yao.zero_state(6;nbatch=1000)
    res = gensample(circ, reg, pepsu, Z)
    @test res[1,1] in [0,1]
    @test all(x -> x in [0,1], res)

 
    corr1 = long_range_coherence(circ, reg, pepsu, 3, 4)
    corr_expect1= long_range_coherence_peps(peps, 3, 4)
    @show corr_expect1
    

    @test isapprox(corr1, corr_expect1, atol=1e-2)
    @show corr1
    @test 0 ≤ corr1 ≤ 1
end
    







g = SimpleDiGraph(16)
g2 = grid([4,4])
edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
for (i,j) in edge_pairs
    add_edge!(g, i, j)
end
    
peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
peps, pepsu = peps2ugate(peps, g)
circ = get_circuit(pepsu, g)
batch_sizes = 1000:1000:60000

    
corrs = Float64[]
    
for batch_size in batch_sizes
    reg = Yao.zero_state(6; nbatch=batch_size)
    corr = long_range_coherence(circ, reg, pepsu, 5, 6)
   
    push!(corrs, corr)
    @show batch_size
end
    
corr_expect = long_range_coherence_peps(peps, 5, 6)
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1], 
title = "Correlation vs Batch Size",
xlabel = "Batch Size",
ylabel = "Correlation")
    
lines!(ax, collect(batch_sizes), corrs, label="1")
hlines!(ax, [corr_expect], label="2", linestyle=:dash)

axislegend()
    
save("correlation_vs_batchsize.png", fig)
    


# Calculate error bars by running multiple trials
n_trials = 10
all_corrs = zeros(length(batch_sizes), n_trials)

for (i, batch_size) in enumerate(batch_sizes)
    for j in 1:n_trials
        reg = Yao.zero_state(6; nbatch=batch_size)
        all_corrs[i,j] = long_range_coherence(circ, reg, pepsu, 5, 6)
    end
    @show batch_size
end

# Calculate mean and standard error
mean_corrs = mean(all_corrs, dims=2)[:,1]
std_errs = std(all_corrs, dims=2)[:,1] ./ sqrt(n_trials)

# Plot with error bars
fig = Figure()
ax = Axis(fig[1,1],
    title = "Correlation vs Batch Size with Error Bars",
    xlabel = "Batch Size",
    ylabel = "Correlation"
)

errorbars!(ax, collect(batch_sizes), mean_corrs, std_errs, 
    whiskerwidth=10)
    scatter!(ax, collect(batch_sizes), mean_corrs, 
    label="S")

    hlines!(ax, [corr_expect], label="E", linestyle=:dash)
axislegend()

save("correlation_with_errors.png", fig)


