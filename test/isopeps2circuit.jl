using IsoPEPS
using Test
using Graphs, GraphPlot, Compose
import Yao

@testset "isopeps2circuit" begin
    g = SimpleDiGraph(4)
    g2 = Graphs.grid([2,2])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    #draw(PNG("g.png", 16cm, 16cm), gplot(g))
    #peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    #peps, pepsu = peps2ugate(peps, g)
    peps,_ = isometric_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = isometric_peps_to_unitary(peps, g)
    
    #@test isapprox(pepsu.vertex_tensors[1]'*pepsu.vertex_tensors[1], Matrix(I, 8, 8),atol=1e-10)  
    #@test isapprox(pepsu.vertex_tensors[1]*pepsu.vertex_tensors[1]', Matrix(I, 8, 8),atol=1e-10)  
    #@test isapprox((reshape(peps.vertex_tensors[2], 4, 2)'*reshape(peps.vertex_tensors[2], 4, 2)), Matrix(I, 2, 2), atol=1e-10)

    circ1 = get_circuit(pepsu, g)
    circ2 = new_get_circuit(pepsu, g)
   
    @test collect_blocks(IsoPEPS.Measure, circ2)|>length == 5
 
    reg1 = Yao.zero_state(5;nbatch=100000)
    reg2 = Yao.zero_state(10;nbatch=100000)
    res1 = gensample(circ1, reg1, pepsu, Yao.Z)
   
    @test res1[1,1] in [0,1]
    @test all(x -> x in [0,1], res1)
   

 
    corr1 = long_range_coherence(circ1, reg1, pepsu, 2, 3)
    corr2 = long_range_coherence(circ2, reg2, pepsu, 2, 3)
    corr_expect1= long_range_coherence_peps(peps, 2, 3) 
    @show corr_expect1
    

    @test isapprox(corr2, corr_expect1, atol=1e-2)
    @show corr1, corr2
    @test 0 ≤ corr1 ≤ 1
end
   



@testset "Bell State" begin
    
    g = SimpleDiGraph(2)
    g2 = grid([1,2])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    peps = zero_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    # Create a Bell state by setting specific tensor values
    # For a 2-qubit system with a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    
    # Reset the tensors to zeros
    peps.vertex_tensors[1] .= 0
    peps.vertex_tensors[2] .= 0
    
    # Set the tensor elements to create a Bell state
    # For the first site tensor (with shape 2×2)
    peps.vertex_tensors[1][1,1] = 1/sqrt(2)  # |0⟩ component
    peps.vertex_tensors[1][2,2] = 1/sqrt(2)  # |1⟩ component
    
    # For the second site tensor (with shape 2×2)
    peps.vertex_tensors[2][1,1] = 1  # |0⟩ component for first site's |0⟩
    peps.vertex_tensors[2][2,2] = 1  # |1⟩ component for first site's |1⟩
    
    # Create ugates with the same structure as peps but with 4x4 tensors
    ugates = deepcopy(peps)
    
    # Initialize tensors with zeros
    ugates.vertex_tensors[1] = zeros(ComplexF64, 4, 4)
    
    # Set values for the first tensor (4x4)
    ugates.vertex_tensors[1][1,1] = 1/sqrt(2)  # |0⟩ component
    ugates.vertex_tensors[1][1,4] = 1/sqrt(2)  # |1⟩ component
    ugates.vertex_tensors[1][2,2] = 1          # Identity part
    ugates.vertex_tensors[1][3,3] = 1          # Identity part
    ugates.vertex_tensors[1][4,1] = 1/sqrt(2)  # |1⟩ component
    ugates.vertex_tensors[1][4,4] = -1/sqrt(2)  # |0⟩ component
    
    
    
    # Display the original tensors for comparison
    @show peps.vertex_tensors[1]
    @show peps.vertex_tensors[2]
    @show ugates.vertex_tensors[1]
    @show ugates.vertex_tensors[2]

    corr_expect1= long_range_coherence_peps(peps, 1, 2)
    @show corr_expect1

    circ = get_circuit(ugates, g)
    reg = Yao.zero_state(3;nbatch=10)

    res = gensample(circ, reg, ugates, Z)
    @show res
    corr1 = long_range_coherence(circ, reg, ugates, 1, 2)
    
    @test isapprox(corr1, corr_expect1, atol=1e-2)
    @show corr1
    @test 0 ≤ corr1 ≤ 1
end





g = SimpleDiGraph(4)
g2 = Graphs.grid([2,2])
edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
for (i,j) in edge_pairs
    add_edge!(g, i, j)
end
    
peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
peps, pepsu = peps2ugate(peps, g)
circ = get_circuit(pepsu, g)
batch_sizes = 1000:1000:100000

    
corrs = Float64[]
    
for batch_size in batch_sizes
    reg = Yao.zero_state(5; nbatch=batch_size)
    corr = long_range_coherence(circ, reg, pepsu, 1, 2)
   
    push!(corrs, corr)
    @show batch_size
end
    
corr_expect = 0.2
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1], 
title = "Correlation vs Batch Size",
xlabel = "Batch Size",
ylabel = "Correlation")
    
lines!(ax, collect(batch_sizes), corrs, label="correlation from circuit")
hlines!(ax, [corr_expect], label="exact correlation", linestyle=:dash)

axislegend()
    
save("Correlation vs Batchsize.png", fig)
    


# Calculate error bars by running multiple trials
n_trials = 10
all_corrs = zeros(length(batch_sizes), n_trials)

for (i, batch_size) in enumerate(batch_sizes)
    for j in 1:n_trials
        reg = Yao.zero_state(5; nbatch=batch_size)
        all_corrs[i,j] = long_range_coherence(circ, reg, pepsu, 1, 2)
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


