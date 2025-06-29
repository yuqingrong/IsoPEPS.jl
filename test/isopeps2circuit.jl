using IsoPEPS
using Test
using Graphs, GraphPlot, Compose
import Yao

@testset "isopeps2circuit" begin
    g = dgrid(2,2)
    #peps = rand_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    #peps, pepsu = peps2ugate(peps, g)
    peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = isometric_peps_to_unitary(peps, g)
    
    #@test isapprox(pepsu.vertex_tensors[1]'*pepsu.vertex_tensors[1], Matrix(I, 8, 8),atol=1e-10)  
    #@test isapprox(pepsu.vertex_tensors[1]*pepsu.vertex_tensors[1]', Matrix(I, 8, 8),atol=1e-10)  
    #@test isapprox((reshape(peps.vertex_tensors[2], 4, 2)'*reshape(peps.vertex_tensors[2], 4, 2)), Matrix(I, 2, 2), atol=1e-10)
    nbit1 =3
    nbit2 = 5
    circ1 = get_reuse_circuit(nbit1, pepsu, g)
    circ2 = get_circuit(nbit2, pepsu, g)
  
    @test collect_blocks(IsoPEPS.Measure, circ2)|>length == 5
 
    reg1 = Yao.zero_state(nbit1;nbatch=100000)
    reg2 = Yao.zero_state(nbit2;nbatch=100000)
    res1 = gensample(circ1, reg1, pepsu, Yao.Z)
    
    @test res1[1,1] in [0,1]
    @test all(x -> x in [0,1], res1)

 
    corr1 = long_range_coherence(circ1, reg1, pepsu, 2, 3)
    corr2 = long_range_coherence(circ2, reg2, pepsu, 2, 3)
    corr_expect= long_range_coherence_peps(peps, 2, 3) 
    
    @test isapprox(corr1, corr_expect, atol=1e-2)
    @test isapprox(corr2, corr_expect, atol=1e-2)
   
    @show corr_expect, corr1, corr2
    @test 0 ≤ corr1 ≤ 1
end
   
@testset "torus_circuit" begin
    g = dtorus(3,3)
    peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = isometric_peps_to_unitary(peps, g)
    circ = Yao.chain(7)
    circ = get_iter_circuit(circ, pepsu, 1, 6, collect(1:1), collect(2:7))
    @test collect_blocks(IsoPEPS.Measure, circ)|>length == 9
end

@testset "Sz_convergence" begin
    all_measurements1 = [zeros(Int,3) for _ in 1:100]
    all_measurements2 = [rand(Int,3) for _ in 1:100]
    @test Sz_convergence(all_measurements1) == true
    @test Sz_convergence(all_measurements2) == false
end

@testset "extract_sz_measurements" begin
    nbit = 7
    g = dtorus(3,3)
    peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = isometric_peps_to_unitary(peps, g)
    circ = Yao.chain(nbit)
    circ = get_iter_circuit(circ, pepsu, 1, 6, collect(1:1), collect(2:7))
    reg = Yao.zero_state(nbit)
    reg |> circ  # TODO: reg should be put to the total circuit
    iter_res = extract_sz_measurements(circ, nbit, g)
    @test size(iter_res) == (nv(g),)
    @test all(x -> x in [0,1], iter_res)
end

@testset "iter_sz_convergence" begin
    g = dtorus(3,3)
    peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = isometric_peps_to_unitary(peps, g)
    circ, converged, converged_iter = iter_sz_convergence(pepsu, g)
    @test converged == true

    nbit = 7
    reg = Yao.zero_state(nbit; nbatch=100000)

    corr_circ = torus_long_range_coherence(circ, reg, pepsu, 2, 3)
    corr_expect = long_range_coherence_peps(peps, 2, 3)  
    @test isapprox(corr_circ, corr_expect, atol=1e-2)
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


