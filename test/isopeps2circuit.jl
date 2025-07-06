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


@testset "sample and extract_res" begin
    nbit = 7
    g = dtorus(3,3)
    peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = isometric_peps_to_unitary(peps, g)
    circ = Yao.chain(nbit)
    circ = get_iter_circuit(circ, pepsu, 1, 6, collect(1:1), collect(2:7))
    reg = Yao.zero_state(nbit; nbatch=1000)
    reg |> circ 
    iter_res = extract_res(circ, reg, g, 1)
    @test size(iter_res) == (1000, nv(g))
    @test all(x -> x in [0,1], iter_res)
end

@testset "iter_sz_convergence" begin
    g = dtorus(3,3)
    peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
    peps = specific_peps(peps, pi/4)
    pepsu = isometric_peps_to_unitary(peps, g)
    circ, converged, converged_iter = iter_sz_convergence(pepsu, g)
    @test converged == true
    reg = Yao.zero_state(7; nbatch=100000)
    corr_circ = torus_long_range_coherence(circ, reg, g, converged_iter, 1, 2)
    corr_expect = long_range_coherence_peps(peps, 1, 2)
    @test isapprox(corr_circ, corr_expect, atol=1e-2)
end




using CairoMakie
# amplitude of quantum state
g = dtorus(3,3)
peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
peps = specific_peps(peps, pi/4, pi/8)
pepsu = isometric_peps_to_unitary(peps, g)
p_exact = pro_amplitude(peps)
circ, converged, converged_iter, p_all, q_all = iter_sz_convergence(pepsu, g)
@test sum(p_all) ≈ 1.0
@test sum(q_all) ≈ 1.0
x_axis = 1:length(p_all)
fig = Figure(size = (1000, 400))
ax = Axis(fig[1, 1], xlabel="measure results", ylabel="probability")
lines!(ax, x_axis, p_exact, color=(:red, 0.6), label="p_exact")
lines!(ax, x_axis, p_all, color=(:blue, 0.6), label="p_all")
lines!(ax, x_axis, q_all, color=(:orange, 0.5), label="q_all")
axislegend(ax)
save("p_all_q_all.png", fig)


# correlation
g = dtorus(3,3)
peps,_ = isometric_peps(Float64, g, 2, 2, TreeSA(), MergeGreedy())
peps = specific_peps(peps, pi/4, pi/4)
pepsu = isometric_peps_to_unitary(peps, g)
corr = all_corr(peps)

circ, converged, converged_iter = iter_sz_convergence(pepsu, g)
reg = Yao.zero_state(7; nbatch=100000)

fig = Figure(size = (800, 600))
ax = Axis3(fig[1, 1], xlabel="Site i", ylabel="Site j", zlabel="Correlation")
surface!(ax, 1:9, 1:9, corr, colormap=:viridis)
save("correlation_3d.png", fig)
