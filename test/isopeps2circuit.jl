using IsoPEPS
using Test
using Graphs, GraphPlot, Compose

@testset "isopeps2circuit" begin
    g = SimpleDiGraph(16)
    g2 = grid([4,4])
    edge_pairs = [(src(e), dst(e)) for e in collect(edges(g2))]
    for (i,j) in edge_pairs
        add_edge!(g, i, j)
    end
    #draw(PNG("g.png", 16cm, 16cm), gplot(g))
    peps = rand_peps(ComplexF64, g, 2, 2, TreeSA(), MergeGreedy())
    pepsu = peps2ugate(peps, g)
    
    @test isapprox(pepsu.vertex_tensors[1]'*pepsu.vertex_tensors[1], Matrix(I, 8, 8),atol=1e-10)  

    circ = get_circuit(pepsu, g)
    @test collect_blocks(IsoPEPS.Measure, circ)|>length == 16

    reg = zero_state(6)
    batch_size = 1024
    res = gensample(circ, reg, batch_size, pepsu, Z)
    @test res[1,1] in [0,1]
    @test all(x -> x in [0,1], res)

    corr = long_range_coherence(circ, reg, pepsu, 6, 5, batch_size)
    @show corr
    @test 0 ≤ corr ≤ 1
end
    

