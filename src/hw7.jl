### 1. (Ground state energy)

using Graphs, Random, LinearAlgebra,ProblemReductions
# Construct the Fullerene graph 
function fullerene()
    th = (1 + sqrt(5)) / 2
    res = NTuple{3,Float64}[]
    for (x, y, z) in ((0.0, 1.0, 3*th), (1.0, 2 + th, 2*th), (th, 2.0, 2*th + 1.0))
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y))
            for loc in ((a, b, c), (a, b, -c), (a, -b, c), (a, -b, -c),
                        (-a, b, c), (-a, b, -c), (-a, -b, c), (-a, -b, -c))
                if loc ∉ res
                    push!(res, loc)
                end
            end
        end
    end
    return res
end

# Generate the graph
fullerene_pos = fullerene()
fullerene_graph = UnitDiskGraph(fullerene_pos, sqrt(5))

# Simulated Annealing
function ising_energy(graph, spins)
    energy = 0
    for e in edges(graph)
        energy += spins[src(e)] * spins[dst(e)]
    end
    return energy
end

function simulated_annealing(graph; steps=10^6, T0=10.0, Tf=0.01)
    n = nv(graph)
    spins = rand([-1, 1], n)
    energy = ising_energy(graph, spins)
    best_energy = energy
    best_spins = copy(spins)

    for step in 1:steps
        T = T0 * (Tf / T0)^(step / steps)  # Cooling schedule
        site = rand(1:n)
        new_spins = copy(spins)
        new_spins[site] *= -1
        new_energy = ising_energy(graph, new_spins)

        if new_energy < energy || exp(-(new_energy - energy) / T) > rand()
            spins = new_spins
            energy = new_energy
            if energy < best_energy
                best_energy = energy
                best_spins = copy(spins)
            end
        end
    end
    return best_energy, best_spins
end

# Run and print result
ground_energy, _ = simulated_annealing(fullerene_graph)
println("Ground state energy estimate: ", ground_energy)
# => -66


### 2. (Spectral gap)
using Graphs, SparseArrays, LinearAlgebra, Arpack, Printf, Plots

# ---- Graph Constructors ----

function build_triangle_graph(n::Int)
    g = SimpleGraph(n)
    rows = div(n, 2)
    for i in 1:rows-1
        add_edge!(g, 2i-1, 2(i+1)-1)
        add_edge!(g, 2i, 2(i+1))
        add_edge!(g, 2i-1, 2i)
        add_edge!(g, 2i, 2(i+1)-1)  # diagonal
    end
    add_edge!(g, n-1, n)  # connect last vertical pair
    return g
end

function build_square_graph(n::Int)
    g = SimpleGraph(n)
    rows = div(n, 2)
    for i in 1:rows-1
        add_edge!(g, 2i-1, 2(i+1)-1)
        add_edge!(g, 2i, 2(i+1))
        add_edge!(g, 2i-1, 2i)
    end
    add_edge!(g, n-1, n)
    return g
end

function build_diamond_graph(n::Int)
    g = SimpleGraph(n)
    rows = div(n, 2)
    for i in 1:rows-1
        add_edge!(g, 2i-1, 2(i+1))
        add_edge!(g, 2i, 2(i+1)-1)
        add_edge!(g, 2i-1, 2i)
    end
    add_edge!(g, n-1, n)
    return g
end

# ---- Ising Hamiltonian ----
function ising_hamiltonian(g::SimpleGraph)
    N = nv(g)
    dim = 2^N
    H = spzeros(Float64, dim, dim)
    for i in 0:dim-1
        spins = [2*((i >> k) & 1) - 1 for k in 0:N-1]
        E = 0.0
        for e in edges(g)
            u, v = src(e), dst(e)
            E += spins[u] * spins[v]
        end
        H[i+1, i+1] = E
    end
    return H
end

# ---- Spectral Gap ----
function spectral_gap(H::SparseMatrixCSC{Float64,Int}, T::Float64)
    β = 1.0 / T
    Tmat = exp.(-β .* H)
    λ, _ = eigs(Tmat; nev=2, which=:LM)
    λ = sort(real.(λ), rev=true)
    return λ[1] - λ[2]
end

# ---- Run Analysis ----
function sweep_temperature(graphfn::Function, N::Int, Ts::Vector{Float64})
    g = graphfn(N)
    H = ising_hamiltonian(g)
    return [spectral_gap(H, T) for T in Ts]
end

function sweep_size(graphfn::Function, Ns::Vector{Int}, T::Float64)
    return [spectral_gap(ising_hamiltonian(graphfn(N)), T) for N in Ns]
end

# ---- Run + Plot ----
temps = 0.1:0.1:2.0
sizes = 6:2:18

triangle_T = sweep_temperature(build_triangle_graph, 12, collect(temps))
square_T = sweep_temperature(build_square_graph, 12, collect(temps))
diamond_T = sweep_temperature(build_diamond_graph, 12, collect(temps))

plot(temps, triangle_T, label="Triangle", xlabel="T", ylabel="Spectral Gap", title="Gap vs Temperature (N=12)")
savefig("spectral_gap_vs_temperature_triangle.png")
plot!(temps, square_T, label="Square")
savefig("spectral_gap_vs_temperature_square.png")
plot!(temps, diamond_T, label="Diamond")
savefig("spectral_gap_vs_temperature_diamond.png")

# ---- Spectral Gap vs Size ----
triangle_N = sweep_size(build_triangle_graph, collect(sizes), 0.1)
square_N = sweep_size(build_square_graph, collect(sizes), 0.1)
diamond_N = sweep_size(build_diamond_graph, collect(sizes), 0.1)

plot(sizes, triangle_N, label="Triangle", xlabel="N", ylabel="Spectral Gap", title="Gap vs System Size (T=0.1)")
plot!(sizes, square_N, label="Square")
plot!(sizes, diamond_N, label="Diamond")
