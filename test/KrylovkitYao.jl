using Test, IsoPEPS, LinearAlgebra
import Optimisers
@testset "KrylovkitYao" begin
    nsites=10
    J=1.0
    h=0.2
    hami = ising_hamiltonian(nsites,J,h)
    E, V = ed_groundstate(hami)
    @test isapprox(E/10, -0.9120354170186685, atol=1e-4)
end

@testset "VariationalCircuit" begin
    nsites=10
    J=1.0
    h=0.2
    hami = ising_hamiltonian(nsites,J,h)
    @test IsoPEPS.ishermitian(hami)
    c = IsoPEPS.variational_circuit(nsites)
    IsoPEPS.dispatch!(c, :random)
    params = IsoPEPS.parameters(c)
    optimizer = Optimisers.setup(Optimisers.Adam(0.01), params)
    niter = 1000
   
    for i = 1:niter
        grad_input, grad_params = IsoPEPS.expect'(hami, IsoPEPS.zero_state(nsites) => c)
        Optimisers.update!(optimizer, params, grad_params)
        IsoPEPS.dispatch!(c, params)
          
    end
    EG = IsoPEPS.expect(hami, IsoPEPS.zero_state(nsites) |> c)/nsites
    @test isapprox(EG, -0.9120354170186685, atol=1e-1) # Q:EG used by VariationalCircuit can't reach 1e-4
end


@testset "Hamiltonian 2D" begin
    h = IsoPEPS.ising_hamiltonian_2d(5, 5, 1.0, 0.2)
    @test length(h[1].content) == 40
    @test length(h[2].content) == 25

    h = IsoPEPS.ising_hamiltonian_2d(2, 2, 1.0, 0.2)
    evals = eigen(Matrix(h)).values
    @test minimum(evals) ≈ -4.040593699203863
end