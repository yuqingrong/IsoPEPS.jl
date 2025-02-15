using Test, isoPEPS, Arpack

@testset "KrylovkitYao" begin
    nsites=10
    J=1.0
    h=0.2

    H_L = Lanczos(nsites,J,h)
    ground_state_energy, ground_state_vector = eigs(H_L, nev=1, which=:SR)
    E_L = ground_state_energy[1]/nsites
    @test isapprox(E_L, -0.9120354170186685, atol=1e-4)
end

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
    c = variational_circuit(nsites)
    dispatch!(c, :random)
    params = parameters(c)
    optimizer = Optimisers.setup(Optimisers.ADAM(0.01), params)
    niter = 100
    y = []
    x = []

    for i = 1:niter
        grad_input, grad_params = expect'(hami, zero_state(nsites) => c)
        Optimisers.update!(optimizer, params, grad_params)
        dispatch!(c, params)
        EG = expect(hami, zero_state(nsites) |> c)/nsites
        deltaE = sqrt((EG-E_L)^2)
        if i>=80
            push!(x,i)
            push!(y,deltaE)
            println(EG)
        end
    end
end