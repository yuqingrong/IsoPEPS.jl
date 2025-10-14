using IsoPEPS
using Test
using Yao
@testset "varcomp" begin
    d=2
    D=32
    J=1.0
    g=1.0
    psi, E = exact_energy(d, D, J, g)
    E_exact = int(g,J)[]
    @test E ≈ E_exact atol = 1e-8
end

@testset "Pauli generators" begin
    generators = pauli_generators_2qubit()
    @test length(generators) == 15
    @test generators[1] == kron(Matrix(Yao.X), Matrix(Yao.I2))
    @test generators[2] == kron(Matrix(Yao.Y), Matrix(Yao.I2))
    @test generators[15] == kron(Matrix(Yao.Z), Matrix(Yao.Z))
end

@testset "su4 gate" begin
    theta = rand(15)
    gate = su4_gate(theta)
    @test size(gate) == (4, 4)
    @test gate*gate' ≈ I
    @test gate'*gate ≈ I
end