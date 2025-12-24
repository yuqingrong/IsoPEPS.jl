using Test
using IsoPEPS

@testset "build_unitary_gate" begin
    for row in 3:6, p in 2:4, nqubits in [3]
        # Shared parameters: all gates should be equal
        params = rand(2 * nqubits * p)
        gates = build_unitary_gate(params, p, row, nqubits)
        @test all(gates[i] == gates[1] for i in 2:row)

        # Independent parameters: all gates should be different
        params = rand(2 * nqubits * p * row)
        gates = build_unitary_gate(params, p, row, nqubits; share_params=false)
        @test all(gates[i] != gates[1] for i in 2:row)
    end
end

@testset "compute_energy" begin
    for row in 3:6
        a = rand()
        b = rand()
        X_samples = a * ones(100)
        Z_samples = b * ones(100)
        g = 1.0
        J = 1.0
        energy = compute_energy(X_samples, Z_samples, g, J, row)
        @test energy ≈ -a - 2 * b^2 atol = 1e-5
    end
end

