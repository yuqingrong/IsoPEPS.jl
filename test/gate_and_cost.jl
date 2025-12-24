using Test
using IsoPEPS

@testset "build_gate_from_params" begin
    for row in 3:6, p in 2:4, nqubits in [3]
        # sharing parameters: all gates should be equal
        params = rand(2 * nqubits * p)
        A_matrix = IsoPEPS.build_gate_from_params(params, p, row, nqubits)
        @test all(A_matrix[i] == A_matrix[1] for i in 2:row)

        # independent parameters: all gates should be different
        params = rand(2 * nqubits * p * row)
        A_matrix = IsoPEPS.build_gate_from_params(params, p, row, nqubits; share_params=false)
        @test all(A_matrix[i] != A_matrix[1] for i in 2:row)
    end
end

@testset "energy_measure" begin
    for row in 3:6
        a = rand()
        b = rand()
        X_list = a * ones(100)
        Z_list = b * ones(100)
        g = 1.0
        J = 1.0
        energy = IsoPEPS.energy_measure(X_list, Z_list, g, J, row)
        @test energy ≈ -a - 2 * b^2 atol = 1e-5
    end
end

