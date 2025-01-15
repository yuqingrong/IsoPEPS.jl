using Test, IsoPEPS

@testset "ImTebd" begin
    nsites = 10
    J=1.0
    h=0.2
    t=20
    H = IsoPEPS.transverse_ising(nsites;J, h) |> IsoPEPS.cache
    p = 1.0

    reg = IsoPEPS.rand_state(IsoPEPS.nqubits(H))
    reg |> itime_groundstate!(H; t,p)
    EG = IsoPEPS.real(IsoPEPS.expect(H, reg)/nsites)
    @test IsoPEPS.ishermitian(H)
    # using imaginary time Evolution
    @test isapprox(EG, -0.9120354170186685, atol=1e-1)
end
