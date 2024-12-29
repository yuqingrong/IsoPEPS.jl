using IsoPEPS
using Test
@testset "mps" begin
    mps=generate_mps(3,10)
    code,result= code_dot(mps,mps)
    @test  isless(0.0,result)
end

@testset "mps_variation" begin
    nsites=10
    bond_dim=2
    h=0.2
    result,f,g! = mps_variation(nsites,bond_dim,h)
    @test isapprox(result.minimum, -9.120354170186685, atol=1e-4)
end

