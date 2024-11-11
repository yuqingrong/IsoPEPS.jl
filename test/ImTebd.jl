using Test, isoPEPS

@testset "ImTebd" begin
    @test ishermitian(h)
    # using imaginary time Evolution
    @test isapprox(EG, -0.4564, atol=1e-4)
end
