using IsoPEPS
using Test

@testset "KrylovkitYao" begin
    include("KrylovkitYao.jl")
end

@testset "LanczosAlgorithm" begin
    include("LanczosAlgorithm.jl")
end

@testset "ImTebd" begin
    include("ImTebd.jl")
end

@testset "inner_product_mps" begin
    include("inner_product_mps.jl")
end
