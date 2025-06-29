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

#@testset "inner_product_mps" begin
    #include("inner_product_mps.jl")
#end

@testset "mps" begin
    include("mps.jl")
end

@testset "peps" begin
    include("peps.jl")
end

@testset "mpsandmpo" begin
    include("mpsandmpo.jl")
end