using isoPEPS
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
