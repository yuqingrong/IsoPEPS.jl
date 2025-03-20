using IsoPEPS
using Test

@testset "IsometricPEPS" begin
    peps = rand_isometricpeps(ComplexF64, 2, 3, 3)
    @test peps.col == 3
    @test peps.vertex_tensors[1][1] isa AbstractArray{ComplexF64, 5}
end


@testset "mose_move_single_column!" begin
    # initialize a peps
    peps = rand_isometricpeps(ComplexF64, 2, 4, 4)
    p1 = mose_move_right_step!(peps, 1)
    @test peps_fidelity(p1, peps) ≈ 1

    p1 = mose_move_right_step!(copy(peps), 2)
    @test peps_fidelity(p1, peps) ≈ 1

    p1 = mose_move_right_step!(copy(peps), 3)
    @test peps_fidelity(p1, peps) ≈ 1
    
end


@testset "mose_move_all_columns!" begin
    peps = rand_isometricpeps(ComplexF64, 2, 4, 4)
    p1 = mose_move_right!(copy(peps))
    @test peps_fidelity(p1, peps) ≈ 1
end 


@testset "isometric_condition" begin

end