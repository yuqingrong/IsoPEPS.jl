using IsoPEPS
using Test

const TEST_FILES = [
    "gates" => "gates.jl",
    "transfer_matrix" => "transfer_matrix.jl",
    "observables" => "observables.jl",
    "quantum_channels" => "quantum_channels.jl",
    "reference" => "reference.jl",
    "training" => "training.jl",
    "visualization" => "visualization.jl",
]

const REQUESTED_TESTS = isempty(ARGS) ? first.(TEST_FILES) : ARGS

for name in REQUESTED_TESTS
    file_idx = findfirst(==(name), first.(TEST_FILES))
    isnothing(file_idx) && error("Unknown test set: $name")

    @testset "$name" begin
        include(TEST_FILES[file_idx].second)
    end
end
