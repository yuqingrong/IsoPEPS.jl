include(joinpath(@__DIR__, "..", "repro", "common.jl"))
using .ReproCommon
using TOML

@testset "release-readiness reproducibility tools" begin
    repro_dir = joinpath(@__DIR__, "..", "repro")

    for filename in ("stage_data.jl", "verify.jl", "check.jl", "reproduce.jl",
                     "data_manifest.toml", "figure_manifest.toml")
        @test isfile(joinpath(repro_dir, filename))
    end

    data_manifest = TOML.parsefile(joinpath(repro_dir, "data_manifest.toml"))
    @test [entry["source"] for entry in data_manifest["directories"]] == [
        "project/results/tfim_abc",
        "project/results/heisenberg",
        "project/results/reference",
    ]
    @test "**/.checkpoint_*" in data_manifest["exclude_globs"]
    @test "reference/states/**" in data_manifest["exclude_globs"]
    @test all(entry -> !isempty(entry["include_files"]), data_manifest["directories"])
    heisenberg_allowlist = only(filter(entry -> entry["source"] == "project/results/heisenberg", data_manifest["directories"]))["include_files"]
    @test !any(filename -> occursin("J2=0.51", filename) || occursin("nqubits=5", filename), heisenberg_allowlist)
    @test length(data_manifest["paper_assets"]) == 17

    figure_manifest = TOML.parsefile(joinpath(repro_dir, "figure_manifest.toml"))
    figures = figure_manifest["figures"]
    @test length(figures) == 17
    @test count(figure -> figure["kind"] == "supplied", figures) == 2
    @test all(figure -> occursin(r"^[0-9a-f]{64}$", figure["baseline_sha256"]), figures)
    @test length(figure_manifest["data_sets"]["tfim_g_scan"]) == 21
    @test length(figure_manifest["data_sets"]["heisenberg_j2_scan"]) == 11

    mktempdir() do directory
        write(joinpath(directory, "payload.txt"), "checksum fixture\n")
        write_sha256_manifest(joinpath(directory, "MANIFEST.sha256"), directory)
        @test verify_sha256_manifest(joinpath(directory, "MANIFEST.sha256"), directory) == 1
        write(joinpath(directory, "payload.txt"), "changed fixture\n")
        @test_throws ErrorException verify_sha256_manifest(joinpath(directory, "MANIFEST.sha256"), directory)
    end
end
