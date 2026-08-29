include(joinpath(@__DIR__, "..", "repro", "common.jl"))
using .ReproCommon
using TOML
using JSON3

module ReproduceModeFixture
    include(joinpath(@__DIR__, "..", "repro", "reproduce.jl"))
end

module ResultsLayoutFixture
    include(joinpath(@__DIR__, "..", "repro", "results_layout.jl"))
end

@testset "release-readiness reproducibility tools" begin
    repro_dir = joinpath(@__DIR__, "..", "repro")

    @testset "release metadata is ready for archival" begin
        project = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
        release_version = project["version"]
        code_metadata_path = joinpath(@__DIR__, "..", ".zenodo.json")
        data_metadata_path = joinpath(repro_dir, "zenodo-data-metadata-template.json")

        @test isfile(code_metadata_path)
        if isfile(code_metadata_path)
            code_metadata = JSON3.read(read(code_metadata_path, String))
            @test code_metadata["version"] == release_version
            @test code_metadata["upload_type"] == "software"
            @test code_metadata["access_right"] == "open"
            @test code_metadata["license"] == "mit"
            @test !isempty(code_metadata["creators"])
        end

        @test isfile(data_metadata_path)
        if isfile(data_metadata_path)
            data_metadata = JSON3.read(read(data_metadata_path, String))["metadata"]
            @test data_metadata["version"] == release_version
            @test data_metadata["upload_type"] == "dataset"
            @test data_metadata["access_right"] == "open"
            @test data_metadata["license"] == "cc-by-4.0"
            @test !isempty(data_metadata["creators"])
        end
    end

    for filename in ("stage_data.jl", "verify.jl", "check.jl", "reproduce.jl",
                     "data_manifest.toml", "figure_manifest.toml")
        @test isfile(joinpath(repro_dir, filename))
    end

    data_manifest = TOML.parsefile(joinpath(repro_dir, "data_manifest.toml"))
    model_entries = filter(entry -> startswith(entry["destination"], "results/tfim_abc/") ||
                                    startswith(entry["destination"], "results/heisenberg/"),
                           data_manifest["directories"])
    @test [entry["destination"] for entry in model_entries] == [
        "results/tfim_abc/raw",
        "results/tfim_abc/processed",
        "results/heisenberg/raw",
        "results/heisenberg/processed",
        "results/heisenberg/processed",
    ]
    raw_entries = filter(entry -> endswith(entry["destination"], "/raw"), model_entries)
    @test all(entry -> all(name -> startswith(name, "circuit_"), entry["include_files"]), raw_entries)
    processed_entries = filter(entry -> endswith(entry["destination"], "/processed"), model_entries)
    raw_filenames = Set(name for entry in raw_entries for name in entry["include_files"])
    @test all(entry -> all(name -> !(name in raw_filenames), entry["include_files"]), processed_entries)
    @test all(entry -> !occursin("figures/", entry["destination"]), processed_entries)
    @test "**/.checkpoint_*" in data_manifest["exclude_globs"]
    @test "reference/states/**" in data_manifest["exclude_globs"]
    @test all(entry -> !isempty(entry["include_files"]), data_manifest["directories"])
    heisenberg_raw_entries = filter(entry -> entry["destination"] == "results/heisenberg/raw", model_entries)
    @test length(heisenberg_raw_entries) == 1
    if length(heisenberg_raw_entries) == 1
        @test !any(filename -> occursin("J2=0.51", filename) || occursin("nqubits=5", filename),
                  only(heisenberg_raw_entries)["include_files"])
    end
    tfim_readouts = filter(entry -> entry["destination"] == "results/tfim_abc/processed/readout_noise_energ_g=3.0.json",
                          data_manifest["files"])
    @test length(tfim_readouts) == 1
    if length(tfim_readouts) == 1
        @test only(tfim_readouts)["source"] == "project/results/tfim_abc/readout_noise_energ_g=3.0.json"
    end
    @test length(data_manifest["paper_assets"]) == 17

    staged_processed_dir = joinpath(@__DIR__, "..", "release-staging", "IsoPEPS-paper-data",
                                    "results", "heisenberg", "processed")
    @test isfile(joinpath(staged_processed_dir, "bond_energy_exact.json"))
    @test isfile(joinpath(staged_processed_dir,
                          "circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json"))

    figure_manifest = TOML.parsefile(joinpath(repro_dir, "figure_manifest.toml"))
    figures = figure_manifest["figures"]
    @test length(figures) == 17
    @test count(figure -> figure["kind"] == "supplied", figures) == 2
    @test all(figure -> occursin(r"^[0-9a-f]{64}$", figure["baseline_sha256"]), figures)
    @test length(figure_manifest["data_sets"]["tfim_g_scan"]) == 21
    @test length(figure_manifest["data_sets"]["heisenberg_j2_scan"]) == 11
    @test all(path -> startswith(path, "results/tfim_abc/raw/circuit_tfim_"),
              figure_manifest["data_sets"]["tfim_g_scan"])
    @test all(path -> startswith(path, "results/tfim_abc/raw/circuit_tfim_"),
              figure_manifest["data_sets"]["tfim_dynamics_scan"])
    @test all(path -> startswith(path, "results/heisenberg/raw/circuit_heisenberg_"),
              figure_manifest["data_sets"]["heisenberg_j2_scan"])
    @test all(path -> startswith(path, "results/heisenberg/raw/circuit_heisenberg_"),
              figure_manifest["data_sets"]["bond_energy_scan"])
    declared_model_sources = String[
        source for figure in figures for source in get(figure, "source_data", String[])
        if startswith(source, "results/tfim_abc/") || startswith(source, "results/heisenberg/")
    ]
    @test all(path -> occursin(r"results/(tfim_abc|heisenberg)/(raw|processed)/", path),
              declared_model_sources)

    bond_figure = only(filter(figure -> figure["manuscript_image"] == "bond_energy_exact.pdf", figures))
    @test bond_figure["data_target"] == "bond-energy-exact-data"
    @test bond_figure["source_data"] == ["results/heisenberg/processed/bond_energy_exact.json"]

    training_figure = only(filter(figure -> occursin("_training_history.pdf", figure["manuscript_image"]), figures))
    @test training_figure["data_target"] == "heisenberg-training-history-data"
    @test training_figure["source_data"] == [
        "results/heisenberg/processed/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2_training_history.json",
        "results/reference/dmrg_bulk_heisenberg_j1j2_Ly4_D2_J2scan.json",
    ]
    @test all(!haskey(figure, "source_data_sets") for figure in figures)
    @test any(haskey(figure, "compute_source_data_sets") for figure in figures)

    for documentation in ("README.md", "docs/reproducibility.md")
        text = read(joinpath(@__DIR__, "..", documentation), String)
        @test occursin("raw/", text)
        @test occursin("processed/", text)
    end

    @testset "plot mode does not schedule processed-data recomputation" begin
        figures = Dict{String,Any}[
            Dict("kind" => "generated", "plot_target" => "plot-energy", "data_target" => "compute-energy"),
            Dict("kind" => "generated", "plot_target" => "plot-observable", "data_target" => "compute-observable"),
            Dict("kind" => "supplied", "plot_target" => ""),
        ]

        plot_plan = ReproduceModeFixture.execution_plan("plot", figures)
        @test plot_plan.data_targets == String[]
        @test plot_plan.plot_targets == ["plot-energy", "plot-observable"]

        compute_plan = ReproduceModeFixture.execution_plan("compute", figures)
        @test compute_plan.data_targets == ["compute-energy", "compute-observable"]
    end

    @testset "processed layout does not require raw directories" begin
        mktempdir() do results_root
            processed_dir = joinpath(results_root, "heisenberg", "processed")
            mkpath(processed_dir)
            processed_file = joinpath(processed_dir, "M2_sampling.json")
            write(processed_file, "{}")
            @test ResultsLayoutFixture.ResultsLayout.result_path(
                results_root, "heisenberg", "M2_sampling.json") == processed_file
        end
    end

    mktempdir() do directory
        write(joinpath(directory, "payload.txt"), "checksum fixture\n")
        write_sha256_manifest(joinpath(directory, "MANIFEST.sha256"), directory)
        @test verify_sha256_manifest(joinpath(directory, "MANIFEST.sha256"), directory) == 1
        write(joinpath(directory, "payload.txt"), "changed fixture\n")
        @test_throws ErrorException verify_sha256_manifest(joinpath(directory, "MANIFEST.sha256"), directory)
    end
end
