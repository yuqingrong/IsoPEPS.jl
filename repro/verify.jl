#!/usr/bin/env julia

"""Validate the checksum manifest and figure baselines in a staged data package."""

include(joinpath(@__DIR__, "common.jl"))
using .ReproCommon
using TOML

function usage()
    println("""
usage: julia --project=. repro/verify.jl --data-dir PATH

Validate every staged file against MANIFEST.sha256 and every manuscript
baseline against the checksum recorded in figure_manifest.toml.
""")
end

function verify_figure_baselines(data_dir::String)
    figure_manifest = TOML.parsefile(joinpath(data_dir, "figure_manifest.toml"))
    figures = get(figure_manifest, "figures", Any[])
    isempty(figures) && error("figure_manifest.toml contains no figures")
    for figure in figures
        baseline = figure["baseline"]
        expected = lowercase(figure["baseline_sha256"])
        path = joinpath(data_dir, baseline)
        isfile(path) || error("missing manuscript baseline: $baseline")
        actual = ReproCommon.file_sha256(path)
        actual == expected || error("baseline checksum mismatch for $baseline")
    end
    return length(figures)
end

function main(args::Vector{String})
    options, positional = ReproCommon.parse_cli(args; flags=["help"])
    if get(options, "help", false)
        usage()
        return
    end
    isempty(positional) || error("unexpected positional arguments: $(join(positional, " "))")
    data_dir = abspath(ReproCommon.require_option(options, "data-dir"))
    isdir(data_dir) || error("--data-dir is not a directory: $data_dir")

    manifest_path = joinpath(data_dir, "MANIFEST.sha256")
    isfile(manifest_path) || error("missing staged checksum manifest: $manifest_path")
    file_count = ReproCommon.verify_sha256_manifest(manifest_path, data_dir)
    figure_count = verify_figure_baselines(data_dir)
    println("Verified $file_count staged files and $figure_count manuscript baselines.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
