#!/usr/bin/env julia

"""
Reproduce paper figures from a staged package.

`--mode archive` is the deterministic default: it restores the exact curated
figure files. `--mode compute` reruns the plotting targets from the staged raw
results and precomputed intermediates, then checks that all expected files are
present. It intentionally never launches an optimization or a DMRG run.
"""

include(joinpath(@__DIR__, "common.jl"))
using .ReproCommon
using TOML

function usage()
    println("""
usage: julia --project=. repro/reproduce.jl --data-dir PATH --output-dir PATH [options]

options:
  --mode archive|compute   archive is byte-identical and is the default
  --help                   show this help
""")
end

function copy_archived_figures!(data_dir::String, output_dir::String, figures)
    for figure in figures
        source = joinpath(data_dir, figure["baseline"])
        destination = joinpath(output_dir, basename(figure["manuscript_image"]))
        mkpath(dirname(destination))
        cp(source, destination; force=true)
    end
end

function compute_figures!(data_dir::String, output_dir::String, figures)
    include(joinpath(ReproCommon.REPO_ROOT, "project", "postprocess.jl"))
    results_root = joinpath(data_dir, "results")
    generated_root = joinpath(output_dir, "generated-results")
    targets = unique(String[figure["plot_target"] for figure in figures if figure["kind"] == "generated"])
    dispatcher = Base.invokelatest(getfield, @__MODULE__, :run_target)
    for target in targets
        # `postprocess.jl` is intentionally loaded here so archive mode needs
        # no plotting dependencies at runtime. `invokelatest` keeps this safe
        # on Julia versions with strict world-age checks after `include`.
        Base.invokelatest(dispatcher, target; results_root=results_root, output_dir=generated_root)
    end

    for figure in figures
        name = basename(figure["manuscript_image"])
        destination = joinpath(output_dir, name)
        if figure["kind"] == "supplied"
            cp(joinpath(data_dir, figure["baseline"]), destination; force=true)
        else
            generated = joinpath(generated_root, figure["generated_output"])
            isfile(generated) || error("plot target did not create expected file: $generated")
            cp(generated, destination; force=true)
        end
    end
end

function main(args::Vector{String})
    options, positional = ReproCommon.parse_cli(args; flags=["help"])
    if get(options, "help", false)
        usage()
        return
    end
    isempty(positional) || error("unexpected positional arguments: $(join(positional, " "))")
    data_dir = abspath(ReproCommon.require_option(options, "data-dir"))
    output_dir = abspath(ReproCommon.require_option(options, "output-dir"))
    mode = get(options, "mode", "archive")
    mode in ("archive", "compute") || error("--mode must be archive or compute")
    isdir(data_dir) || error("--data-dir is not a directory: $data_dir")
    mkpath(output_dir)

    figures = TOML.parsefile(joinpath(data_dir, "figure_manifest.toml"))["figures"]
    if mode == "archive"
        copy_archived_figures!(data_dir, output_dir, figures)
        println("Restored $(length(figures)) exact curated figure files to: $output_dir")
    else
        compute_figures!(data_dir, output_dir, figures)
        println("Regenerated $(length(figures)) paper figures from staged results to: $output_dir")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
