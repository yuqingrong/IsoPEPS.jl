#!/usr/bin/env julia

"""
Reproduce paper figures from a staged package.

`--mode plot` is the default: it renders paper figures directly from the
canonical processed JSON files in the staged package. `--mode archive` restores
the exact curated figure files. `--mode compute` is a separate validation path
that regenerates declared processed-data targets without replacing or plotting
from the canonical processed JSON files. Neither mode launches an optimization
or a DMRG run.
"""

include(joinpath(@__DIR__, "common.jl"))
using .ReproCommon
using TOML

function usage()
    println("""
usage: julia --project=. repro/reproduce.jl --data-dir PATH --output-dir PATH [options]

options:
  --mode plot|archive|compute
                           plot uses canonical processed data and is the default;
                           archive restores exact curated figures; compute writes
                           recomputed data for validation only
  --help                   show this help
""")
end

"""Return the data and plot targets required by a reproduction mode."""
function execution_plan(mode::AbstractString, figures)
    mode in ("plot", "compute") || error("execution plan is only defined for plot or compute mode")
    plot_targets = unique(String[
        figure["plot_target"] for figure in figures if figure["kind"] == "generated"
    ])
    data_targets = mode == "compute" ? unique(String[
        get(figure, "data_target", "") for figure in figures
        if figure["kind"] == "generated" && !isempty(get(figure, "data_target", ""))
    ]) : String[]
    return (; data_targets, plot_targets)
end

function copy_archived_figures!(data_dir::String, output_dir::String, figures)
    for figure in figures
        source = joinpath(data_dir, figure["baseline"])
        destination = joinpath(output_dir, basename(figure["manuscript_image"]))
        mkpath(dirname(destination))
        cp(source, destination; force=true)
    end
end

function plot_figures!(data_dir::String, output_dir::String, figures)
    include(joinpath(ReproCommon.REPO_ROOT, "project", "postprocess.jl"))
    results_root = joinpath(data_dir, "results")
    plan = execution_plan("plot", figures)
    dispatcher = Base.invokelatest(getfield, @__MODULE__, :run_target)
    mktempdir() do rendered_root
        for target in plan.plot_targets
            # `postprocess.jl` is intentionally loaded here so archive mode needs
            # no plotting dependencies at runtime. `invokelatest` keeps this safe
            # on Julia versions with strict world-age checks after `include`.
            Base.invokelatest(dispatcher, target; results_root=results_root, output_dir=rendered_root)
        end

        for figure in figures
            name = basename(figure["manuscript_image"])
            destination = joinpath(output_dir, name)
            if figure["kind"] == "supplied"
                cp(joinpath(data_dir, figure["baseline"]), destination; force=true)
            else
                generated = joinpath(rendered_root, figure["generated_output"])
                isfile(generated) || error("plot target did not create expected file: $generated")
                cp(generated, destination; force=true)
            end
        end
    end
end

function recompute_processed_data!(data_dir::String, output_dir::String, figures)
    include(joinpath(ReproCommon.REPO_ROOT, "project", "postprocess.jl"))
    results_root = joinpath(data_dir, "results")
    recomputed_root = joinpath(output_dir, "recomputed-data")
    plan = execution_plan("compute", figures)
    dispatcher = Base.invokelatest(getfield, @__MODULE__, :run_target)
    for target in plan.data_targets
        Base.invokelatest(dispatcher, target; results_root=results_root, output_dir=recomputed_root)
    end
    return recomputed_root
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
    mode = get(options, "mode", "plot")
    mode in ("plot", "archive", "compute") || error("--mode must be plot, archive, or compute")
    isdir(data_dir) || error("--data-dir is not a directory: $data_dir")
    mkpath(output_dir)

    figures = TOML.parsefile(joinpath(data_dir, "figure_manifest.toml"))["figures"]
    if mode == "archive"
        copy_archived_figures!(data_dir, output_dir, figures)
        println("Restored $(length(figures)) exact curated figure files to: $output_dir")
    elseif mode == "plot"
        plot_figures!(data_dir, output_dir, figures)
        println("Rendered $(length(figures)) paper figures from canonical processed data to: $output_dir")
    else
        recomputed_root = recompute_processed_data!(data_dir, output_dir, figures)
        println("Recomputed declared processed-data targets to: $recomputed_root")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
