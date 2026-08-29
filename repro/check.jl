#!/usr/bin/env julia

"""Cross-check manuscript figures, staged inputs, and optional rendered outputs."""

include(joinpath(@__DIR__, "common.jl"))
using .ReproCommon
using TOML

function usage()
    println("""
usage: julia --project=. repro/check.jl --data-dir PATH --tex PATH [options]

Confirm that figure_manifest.toml exactly covers main.tex's image list and
that all declared source data are present. With --rendered-dir, check that
each reproduced figure is non-empty; add --exact-baselines for archive-mode
reproduction, where byte-identical copies are required.

options:
  --rendered-dir PATH    directory produced by repro/reproduce.jl
  --exact-baselines      require rendered files to match baseline checksums
  --raster-tolerance N   mean pixel error allowed for regenerated PDFs (default: 0.15)
  --help                 show this help
""")
end

function manifest_figure_names(figures)
    return sort!(String[basename(figure["manuscript_image"]) for figure in figures])
end

function declared_sources(figure_manifest::Dict{String,Any}, figure::Dict{String,Any})
    sources = String.(get(figure, "source_data", String[]))
    data_sets = get(figure_manifest, "data_sets", Dict{String,Any}())
    for set_name in String.(get(figure, "source_data_sets", String[]))
        haskey(data_sets, set_name) || error("unknown source data set '$set_name' for $(figure["manuscript_image"])")
        append!(sources, String.(data_sets[set_name]))
    end
    return unique(sources)
end

function check_source_data(data_dir::String, figure_manifest::Dict{String,Any}, figures)
    for figure in figures
        for source in declared_sources(figure_manifest, figure)
            path = joinpath(data_dir, source)
            isfile(path) || error("missing source data for $(figure["manuscript_image"]): $source")
        end
    end
end

function ppm_payload(path::String)
    bytes = read(path)
    tokens = String[]
    index = 1
    while length(tokens) < 4
        while index <= length(bytes) && bytes[index] in (UInt8(' '), UInt8('\t'), UInt8('\r'), UInt8('\n'))
            index += 1
        end
        if index <= length(bytes) && bytes[index] == UInt8('#')
            while index <= length(bytes) && bytes[index] != UInt8('\n')
                index += 1
            end
            continue
        end
        start = index
        while index <= length(bytes) && !(bytes[index] in (UInt8(' '), UInt8('\t'), UInt8('\r'), UInt8('\n')))
            index += 1
        end
        push!(tokens, String(bytes[start:index - 1]))
    end
    tokens[1] == "P6" || error("expected binary PPM from renderer: $path")
    width, height, max_value = parse.(Int, tokens[2:4])
    max_value == 255 || error("unsupported PPM color range in $path")
    # Consume only the header delimiter; skipping arbitrary whitespace here
    # would incorrectly discard valid pixel values equal to whitespace bytes.
    index <= length(bytes) && bytes[index] == UInt8('\r') && (index += 1)
    index <= length(bytes) && bytes[index] == UInt8('\n') && (index += 1)
    payload = @view bytes[index:end]
    length(payload) == 3 * width * height || error("unexpected raster length in $path")
    return width, height, payload
end

function rendered_pdf_error(baseline::String, rendered::String)
    renderer = Sys.which("pdftoppm")
    isnothing(renderer) && error("pdftoppm is required for regenerated-PDF comparison")
    return mktempdir() do directory
        baseline_base = joinpath(directory, "baseline")
        rendered_base = joinpath(directory, "rendered")
        run(`$renderer -f 1 -l 1 -singlefile -r 144 $baseline $baseline_base`)
        run(`$renderer -f 1 -l 1 -singlefile -r 144 $rendered $rendered_base`)
        width_a, height_a, pixels_a = ppm_payload("$baseline_base.ppm")
        width_b, height_b, pixels_b = ppm_payload("$rendered_base.ppm")
        dimension_error = max(abs(width_a - width_b) / width_a,
                              abs(height_a - height_b) / height_a)
        dimension_error <= 0.02 || error(
            "rendered PDF dimensions differ too much: baseline $(width_a)×$(height_a), " *
            "rendered $(width_b)×$(height_b)")

        # Compare the baseline grid against the corresponding nearest rendered
        # pixel. This accepts sub-percent PDF-point rounding while still
        # detecting a changed page layout or plotted content.
        total_error = 0
        for y in 0:(height_a - 1), x in 0:(width_a - 1), channel in 0:2
            rendered_y = min(height_b - 1, floor(Int, (y + 0.5) * height_b / height_a))
            rendered_x = min(width_b - 1, floor(Int, (x + 0.5) * width_b / width_a))
            baseline_index = 3 * (y * width_a + x) + channel + 1
            rendered_index = 3 * (rendered_y * width_b + rendered_x) + channel + 1
            total_error += abs(Int(pixels_a[baseline_index]) - Int(pixels_b[rendered_index]))
        end
        return total_error / (255 * length(pixels_a))
    end
end

function check_rendered(data_dir::String, rendered_dir::String, figures;
                        exact_baselines::Bool=false, raster_tolerance::Union{Nothing,Float64}=nothing)
    for figure in figures
        name = basename(figure["manuscript_image"])
        rendered = joinpath(rendered_dir, name)
        isfile(rendered) || error("reproduced figure is missing: $name")
        filesize(rendered) > 0 || error("reproduced figure is empty: $name")
        if exact_baselines
            expected = lowercase(figure["baseline_sha256"])
            ReproCommon.file_sha256(rendered) == expected ||
                error("archive-mode figure differs from its baseline: $name")
        elseif figure["kind"] == "supplied"
            ReproCommon.file_sha256(rendered) == lowercase(figure["baseline_sha256"]) ||
                error("supplied illustration differs from its curated baseline: $name")
        elseif endswith(lowercase(name), ".pdf") && !isnothing(raster_tolerance)
            baseline = joinpath(data_dir, figure["baseline"])
            error_fraction = rendered_pdf_error(baseline, rendered)
            error_fraction <= raster_tolerance || error(
                "rendered baseline error for $name is $(round(error_fraction; digits=4)); " *
                "limit is $raster_tolerance")
        end
    end
end

function main(args::Vector{String})
    options, positional = ReproCommon.parse_cli(args; flags=["help", "exact-baselines"])
    if get(options, "help", false)
        usage()
        return
    end
    isempty(positional) || error("unexpected positional arguments: $(join(positional, " "))")
    data_dir = abspath(ReproCommon.require_option(options, "data-dir"))
    tex_path = abspath(ReproCommon.require_option(options, "tex"))
    isdir(data_dir) || error("--data-dir is not a directory: $data_dir")
    isfile(tex_path) || error("--tex is not a file: $tex_path")

    figure_manifest = TOML.parsefile(joinpath(data_dir, "figure_manifest.toml"))
    figures = get(figure_manifest, "figures", Any[])
    ReproCommon.verify_sha256_manifest(joinpath(data_dir, "MANIFEST.sha256"), data_dir)
    manifest_names = manifest_figure_names(figures)
    tex_names = ReproCommon.parse_includegraphics(tex_path)
    manifest_names == tex_names || error(
        "figure manifest and LaTeX image list differ\n" *
        "  only in manifest: $(join(setdiff(manifest_names, tex_names), ", "))\n" *
        "  only in LaTeX: $(join(setdiff(tex_names, manifest_names), ", "))")
    check_source_data(data_dir, figure_manifest, figures)

    if haskey(options, "rendered-dir")
        raster_tolerance = if get(options, "exact-baselines", false)
            nothing
        else
            parse(Float64, get(options, "raster-tolerance", "0.15"))
        end
        check_rendered(data_dir, abspath(String(options["rendered-dir"])), figures;
                       exact_baselines=get(options, "exact-baselines", false),
                       raster_tolerance=raster_tolerance)
    elseif get(options, "exact-baselines", false)
        error("--exact-baselines requires --rendered-dir")
    end

    println("Checked $(length(figures)) manuscript figures against $(basename(tex_path)).")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
