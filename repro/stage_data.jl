#!/usr/bin/env julia

"""
Build a private, release-ready data package without modifying the source tree.

Example:
    julia --project=. repro/stage_data.jl \
        --source-root /path/to/IsoPEPS.jl \
        --paper-image-dir /path/to/IsoPEPS-Notes/arxiv_submit/image \
        --destination release-staging/IsoPEPS-paper-data
"""

include(joinpath(@__DIR__, "common.jl"))
using .ReproCommon
using TOML

function usage()
    println("""
usage: julia --project=. repro/stage_data.jl --source-root PATH --paper-image-dir PATH [options]

Copy the strict allowlist in repro/data_manifest.toml into a new, empty local
package. The source results are read only. The default destination is
release-staging/IsoPEPS-paper-data beneath this repository.

options:
  --destination PATH     destination package directory
  --manifest PATH        allowlist manifest (default: repro/data_manifest.toml)
  --help                 show this help
""")
end

function contains_path(parent::AbstractString, child::AbstractString)
    parent_abs = abspath(parent)
    child_abs = abspath(child)
    return child_abs == parent_abs || startswith(child_abs, parent_abs * Base.Filesystem.path_separator)
end

function is_excluded(relative_path::AbstractString, excludes::Vector{String})
    rel = replace(relative_path, '\\' => '/')
    base = basename(rel)
    base == ".DS_Store" && return true
    startswith(base, ".checkpoint_") && return true
    # The only directory-level exclusion in the allowlist is reference/states.
    # It can appear as `states/...` beneath its selected reference directory.
    (rel == "states" || startswith(rel, "states/")) && return true
    return any(pattern -> occursin(".DS_Store", pattern) && base == ".DS_Store", excludes)
end

function selected_files(source::AbstractString, include_files::Vector{String}, excludes::Vector{String})
    isempty(include_files) && error("every allowlisted directory must declare non-empty include_files")
    files = String[]
    for relative in include_files
        isabspath(relative) && error("allowlisted relative path must not be absolute: $relative")
        normalized = normpath(relative)
        startswith(normalized, ".." * Base.Filesystem.path_separator) &&
            error("allowlisted relative path escapes its directory: $relative")
        path = joinpath(source, normalized)
        isfile(path) || error("allowlisted file is missing: $path")
        is_excluded(normalized, excludes) && error("allowlist includes an excluded file: $relative")
        push!(files, path)
    end
    length(unique(files)) == length(files) || error("allowlisted directory contains duplicate include_files entries")
    sort!(files)
    return files
end

function copy_selected_tree!(source_root::String, destination_root::String, entry::Dict{String,Any}, excludes::Vector{String})
    source = normpath(joinpath(source_root, entry["source"]))
    contains_path(source_root, source) || error("allowlist source escapes --source-root: $source")
    isdir(source) || error("allowlisted source directory is missing: $source")
    destination = joinpath(destination_root, entry["destination"])
    include_files = String.(get(entry, "include_files", String[]))
    for source_file in selected_files(source, include_files, excludes)
        relative = relpath(source_file, source)
        destination_file = joinpath(destination, relative)
        mkpath(dirname(destination_file))
        cp(source_file, destination_file)
    end
end

function copy_selected_file!(source_root::String, destination_root::String, entry::Dict{String,Any})
    source = normpath(joinpath(source_root, entry["source"]))
    contains_path(source_root, source) || error("allowlist source escapes --source-root: $source")
    isfile(source) || error("allowlisted source file is missing: $source")
    destination = joinpath(destination_root, entry["destination"])
    mkpath(dirname(destination))
    cp(source, destination)
end

function copy_paper_asset!(paper_image_dir::String, destination_root::String, entry::Dict{String,Any})
    source = normpath(joinpath(paper_image_dir, entry["source"]))
    contains_path(paper_image_dir, source) || error("paper asset escapes --paper-image-dir: $source")
    isfile(source) || error("paper baseline is missing: $source")
    destination = joinpath(destination_root, entry["destination"])
    mkpath(dirname(destination))
    cp(source, destination)
end

function source_checksums(source_root::String, paper_image_dir::String, manifest::Dict{String,Any})
    excludes = String.(get(manifest, "exclude_globs", String[]))
    checksums = Dict{String,String}()
    for entry in manifest["directories"]
        source = joinpath(source_root, entry["source"])
        include_files = String.(get(entry, "include_files", String[]))
        for path in selected_files(source, include_files, excludes)
            key = "source/" * ReproCommon.normalized_relpath(source_root, path)
            checksums[key] = ReproCommon.file_sha256(path)
        end
    end
    for entry in manifest["files"]
        path = joinpath(source_root, entry["source"])
        checksums["source/" * entry["source"]] = ReproCommon.file_sha256(path)
    end
    for entry in manifest["paper_assets"]
        path = joinpath(paper_image_dir, entry["source"])
        checksums["paper-image/" * entry["source"]] = ReproCommon.file_sha256(path)
    end
    return checksums
end

function write_source_checksums(path::String, checksums::Dict{String,String})
    open(path, "w") do io
        for key in sort!(collect(keys(checksums)))
            println(io, checksums[key], "  ", key)
        end
    end
end

function write_package_readme(path::String)
    write(path, """
# IsoPEPS paper data package (local staging copy)

This directory is a private, local staging package for the IsoPEPS Notes
paper. It was produced by `repro/stage_data.jl` from the strict allowlist in
`data_manifest.toml`; it is not a public release or a Zenodo deposit.

- `results/tfim_abc/raw/` and `results/heisenberg/raw/` contain saved
  optimized circuit inputs. Their sibling `processed/` directories contain
  the canonical figure-ready and sampling-summary JSONs. Readers should use
  these processed files with the default `repro/reproduce.jl --mode plot`
  workflow. The raw circuit inputs are retained only for optional `--mode
  compute` validation. `results/reference/` contains the DMRG/VUMPS/PEPSKit
  references required by the paper figures.
- `figures/published/` contains the exact figure files used by the manuscript.
- `figure_manifest.toml` maps every manuscript figure to its inputs, command,
  and baseline checksum.
- `MANIFEST.sha256` verifies every file in this package.

The data are intended for future release under CC BY 4.0; see
`LICENSE-CC-BY-4.0.txt`. Until the authors approve a release, keep this
directory private.
""")
end

function write_provenance(path::String, source_root::String, paper_image_dir::String)
    write(path, """
schema_version = 1
package_purpose = "Private local staging for future IsoPEPS Notes data release"
code_license = "MIT"
data_license = "CC-BY-4.0"
source_results_root = "project/results (relative to the private source checkout)"
paper_image_root = "arxiv_submit/image (relative to the manuscript checkout)"
source_layout = "Source-file SHA-256 values are recorded in SOURCE-MANIFEST.sha256; local absolute paths are intentionally omitted."
release_status = "not published; no DOI, Zenodo record, tag, or upload has been created"
""")
end

function main(args::Vector{String})
    options, positional = ReproCommon.parse_cli(args; flags=["help"])
    if get(options, "help", false)
        usage()
        return
    end
    isempty(positional) || error("unexpected positional arguments: $(join(positional, " "))")

    source_root = abspath(ReproCommon.require_option(options, "source-root"))
    paper_image_dir = abspath(ReproCommon.require_option(options, "paper-image-dir"))
    manifest_path = abspath(get(options, "manifest", joinpath(@__DIR__, "data_manifest.toml")))
    destination = abspath(get(options, "destination", joinpath(ReproCommon.REPO_ROOT, "release-staging", "IsoPEPS-paper-data")))
    isdir(source_root) || error("--source-root is not a directory: $source_root")
    isdir(paper_image_dir) || error("--paper-image-dir is not a directory: $paper_image_dir")
    isfile(manifest_path) || error("allowlist manifest is missing: $manifest_path")

    manifest = TOML.parsefile(manifest_path)
    destination = ReproCommon.ensure_empty_directory(destination)
    excludes = String.(get(manifest, "exclude_globs", String[]))

    before = source_checksums(source_root, paper_image_dir, manifest)
    for entry in manifest["directories"]
        copy_selected_tree!(source_root, destination, entry, excludes)
    end
    for entry in manifest["files"]
        copy_selected_file!(source_root, destination, entry)
    end
    for entry in manifest["paper_assets"]
        copy_paper_asset!(paper_image_dir, destination, entry)
    end

    cp(manifest_path, joinpath(destination, "data_manifest.toml"))
    cp(joinpath(@__DIR__, "figure_manifest.toml"), joinpath(destination, "figure_manifest.toml"))
    cp(joinpath(@__DIR__, "licenses", "CC-BY-4.0.txt"), joinpath(destination, "LICENSE-CC-BY-4.0.txt"))
    write_package_readme(joinpath(destination, "README.md"))
    write_provenance(joinpath(destination, "provenance.toml"), source_root, paper_image_dir)
    write_source_checksums(joinpath(destination, "SOURCE-MANIFEST.sha256"), before)

    after = source_checksums(source_root, paper_image_dir, manifest)
    before == after || error("source-result checksums changed while staging; package retained for inspection")
    ReproCommon.write_sha256_manifest(joinpath(destination, "MANIFEST.sha256"), destination)
    println("Staged $(length(before)) source files in: $destination")
    println("Verified source-result checksums before and after staging.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
