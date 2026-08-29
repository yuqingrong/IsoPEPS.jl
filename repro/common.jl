module ReproCommon

using SHA

export REPO_ROOT, file_sha256, normalized_relpath, parse_cli, require_option,
       write_sha256_manifest, read_sha256_manifest, verify_sha256_manifest,
       list_manifest_files, parse_includegraphics, ensure_empty_directory

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

"""Return the lower-case SHA-256 digest of `path`."""
function file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

"""Return a portable, forward-slash relative path."""
normalized_relpath(root::AbstractString, path::AbstractString) =
    replace(relpath(path, root), '\\' => '/')

"""
    parse_cli(args; flags=String[])

Parse the small `--key value` command-line interface shared by the local
reproducibility tools. Flag names listed in `flags` are stored as `true`.
"""
function parse_cli(args::Vector{String}; flags::Vector{String}=String[])
    options = Dict{String,Any}()
    positional = String[]
    flag_set = Set(flags)
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if key in flag_set
                options[key] = true
            elseif i < length(args) && !startswith(args[i + 1], "--")
                options[key] = args[i + 1]
                i += 1
            else
                error("option --$key requires a value")
            end
        else
            push!(positional, arg)
        end
        i += 1
    end
    return options, positional
end

function require_option(options::Dict{String,Any}, key::String)
    haskey(options, key) || error("missing required option --$key")
    return String(options[key])
end

function list_manifest_files(root::AbstractString; skip::Set{String}=Set(["MANIFEST.sha256"]))
    files = String[]
    for (directory, _, filenames) in walkdir(root)
        for filename in filenames
            path = joinpath(directory, filename)
            rel = normalized_relpath(root, path)
            rel in skip || push!(files, rel)
        end
    end
    sort!(files)
    return files
end

"""Write a deterministic SHA-256 manifest and return its file list."""
function write_sha256_manifest(manifest_path::AbstractString, root::AbstractString)
    files = list_manifest_files(root)
    open(manifest_path, "w") do io
        for rel in files
            println(io, file_sha256(joinpath(root, rel)), "  ", rel)
        end
    end
    return files
end

function read_sha256_manifest(path::AbstractString)
    entries = Dict{String,String}()
    for (line_number, line) in enumerate(eachline(path))
        isempty(strip(line)) && continue
        match_result = match(r"^([0-9a-f]{64})  (.+)$", line)
        isnothing(match_result) && error("invalid SHA-256 manifest line $line_number in $path")
        entries[match_result.captures[2]] = match_result.captures[1]
    end
    return entries
end

"""Validate file hashes and reject files absent from the manifest."""
function verify_sha256_manifest(manifest_path::AbstractString, root::AbstractString)
    expected = read_sha256_manifest(manifest_path)
    actual_files = Set(list_manifest_files(root))
    expected_files = Set(keys(expected))
    missing = sort!(collect(setdiff(expected_files, actual_files)))
    unexpected = sort!(collect(setdiff(actual_files, expected_files)))
    isempty(missing) || error("files missing from staged package: $(join(missing, ", "))")
    isempty(unexpected) || error("files absent from MANIFEST.sha256: $(join(unexpected, ", "))")

    for rel in sort!(collect(expected_files))
        actual = file_sha256(joinpath(root, rel))
        actual == expected[rel] || error("checksum mismatch for $rel")
    end
    return length(expected)
end

"""Extract every image filename referenced by LaTeX `\\includegraphics` commands."""
function parse_includegraphics(tex_path::AbstractString)
    text = read(tex_path, String)
    references = String[]
    for match_result in eachmatch(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
        push!(references, basename(match_result.captures[1]))
    end
    return sort!(unique(references))
end

"""Create `directory`, refusing to overwrite an existing non-empty staging area."""
function ensure_empty_directory(directory::AbstractString)
    if ispath(directory)
        isdir(directory) || error("staging destination exists and is not a directory: $directory")
        isempty(readdir(directory)) || error(
            "staging destination is not empty: $directory\n" *
            "Choose a new --destination; this tool never overwrites a package.")
    else
        mkpath(directory)
    end
    return abspath(directory)
end

end # module
