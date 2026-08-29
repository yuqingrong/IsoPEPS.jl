module ResultsLayout

export result_path

const STAGED_MODELS = ("tfim_abc", "heisenberg")

function result_path(results_root::AbstractString, pieces::AbstractString...)::String
    requested = joinpath(results_root, pieces...)
    length(pieces) > 1 && (isfile(requested) || isdir(requested)) && return requested
    isempty(pieces) && return requested

    model = first(pieces)
    model in STAGED_MODELS || return requested
    model_root = joinpath(results_root, model)
    staged_raw = joinpath(model_root, "raw")
    staged_processed = joinpath(model_root, "processed")
    isdir(staged_processed) || return requested

    length(pieces) == 1 && return staged_raw
    remaining = pieces[2:end]
    first(remaining) == "figures" && return joinpath(model_root, "processed", remaining[2:end]...)
    startswith(first(remaining), "circuit_") && return joinpath(staged_raw, remaining...)
    return joinpath(model_root, "processed", remaining...)
end

end
