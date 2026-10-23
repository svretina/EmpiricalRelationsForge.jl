"""A research-candidate model together with fit, validation, and provenance records."""
struct ModelResult{M,F,V}
    model::M
    fit_metrics::F
    validation::V
    provenance::Dict{Symbol,Any}
    reproducibility::Dict{Symbol,Any}
end

function model_result(model, dataset::RegressionDataset; validation=nothing,
                      provenance=Dict{Symbol,Any}(), reproducibility=Dict{Symbol,Any}())
    defaults = Dict{Symbol,Any}(
        :target => dataset.target_name,
        :features => copy(dataset.feature_names),
        :row_ids => copy(dataset.row_ids),
        :source_papers => unique(dataset.source_papers),
        :units => copy(dataset.units),
    )
    ModelResult(model, metrics(model, dataset), validation, merge(defaults, provenance),
                merge(Dict{Symbol,Any}(:julia_version => string(VERSION)), reproducibility))
end

function _toml_safe(value)
    value === missing && return "missing"
    value === nothing && return "nothing"
    value isa Symbol && return string(value)
    value isa AbstractDict && return Dict(string(k) => _toml_safe(v) for (k, v) in value)
    value isa AbstractVector && return [_toml_safe(v) for v in value]
    value isa Union{String,Number,Bool} && return value
    return string(value)
end

"""Write portable model metadata. Native model objects are deliberately not reconstructed from this file."""
function save_metadata(path::AbstractString, result::ModelResult)
    model = result.model
    payload = Dict{String,Any}(
        "model" => Dict(
            "type" => string(typeof(model)),
            "formula" => formula(model),
            "coefficients" => applicable(coefficients, model) ? coefficients(model) : Any[],
        ),
        "fit_metrics" => Dict(string(name) => _toml_safe(getfield(result.fit_metrics, name))
                              for name in fieldnames(typeof(result.fit_metrics))),
        "provenance" => _toml_safe(result.provenance),
        "reproducibility" => _toml_safe(result.reproducibility),
    )
    open(path, "w") do io
        TOML.print(io, payload; sorted=true)
    end
    path
end
