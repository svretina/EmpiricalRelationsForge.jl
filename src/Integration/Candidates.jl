struct PublishedPredictor{F,D} <: AbstractFittedModel
    relation_id::String
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    evaluator::F
    domain_check::D
    expression::String
    provenance::Dict{Symbol,Any}
end

"""Construct a fixed predictor from a supported published-relation package."""
function published_predictor end

function PublishedPredictor(relation_id, feature_names, target_name, evaluator;
                            units=Dict{Symbol,String}(), domain_check=(values -> true),
                            expression=string(relation_id), provenance=Dict{Symbol,Any}())
    PublishedPredictor(String(relation_id), collect(Symbol.(feature_names)), Symbol(target_name),
        Dict{Symbol,String}(units), evaluator, domain_check, String(expression),
        Dict{Symbol,Any}(provenance))
end
function predict(model::PublishedPredictor, X::AbstractMatrix)
    size(X, 2) == length(model.feature_names) || throw(DimensionMismatch("published model feature count differs"))
    [begin
        values = NamedTuple{Tuple(model.feature_names)}(Tuple(X[row, :]))
        model.domain_check(values) || throw(DomainError(values, "published calibration domain violation"))
        Float64(model.evaluator(values))
    end for row in axes(X, 1)]
end
predict(model::PublishedPredictor, data::RegressionDataset) = begin
    data.feature_names == model.feature_names || throw(ArgumentError("dataset feature order does not match published model"))
    predict(model, data.X)
end
formula(model::PublishedPredictor) = model.expression
coefficients(::PublishedPredictor) = Float64[]
parameter_names(::PublishedPredictor) = Symbol[]
parameter_vector(::PublishedPredictor) = Float64[]
model_structure(model::PublishedPredictor) = ModelStructure(:published, copy(model.feature_names),
    model.target_name, Symbol[], model.expression, _digest(model.relation_id * model.expression))

struct CandidateExport
    path::String
    model_file::String
    review_file::String
end

"""Export a candidate model artifact and a scientific review template; never registers it."""
function export_candidate(model, report; path::AbstractString="candidate_relation", overwrite::Bool=false,
                          uncertainty=nothing, applicability::AbstractString="Not yet established",
                          notes::AbstractString="")
    ispath(path) && !overwrite && throw(ArgumentError("export path already exists; pass overwrite=true to replace files"))
    mkpath(path)
    model_file = joinpath(path, "model.toml")
    save_model(model_file, model; metadata=Dict(:applicability=>applicability))
    review_file = joinpath(path, "review.md")
    validation = report isa ValidationReport ? report.aggregate_metrics :
                 report isa ModelResult ? report.validation : nothing
    open(review_file, "w") do io
        println(io, "# Candidate empirical relation review\n")
        println(io, "## Formula\n\n`$(formula(model))`\n")
        println(io, "## Parameters\n\n", parameter_vector(model), "\n")
        println(io, "## Applicability\n\n", applicability, "\n")
        println(io, "## Validation\n")
        if validation isa MetricReport
            println(io, "- RMSE: ", validation.rmse)
            println(io, "- MAE: ", validation.mae)
            println(io, "- Maximum absolute residual: ", validation.max_absolute_error)
        else
            println(io, "Validation metrics were not supplied.")
        end
        println(io, "\n## Uncertainty\n")
        println(io, isnothing(uncertainty) ? "Not estimated." : string(uncertainty))
        println(io, "\n## Scientific limitations\n\n", isempty(notes) ?
            "Independent review, domain assessment, structural stability, and extrapolation tests are required." : notes)
        println(io, "\nThis artifact is a research candidate and is not a registered published relation.")
    end
    CandidateExport(String(path), model_file, review_file)
end
