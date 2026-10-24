const DISCOVERY_SCHEMA_VERSION = 1

_metric_value(value) = ismissing(value) ? "missing" : Float64(value)
_load_metric_value(value) = value == "missing" ? missing : Float64(value)

function _metric_payload(report::MetricReport)
    Dict("n"=>report.n, "mae"=>report.mae, "rmse"=>report.rmse,
        "max_absolute_error"=>report.max_absolute_error,
        "mean_signed_residual"=>report.mean_signed_residual,
        "r_squared"=>_metric_value(report.r_squared),
        "mean_absolute_relative_error"=>_metric_value(report.mean_absolute_relative_error))
end

function _load_metric(payload)
    MetricReport(Int(payload["n"]), Float64(payload["mae"]), Float64(payload["rmse"]),
        Float64(payload["max_absolute_error"]), Float64(payload["mean_signed_residual"]),
        _load_metric_value(payload["r_squared"]),
        _load_metric_value(payload["mean_absolute_relative_error"]))
end

function _problem_payload(problem::RelationDiscoveryProblem)
    Dict("target"=>string(problem.target),
        "candidate_features"=>string.(problem.candidate_features),
        "mandatory_features"=>string.(problem.mandatory_features),
        "forbidden_features"=>string.(problem.forbidden_features),
        "min_features"=>problem.min_features, "max_features"=>problem.max_features,
        "transformations"=>[Dict("name"=>string(t.name), "source"=>string(t.source),
            "kind"=>string(t.kind), "parameter"=>t.parameter) for t in problem.transformations],
        "binary_operators"=>string.(problem.binary_operators),
        "unary_operators"=>string.(problem.unary_operators),
        "max_expression_complexity"=>problem.max_expression_complexity,
        "domains"=>Dict(string(k)=>[v[1], v[2]] for (k,v) in problem.domains),
        "units"=>Dict(string(k)=>v for (k,v) in problem.units),
        "variables"=>[Dict("name"=>string(v.name), "role"=>string(v.role),
            "accessible"=>v.accessible, "description"=>v.description)
            for v in values(problem.variables)],
        "objective"=>string(problem.objective),
        "complexity_penalty"=>problem.complexity_penalty,
        "variable_penalty"=>problem.variable_penalty)
end

function _load_problem(payload)
    transformations = FeatureTransformation[FeatureTransformation(Symbol(t["source"]), Symbol(t["kind"]);
        name=Symbol(t["name"]), parameter=Float64(t["parameter"]))
        for t in payload["transformations"]]
    variables = VariableDescriptor[VariableDescriptor(Symbol(v["name"]); role=Symbol(v["role"]),
        accessible=Bool(v["accessible"]), description=String(v["description"]))
        for v in payload["variables"]]
    symbols(values) = Symbol[Symbol(value) for value in values]
    RelationDiscoveryProblem(target=Symbol(payload["target"]),
        candidate_features=symbols(payload["candidate_features"]),
        mandatory_features=symbols(payload["mandatory_features"]),
        forbidden_features=symbols(payload["forbidden_features"]),
        min_features=Int(payload["min_features"]), max_features=Int(payload["max_features"]),
        transformations=transformations, binary_operators=symbols(payload["binary_operators"]),
        unary_operators=symbols(payload["unary_operators"]),
        max_expression_complexity=Int(payload["max_expression_complexity"]),
        domains=Dict(Symbol(k)=>(Float64(v[1]), Float64(v[2])) for (k,v) in payload["domains"]),
        units=Dict(Symbol(k)=>String(v) for (k,v) in payload["units"]), variables=variables,
        objective=Symbol(payload["objective"]),
        complexity_penalty=Float64(payload["complexity_penalty"]),
        variable_penalty=Float64(payload["variable_penalty"]))
end

function _evaluation_payload(evaluation::CandidateEvaluation)
    Dict("training"=>_metric_payload(evaluation.training),
        "has_validation"=>!isnothing(evaluation.validation),
        "validation"=>isnothing(evaluation.validation) ? Dict{String,Any}() :
            _metric_payload(evaluation.validation),
        "folds"=>[Dict("train"=>fold.train, "test"=>fold.test) for fold in evaluation.folds],
        "predictions"=>evaluation.predictions, "observed"=>evaluation.observed,
        "row_ids"=>_toml_safe(evaluation.row_ids))
end

function _load_evaluation(payload)
    validation = payload["has_validation"] ? _load_metric(payload["validation"]) : nothing
    CandidateEvaluation(_load_metric(payload["training"]), validation,
        [DataSplit(Int.(fold["train"]), Int.(fold["test"])) for fold in payload["folds"]],
        Float64.(payload["predictions"]), Float64.(payload["observed"]),
        Any[payload["row_ids"]...])
end

"""Save a complete, reconstructible discovery campaign and all candidate models."""
function save_discovery(path::AbstractString, result::RelationDiscoveryResult;
                        overwrite::Bool=false)
    ispath(path) && !overwrite && throw(ArgumentError(
        "discovery artifact already exists; pass overwrite=true to replace files"))
    mkpath(path)
    model_directory = joinpath(path, "models")
    mkpath(model_directory)
    candidates = Dict{String,Any}[]
    for (index, candidate) in enumerate(result.candidates)
        filename = "candidate-$(lpad(index, 4, '0')).toml"
        save_model(joinpath(model_directory, filename), candidate.model;
            metadata=Dict(:candidate_id=>candidate.id, :dataset_hash=>result.dataset_hash))
        push!(candidates, Dict("id"=>candidate.id, "model"=>joinpath("models", filename),
            "family"=>string(candidate.family),
            "selected_features"=>string.(candidate.selected_features),
            "input_features"=>string.(candidate.input_features),
            "source_features"=>string.(candidate.source_features),
            "transformations"=>[Dict("name"=>string(t.name), "source"=>string(t.source),
                "kind"=>string(t.kind), "parameter"=>t.parameter) for t in candidate.transformations],
            "complexity"=>candidate.complexity, "evaluation"=>_evaluation_payload(candidate.evaluation),
            "score"=>candidate.score, "conditioning"=>candidate.conditioning,
            "physical_validity"=>string(candidate.physical_validity),
            "applicability"=>candidate.applicability, "status"=>string(candidate.status),
            "provenance"=>_toml_safe(candidate.provenance)))
    end
    payload = Dict("schema_version"=>DISCOVERY_SCHEMA_VERSION,
        "problem"=>_problem_payload(result.problem), "candidates"=>candidates,
        "pareto_indices"=>result.pareto_indices, "subset_strategy"=>string(result.subset_strategy),
        "validation_strategy"=>result.validation_strategy, "seed"=>result.seed,
        "dataset_hash"=>result.dataset_hash, "training_row_ids"=>_toml_safe(result.training_row_ids),
        "configuration"=>_toml_safe(result.configuration),
        "julia_version"=>string(VERSION), "package_version"=>"0.1.0")
    open(joinpath(path, "discovery.toml"), "w") do io
        TOML.print(io, payload; sorted=true)
    end
    path
end

"""Load a discovery artifact written by `save_discovery`."""
function load_discovery(path::AbstractString)
    manifest = joinpath(path, "discovery.toml")
    isfile(manifest) || throw(ArgumentError("discovery manifest does not exist: $manifest"))
    payload = try TOML.parsefile(manifest) catch error
        throw(ArgumentError("invalid discovery artifact: $(sprint(showerror, error))"))
    end
    get(payload, "schema_version", nothing) == DISCOVERY_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported discovery artifact schema version"))
    candidates = RelationCandidate[]
    for candidate in payload["candidates"]
        model = load_model(joinpath(path, candidate["model"]))
        transformations = FeatureTransformation[FeatureTransformation(Symbol(t["source"]), Symbol(t["kind"]);
            name=Symbol(t["name"]), parameter=Float64(t["parameter"]))
            for t in candidate["transformations"]]
        push!(candidates, RelationCandidate(String(candidate["id"]), model,
            Symbol(candidate["family"]), Symbol.(candidate["selected_features"]),
            Symbol.(candidate["input_features"]), Symbol.(candidate["source_features"]),
            transformations, Int(candidate["complexity"]),
            _load_evaluation(candidate["evaluation"]), Float64(candidate["score"]),
            Float64(candidate["conditioning"]), Symbol(candidate["physical_validity"]),
            String(candidate["applicability"]), Symbol(candidate["status"]),
            Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in candidate["provenance"])))
    end
    RelationDiscoveryResult(_load_problem(payload["problem"]), candidates,
        Int.(payload["pareto_indices"]), Symbol(payload["subset_strategy"]),
        String(payload["validation_strategy"]), Int(payload["seed"]),
        String(payload["dataset_hash"]), Any[payload["training_row_ids"]...],
        Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in payload["configuration"]))
end
