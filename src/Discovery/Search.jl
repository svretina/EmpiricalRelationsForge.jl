struct CandidateEvaluation
    training::MetricReport{Float64}
    validation::Union{Nothing,MetricReport{Float64}}
    folds::Vector{DataSplit}
    predictions::Vector{Float64}
    observed::Vector{Float64}
    row_ids::Vector{Any}
end

"""One fitted relation produced by a discovery campaign."""
struct RelationCandidate{M} <: AbstractFittedModel
    id::String
    model::M
    family::Symbol
    selected_features::Vector{Symbol}
    input_features::Vector{Symbol}
    source_features::Vector{Symbol}
    transformations::Vector{FeatureTransformation}
    complexity::Int
    evaluation::CandidateEvaluation
    score::Float64
    conditioning::Float64
    physical_validity::Symbol
    applicability::String
    status::Symbol
    provenance::Dict{Symbol,Any}
end

struct RelationDiscoveryResult
    problem::RelationDiscoveryProblem
    candidates::Vector{RelationCandidate}
    pareto_indices::Vector{Int}
    subset_strategy::Symbol
    validation_strategy::String
    seed::Int
    dataset_hash::String
    training_row_ids::Vector{Any}
    configuration::Dict{Symbol,Any}
end

"""Outer-fold assessment with the complete discovery procedure repeated per fold."""
struct DiscoveryValidationReport{S<:AbstractValidationStrategy}
    strategy::S
    folds::Vector{DataSplit}
    fold_metrics::Vector{MetricReport{Float64}}
    aggregate_metrics::MetricReport{Float64}
    predictions::Vector{Float64}
    observed::Vector{Float64}
    row_ids::Vector{Any}
    selected_formulas::Vector{String}
    selected_features::Vector{Vector{Symbol}}
end

function _source_matrix(source_features, transforms, X, input_names)
    lookup = Dict(name => index for (index, name) in enumerate(input_names))
    transform_lookup = Dict(transform.name => transform for transform in transforms)
    columns = Vector{Vector{Float64}}()
    for name in source_features
        if haskey(lookup, name)
            push!(columns, Float64.(view(X, :, lookup[name])))
        elseif haskey(transform_lookup, name)
            transform = transform_lookup[name]
            haskey(lookup, transform.source) || throw(ArgumentError("missing source feature $(transform.source)"))
            push!(columns, _transform_values(transform, view(X, :, lookup[transform.source])))
        else
            throw(ArgumentError("model feature :$name cannot be constructed"))
        end
    end
    hcat(columns...)
end

predict(candidate::RelationCandidate, data::RegressionDataset) = begin
    candidate.model.target_name == data.target_name || throw(ArgumentError("target identity differs"))
    predict(candidate.model, _source_matrix(candidate.source_features, candidate.transformations,
        data.X, data.feature_names))
end
function predict(candidate::RelationCandidate, X::AbstractMatrix)
    size(X, 2) == length(candidate.input_features) || throw(DimensionMismatch(
        "matrix prediction requires columns in input-feature order"))
    predict(candidate.model, _source_matrix(candidate.source_features,
        candidate.transformations, X, candidate.input_features))
end
formula(candidate::RelationCandidate) = formula(candidate.model)
coefficients(candidate::RelationCandidate) = coefficients(candidate.model)
parameter_names(candidate::RelationCandidate) = parameter_names(candidate.model)
parameter_vector(candidate::RelationCandidate) = parameter_vector(candidate.model)
function with_parameters(candidate::RelationCandidate, values)
    RelationCandidate(candidate.id, with_parameters(candidate.model, values), candidate.family,
        copy(candidate.selected_features), copy(candidate.input_features), copy(candidate.source_features),
        copy(candidate.transformations),
        candidate.complexity, candidate.evaluation, candidate.score, candidate.conditioning,
        candidate.physical_validity, candidate.applicability, candidate.status, copy(candidate.provenance))
end
function model_structure(candidate::RelationCandidate)
    structure = model_structure(candidate.model)
    signature = structure.fingerprint * ";selected=" * join(string.(candidate.selected_features), ",")
    ModelStructure(:relation_candidate, copy(candidate.selected_features), structure.target_name,
        structure.parameter_names, structure.expression, _digest(signature))
end

function _column_dataset(dataset, feature_names; transforms=FeatureTransformation[])
    X = _source_matrix(feature_names, transforms, dataset.X, dataset.feature_names)
    units = Dict{Symbol,String}(dataset.target_name => get(dataset.units, dataset.target_name, ""))
    transform_lookup = Dict(t.name => t for t in transforms)
    for name in feature_names
        units[name] = if name in dataset.feature_names
            get(dataset.units, name, "")
        else
            get(dataset.units, transform_lookup[name].source, "")
        end
    end
    RegressionDataset(X, dataset.y; features=feature_names, target=dataset.target_name,
        units, row_ids=dataset.row_ids, eos_ids=dataset.eos_ids,
        simulation_ids=dataset.simulation_ids, source_papers=dataset.source_papers,
        observation_kinds=dataset.observation_kinds, uncertainties=dataset.uncertainties,
        uncertainty_kind=dataset.uncertainty_kind, weights=dataset.weights,
        transformations=dataset.transformations)
end

function _combinations(values, k)
    k == 0 && return [eltype(values)[]]
    k > length(values) && return Vector{Vector{eltype(values)}}()
    result = Vector{Vector{eltype(values)}}()
    for index in 1:(length(values) - k + 1)
        for tail in _combinations(values[(index + 1):end], k - 1)
            push!(result, vcat(values[index], tail))
        end
    end
    result
end

function _all_subsets(problem::RelationDiscoveryProblem)
    available = setdiff(problem.candidate_features, problem.forbidden_features)
    optional = setdiff(available, problem.mandatory_features)
    subsets = Vector{Vector{Symbol}}()
    for count in problem.min_features:problem.max_features
        optional_count = count - length(problem.mandatory_features)
        optional_count < 0 && continue
        for chosen in _combinations(optional, optional_count)
            selected = [name for name in available if name in union(problem.mandatory_features, chosen)]
            push!(subsets, selected)
        end
    end
    unique(subsets)
end

function _check_problem(problem, dataset)
    problem.target == dataset.target_name || throw(ArgumentError(
        "problem target $(problem.target) differs from dataset target $(dataset.target_name)"))
    missing_features = setdiff(problem.candidate_features, dataset.feature_names)
    isempty(missing_features) || throw(ArgumentError("dataset is missing candidate features: $(join(missing_features, ", "))"))
    for (name, unit) in problem.units
        haskey(dataset.units, name) && dataset.units[name] != unit && throw(ArgumentError(
            "unit convention for :$name differs between problem and dataset"))
    end
    for (name, (lower, upper)) in problem.domains
        values = if name == dataset.target_name
            dataset.y
        else
            column = findfirst(==(name), dataset.feature_names)
            isnothing(column) && continue
            view(dataset.X, :, column)
        end
        all(value -> lower <= value <= upper, values) || throw(DomainError(name,
            "dataset values fall outside the discovery problem domain"))
    end
end

_conditioning(model::FittedPolynomialModel) = model.condition_number
_conditioning(model::FittedBasisModel) = model.condition_number
_conditioning(model::FittedNonlinearModel) = NaN
_conditioning(model::RefittedSymbolicModel) = model.diagnostics.jacobian_condition
_conditioning(model) = NaN

function _candidate_score(problem, training, validation, complexity, nvariables)
    error = problem.objective == :training_rmse || isnothing(validation) ? training.rmse : validation.rmse
    penalized = problem.objective == :penalized ||
        problem.complexity_penalty > 0 || problem.variable_penalty > 0
    penalized ?
        error + problem.complexity_penalty * complexity + problem.variable_penalty * nvariables : error
end

function _float_metrics(report)
    MetricReport(report.n, Float64(report.mae), Float64(report.rmse),
        Float64(report.max_absolute_error), Float64(report.mean_signed_residual),
        ismissing(report.r_squared) ? missing : Float64(report.r_squared),
        ismissing(report.mean_absolute_relative_error) ? missing :
            Float64(report.mean_absolute_relative_error))
end

function _evaluation(model, data, validation; estimator=model)
    training = _float_metrics(metrics(model, data))
    isnothing(validation) && return CandidateEvaluation(training, nothing, DataSplit[], Float64[], Float64[], Any[])
    folds = _splits(validation, data)
    report = _validation_from_folds(estimator, data, validation, folds)
    converted = _float_metrics(report.aggregate_metrics)
    CandidateEvaluation(training, converted, report.folds, Float64.(report.predictions),
        Float64.(report.observed), copy(report.row_ids))
end

function _polynomial_candidates(problem, dataset, subset, search, validation, counter)
    transforms = search.include_transformations ?
        filter(t -> t.source in subset, problem.transformations) : FeatureTransformation[]
    model_features = Symbol.(vcat(subset, getfield.(transforms, :name)))
    working = _column_dataset(dataset, model_features; transforms)
    results = RelationCandidate[]
    for degree in search.min_degree:search.max_degree
        method = PolynomialRegression(; degree, cross_terms=search.interactions,
            weighting=search.weighting, standardize=search.standardize, solver=search.solver)
        try
            model = fit(working, method)
            evaluation = _evaluation(model, working, validation; estimator=method)
            complexity = length(model.beta) + degree
            score = _candidate_score(problem, evaluation.training, evaluation.validation,
                complexity, length(subset))
            push!(results, RelationCandidate("candidate-$(counter + length(results))", model,
                :polynomial, copy(subset), copy(subset), model_features, transforms, complexity, evaluation,
                score, model.condition_number, :unchecked, "Defined by the supplied data domain",
                :success, Dict{Symbol,Any}(:degree=>degree, :interactions=>search.interactions)))
        catch error
            @warn "polynomial candidate failed" subset degree exception=(error, catch_backtrace())
        end
    end
    results
end

function _used_symbolic_features(candidate::SymbolicCandidate)
    indices = Int[]
    function visit(node)
        if node.degree == 0
            !node.constant && push!(indices, Int(node.feature))
            return
        end
        foreach(visit, _node_children(node))
    end
    visit(_raw_tree(candidate.expression))
    unique(candidate.feature_names[sort(unique(indices))])
end

function _symbolic_candidates(problem, dataset, subset, search, validation, counter)
    working = _column_dataset(dataset, subset)
    discovery = discover(working, search)
    results = RelationCandidate[]
    for symbolic in discovery.candidates
        used = _used_symbolic_features(symbolic)
        length(used) < problem.min_features && continue
        all(in(used), problem.mandatory_features) || continue
        symbolic.complexity <= problem.max_expression_complexity || continue
        try
            refined = refit(symbolic, working)
            refined.status == :success || continue
            evaluation = _evaluation(refined, working, validation; estimator=refined)
            score = _candidate_score(problem, evaluation.training, evaluation.validation,
                symbolic.complexity, length(used))
            push!(results, RelationCandidate("candidate-$(counter + length(results))", refined,
                :symbolic, used, copy(subset), copy(subset), FeatureTransformation[], symbolic.complexity,
                evaluation, score, refined.diagnostics.jacobian_condition, :unchecked,
                "Defined by the supplied data domain", :success,
                Dict{Symbol,Any}(:backend_version=>string(discovery.backend_version),
                    :search=>discovery.search_configuration)))
        catch error
            @warn "symbolic candidate refit failed" exception=(error, catch_backtrace())
        end
    end
    results
end

function _fixed_candidate(problem, dataset, model, validation, counter)
    all(name -> name in dataset.feature_names, model.feature_names) ||
        throw(ArgumentError("fixed predictor requires unavailable inputs"))
    working = _column_dataset(dataset, model.feature_names)
    evaluation = _evaluation(model, working, validation)
    score = _candidate_score(problem, evaluation.training, evaluation.validation, 0, length(model.feature_names))
    RelationCandidate("candidate-$counter", model, :published, copy(model.feature_names),
        copy(model.feature_names),
        copy(model.feature_names), FeatureTransformation[], 0, evaluation, score, NaN,
        :externally_defined, get(model.provenance, :applicability, "Published calibration domain"),
        :success, copy(model.provenance))
end

function _basis_candidate(problem, dataset, subset, method, validation, counter)
    required = unique(reduce(vcat,
        [collect(keys(term.powers)) for term in method.basis.terms]; init=Symbol[]))
    Set(required) == Set(subset) || return RelationCandidate[]
    working = _column_dataset(dataset, subset)
    model = fit(working, method)
    evaluation = _evaluation(model, working, validation; estimator=method)
    complexity = length(model.beta) + sum(length(term.powers) for term in method.basis.terms)
    score = _candidate_score(problem, evaluation.training, evaluation.validation,
        complexity, length(subset))
    [RelationCandidate("candidate-$counter", model, :linear_basis, copy(subset),
        copy(subset), copy(subset), FeatureTransformation[], complexity, evaluation,
        score, model.condition_number, :unchecked, "Defined by the supplied data domain",
        :success, Dict{Symbol,Any}(:basis_terms=>string.(getfield.(method.basis.terms, :id))))]
end

function _discover_subset(problem, dataset, subset, methods, validation, counter)
    candidates = RelationCandidate[]
    for method in methods
        found = if method isa PolynomialSearch
            _polynomial_candidates(problem, dataset, subset, method, validation, counter + length(candidates))
        elseif method isa LinearBasisRegression
            _basis_candidate(problem, dataset, subset, method, validation, counter + length(candidates))
        elseif method isa SymbolicSearch
            _symbolic_candidates(problem, dataset, subset, method, validation, counter + length(candidates))
        elseif method isa PublishedPredictor
            Set(subset) == Set(method.feature_names) ?
                [_fixed_candidate(problem, dataset, method, validation, counter + length(candidates))] : RelationCandidate[]
        elseif method isa Function
            configured = method(subset)
            configured isa AbstractRegressionMethod || throw(ArgumentError("method factory must return a regression method"))
            working = _column_dataset(dataset, subset)
            model = fit(working, configured)
            evaluation = _evaluation(model, working, validation; estimator=configured)
            complexity = length(parameter_vector(model))
            score = _candidate_score(problem, evaluation.training, evaluation.validation, complexity, length(subset))
            [RelationCandidate("candidate-$(counter + length(candidates))", model, :custom,
                copy(subset), copy(subset), copy(subset), FeatureTransformation[], complexity, evaluation,
                score, _conditioning(model), :unchecked, "Defined by the supplied data domain",
                :success, Dict{Symbol,Any}())]
        else
            throw(ArgumentError("unsupported discovery method $(typeof(method))"))
        end
        append!(candidates, found)
    end
    candidates
end

function pareto_frontier(candidates::AbstractVector{<:RelationCandidate})
    indices = Int[]
    for i in eachindex(candidates)
        ci = candidates[i]
        error_i = isnothing(ci.evaluation.validation) ? ci.evaluation.training.rmse : ci.evaluation.validation.rmse
        dominated = any(eachindex(candidates)) do j
            i == j && return false
            cj = candidates[j]
            error_j = isnothing(cj.evaluation.validation) ? cj.evaluation.training.rmse : cj.evaluation.validation.rmse
            (error_j <= error_i && cj.complexity <= ci.complexity &&
             length(cj.selected_features) <= length(ci.selected_features)) &&
            (error_j < error_i || cj.complexity < ci.complexity ||
             length(cj.selected_features) < length(ci.selected_features))
        end
        dominated || push!(indices, i)
    end
    indices
end

pareto_frontier(result::RelationDiscoveryResult) = result.candidates[result.pareto_indices]

function _best_score(candidates)
    isempty(candidates) ? Inf : minimum(candidate.score for candidate in candidates)
end

"""Explore feature subsets and model families, retaining all successful relations."""
function discover_relations(problem::RelationDiscoveryProblem, dataset::RegressionDataset;
        methods=[PolynomialSearch()], subset_strategy::Symbol=:exhaustive,
        validation=nothing, seed::Integer=0)
    _check_problem(problem, dataset)
    subset_strategy in (:exhaustive, :penalized, :forward, :backward, :symbolic) ||
        throw(ArgumentError("unsupported subset strategy :$subset_strategy"))
    length(dataset) >= 2 || throw(ArgumentError("relation discovery requires at least two observations"))
    chosen_validation = isnothing(validation) ? KFold(k=min(5, length(dataset)), seed=seed) : validation
    candidates = RelationCandidate[]
    if subset_strategy in (:exhaustive, :penalized)
        for subset in _all_subsets(problem)
            append!(candidates, _discover_subset(problem, dataset, subset, methods,
                chosen_validation, length(candidates) + 1))
        end
    elseif subset_strategy == :symbolic
        searches = filter(method -> method isa SymbolicSearch, methods)
        isempty(searches) && throw(ArgumentError(":symbolic subset strategy requires SymbolicSearch"))
        subset = setdiff(problem.candidate_features, problem.forbidden_features)
        append!(candidates, _discover_subset(problem, dataset, subset, searches,
            chosen_validation, 1))
    elseif subset_strategy == :forward
        selected = copy(problem.mandatory_features)
        available = setdiff(setdiff(problem.candidate_features, problem.forbidden_features), selected)
        while length(selected) < problem.max_features && !isempty(available)
            trials = [(feature, _discover_subset(problem, dataset, vcat(selected, feature), methods,
                chosen_validation, length(candidates) + 1)) for feature in available]
            best = argmin(_best_score(last(trial)) for trial in trials)
            feature, found = trials[best]
            length(selected) + 1 >= problem.min_features && append!(candidates, found)
            push!(selected, feature)
            filter!(!=(feature), available)
        end
    else
        selected = setdiff(problem.candidate_features, problem.forbidden_features)
        while length(selected) >= max(problem.min_features, length(problem.mandatory_features))
            if length(selected) <= problem.max_features
                append!(candidates, _discover_subset(problem, dataset, selected, methods,
                    chosen_validation, length(candidates) + 1))
            end
            length(selected) == max(problem.min_features, length(problem.mandatory_features)) && break
            removable = setdiff(selected, problem.mandatory_features)
            trials = [(feature, _discover_subset(problem, dataset, setdiff(selected, [feature]), methods,
                chosen_validation, length(candidates) + 1)) for feature in removable]
            best = argmin(_best_score(last(trial)) for trial in trials)
            selected = setdiff(selected, [first(trials[best])])
        end
    end
    isempty(candidates) && throw(ErrorException("discovery produced no valid candidates"))
    sort!(candidates; by=candidate -> candidate.score)
    RelationDiscoveryResult(problem, candidates, pareto_frontier(candidates), subset_strategy,
        string(typeof(chosen_validation)), Int(seed), dataset_fingerprint(dataset),
        copy(dataset.row_ids), Dict(:methods=>string.(typeof.(methods))))
end

"""
    validate_discovery(problem, dataset; methods, outer, inner, ...)

Estimate generalization error without feature- or structure-selection leakage. Every
outer training partition repeats the complete subset and model-family search.
"""
function validate_discovery(problem::RelationDiscoveryProblem, dataset::RegressionDataset;
        methods=[PolynomialSearch()], outer::AbstractValidationStrategy=KFold(k=5),
        inner::Union{Nothing,AbstractValidationStrategy}=nothing,
        subset_strategy::Symbol=:exhaustive, criterion=:validation_rmse, seed::Integer=0)
    _check_problem(problem, dataset)
    folds = _splits(outer, dataset)
    predictions = Float64[]
    observed = Float64[]
    row_ids = Any[]
    fold_reports = MetricReport{Float64}[]
    selected_formulas = String[]
    selected_features = Vector{Symbol}[]
    for (fold_index, partition) in enumerate(folds)
        training = subset(dataset, partition.train)
        testing = subset(dataset, partition.test)
        length(training) >= 2 || throw(ArgumentError(
            "outer training partitions must contain at least two observations"))
        inner_strategy = isnothing(inner) ? KFold(k=min(5, length(training)),
            seed=Int(seed) + fold_index) : inner
        result = discover_relations(problem, training; methods, subset_strategy,
            validation=inner_strategy, seed=Int(seed) + fold_index)
        candidate = select_model(result; criterion)
        fold_predictions = Float64.(predict(candidate, testing))
        append!(predictions, fold_predictions)
        append!(observed, Float64.(testing.y))
        append!(row_ids, testing.row_ids)
        push!(fold_reports, _float_metrics(metrics(fold_predictions, testing.y)))
        push!(selected_formulas, formula(candidate))
        push!(selected_features, copy(candidate.selected_features))
    end
    DiscoveryValidationReport(outer, folds, fold_reports,
        _float_metrics(metrics(predictions, observed)), predictions, observed, row_ids,
        selected_formulas, selected_features)
end

function _candidate_group_key(candidate::RelationCandidate)
    model = candidate.model
    fingerprint = model isa RefittedSymbolicModel ?
        structural_fingerprint(model.candidate) : model_structure(model).fingerprint
    string(candidate.family, ":", join(candidate.selected_features, ","), ":", fingerprint)
end

"""Return a sortable candidate DataFrame; safely equivalent structures are grouped by default."""
function leaderboard(result::RelationDiscoveryResult; pareto::Bool=false, unique::Bool=true)
    indices = pareto ? result.pareto_indices : collect(eachindex(result.candidates))
    if unique
        seen = Set{String}()
        indices = filter(indices) do index
            key = _candidate_group_key(result.candidates[index])
            key in seen && return false
            push!(seen, key)
            true
        end
    end
    DataFrame([
        (; rank=rank, id=candidate.id, family=candidate.family,
           target=result.problem.target, features=join(string.(candidate.selected_features), ", "),
           nvariables=length(candidate.selected_features), nparameters=length(parameter_vector(candidate)),
           complexity=candidate.complexity, training_rmse=candidate.evaluation.training.rmse,
           validation_rmse=isnothing(candidate.evaluation.validation) ? missing : candidate.evaluation.validation.rmse,
           mae=isnothing(candidate.evaluation.validation) ? candidate.evaluation.training.mae : candidate.evaluation.validation.mae,
           max_error=isnothing(candidate.evaluation.validation) ? candidate.evaluation.training.max_absolute_error : candidate.evaluation.validation.max_absolute_error,
           conditioning=candidate.conditioning, score=candidate.score,
           formula=formula(candidate))
        for (rank, index) in enumerate(indices) for candidate in (result.candidates[index],)
    ])
end

function select_model(result::RelationDiscoveryResult; criterion=:validation_rmse)
    criterion isa Function && return criterion(result.candidates)
    criterion == :score && return result.candidates[argmin(candidate.score for candidate in result.candidates)]
    criterion == :training_rmse && return result.candidates[argmin(candidate.evaluation.training.rmse for candidate in result.candidates)]
    criterion == :validation_rmse && return result.candidates[argmin(
        isnothing(candidate.evaluation.validation) ? Inf : candidate.evaluation.validation.rmse
        for candidate in result.candidates)]
    criterion == :complexity && return result.candidates[argmin(candidate.complexity for candidate in result.candidates)]
    throw(ArgumentError("unknown discovery selection criterion $criterion"))
end

function filter_candidates(result::RelationDiscoveryResult; roles=nothing,
                           accessible_only::Bool=false, families=nothing)
    filter(result.candidates) do candidate
        isnothing(families) || candidate.family in families || return false
        descriptors = [get(result.problem.variables, feature,
            VariableDescriptor(feature)) for feature in candidate.selected_features]
        accessible_only && !all(descriptor.accessible for descriptor in descriptors) && return false
        isnothing(roles) || all(descriptor.role in roles for descriptor in descriptors)
    end
end
