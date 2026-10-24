struct ModelComparison
    name::Symbol
    source::Symbol
    validation::Union{Nothing,ValidationReport}
    complexity::Int
    nparameters::Int
    status::Symbol
    message::String
end

struct ComparisonReport{S<:AbstractValidationStrategy}
    strategy::S
    folds::Vector{DataSplit}
    models::Vector{ModelComparison}
end

_estimator_source(::AbstractRegressionMethod) = :trainable
_estimator_source(::SymbolicSearch) = :discovery
_estimator_source(::SymbolicCandidate) = :fixed_structure
_estimator_source(::RefittedSymbolicModel) = :fixed_structure
_estimator_source(::AbstractFittedModel) = :fixed_predictor

_fit_estimator(method::AbstractRegressionMethod, data) = fit(data, method)
_fit_estimator(candidate::SymbolicCandidate, data) = refit(candidate, data)
_fit_estimator(model::RefittedSymbolicModel, data) = refit(model.candidate, data;
    method=model.method, weighting=model.weighting, target_transform=model.target_transform,
    loss_space=model.loss_space, lower_bounds=model.lower_bounds, upper_bounds=model.upper_bounds,
    initial_guesses=model.parameters)
_fit_estimator(model::AbstractFittedModel, data) = model
function _fit_estimator(search::SymbolicSearch, data)
    result = discover(data, search)
    refit(best_model(result), data)
end

function _validation_from_folds(estimator, dataset, strategy, folds)
    predictions = eltype(dataset.y)[]
    observed = eltype(dataset.y)[]
    row_ids = Any[]
    groups = Any[]
    reports = MetricReport{eltype(dataset.y)}[]
    for partition in folds
        training = subset(dataset, partition.train)
        testing = subset(dataset, partition.test)
        fitted = _fit_estimator(estimator, training)
        fitted isa RefittedSymbolicModel && fitted.status != :success &&
            throw(ErrorException("symbolic refit failed: $(fitted.diagnostics.message)"))
        prediction = predict(fitted, testing)
        append!(predictions, prediction)
        append!(observed, testing.y)
        append!(row_ids, testing.row_ids)
        append!(groups, testing.eos_ids)
        push!(reports, metrics(prediction, testing.y))
    end
    group_reports = any(ismissing, groups) ? Dict{Any,MetricReport{eltype(dataset.y)}}() :
        Dict{Any,MetricReport{eltype(dataset.y)}}(group => report for (group, report) in
            grouped_metrics(predictions, observed, groups))
    ValidationReport(strategy, folds, reports, metrics(predictions, observed), predictions,
                     observed, row_ids, group_reports)
end

function validate(candidate::Union{SymbolicCandidate,RefittedSymbolicModel},
                  dataset::RegressionDataset;
                  strategy::AbstractValidationStrategy=LeaveOneEOSOut())
    folds = _splits(strategy, dataset)
    _validation_from_folds(candidate, dataset, strategy, folds)
end

function _model_complexity(estimator)
    estimator isa SymbolicCandidate && return estimator.complexity
    estimator isa RefittedSymbolicModel && return estimator.candidate.complexity
    estimator isa PolynomialRegression && return estimator.degree
    estimator isa LinearBasisRegression && return length(estimator.basis.terms)
    0
end

function _model_nparameters(estimator, dataset)
    try
        fitted = _fit_estimator(estimator, dataset)
        return length(parameter_vector(fitted))
    catch
        return 0
    end
end

"""Compare trainable estimators and fixed predictors on exactly the same folds."""
function compare(dataset::RegressionDataset, models;
                 validation::AbstractValidationStrategy=LeaveOneEOSOut())
    folds = _splits(validation, dataset)
    entries = ModelComparison[]
    for (index, entry) in enumerate(models)
        name, estimator = entry isa Pair ? (Symbol(first(entry)), last(entry)) :
                          (Symbol("model_", index), entry)
        source = _estimator_source(estimator)
        try
            report = _validation_from_folds(estimator, dataset, validation, folds)
            push!(entries, ModelComparison(name, source, report, _model_complexity(estimator),
                                            _model_nparameters(estimator, dataset), :success, "ok"))
        catch error
            push!(entries, ModelComparison(name, source, nothing, _model_complexity(estimator),
                                            0, :failed, sprint(showerror, error)))
        end
    end
    ComparisonReport(validation, folds, entries)
end

function select_model(report::ComparisonReport; policy=:minimum_rmse,
                      target_error::Real=Inf)
    successful = filter(entry -> entry.status == :success, report.models)
    isempty(successful) && throw(ArgumentError("comparison has no successful models"))
    policy isa Function && return policy(successful)
    policy == :minimum_rmse && return successful[argmin(m.validation.aggregate_metrics.rmse for m in successful)]
    if policy == :minimum_worst_group_rmse
        worst = map(successful) do model
            reports = collect(values(model.validation.group_metrics))
            isempty(reports) ? model.validation.aggregate_metrics.rmse : maximum(x.rmse for x in reports)
        end
        return successful[argmin(worst)]
    end
    if policy == :lowest_complexity_below_error
        feasible = filter(m -> m.validation.aggregate_metrics.rmse <= target_error, successful)
        isempty(feasible) && throw(ArgumentError("no model satisfies target_error"))
        return feasible[argmin(m.complexity for m in feasible)]
    end
    throw(ArgumentError("unknown model-selection policy $policy"))
end

struct NestedValidationReport{S<:AbstractValidationStrategy,I<:AbstractValidationStrategy}
    outer_strategy::S
    inner_strategy::I
    outer_folds::Vector{DataSplit}
    selected_formulas::Vector{String}
    inner_reports::Vector{ComparisonReport}
    outer_metrics::Vector{MetricReport{Float64}}
    aggregate_metrics::MetricReport{Float64}
    predictions::Vector{Float64}
    observed::Vector{Float64}
    row_ids::Vector{Any}
end

"""Nested symbolic discovery and fixed-structure selection without outer-fold leakage."""
function nested_validate(search::SymbolicSearch, dataset::RegressionDataset;
                         outer::AbstractValidationStrategy=LeaveOneEOSOut(),
                         inner::AbstractValidationStrategy=GroupedKFold(k=3, by=:eos, seed=0),
                         policy::Symbol=:minimum_rmse)
    outer_folds = _splits(outer, dataset)
    formulas = String[]
    inner_reports = ComparisonReport[]
    fold_metrics = MetricReport{Float64}[]
    predictions = Float64[]
    observed = Float64[]
    row_ids = Any[]
    for partition in outer_folds
        training = subset(dataset, partition.train)
        testing = subset(dataset, partition.test)
        discovery = discover(training, search)
        isempty(discovery.candidates) && throw(ErrorException("symbolic discovery returned no candidates"))
        named = [Symbol("candidate_", i) => candidate for (i, candidate) in enumerate(discovery.candidates)]
        comparison = compare(training, named; validation=inner)
        selected = select_model(comparison; policy)
        selected_index = parse(Int, split(string(selected.name), "_")[end])
        chosen = discovery.candidates[selected_index]
        fitted = refit(chosen, training)
        fitted.status == :success || throw(ErrorException("selected symbolic structure could not be refitted"))
        prediction = Float64.(predict(fitted, testing))
        append!(predictions, prediction)
        append!(observed, testing.y)
        append!(row_ids, testing.row_ids)
        push!(fold_metrics, metrics(prediction, testing.y))
        push!(formulas, formula(fitted))
        push!(inner_reports, comparison)
    end
    NestedValidationReport(outer, inner, outer_folds, formulas, inner_reports, fold_metrics,
        metrics(predictions, observed), predictions, observed, row_ids)
end
