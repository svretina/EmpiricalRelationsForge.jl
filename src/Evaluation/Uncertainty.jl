struct CovarianceEstimate
    covariance::Union{Nothing,Matrix{Float64}}
    standard_errors::Union{Nothing,Vector{Float64}}
    parameter_names::Vector{Symbol}
    rank::Int
    condition_number::Float64
    residual_variance::Float64
    assumptions::Vector{String}
    status::Symbol
    message::String
end

function _prediction_jacobian(model, X)
    parameters = parameter_vector(model)
    base = Float64.(predict(model, X))
    J = Matrix{Float64}(undef, length(base), length(parameters))
    for j in eachindex(parameters)
        step = sqrt(eps(Float64)) * max(abs(parameters[j]), 1.0)
        shifted = copy(parameters)
        shifted[j] += step
        J[:, j] = (Float64.(predict(with_parameters(model, shifted), X)) .- base) ./ step
    end
    J
end

_weighting(model::FittedPolynomialModel) = model.method.weighting
_weighting(model::FittedBasisModel) = model.method.weighting
_weighting(model::FittedNonlinearModel) = model.specification.weighting
_weighting(model::RefittedSymbolicModel) = model.weighting
_weighting(model) = :none

"""Local parameter covariance conditional on a fixed model structure."""
function parameter_covariance(model, dataset::RegressionDataset; known_variance=nothing)
    J = _prediction_jacobian(model, dataset.X)
    weighting = _weighting(model)
    weights = _fit_weights(dataset, weighting)
    Jw = isnothing(weights) ? J : J .* sqrt.(weights)
    singular = svdvals(Jw)
    tolerance = isempty(singular) ? 0.0 : sqrt(eps(Float64)) * max(size(Jw)...) * max(first(singular), 1.0)
    r = count(>(tolerance), singular)
    condition = isempty(singular) || iszero(last(singular)) ? Inf : first(singular) / last(singular)
    p = size(Jw, 2)
    assumptions = ["independent residuals", "fixed mathematical structure", "local linear approximation"]
    if r < p
        return CovarianceEstimate(nothing, nothing, parameter_names(model), r, condition, NaN,
                                  assumptions, :singular, "parameter Jacobian is rank deficient")
    end
    residual = Float64.(predict(model, dataset) .- dataset.y)
    weighted_residual = isnothing(weights) ? residual : sqrt.(weights) .* residual
    dof = length(residual) - p
    variance = if !isnothing(known_variance)
        push!(assumptions, "known residual variance")
        Float64(known_variance)
    elseif weighting == :inverse_variance
        push!(assumptions, "known observation variances supplied")
        1.0
    else
        dof > 0 || return CovarianceEstimate(nothing, nothing, parameter_names(model), r,
            condition, NaN, assumptions, :insufficient_dof, "residual degrees of freedom are not positive")
        push!(assumptions, "common unknown residual variance estimated from residuals")
        sum(abs2, weighted_residual) / dof
    end
    factor = svd(Jw)
    covariance = variance .* (factor.V * Diagonal(inv.(factor.S .^ 2)) * factor.V')
    CovarianceEstimate(covariance, sqrt.(max.(diag(covariance), 0.0)), parameter_names(model),
                       r, condition, variance, assumptions, :success, "ok")
end

function prediction_uncertainty(model, covariance::CovarianceEstimate, X::AbstractMatrix;
                                level::Real=0.95)
    covariance.status == :success || throw(ArgumentError("covariance estimate is unavailable"))
    0 < level < 1 || throw(ArgumentError("level must lie between zero and one"))
    J = _prediction_jacobian(model, X)
    variance = vec(sum((J * covariance.covariance) .* J; dims=2))
    standard_error = sqrt.(max.(variance, 0.0))
    (; prediction=Float64.(predict(model, X)), standard_error,
       kind=:parameter_induced, level=Float64(level))
end

struct BootstrapResult
    parameter_names::Vector{Symbol}
    parameter_samples::Matrix{Float64}
    prediction_samples::Matrix{Float64}
    parameter_intervals::Matrix{Float64}
    prediction_intervals::Matrix{Float64}
    failures::Int
    unique_group_counts::Vector{Int}
    group::Symbol
    seed::Int
    confidence_level::Float64
    diagnostics::Vector{String}
end

_refit_like(model::FittedPolynomialModel, data) = fit(data, model.method)
_refit_like(model::FittedBasisModel, data) = fit(data, model.method)
_refit_like(model::FittedNonlinearModel, data) = fit(data, model.specification)
_refit_like(model::SymbolicCandidate, data) = refit(model, data)
_refit_like(model::RefittedSymbolicModel, data) = refit(model.candidate, data;
    method=model.method, weighting=model.weighting, target_transform=model.target_transform,
    loss_space=model.loss_space, lower_bounds=model.lower_bounds, upper_bounds=model.upper_bounds,
    initial_guesses=model.parameters)

function _bootstrap_dataset(dataset, indices, replicate)
    RegressionDataset(dataset.X[indices, :], dataset.y[indices];
        features=dataset.feature_names, target=dataset.target_name, units=dataset.units,
        row_ids=[(dataset.row_ids[index], replicate, position) for (position, index) in enumerate(indices)],
        eos_ids=dataset.eos_ids[indices], simulation_ids=dataset.simulation_ids[indices],
        source_papers=dataset.source_papers[indices], observation_kinds=dataset.observation_kinds[indices],
        uncertainties=isnothing(dataset.uncertainties) ? nothing : dataset.uncertainties[indices],
        uncertainty_kind=dataset.uncertainty_kind,
        weights=isnothing(dataset.weights) ? nothing : dataset.weights[indices],
        transformations=dataset.transformations)
end

"""Refit a fixed structure on complete groups sampled with replacement."""
function bootstrap(model, dataset::RegressionDataset; group::Symbol=:eos,
                   n::Integer=1000, seed::Integer=0, confidence_level::Real=0.95,
                   prediction_data::RegressionDataset=dataset, on_failure::Symbol=:record)
    n > 0 || throw(ArgumentError("n must be positive"))
    0 < confidence_level < 1 || throw(ArgumentError("confidence_level must lie between zero and one"))
    on_failure in (:record, :error) || throw(ArgumentError("on_failure must be :record or :error"))
    groups = _groups(dataset, group)
    any(ismissing, groups) && throw(ArgumentError("bootstrap group identifiers cannot be missing"))
    unique_groups = unique(groups)
    length(unique_groups) >= 2 || throw(ArgumentError("grouped bootstrap requires at least two groups"))
    rng = MersenneTwister(seed)
    parameter_rows = Vector{Vector{Float64}}()
    prediction_rows = Vector{Vector{Float64}}()
    unique_counts = Int[]
    failures = 0
    diagnostics = String[]
    for replicate in 1:n
        sampled = rand(rng, unique_groups, length(unique_groups))
        indices = reduce(vcat, [findall(==(selected), groups) for selected in sampled])
        push!(unique_counts, length(unique(sampled)))
        try
            fitted = _refit_like(model, _bootstrap_dataset(dataset, indices, replicate))
            fitted isa RefittedSymbolicModel && fitted.status != :success && error(fitted.diagnostics.message)
            push!(parameter_rows, Float64.(parameter_vector(fitted)))
            push!(prediction_rows, Float64.(predict(fitted, prediction_data)))
        catch error
            failures += 1
            push!(diagnostics, "replicate $replicate: $(sprint(showerror, error))")
            on_failure == :error && rethrow()
        end
    end
    isempty(parameter_rows) && throw(ErrorException("every bootstrap fit failed"))
    parameters = reduce(hcat, parameter_rows)'
    predictions = reduce(hcat, prediction_rows)'
    alpha = (1 - confidence_level) / 2
    p_intervals = hcat([quantile(view(parameters, :, j), [alpha, 1 - alpha]) for j in axes(parameters, 2)]...)'
    y_intervals = hcat([quantile(view(predictions, :, j), [alpha, 1 - alpha]) for j in axes(predictions, 2)]...)'
    minimum(unique_counts) < 2 && push!(diagnostics, "at least one replicate contains fewer than two unique groups")
    BootstrapResult(parameter_names(model), Matrix(parameters), Matrix(predictions),
        p_intervals, y_intervals, failures, unique_counts, group, Int(seed),
        Float64(confidence_level), diagnostics)
end
