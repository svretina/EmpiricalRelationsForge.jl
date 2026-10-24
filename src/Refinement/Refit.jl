"""Numerical identifiability and validity information from symbolic refitting."""
struct RefitDiagnostics
    valid::Bool
    message::String
    jacobian_rank::Int
    jacobian_condition::Float64
    unused_parameters::Vector{Symbol}
end

"""A SymbolicRegression structure whose constants were independently refitted."""
struct RefittedSymbolicModel{C,T<:AbstractFloat} <: AbstractFittedModel
    original::C
    candidate::C
    original_parameters::Vector{T}
    parameters::Vector{T}
    original_objective::T
    objective::T
    method::Symbol
    status::Symbol
    solver_name::String
    weighting::Symbol
    target_transform::AbstractTargetTransformation
    loss_space::Symbol
    lower_bounds::Union{Nothing,Vector{T}}
    upper_bounds::Union{Nothing,Vector{T}}
    diagnostics::RefitDiagnostics
    structure_fingerprint::String
    training_row_ids::Vector{Any}
end

parameter_names(model::RefittedSymbolicModel) = parameter_names(model.candidate)
parameter_vector(model::RefittedSymbolicModel) = copy(model.parameters)
coefficients(model::RefittedSymbolicModel) = parameter_vector(model)
formula(model::RefittedSymbolicModel) = formula(model.candidate)
predict(model::RefittedSymbolicModel, X::AbstractMatrix) = predict(model.candidate, X)
predict(model::RefittedSymbolicModel, data::RegressionDataset) = predict(model.candidate, data)
model_structure(model::RefittedSymbolicModel) = model_structure(model.candidate)

function with_parameters(model::RefittedSymbolicModel, values)
    candidate = with_parameters(model.candidate, values)
    RefittedSymbolicModel(model.original, candidate, copy(model.original_parameters),
        Float64.(values), model.original_objective, model.objective, model.method,
        model.status, model.solver_name, model.weighting, model.target_transform,
        model.loss_space, model.lower_bounds, model.upper_bounds, model.diagnostics,
        model.structure_fingerprint, copy(model.training_row_ids))
end

function _symbolic_predictions(candidate, parameters, X)
    configured = with_parameters(candidate, parameters)
    predict(configured, X)
end

function _symbolic_residuals(candidate, parameters, dataset, weighting, transform, loss_space)
    prediction = _symbolic_predictions(candidate, parameters, dataset.X)
    all(isfinite, prediction) || throw(DomainError(prediction, "symbolic prediction is non-finite"))
    scale = _target_scale(transform, dataset.X, dataset.feature_names)
    residual = loss_space == :original ? prediction .- dataset.y : scale .* (prediction .- dataset.y)
    weights = _fit_weights(dataset, weighting)
    isnothing(weights) ? residual : sqrt.(weights) .* residual
end

function _symbolic_parameter_degree(node, options)
    node.degree == 0 && return node.constant ? 1 : 0
    degrees = [_symbolic_parameter_degree(child, options) for child in _node_children(node)]
    any(<(0), degrees) && return -1
    op = options.operators.ops[node.degree][node.op]
    if node.degree == 2 && (op === (+) || op === (-))
        return max(degrees...)
    elseif node.degree == 2 && op === (*)
        total = sum(degrees)
        return total <= 1 ? total : -1
    elseif node.degree == 2 && op === (/)
        return degrees[2] == 0 ? degrees[1] : -1
    end
    return all(iszero, degrees) ? 0 : -1
end

_linear_in_symbolic_parameters(candidate) =
    _symbolic_parameter_degree(_raw_tree(candidate.expression), candidate.options) in (0, 1)

function _finite_jacobian(candidate, parameters, dataset)
    base = _symbolic_predictions(candidate, parameters, dataset.X)
    J = Matrix{Float64}(undef, length(base), length(parameters))
    for j in eachindex(parameters)
        step = sqrt(eps(Float64)) * max(abs(parameters[j]), 1.0)
        shifted = copy(parameters)
        shifted[j] += step
        J[:, j] = (_symbolic_predictions(candidate, shifted, dataset.X) .- base) ./ step
    end
    J
end

function _refit_diagnostics(candidate, parameters, dataset; message="ok")
    try
        J = _finite_jacobian(candidate, parameters, dataset)
        singular = svdvals(J)
        tolerance = isempty(singular) ? 0.0 : sqrt(eps(Float64)) * max(size(J)...) * max(first(singular), 1.0)
        r = count(>(tolerance), singular)
        condition = isempty(singular) || iszero(last(singular)) ? Inf : first(singular) / last(singular)
        unused = parameter_names(candidate)[findall(j -> all(abs.(view(J, :, j)) .<= sqrt(eps(Float64))), axes(J, 2))]
        RefitDiagnostics(all(isfinite, J), String(message), r, condition, unused)
    catch error
        RefitDiagnostics(false, sprint(showerror, error), 0, Inf, Symbol[])
    end
end

"""
    refit(candidate, dataset; method=:auto, ...)

Refit the scalar constants of a symbolic candidate without changing its operator tree.
`method=:auto` uses QR/SVD only when the tree is provably affine in every constant;
otherwise it delegates residual least squares to NonlinearSolve.jl.
"""
function refit(candidate::SymbolicCandidate, dataset::RegressionDataset;
               method::Symbol=:auto, solver::Symbol=:qr, weighting::Symbol=:none,
               target_transform::AbstractTargetTransformation=IdentityTarget(),
               loss_space::Symbol=:original, lower_bounds=nothing, upper_bounds=nothing,
               initial_guesses=nothing, parameter_scale=nothing,
               algorithm=nothing,
               solver_options::NamedTuple=(;))
    dataset.feature_names == candidate.feature_names || throw(ArgumentError("dataset feature order does not match candidate"))
    method in (:auto, :linear, :nonlinear) || throw(ArgumentError("method must be :auto, :linear, or :nonlinear"))
    solver in (:qr, :svd) || throw(ArgumentError("solver must be :qr or :svd"))
    loss_space in (:original, :transformed) || throw(ArgumentError("loss_space must be :original or :transformed"))
    _check_weighting(weighting)
    original = parameter_vector(candidate)
    isempty(original) && throw(ArgumentError("symbolic candidate has no free scalar constants"))
    linear = _linear_in_symbolic_parameters(candidate)
    method == :linear && !linear && throw(ArgumentError("candidate is not linear in its scalar constants"))
    selected = method == :auto ? (linear && isnothing(lower_bounds) ? :linear : :nonlinear) : method
    original_objective = try
        sum(abs2, _symbolic_residuals(candidate, original, dataset, weighting, target_transform, loss_space))
    catch
        Inf
    end
    fingerprint = model_structure(candidate).fingerprint

    if selected == :linear
        zero_parameters = zeros(length(original))
        baseline = _symbolic_predictions(candidate, zero_parameters, dataset.X)
        A = hcat([_symbolic_predictions(candidate, [i == j ? 1.0 : 0.0 for i in eachindex(original)], dataset.X) .- baseline
                  for j in eachindex(original)]...)
        scale = _target_scale(target_transform, dataset.X, dataset.feature_names)
        yfit = scale .* (dataset.y .- baseline)
        Afit = scale .* A
        if loss_space == :original
            Afit ./= scale
            yfit ./= scale
        end
        weights = _fit_weights(dataset, weighting)
        if !isnothing(weights)
            rootweights = sqrt.(weights)
            Afit .*= rootweights
            yfit .*= rootweights
        end
        parameters = solver == :qr ? qr(Afit, ColumnNorm()) \ yfit : svd(Afit) \ yfit
        candidate2 = with_parameters(candidate, parameters)
        objective = sum(abs2, _symbolic_residuals(candidate2, parameters, dataset, weighting, target_transform, loss_space))
        diagnostics = _refit_diagnostics(candidate2, parameters, dataset)
        return RefittedSymbolicModel(candidate, candidate2, original, collect(parameters),
            original_objective, objective, :linear, :success, string(solver), weighting,
            target_transform, loss_space, nothing, nothing, diagnostics, fingerprint,
            copy(dataset.row_ids))
    end

    guesses = isnothing(initial_guesses) ? original : initial_guesses
    model_function = (X, p) -> begin
        prediction = _symbolic_predictions(candidate, p, X)
        _target_scale(target_transform, X, candidate.feature_names) .* prediction
    end
    selected_algorithm = if isnothing(algorithm)
        isnothing(lower_bounds) ?
            NonlinearSolve.TrustRegion(autodiff=NonlinearSolve.AutoFiniteDiff()) :
            NonlinearSolve.TrustRegionReflective(autodiff=NonlinearSolve.AutoFiniteDiff())
    else
        algorithm
    end
    specification = NonlinearModelSpec(model_function, parameter_names(candidate), guesses;
        expression=formula(candidate), target_transform, loss_space, weighting,
        lower_bounds, upper_bounds, parameter_scale, algorithm=selected_algorithm, solver_options)
    try
        fitted = fit(dataset, specification)
        candidate2 = with_parameters(candidate, fitted.parameters)
        objective = sum(abs2, _symbolic_residuals(candidate2, fitted.parameters, dataset,
                                                   weighting, target_transform, loss_space))
        diagnostics = _refit_diagnostics(candidate2, fitted.parameters, dataset;
                                          message=fitted.termination_status)
        status = fitted.converged ? :success : :failed
        RefittedSymbolicModel(candidate, candidate2, original, copy(fitted.parameters),
            original_objective, objective, :nonlinear, status, fitted.solver_name, weighting,
            target_transform, loss_space,
            isnothing(lower_bounds) ? nothing : Float64.(lower_bounds),
            isnothing(upper_bounds) ? nothing : Float64.(upper_bounds), diagnostics,
            fingerprint, copy(dataset.row_ids))
    catch error
        diagnostics = RefitDiagnostics(false, sprint(showerror, error), 0, Inf, Symbol[])
        RefittedSymbolicModel(candidate, candidate, original, copy(original), original_objective,
            Inf, :nonlinear, :failed, string(typeof(selected_algorithm)), weighting, target_transform,
            loss_space, isnothing(lower_bounds) ? nothing : Float64.(lower_bounds),
            isnothing(upper_bounds) ? nothing : Float64.(upper_bounds), diagnostics,
            fingerprint, copy(dataset.row_ids))
    end
end
