const _WEIGHTING_CONVENTIONS = (:none, :observation, :inverse_variance)

abstract type AbstractTargetTransformation end

"""No transformation of the dependent variable."""
struct IdentityTarget <: AbstractTargetTransformation end

"""Fit `y′ = y * feature^power`; inverse prediction divides by the same scale."""
struct FeatureScaleTarget <: AbstractTargetTransformation
    feature::Symbol
    power::Float64
end
FeatureScaleTarget(feature::Symbol; power::Real=1) = FeatureScaleTarget(feature, Float64(power))

_target_scale(::IdentityTarget, X, feature_names) = ones(eltype(X), size(X, 1))
function _target_scale(transform::FeatureScaleTarget, X, feature_names)
    column = findfirst(==(transform.feature), feature_names)
    isnothing(column) && throw(ArgumentError("target transformation feature $(transform.feature) is absent"))
    scale = X[:, column] .^ transform.power
    all(x -> isfinite(x) && !iszero(x), scale) || throw(ArgumentError(
        "target transformation produced non-finite or zero scale factors"))
    scale
end

function _check_weighting(weighting::Symbol)
    weighting in _WEIGHTING_CONVENTIONS || throw(ArgumentError(
        "weighting must be :none, :observation, or :inverse_variance"))
    weighting
end

"""Ordinary or diagonal weighted linear least squares."""
struct LinearRegression{TT<:AbstractTargetTransformation} <: AbstractRegressionMethod
    intercept::Bool
    weighting::Symbol
    standardize::Bool
    condition_warning::Float64
    target_transform::TT
    loss_space::Symbol
    solver::Symbol
end

function LinearRegression(; intercept::Bool=true, weighting::Symbol=:none,
                          standardize::Bool=false, condition_warning::Real=1e10,
                          target_transform::AbstractTargetTransformation=IdentityTarget(),
                          loss_space::Symbol=:original, solver::Symbol=:qr)
    condition_warning > 0 || throw(ArgumentError("condition_warning must be positive"))
    loss_space in (:original, :transformed) || throw(ArgumentError("loss_space must be :original or :transformed"))
    solver in (:qr, :svd) || throw(ArgumentError("solver must be :qr or :svd"))
    LinearRegression(intercept, _check_weighting(weighting), standardize,
                     Float64(condition_warning), target_transform, loss_space, solver)
end

"""Complete multivariate polynomial least squares through a chosen total degree."""
struct PolynomialRegression{TT<:AbstractTargetTransformation} <: AbstractRegressionMethod
    degree::Int
    intercept::Bool
    cross_terms::Bool
    weighting::Symbol
    standardize::Bool
    condition_warning::Float64
    target_transform::TT
    loss_space::Symbol
    solver::Symbol
end

function PolynomialRegression(; degree::Integer, intercept::Bool=true,
                              cross_terms::Bool=true, weighting::Symbol=:none,
                              standardize::Bool=false, condition_warning::Real=1e10,
                              target_transform::AbstractTargetTransformation=IdentityTarget(),
                              loss_space::Symbol=:original, solver::Symbol=:qr)
    degree >= 1 || throw(ArgumentError("degree must be at least one"))
    condition_warning > 0 || throw(ArgumentError("condition_warning must be positive"))
    loss_space in (:original, :transformed) || throw(ArgumentError("loss_space must be :original or :transformed"))
    solver in (:qr, :svd) || throw(ArgumentError("solver must be :qr or :svd"))
    PolynomialRegression(Int(degree), intercept, cross_terms, _check_weighting(weighting),
                         standardize, Float64(condition_warning), target_transform, loss_space, solver)
end

"""A fitted linear-in-coefficients polynomial model."""
struct FittedPolynomialModel{T<:AbstractFloat,M<:AbstractRegressionMethod,
                             TT<:AbstractTargetTransformation,R<:AbstractVector} <: AbstractFittedModel
    method::M
    beta::Vector{T}
    exponents::Matrix{Int}
    term_names::Vector{String}
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    standardizer::Union{Nothing,Standardizer{T}}
    transformations::Vector{TransformationRecord}
    rank::Int
    condition_number::T
    training_row_ids::R
    target_transform::TT
    loss_space::Symbol
    objective::T
end

function _compositions(total::Int, p::Int)
    p == 1 && return [[total]]
    result = Vector{Vector{Int}}()
    for first_power in total:-1:0
        for tail in _compositions(total - first_power, p - 1)
            push!(result, vcat(first_power, tail))
        end
    end
    result
end

function _basis_exponents(p::Int, degree::Int; intercept::Bool, cross_terms::Bool)
    p > 0 || throw(ArgumentError("at least one feature is required"))
    rows = Vector{Vector{Int}}()
    intercept && push!(rows, zeros(Int, p))
    for d in 1:degree
        if cross_terms
            append!(rows, _compositions(d, p))
        else
            for j in 1:p
                exponent = zeros(Int, p)
                exponent[j] = d
                push!(rows, exponent)
            end
        end
    end
    reduce(vcat, permutedims.(rows))
end

function _term_name(exponents, features)
    factors = String[]
    for (power, feature) in zip(exponents, features)
        power == 0 && continue
        push!(factors, power == 1 ? string(feature) : "$(feature)^$(power)")
    end
    isempty(factors) ? "1" : join(factors, "*")
end

function _design_matrix(X::AbstractMatrix, exponents::AbstractMatrix{<:Integer})
    n = size(X, 1)
    A = ones(promote_type(eltype(X), Float64), n, size(exponents, 1))
    for term in axes(exponents, 1), feature in axes(exponents, 2)
        power = exponents[term, feature]
        power == 0 || (A[:, term] .*= X[:, feature] .^ power)
    end
    A
end

function _fit_weights(dataset::RegressionDataset, convention::Symbol)
    convention == :none && return nothing
    if convention == :observation
        isnothing(dataset.weights) && throw(ArgumentError("observation weighting requires dataset.weights"))
        return dataset.weights
    end
    isnothing(dataset.uncertainties) && throw(ArgumentError("inverse-variance weighting requires dataset.uncertainties"))
    return inv.(dataset.uncertainties .^ 2)
end

function _fit_polynomial(dataset::RegressionDataset, method::AbstractRegressionMethod,
                         degree::Int, intercept::Bool, cross_terms::Bool,
                         weighting::Symbol, standardize::Bool, threshold::Float64)
    training = dataset
    scaler = nothing
    if standardize
        scaler = fit_standardizer(dataset)
        training = transform_features(dataset, scaler)
    end
    exponents = _basis_exponents(size(training.X, 2), degree; intercept, cross_terms)
    A = _design_matrix(training.X, exponents)
    scale = _target_scale(method.target_transform, dataset.X, dataset.feature_names)
    fitted_target = scale .* training.y
    weights = _fit_weights(training, weighting)
    if method.loss_space == :original
        loss_weights = inv.(scale .^ 2)
        weights = isnothing(weights) ? loss_weights : weights .* loss_weights
    end
    Aw, yw = if isnothing(weights)
        (A, fitted_target)
    else
        rootweights = sqrt.(weights)
        (A .* rootweights, fitted_target .* rootweights)
    end
    singular_values = svdvals(Aw)
    numerical_rank = rank(Aw)
    condition_number = isempty(singular_values) || iszero(last(singular_values)) ? Inf : first(singular_values) / last(singular_values)
    numerical_rank < size(Aw, 2) && @warn "rank-deficient design matrix" rank=numerical_rank columns=size(Aw, 2)
    condition_number > threshold && @warn "ill-conditioned design matrix" condition_number threshold
    beta = method.solver == :qr ? qr(Aw, ColumnNorm()) \ yw : svd(Aw) \ yw
    term_names = [_term_name(view(exponents, i, :), dataset.feature_names) for i in axes(exponents, 1)]
    FittedPolynomialModel(
        method, beta, exponents, term_names, copy(dataset.feature_names), dataset.target_name,
        copy(dataset.units), scaler, copy(training.transformations), numerical_rank,
        convert(eltype(beta), condition_number), copy(dataset.row_ids),
        method.target_transform, method.loss_space, sum(abs2, Aw * beta - yw),
    )
end

fit(dataset::RegressionDataset, method::LinearRegression) =
    _fit_polynomial(dataset, method, 1, method.intercept, false, method.weighting,
                    method.standardize, method.condition_warning)

fit(dataset::RegressionDataset, method::PolynomialRegression) =
    _fit_polynomial(dataset, method, method.degree, method.intercept, method.cross_terms,
                    method.weighting, method.standardize, method.condition_warning)

function _prediction_matrix(model::FittedPolynomialModel, X::AbstractMatrix)
    size(X, 2) == length(model.feature_names) || throw(DimensionMismatch(
        "new data must have $(length(model.feature_names)) columns in model feature order"))
    values = Matrix{eltype(model.beta)}(X)
    if !isnothing(model.standardizer)
        values = (values .- model.standardizer.means') ./ model.standardizer.scales'
    end
    values
end

function predict(model::FittedPolynomialModel, X::AbstractMatrix)
    transformed = _design_matrix(_prediction_matrix(model, X), model.exponents) * model.beta
    transformed ./ _target_scale(model.target_transform, X, model.feature_names)
end

function predict(model::FittedPolynomialModel, dataset::RegressionDataset)
    dataset.feature_names == model.feature_names || throw(ArgumentError("dataset feature order does not match model"))
    predict(model, dataset.X)
end

function predict(model::FittedPolynomialModel, table)
    columns = Tables.columntable(table)
    missing_names = setdiff(model.feature_names, Symbol.(propertynames(columns)))
    isempty(missing_names) || throw(ArgumentError("missing model features: $(join(missing_names, ", "))"))
    X = hcat((Float64.(getproperty(columns, name)) for name in model.feature_names)...)
    any(ismissing, X) && throw(ArgumentError("prediction features cannot contain missing values"))
    predict(model, X)
end

coefficients(model::FittedPolynomialModel) = copy(model.beta)

function formula(model::FittedPolynomialModel)
    terms = String[]
    for (coefficient, term) in zip(model.beta, model.term_names)
        label = isnothing(model.standardizer) ? term : replace(term, r"([A-Za-z_]\w*)" => s"z(\1)")
        push!(terms, "($(coefficient))*$(label)")
    end
    "$(model.target_name) = " * join(terms, " + ")
end
