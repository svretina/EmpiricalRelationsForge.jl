"""
A serializable monomial-like basis term `coefficient * ∏ xᵢ^powerᵢ`.

Powers may be integer, negative, or fractional. `id` and `label` provide a
stable machine identifier and mathematical representation.
"""
struct PowerProductTerm
    id::Symbol
    label::String
    powers::Dict{Symbol,Float64}
    coefficient::Float64
    unit::String
end

function PowerProductTerm(id::Symbol, powers::AbstractDict{Symbol,<:Real};
                          label::AbstractString=string(id), coefficient::Real=1,
                          unit::AbstractString="")
    PowerProductTerm(id, String(label), Dict(k => Float64(v) for (k, v) in powers),
                     Float64(coefficient), String(unit))
end

PowerProductTerm(id::Symbol; label::AbstractString="1", coefficient::Real=1,
                 unit::AbstractString="") =
    PowerProductTerm(id, Dict{Symbol,Float64}(); label, coefficient, unit)

struct BasisSet
    terms::Vector{PowerProductTerm}
    function BasisSet(terms::AbstractVector{PowerProductTerm})
        isempty(terms) && throw(ArgumentError("basis must contain at least one term"))
        ids = getfield.(terms, :id)
        allunique(ids) || throw(ArgumentError("basis term identifiers must be unique"))
        new(collect(terms))
    end
end
BasisSet(terms::PowerProductTerm...) = BasisSet(collect(terms))

struct LinearBasisRegression <: AbstractRegressionMethod
    basis::BasisSet
    weighting::Symbol
    standardize::Bool
    condition_warning::Float64
    target_transform::AbstractTargetTransformation
    loss_space::Symbol
    solver::Symbol
end

function LinearBasisRegression(basis::BasisSet; weighting::Symbol=:none,
                               standardize::Bool=false, condition_warning::Real=1e10,
                               target_transform::AbstractTargetTransformation=IdentityTarget(),
                               loss_space::Symbol=:original, solver::Symbol=:qr)
    _check_weighting(weighting)
    loss_space in (:original, :transformed) || throw(ArgumentError("loss_space must be :original or :transformed"))
    solver in (:qr, :svd) || throw(ArgumentError("solver must be :qr or :svd"))
    LinearBasisRegression(basis, weighting, standardize, Float64(condition_warning),
                          target_transform, loss_space, solver)
end

struct FittedBasisModel{T<:AbstractFloat} <: AbstractFittedModel
    method::LinearBasisRegression
    beta::Vector{T}
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    standardizer::Union{Nothing,Standardizer{T}}
    rank::Int
    condition_number::T
    objective::T
    training_row_ids::Vector{Any}
end

function _basis_matrix(basis::BasisSet, X::AbstractMatrix, feature_names)
    lookup = Dict(name => index for (index, name) in enumerate(feature_names))
    A = ones(promote_type(eltype(X), Float64), size(X, 1), length(basis.terms))
    for (column, term) in enumerate(basis.terms)
        A[:, column] .*= term.coefficient
        for (feature, power) in term.powers
            index = get(lookup, feature, nothing)
            isnothing(index) && throw(ArgumentError("basis term $(term.id) requires absent feature :$feature"))
            values = X[:, index] .^ power
            all(isfinite, values) || throw(DomainError(values, "basis term $(term.id) is not finite"))
            A[:, column] .*= values
        end
    end
    A
end

function fit(dataset::RegressionDataset, method::LinearBasisRegression)
    scaler = method.standardize ? fit_standardizer(dataset) : nothing
    Xfit = isnothing(scaler) ? dataset.X : transform_features(dataset, scaler).X
    A = _basis_matrix(method.basis, Xfit, dataset.feature_names)
    scale = _target_scale(method.target_transform, dataset.X, dataset.feature_names)
    yfit = scale .* dataset.y
    weights = _fit_weights(dataset, method.weighting)
    if method.loss_space == :original
        loss_weights = inv.(scale .^ 2)
        weights = isnothing(weights) ? loss_weights : weights .* loss_weights
    end
    Aw, yw = isnothing(weights) ? (A, yfit) :
             (A .* sqrt.(weights), yfit .* sqrt.(weights))
    singular_values = svdvals(Aw)
    numerical_rank = rank(Aw)
    condition_number = iszero(last(singular_values)) ? Inf : first(singular_values) / last(singular_values)
    numerical_rank < size(Aw, 2) && @warn "rank-deficient design matrix" rank=numerical_rank columns=size(Aw, 2)
    condition_number > method.condition_warning && @warn "ill-conditioned design matrix" condition_number threshold=method.condition_warning
    beta = method.solver == :qr ? qr(Aw, ColumnNorm()) \ yw : svd(Aw) \ yw
    FittedBasisModel(method, beta, copy(dataset.feature_names), dataset.target_name,
                     copy(dataset.units), scaler, numerical_rank,
                     convert(eltype(beta), condition_number), sum(abs2, Aw * beta - yw),
                     copy(dataset.row_ids))
end

function predict(model::FittedBasisModel, X::AbstractMatrix)
    size(X, 2) == length(model.feature_names) || throw(DimensionMismatch("new data feature count differs"))
    Xfit = isnothing(model.standardizer) ? X :
           (X .- model.standardizer.means') ./ model.standardizer.scales'
    transformed = _basis_matrix(model.method.basis, Xfit, model.feature_names) * model.beta
    transformed ./ _target_scale(model.method.target_transform, X, model.feature_names)
end
predict(model::FittedBasisModel, dataset::RegressionDataset) = begin
    dataset.feature_names == model.feature_names || throw(ArgumentError("dataset feature order does not match model"))
    predict(model, dataset.X)
end
coefficients(model::FittedBasisModel) = copy(model.beta)
formula(model::FittedBasisModel) = "$(model.target_name) = " * join(
    ["($(coefficient))*$(term.label)" for (coefficient, term) in zip(model.beta, model.method.basis.terms)], " + ")
