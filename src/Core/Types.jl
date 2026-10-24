"""A recorded operation applied to scientific data."""
struct TransformationRecord
    name::Symbol
    variables::Vector{Symbol}
    parameters::Dict{Symbol,Any}
end

"""
    RegressionDataset

An observations-by-features numerical dataset with scientific provenance. Matrix
orientation is deliberately fixed: rows are observations and columns follow
`feature_names`.
"""
struct RegressionDataset{T<:AbstractFloat,
                         R<:AbstractVector,E<:AbstractVector,
                         S<:AbstractVector,P<:AbstractVector}
    X::Matrix{T}
    y::Vector{T}
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    row_ids::R
    eos_ids::E
    simulation_ids::S
    source_papers::P
    observation_kinds::Vector{Symbol}
    uncertainties::Union{Nothing,Vector{T}}
    uncertainty_kind::Symbol
    weights::Union{Nothing,Vector{T}}
    transformations::Vector{TransformationRecord}
end

Base.length(dataset::RegressionDataset) = length(dataset.y)
Base.size(dataset::RegressionDataset) = size(dataset.X)

function RegressionDataset(
    X::AbstractMatrix,
    y::AbstractVector;
    features,
    target,
    units=Dict{Symbol,String}(),
    row_ids=collect(axes(X, 1)),
    eos_ids=fill(missing, size(X, 1)),
    simulation_ids=fill(missing, size(X, 1)),
    source_papers=fill(missing, size(X, 1)),
    observation_kinds=fill(:measured, size(X, 1)),
    uncertainties=nothing,
    uncertainty_kind::Symbol=:unspecified,
    weights=nothing,
    transformations=TransformationRecord[],
)
    T = promote_type(float(eltype(X)), float(eltype(y)))
    dataset = RegressionDataset(
        Matrix{T}(X), Vector{T}(y), Symbol.(features), Symbol(target),
        Dict{Symbol,String}(Symbol(k) => string(v) for (k, v) in pairs(units)),
        collect(row_ids), collect(eos_ids), collect(simulation_ids),
        collect(source_papers), Symbol[observation_kinds...],
        isnothing(uncertainties) ? nothing : Vector{T}(uncertainties),
        uncertainty_kind,
        isnothing(weights) ? nothing : Vector{T}(weights),
        collect(transformations),
    )
    validate_dataset(dataset)
    return dataset
end

"""Return an index-selected dataset while preserving all metadata."""
function subset(dataset::RegressionDataset, indices::AbstractVector{<:Integer})
    RegressionDataset(
        dataset.X[indices, :], dataset.y[indices];
        features=dataset.feature_names,
        target=dataset.target_name,
        units=dataset.units,
        row_ids=dataset.row_ids[indices],
        eos_ids=dataset.eos_ids[indices],
        simulation_ids=dataset.simulation_ids[indices],
        source_papers=dataset.source_papers[indices],
        observation_kinds=dataset.observation_kinds[indices],
        uncertainties=isnothing(dataset.uncertainties) ? nothing : dataset.uncertainties[indices],
        uncertainty_kind=dataset.uncertainty_kind,
        weights=isnothing(dataset.weights) ? nothing : dataset.weights[indices],
        transformations=dataset.transformations,
    )
end
