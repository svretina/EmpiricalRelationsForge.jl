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
struct RegressionDataset{T<:AbstractFloat}
    X::Matrix{T}
    y::Vector{T}
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    row_ids::Vector{Any}
    eos_ids::Vector{Any}
    simulation_ids::Vector{Any}
    source_papers::Vector{Any}
    uncertainties::Union{Nothing,Vector{T}}
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
    uncertainties=nothing,
    weights=nothing,
    transformations=TransformationRecord[],
)
    T = promote_type(float(eltype(X)), float(eltype(y)))
    dataset = RegressionDataset{T}(
        Matrix{T}(X), Vector{T}(y), Symbol.(features), Symbol(target),
        Dict{Symbol,String}(Symbol(k) => string(v) for (k, v) in pairs(units)),
        Any[row_ids...], Any[eos_ids...], Any[simulation_ids...],
        Any[source_papers...],
        isnothing(uncertainties) ? nothing : Vector{T}(uncertainties),
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
        uncertainties=isnothing(dataset.uncertainties) ? nothing : dataset.uncertainties[indices],
        weights=isnothing(dataset.weights) ? nothing : dataset.weights[indices],
        transformations=dataset.transformations,
    )
end
