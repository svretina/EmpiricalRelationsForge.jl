"""Feature-wise standardization learned from training data."""
struct Standardizer{T<:AbstractFloat}
    feature_names::Vector{Symbol}
    means::Vector{T}
    scales::Vector{T}
end

function fit_standardizer(dataset::RegressionDataset)
    means = vec(mean(dataset.X; dims=1))
    scales = vec(std(dataset.X; dims=1, corrected=false))
    any(iszero, scales) && throw(ArgumentError("cannot standardize a constant feature"))
    Standardizer(copy(dataset.feature_names), means, scales)
end

function transform_features(dataset::RegressionDataset, transform::Standardizer)
    dataset.feature_names == transform.feature_names || throw(ArgumentError("feature order does not match standardizer"))
    X = (dataset.X .- transform.means') ./ transform.scales'
    record = TransformationRecord(:standardize, copy(dataset.feature_names), Dict(
        :means => copy(transform.means), :scales => copy(transform.scales)))
    RegressionDataset(
        X, dataset.y; features=dataset.feature_names, target=dataset.target_name,
        units=dataset.units, row_ids=dataset.row_ids, eos_ids=dataset.eos_ids,
        simulation_ids=dataset.simulation_ids, source_papers=dataset.source_papers,
        observation_kinds=dataset.observation_kinds,
        uncertainties=dataset.uncertainties, uncertainty_kind=dataset.uncertainty_kind,
        weights=dataset.weights,
        transformations=vcat(dataset.transformations, [record]),
    )
end
