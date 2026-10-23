"""Validate dimensions, identifiers, and numerical metadata of a dataset."""
function validate_dataset(dataset::RegressionDataset)
    n, p = size(dataset.X)
    p == length(dataset.feature_names) || throw(DimensionMismatch("feature names must match the columns of X"))
    length(unique(dataset.feature_names)) == p || throw(ArgumentError("feature names must be unique"))
    dataset.target_name in dataset.feature_names && throw(ArgumentError("target cannot also be a feature"))
    for (name, values) in (
        (:target, dataset.y), (:row_ids, dataset.row_ids), (:eos_ids, dataset.eos_ids),
        (:simulation_ids, dataset.simulation_ids), (:source_papers, dataset.source_papers),
    )
        length(values) == n || throw(DimensionMismatch("$name must have one value per observation"))
    end
    length(unique(dataset.row_ids)) == n || throw(ArgumentError("row identifiers must be unique"))
    all(isfinite, dataset.X) || throw(ArgumentError("features must be finite"))
    all(isfinite, dataset.y) || throw(ArgumentError("target values must be finite"))
    for (name, values) in ((:uncertainties, dataset.uncertainties), (:weights, dataset.weights))
        isnothing(values) && continue
        length(values) == n || throw(DimensionMismatch("$name must have one value per observation"))
        all(x -> isfinite(x) && x > 0, values) || throw(ArgumentError("$name must be finite and positive"))
    end
    return dataset
end
