"""Validate dimensions, identifiers, and numerical metadata of a dataset."""
function validate_dataset(dataset::RegressionDataset)
    n, p = size(dataset.X)
    p == length(dataset.feature_names) || throw(DimensionMismatch("feature names must match the columns of X"))
    length(unique(dataset.feature_names)) == p || throw(ArgumentError("feature names must be unique"))
    dataset.target_name in dataset.feature_names && throw(ArgumentError("target cannot also be a feature"))
    for (name, values) in (
        (:target, dataset.y), (:row_ids, dataset.row_ids), (:eos_ids, dataset.eos_ids),
        (:simulation_ids, dataset.simulation_ids), (:source_papers, dataset.source_papers),
        (:observation_kinds, dataset.observation_kinds),
    )
        length(values) == n || throw(DimensionMismatch("$name must have one value per observation"))
    end
    length(unique(dataset.row_ids)) == n || throw(ArgumentError("row identifiers must be unique"))
    allowed_kinds = (:measured, :predicted, :unreviewed)
    all(in(allowed_kinds), dataset.observation_kinds) || throw(ArgumentError(
        "observation kinds must be :measured, :predicted, or :unreviewed"))
    all(isfinite, dataset.X) || throw(ArgumentError("features must be finite"))
    all(isfinite, dataset.y) || throw(ArgumentError("target values must be finite"))
    for (name, values) in ((:uncertainties, dataset.uncertainties), (:weights, dataset.weights))
        isnothing(values) && continue
        length(values) == n || throw(DimensionMismatch("$name must have one value per observation"))
        all(x -> isfinite(x) && x > 0, values) || throw(ArgumentError("$name must be finite and positive"))
    end
    return dataset
end

"""Return repeated non-missing identifiers and their row indices."""
function duplicate_identifiers(dataset::RegressionDataset; by::Symbol=:simulation)
    values = by == :row ? dataset.row_ids :
             by == :simulation ? dataset.simulation_ids :
             by == :eos ? dataset.eos_ids :
             throw(ArgumentError("by must be :row, :simulation, or :eos"))
    Dict(value => findall(==(value), values) for value in unique(skipmissing(values))
         if count(==(value), values) > 1)
end

"""Flag known deterministic BNS feature combinations without removing variables."""
function physical_dependency_warnings(features::AbstractVector{Symbol})
    present = Set(features)
    warnings = String[]
    all(in(present), (:mass_1, :mass_2, :M_chirp)) && push!(warnings,
        "M_chirp is determined by mass_1 and mass_2")
    all(in(present), (:mass_ratio, :M_chirp, :total_mass)) && push!(warnings,
        "mass_ratio and M_chirp determine total_mass")
    all(in(present), (:mass_1, :mass_2, :Lambda_1, :Lambda_2, :Lambda_tilde)) && push!(warnings,
        "Lambda_tilde is determined by component masses and tidal deformabilities")
    warnings
end
