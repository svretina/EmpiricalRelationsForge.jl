# Scientific data

## Orientation and provenance

`RegressionDataset.X` is always observations by features. The order of its
columns is exactly `feature_names`. SymbolicRegression.jl uses the opposite
orientation internally; `discover` and symbolic `predict` perform that
transpose at the adapter boundary.

Each row retains an original row identifier, EOS identifier, simulation
identifier, source-paper identifier, and observation class. Allowed classes are
`:measured`, `:predicted`, and `:unreviewed`. A table constructor supplied with
an `observation_kind` column includes only `:measured` rows by default:

```julia
dataset = RegressionDataset(
    table;
    target=:f_peak,
    features=[:M_chirp, :Lambda_tilde],
    observation_kind=:kind,
)
```

Use `include_kinds` explicitly to change this policy. Model predictions are
therefore not silently promoted to ground truth.

Missing feature and target values are dropped by default. Set
`dropmissing=false` to require complete input and raise an error otherwise.
Row identifiers must be unique. `duplicate_identifiers(dataset;
by=:simulation)` reports repeated simulation identifiers without deleting
scientifically meaningful resolution variants.

## Uncertainty and weights

`uncertainties` are positive standard deviations. `uncertainty_kind` records
their interpretation, such as `:measurement` or `:numerical`. `weights` are
separate positive observation-importance values. The fitting method determines
which quantity enters its loss; the two are never interchanged implicitly.

## Physical dependencies

`physical_dependency_warnings` flags known deterministic BNS combinations,
including component masses with chirp mass and component tidal deformabilities
with effective tidal deformability. It returns warnings only. Variables are not
removed because redundant physical coordinates may still be useful in a
controlled model comparison.

## Standardization

For feature column ``x_j``, standardization uses

```math
z_j = \frac{x_j-\mu_j}{s_j},
```

where ``\mu_j`` and ``s_j`` are learned by `fit_standardizer`. A zero-scale
feature is rejected. During validation these statistics are fitted on the
training fold, which prevents information from the held-out fold entering the
model.
