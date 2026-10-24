# Reproducibility and result records

Fitted models retain feature order, target name, units, transformations,
training row IDs, fitted parameters, method configuration, objective, rank or
solver diagnostics, and relevant backend versions. Validation reports retain
the exact row indices assigned to every fold.

`ModelResult` combines a fitted model with in-sample metrics, an optional
validation report, provenance, and reproducibility metadata. `save_metadata`
writes portable TOML containing the displayed formula, coefficients, metrics,
provenance, and Julia version.

```julia
validation = validate(method, dataset; strategy=LeaveOneEOSOut())
record = model_result(
    fit(dataset, method), dataset;
    validation,
    provenance=Dict(:analysis => "postmerger-frequency-baseline"),
    reproducibility=Dict(:seed => 42),
)
save_metadata("candidate.toml", record)
```

`save_metadata` is an audit record rather than executable model serialization. Use
`save_model` and `load_model` for reconstructible polynomial, named-basis, normalized,
and refitted symbolic models. Symbolic operator trees are stored explicitly and are
restored without evaluating saved Julia source or parsing display formulas.

```julia
save_model("model.toml", model;
    metadata=Dict(:dataset_hash => dataset_fingerprint(dataset)))
restored = load_model("model.toml")
```

Use `save_discovery` and `load_discovery` for a complete portable campaign whose
candidate models are all reconstructible. Arbitrary nonlinear Julia closures and
published predictor closures remain non-portable. Campaign artifacts store row
identities and a dataset hash, not a private copy of the raw scientific database, so
the immutable source dataset must be archived separately.

For every reported result, record the complete search grammar, computational budget,
random seeds, validation groups, retained frontier, unsuccessful fits, software
revision, and backend versions. Selecting and reporting only the best stochastic run
does not constitute a reproducible discovery analysis.
