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

The TOML export is an audit record, not a universal executable serialization.
Power-product bases are reconstructible from their structured specification,
but arbitrary nonlinear Julia callables and native symbolic backend trees need
a versioned expression/checkpoint format for lossless restoration. This remains
an explicit limitation.
