# Unified comparison and nested validation

`compare` evaluates trainable methods, fixed symbolic structures, symbolic search
specifications, and fixed predictors on the same stored `DataSplit` objects.

```julia
report = compare(dataset, [
    :linear => LinearRegression(),
    :published => published_model,
    :candidate => candidate,
]; validation=LeaveOneEOSOut())

selected = select_model(report; policy=:minimum_rmse)
```

Available policies are minimum aggregate RMSE, minimum worst-group RMSE, and the
lowest-complexity model below a supplied error threshold. The policy is explicit;
the `policy` keyword may also be a function receiving the successful comparison
entries. The package does not construct an undocumented composite score. A
`PublishedPredictor` is marked as a fixed predictor and is never presented as a
newly trained model.

Fixed-structure validation asks how well a previously chosen formula performs
when its constants are fitted on each training fold. It does not measure the
generalization error of the discovery procedure that selected the structure.
For that question use `nested_validate`. Every outer training partition performs
discovery, inner grouped comparison and selection, then a final refit before the
outer holdout is evaluated. Outer targets are unavailable to inner selection.

Fold membership, fold metrics, predictions, row identities, and group-resolved
metrics remain in the report. Fold errors are not treated as independent samples
for formal confidence intervals.
