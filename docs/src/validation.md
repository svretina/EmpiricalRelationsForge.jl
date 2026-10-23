# Validation and metrics

## Residual convention

Residuals are prediction minus observation,

```math
r_i=\hat y_i-y_i.
```

`metrics` reports

```math
\operatorname{MAE}=\frac1n\sum_i|r_i|,
\quad
\operatorname{RMSE}=\sqrt{\frac1n\sum_i r_i^2},
```

maximum absolute error, mean signed residual, and

```math
R^2=1-\frac{\sum_i r_i^2}{\sum_i(y_i-\bar y)^2}.
```

``R^2`` is `missing` for a constant observed target. Mean absolute relative
error uses ``|r_i/y_i|`` and is `missing` if any denominator is zero. Absolute
metrics inherit the target unit; for a kHz target they are in kHz. Multiply
such values by 1000 for presentation in Hz.

## Strategies

- `RandomHoldout` provides a reproducible random train/test split.
- `KFold` provides reproducibly shuffled observation-level folds.
- `GroupedKFold` partitions EOS, simulation, or source-paper identifiers.
- `LeaveOneEOSOut` holds out each EOS in turn.

Grouped strategies reject missing group identifiers and ensure that no group
crosses a train/test boundary. A `ValidationReport` stores exact fold indices,
fold metrics, aggregate out-of-fold metrics, predictions, observations, row
IDs, and per-EOS residual summaries.

For quasi-universal BNS relations, leave-one-EOS-out validation is preferred
over random splitting because nearby configurations from one EOS are strongly
correlated. Resolution variants should additionally be grouped by simulation
family when that identifier is available.
