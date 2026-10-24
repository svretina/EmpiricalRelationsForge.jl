# Parameter and prediction uncertainty

`parameter_covariance(model, dataset)` estimates covariance conditional on the
chosen mathematical structure. It forms a numerical prediction Jacobian and uses
an SVD, avoiding an explicit inverse of the normal matrix. Rank-deficient problems
return a structured `:singular` result rather than spurious standard errors.

```julia
covariance = parameter_covariance(model, dataset)
propagated = prediction_uncertainty(model, covariance, new_X)
```

The recorded assumptions identify independent residuals, a fixed structure, and
a local linear approximation. With inverse-variance weighting, supplied
observation variances are treated as known. Otherwise a common variance is
estimated from residual degrees of freedom. The propagated values quantify only
parameter-induced prediction uncertainty; they are not prediction intervals for
new physical systems and do not include model discrepancy.

`bootstrap(model, dataset; group=:eos, n=1000, seed=0)` samples complete groups
with replacement, preserving repeated groups and all of their observations. It
refits the fixed structure and returns coefficient samples, prediction samples,
percentile intervals, failure diagnostics, and the effective number of unique
groups in each replicate. This estimates fixed-structure stability. Repeating a
full symbolic search is intentionally left to an explicit, computationally
budgeted experiment.
