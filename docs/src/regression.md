# Regression theory

## Linear-in-coefficients models

A model may be nonlinear in physical predictors while remaining linear in its
unknown coefficients:

```math
\hat y_i = \sum_{j=1}^{p} \beta_j\phi_j(\boldsymbol{x}_i).
```

Once the named basis functions ``\phi_j`` are evaluated, fitting reduces to

```math
\min_{\boldsymbol\beta}\|A\boldsymbol\beta-\boldsymbol y\|_2^2,
\qquad A_{ij}=\phi_j(\boldsymbol{x}_i).
```

`LinearBasisRegression` implements this construction. `PowerProductTerm`
represents

```math
c\prod_k x_k^{q_k},
```

where powers ``q_k`` may be positive, negative, or fractional. Stable term
identifiers, printed labels, powers, coefficients, and units are retained; no
source code must be evaluated to reconstruct such a basis.

## Polynomial basis

`PolynomialRegression(degree=d)` includes all monomials with total degree at
most ``d``. Terms are ordered first by total degree and then by descending
power of the first feature. For ``x_1,x_2`` at degree two, the order is

```text
1, x1, x2, x1^2, x1*x2, x2^2
```

Set `cross_terms=false` to retain only univariate powers.

## Numerical solution

The default solver is column-pivoted QR. Set `solver=:svd` when a
rank-deficient or severely ill-conditioned design is expected. Neither path
forms ``(A^T A)^{-1}``. Models report numerical rank, the singular-value
condition number, weighted residual objective, and warnings when configured
thresholds are exceeded.

Standardization often reduces the condition number when predictors have very
different scales. It cannot repair exact linear dependence; rank diagnostics
remain necessary.

## Weighted least squares

With explicit observation weights, Forge minimizes

```math
\sum_i w_i r_i^2.
```

Choose `weighting=:observation` to use `dataset.weights`. Choose
`weighting=:inverse_variance` to set ``w_i=1/\sigma_i^2`` from
`dataset.uncertainties`. Standard deviations themselves are never used as
weights. Full covariance weighting is not yet implemented.

## Target transformations and loss space

`FeatureScaleTarget(:M_chirp)` defines

```math
y'_i=M_{{\rm chirp},i}y_i.
```

Fitting in transformed space minimizes ``\sum(y'_i-\hat y'_i)^2``. This is
equivalent to mass-squared weighting of residuals in the original observable.
With `loss_space=:original`, Forge instead divides transformed residuals by the
row scale before solving, so the optimized objective is expressed in the
original physical target. Predictions are always returned in the original
target space.
