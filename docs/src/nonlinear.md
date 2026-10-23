# Nonlinear fitting

Use nonlinear least squares only when fitted parameters enter nonlinearly, for
example

```math
y=a_0+a_1x_1^{a_2}x_2^{a_3}.
```

`NonlinearModelSpec` accepts a vectorized model, named parameters, one or more
initial guesses, a stable mathematical expression, optional bounds, target and
loss-space definitions, a NonlinearSolve algorithm, and solver options.

```julia
model_function = (X, p) -> @. p[1] + p[2] * X[:, 1]^p[3] * X[:, 2]^p[4]

specification = NonlinearModelSpec(
    model_function,
    [:a0, :a1, :a2, :a3],
    [[0.5, 1.5, 1.0, -0.2], [1.0, 2.0, 0.8, -0.6]];
    expression="y = a0 + a1*x1^a2*x2^a3",
    solver_options=(abstol=1e-10, reltol=1e-10, maxiters=500),
)

model = fit(dataset, specification)
```

The default `FastShortcutNLLSPolyalg` combines fast and robust strategies from
NonlinearSolve. Researchers may pass `GaussNewton()`,
`LevenbergMarquardt()`, `TrustRegion()`, or an appropriate bounded algorithm.
Bounds are forwarded through `NonlinearLeastSquaresProblem`.

Every supplied initial guess is solved independently and the smallest final
objective is retained. The result records termination status, convergence flag,
step count, residual norm, objective, solver type and version, attempted-start
count, fitted parameters, and training row IDs. Convergence establishes a
stationary solution under the solver criteria; it does not establish global
optimality.

The callable model itself is exploratory Julia code and is not reconstructible
from TOML. A future expression specification must cover general nonlinear
models before lossless portable restoration can be claimed.
