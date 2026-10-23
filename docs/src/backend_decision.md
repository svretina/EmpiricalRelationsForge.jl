# Regression backend decision

## Decision

Forge uses `LinearAlgebra` for linear-in-parameter fits, `NonlinearSolve.jl` for
nonlinear least squares, and `SymbolicRegression.jl` for expression discovery.
The public scientific specification is independent of these solvers.

## Evaluated packages

The evaluation was run with Julia 1.13.0-rc1 on 2026-10-23. Timings below are
single warmed measurements for an 80-observation synthetic nonlinear power-law
problem; they characterize this check, not a general benchmark.

| Package | Version checked | Finding |
| --- | ---: | --- |
| LinearAlgebra | 1.13 | Pivoted QR and SVD cover linear-in-coefficient models without normal equations. |
| NonlinearSolve | 4.32.0 | Adopted. Documented least-squares problem, polyalgorithm, LM, GN, trust-region, bounds, AD, diagnostics. |
| CurveFit | 0.6.1 | Rejected as a dependency. The registered compatible package exposed an older custom `nonlinear_fit` API, not the documented SciML problem interface. |
| LsqFit | 0.16.1 | Numerically accurate and lightweight, but centered on LM and less suitable for the requested replaceable algorithm layer. |
| GLM | 1.9.5 | Strong statistical linear-model interface; unnecessary for basis construction already expressed directly as a design matrix. |
| SymbolicRegression | 2.7.2 | Adopted for discovery; native expressions and Pareto members are retained. |

The documentation currently published for CurveFit describes
`NonlinearCurveFitProblem` backed by NonlinearSolve. That interface was absent
from the registered v0.6.1 resolved in this environment. Depending directly on
NonlinearSolve avoids coupling Forge to this version discrepancy.

## Representative nonlinear comparison

For exact data from

```math
y=0.8+1.7x_1^{1.2}x_2^{-0.4},
```

all tested nonlinear solvers recovered the generating parameters from the same
initial guess.

| Backend | Residual norm | Warm time | Allocated bytes |
| --- | ---: | ---: | ---: |
| CurveFit 0.6.1 `nonlinear_fit` | ``7.47\times10^{-10}`` | 0.241 ms | 277,656 |
| NonlinearSolve 4.32 LM | ``9.89\times10^{-14}`` | 1.383 ms | 273,112 |
| LsqFit 0.16.1 | ``3.11\times10^{-15}`` | 4.083 ms | 1,803,576 |

The small sample makes speed differences immaterial. NonlinearSolve was chosen
for algorithm coverage, bounds, diagnostics, and backend flexibility. The old
CurveFit path was fast on this easy case but implements its own numerical
iteration and does not satisfy the architectural requirement.

## Failure modes

- Linear bases can be rank deficient or ill-conditioned. Forge reports rank
  and condition number and provides SVD as an explicit alternative.
- Fractional and negative powers have restricted real domains. Non-finite basis
  values are rejected.
- Nonlinear fits depend on initialization and may converge to local stationary
  points. Multiple starts are supported and all failures remain visible.
- Symbolic searches are stochastic. Tests require valid candidates rather than
  one exact recovered expression.

Official references: [CurveFit](https://docs.sciml.ai/CurveFit/stable/),
[NonlinearSolve least squares](https://docs.sciml.ai/NonlinearSolve/stable/solvers/nonlinear_least_squares_solvers/),
[LsqFit](https://julianlsolvers.github.io/LsqFit.jl/latest/), and
[GLM](https://juliastats.org/GLM.jl/stable/).
