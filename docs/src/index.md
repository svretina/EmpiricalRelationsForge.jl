# EmpiricalRelationsForge.jl

```@raw html
<p style="text-align: center;">
  <img src="assets/images/logo.png" alt="EmpiricalRelationsForge.jl — binary neutron-star empirical-relation fitting and discovery logo" style="width: min(100%, 720px); height: auto;">
</p>
```

This package provides metadata-preserving data preparation, regression,
equation discovery, and EOS-aware validation for scientific empirical
relations. Its first application is binary-neutron-star postmerger research,
but the numerical layer is domain independent.

`RelationDiscoveryProblem` provides the top-level workflow for searching over
explanatory-variable subsets and mathematical structures. The target is arbitrary:
forward postmerger-frequency relations and inverse stellar-property relations are fitted
and validated as independent statistical problems.

The production dependency stack is deliberately narrow:

- `LinearAlgebra` supplies QR and SVD for models linear in their coefficients.
- `NonlinearSolve.jl` supplies nonlinear least-squares algorithms.
- `SymbolicRegression.jl` supplies equation discovery and Pareto frontiers.
- `Tables.jl` and `DataFrames.jl` supply tabular interoperability.

`EmpiricalRelations.jl` remains the catalogue of published relations. Forge
models are research candidates and are never inserted into that catalogue
automatically.
