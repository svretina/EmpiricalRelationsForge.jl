# EmpiricalRelationsForge.jl

This package provides metadata-preserving data preparation, regression,
equation discovery, and EOS-aware validation for scientific empirical
relations. Its first application is binary-neutron-star postmerger research,
but the numerical layer is domain independent.

The production dependency stack is deliberately narrow:

- `LinearAlgebra` supplies QR and SVD for models linear in their coefficients.
- `NonlinearSolve.jl` supplies nonlinear least-squares algorithms.
- `SymbolicRegression.jl` supplies equation discovery and Pareto frontiers.
- `Tables.jl` and `DataFrames.jl` supply tabular interoperability.

`EmpiricalRelations.jl` remains the catalogue of published relations. Forge
models are research candidates and are never inserted into that catalogue
automatically.
