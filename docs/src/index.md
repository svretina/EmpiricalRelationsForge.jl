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

## Scientific purpose

EmpiricalRelationsForge.jl is designed to answer four separate questions:

1. Which physical quantity is the target?
2. Which explanatory quantities are scientifically available or admissible?
3. Which mathematical structures should be fitted or searched?
4. Which validation design represents the intended use of the relation?

Keeping these questions explicit permits both forward relations, such as predicting a
postmerger frequency, and inverse relations, such as inferring a stellar radius. An
inverse relation is fitted in its own target space; it is not assumed to be an
algebraic rearrangement of the best forward equation.

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

## Recommended path through the manual

- [Getting started](@ref) constructs and validates a first physical-basis fit.
- [Capability cookbook](@ref) gives runnable examples for every major public workflow.
- [Relation discovery](@ref) explains joint feature and structure exploration.
- [Validation and metrics](@ref) separates fixed-model validation from validation of a
  complete selection procedure.
- [Parameter and prediction uncertainty](@ref) states the assumptions and limitations
  of covariance propagation and grouped resampling.
- [Vretinaris2020 reproduction benchmark](@ref) documents the real-data reproduction
  boundary and committed numerical results.

## Interpretation boundary

The package produces auditable candidate relations, not automatic scientific
conclusions. A final claim requires an analysis-specific calibration domain, repeated
stochastic searches, appropriate grouped or external validation, uncertainty from the
underlying simulations or measurements, and physical interpretation. The complete
candidate frontier and failed fits should remain part of the research record.
