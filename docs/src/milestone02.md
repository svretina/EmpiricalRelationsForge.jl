# Milestone 02 implementation report

Milestone 02 adds a common parameter/structure interface, symbolic constant refitting,
shared-fold comparison, nested symbolic validation, covariance and grouped bootstrap,
physical normalization and dimensions, safe model artifacts, conservative candidate
equivalence, a published-relation extension, and review exports.

## Numerical and dependency decisions

- Pivoted QR and SVD remain the linear solvers.
- Nonlinear residual least squares remains delegated to NonlinearSolve.jl. Symbolic
  node constants require finite-difference Jacobians; reconstructible user models may
  use compatible automatic-differentiation backends.
- Native SymbolicRegression.jl trees, evaluation, Pareto extraction, and dimensional
  constraints are reused. DynamicQuantities.jl supplies backend dimensions.
- SHA stdlib fingerprints stable structures and dataset definitions. TOML stores only
  explicitly reconstructed model forms.
- EmpiricalRelations.jl is a weak dependency loaded through a package extension.

## Scope and limitations

Uncertainty is conditional on a fixed selected structure. Covariance assumes a local
linear approximation, and grouped percentile bootstrap does not decompose all levels
of the simulation hierarchy. Rediscovery bootstrap and large nested searches remain
explicit expensive analyses rather than routine CI.

The dimensional checker is intentionally conservative because fit coefficients may
carry units. It is not a replacement for the backend's dimensional search constraints
or a general unit algebra system. Equivalence canonicalization is likewise restricted
to transformations that do not cancel singularities or broaden a formula's domain.

Portable serialization excludes arbitrary Julia closures. Published relations remain
fixed external benchmarks and require matching observable definitions, units, and
calibration domains. Candidate export is preparation for independent review, not
automatic promotion to the companion catalogue.

Fast tests exercise deterministic tree operations, refitting, shared folds, local
uncertainty, grouped bootstrap, normalization, serialization, equivalence, and export.
Short symbolic discovery and nonlinear fits are integration tests. Full BNS searches,
large bootstrap ensembles, and nested discovery campaigns should run manually or on a
scheduled workflow with recorded budgets and seeds.
