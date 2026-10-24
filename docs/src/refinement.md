# Symbolic-structure refitting

A discovered expression contains a mathematical tree and provisional scalar
constants. `refit` optimizes only those constants; it does not mutate the input
candidate or change its operators, variables, or topology.

```julia
refined = refit(candidate, dataset; method=:auto)
coefficients(refined)
refined.original_parameters
refined.original_objective, refined.objective
refined.diagnostics
```

Constants are enumerated by the native expression-tree traversal, so two nodes
with equal values remain distinct parameters. `model_structure(candidate)`
records the feature order, target, formula, ordered parameter names, and a
value-independent SHA-256 structure fingerprint.

The automatic method proves only simple affine constructions (`+`, `-`, products
by parameter-free expressions, and division by parameter-free expressions) to be
linear in their constants. These are solved by pivoted QR or SVD. Every uncertain
case uses residual least squares through NonlinearSolve.jl. Symbolic trees have
concrete scalar storage, so their nonlinear refits use finite-difference Jacobians;
ordinary reconstructible nonlinear models may use automatic differentiation.

Bounds, multiple starting values, parameter scales, observation or inverse-variance
weighting, target transformations, and original/transformed loss spaces are
forwarded to the established fitting layer. A failed optimization returns a
`RefittedSymbolicModel` with `status == :failed` and retained diagnostics.

Jacobian rank and condition are local numerical diagnostics, not proofs of global
identifiability. Near-singular or redundant parameterizations must be reported as
such even when one numerical solution is returned.
