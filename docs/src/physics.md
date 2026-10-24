# Physical normalization and dimensional constraints

`PhysicalQuantity` records a name, mass-length-time dimension, canonical unit,
fixed normalization scale, and valid numerical domain. `BNS_QUANTITIES` contains
the common binary-neutron-star masses, radii, frequencies, tidal deformability,
mass ratio, and compactness. User registries keep this mechanism domain independent.

```julia
normalized, convention = normalize_physical(dataset, BNS_QUANTITIES)
normalized_model = fit(normalized, LinearRegression())
model = physical_model(normalized_model, convention)
predict(model, dataset)  # original target units
```

These scales are physical conventions, not statistics learned from all rows, and
therefore do not leak validation information. Statistical standardization remains
a fold-local fitted transformation.

`check_dimensions(candidate, registry)` implements a conservative inspection of
addition, multiplication, division, constant powers, roots, and transcendental
functions. Scalar fit coefficients may carry dimensions, so an indeterminate
coefficient dimension is admissible. The result is a screening diagnostic, not a
general symbolic proof. `expression_validity` separately checks evaluation on the
observed domain and reports non-finite output; finite sampling does not prove global
domain validity.

For discovery, `SymbolicSearch(dimensional_constraints=true)` uses the native
SymbolicRegression.jl dimensional constraint system with DynamicQuantities.jl.
Physical templates and restricted operator families can additionally be supplied
through the backend configuration.
