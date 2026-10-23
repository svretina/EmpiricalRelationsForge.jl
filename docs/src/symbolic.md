# Symbolic discovery

`SymbolicSearch` is a thin adapter over SymbolicRegression.jl. It accepts
operator sets, maximum expression size, iteration count, parallelism, seed,
weighting convention, and backend options. A fully configured backend `Options`
object may be supplied for advanced control.

```julia
search = SymbolicSearch(
    binary_operators=[+, -, *, /],
    unary_operators=[sqrt],
    max_complexity=20,
    niterations=100,
    seed=42,
)
result = discover(dataset, search)
leaderboard(result)
candidate = best_model(result; criterion=:loss)
```

The adapter transposes the observations-by-features Forge matrix to the
features-by-observations backend convention. It preserves the native expression
objects, hall of fame, Pareto candidates, losses, computed complexities,
variable order, units, training IDs, search configuration, seed, and backend
version. Candidate evaluation uses the backend expression evaluator, not a
parsed display string.

`best_model` is a convenience selection by loss or complexity. It is not a
claim that a candidate is scientifically preferred. Candidate choice must use
training/validation data and an untouched test set must remain unavailable to
formula selection.
