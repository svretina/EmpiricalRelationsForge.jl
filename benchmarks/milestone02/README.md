# Milestone 02 scientific workflow benchmark

Run the deterministic known-structure workflow from the repository root:

```sh
julia --project=. examples/scientific_refinement.jl
```

It generates grouped noisy observations from
`y = a + b*x1 + c/sqrt(x2)`, fits the named basis, refits an independently
constructed symbolic tree, performs leave-one-group-out comparison, estimates
local covariance and grouped bootstrap intervals, checks dimensions, round-trips
the selected model, and exports a candidate review. It does not rely on stochastic
recovery of one exact expression.

Set `ERF_RUN_SYMBOLIC_SEARCH=true` to append a bounded symbolic search. Full nested
discovery and large bootstrap ensembles are intentionally excluded from ordinary CI.
