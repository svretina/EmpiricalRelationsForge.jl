# Candidate equivalence and deduplication

Equivalence is reported by level rather than reduced to one unrestricted Boolean:

- `:identical_tree` includes constants and exact tree order;
- `:canonical_structure` safely orders commutative `+` and `*` children while
  treating constants as distinct free parameter positions;
- `:numerical_sample` reports agreement on supplied inputs and tolerances;
- different variables or structures remain explicit outcomes.

```julia
assessment = equivalence(first_candidate, second_candidate; X=domain_samples)
catalogue = deduplicate_candidates(result.candidates)
```

The canonicalizer does not apply domain-changing cancellation, arbitrary algebraic
simplification, or floating-point reassociation. Numerical agreement is evidence on
the sample, not a proof. `deduplicate_candidates` preserves every original candidate,
its loss, and the member indices of each equivalence group while also identifying a
lowest-loss representative.
