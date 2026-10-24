# Published relations and candidate export

EmpiricalRelationsForge does not alter the reviewed catalogue in
EmpiricalRelations.jl. When the companion package is loaded, its optional extension
provides:

```julia
using EmpiricalRelations
spec = relation(:Vretinaris2020; quantity=:f_peak,
    inputs=(:M_chirp, :Lambda_tilde))
baseline = published_predictor(spec; strict_domain=true)
report = compare(dataset, [:published => baseline, :candidate => candidate])
```

The adapter uses the companion package's public evaluator, formula, domain checking,
stable identifier, and source metadata. It is classified as a fixed benchmark
predictor. Dataset columns must have the same identities, order, and canonical unit
conventions required by the published relation.

`export_candidate(model, report; path=...)` writes a versioned model artifact and a
scientific review document containing the formula, parameters, applicability,
validation errors, uncertainty material supplied by the researcher, limitations,
and candidate status. Export never registers or verifies a relation.
