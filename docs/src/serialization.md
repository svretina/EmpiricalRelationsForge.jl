# Model serialization

```julia
save_model("candidate.toml", model;
    metadata=Dict(:dataset_hash => dataset_fingerprint(dataset)))
restored = load_model("candidate.toml")
```

Artifacts use versioned TOML and explicit reconstruction adapters. Polynomial,
named-basis, normalized, and refitted symbolic models retain full-precision
parameters, feature order, units, transforms, training row identities, operator
identities, and backend/package versions as applicable. Symbolic trees are stored
recursively; restoration never evaluates saved Julia source or parses a display
formula.

Arbitrary nonlinear Julia closures are deliberately rejected because their code
and captured state cannot be represented portably. Unsupported schema versions,
unknown operators, missing files, and malformed or truncated TOML produce
informative errors instead of silent reinterpretation.

`dataset_fingerprint` hashes the numerical values and defining row/variable
identities. Store it and the split/search/solver configuration as artifact metadata;
the complete raw scientific database need not be embedded in each model.
