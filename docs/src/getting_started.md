# Getting started

```julia
using DataFrames
using EmpiricalRelationsForge

table = DataFrame(
    id = 1:8,
    eos = repeat([:APR, :SLy, :DD2, :SFHo], inner=2),
    M_chirp = range(1.1, 1.3; length=8),
    Lambda_tilde = range(900.0, 300.0; length=8),
)
table.f_peak = @. 2.1 + 0.7 * table.M_chirp - 4 / sqrt(table.Lambda_tilde)

dataset = RegressionDataset(
    table;
    target=:f_peak,
    features=[:M_chirp, :Lambda_tilde],
    row_id=:id,
    eos_id=:eos,
    units=Dict(:M_chirp => "M_sun", :Lambda_tilde => "1", :f_peak => "kHz"),
)

basis = BasisSet(
    PowerProductTerm(:constant),
    PowerProductTerm(:chirp_mass, Dict(:M_chirp => 1); label="M_chirp"),
    PowerProductTerm(:inverse_sqrt_tidal, Dict(:Lambda_tilde => -0.5);
                     label="Lambda_tilde^(-1/2)"),
)

model = fit(dataset, LinearBasisRegression(basis))
formula(model)
metrics(model, dataset)

report = validate(
    LinearBasisRegression(basis), dataset;
    strategy=LeaveOneEOSOut(),
)
```

The last call refits both preprocessing and coefficients once per held-out EOS.
It does not reuse the full-data model.
