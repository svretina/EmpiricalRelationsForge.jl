# EmpiricalRelationsForge.jl

EmpiricalRelationsForge.jl is a Julia package for preparing, validating, and
eventually fitting scientific empirical relations. It complements
EmpiricalRelations.jl, which remains the catalogue of published relations and
their evidence.

The package implements scientific datasets, linear and polynomial regression,
named basis functions, nonlinear least squares, symbolic discovery, metrics,
and leakage-safe grouped validation.

```julia
using DataFrames, EmpiricalRelationsForge, Random

measurements = DataFrame(
    simulation = 1:6,
    eos = repeat([:APR, :SLy, :DD2], inner=2),
    chirp_mass = [1.16, 1.18, 1.20, 1.22, 1.24, 1.26],
    lambda_tilde = [800, 720, 640, 560, 480, 400],
    f_peak = [2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
)

dataset = RegressionDataset(
    measurements;
    target=:f_peak,
    features=[:chirp_mass, :lambda_tilde],
    row_id=:simulation,
    eos_id=:eos,
    units=Dict(:chirp_mass => "M_sun", :f_peak => "kHz"),
)

folds = leave_one_group_out(dataset; by=:eos)

model = fit(dataset, PolynomialRegression(degree=2))
report = validate(PolynomialRegression(degree=2), dataset;
                  strategy=LeaveOneEOSOut())
```

Julia 1.10 or later is supported. The documentation covers the mathematical
objectives, weighting conventions, target transformations, numerical backends,
validation methodology, scientific conventions, and reproducibility limits.
