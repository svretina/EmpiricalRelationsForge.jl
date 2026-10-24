# EmpiricalRelationsForge.jl

[![CI](https://github.com/svretina/EmpiricalRelationsForge.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/svretina/EmpiricalRelationsForge.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![GitHub release](https://img.shields.io/github/v/release/svretina/EmpiricalRelationsForge.jl)](https://github.com/svretina/EmpiricalRelationsForge.jl/releases/latest)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Documentation](https://github.com/svretina/EmpiricalRelationsForge.jl/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/svretina/EmpiricalRelationsForge.jl/actions/workflows/docs.yml?query=branch%3Amaster)
[![Documentation site](https://img.shields.io/badge/docs-GitHub%20Pages-176b87)](https://svretina.github.io/EmpiricalRelationsForge.jl/)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia)](Project.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Data: CSV](https://img.shields.io/badge/data-CSV-2b7a4b)](benchmarks/vretinaris2020/README.md)

EmpiricalRelationsForge.jl provides a reproducible numerical workflow for fitting,
discovering, and validating scientific empirical relations. It combines
metadata-preserving datasets, stable regression methods, nonlinear least squares,
symbolic regression, and group-aware validation. The initial application is
binary-neutron-star postmerger research, but the numerical layer is domain independent.

The package complements
[EmpiricalRelations.jl](https://github.com/svretina/EmpiricalRelations.jl), which is the
reviewed catalogue of published measurements and relations. Forge constructs and tests
research candidates; it never inserts them into the published catalogue automatically.

**[Read the documentation](https://svretina.github.io/EmpiricalRelationsForge.jl/)**
· [Getting started](https://svretina.github.io/EmpiricalRelationsForge.jl/getting_started/)
· [Regression theory](https://svretina.github.io/EmpiricalRelationsForge.jl/regression/)
· [Symbolic discovery](https://svretina.github.io/EmpiricalRelationsForge.jl/symbolic/)
· [Vretinaris2020 benchmark](https://svretina.github.io/EmpiricalRelationsForge.jl/vretinaris2020/)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/svretina/EmpiricalRelationsForge.jl")
```

Julia 1.10 or later is supported.

## Capabilities

- Tabular ingestion through Tables.jl and DataFrames.jl with row identifiers, EOS or
  group labels, units, uncertainties, provenance, and transformation history.
- Linear, polynomial, and named-basis regression solved by pivoted QR or SVD without
  forming the normal-equation inverse.
- Weighted least squares and target transformations with an explicit choice of fitting
  in transformed or original physical space.
- Nonlinear least squares through NonlinearSolve.jl, including parameter bounds,
  multiple starting points, solver metadata, and convergence diagnostics.
- Equation discovery through SymbolicRegression.jl with configurable operators,
  complexity limits, weighting, random seeds, and preservation of the Pareto frontier.
- Random, grouped, and leave-one-EOS-out validation in which preprocessing and model
  coefficients are refitted independently inside every training fold.
- Global and group-resolved error metrics, design rank and condition diagnostics, and
  auditable model metadata export.

## Quick start

The following example fits a physically motivated basis and evaluates it without
leaking EOS groups between training and validation sets:

```julia
using DataFrames
using EmpiricalRelationsForge

measurements = DataFrame(
    simulation = 1:8,
    eos = repeat([:APR, :SLy, :DD2, :SFHo], inner=2),
    M_chirp = range(1.10, 1.30; length=8),
    Lambda_tilde = range(900.0, 300.0; length=8),
)
measurements.f_peak = @. 2.1 + 0.7 * measurements.M_chirp -
                           4 / sqrt(measurements.Lambda_tilde)

dataset = RegressionDataset(
    measurements;
    target=:f_peak,
    features=[:M_chirp, :Lambda_tilde],
    row_id=:simulation,
    eos_id=:eos,
    units=Dict(
        :M_chirp => "M_sun",
        :Lambda_tilde => "1",
        :f_peak => "kHz",
    ),
)

basis = BasisSet(
    PowerProductTerm(:constant),
    PowerProductTerm(:chirp_mass, Dict(:M_chirp => 1); label="M_chirp"),
    PowerProductTerm(
        :inverse_sqrt_tidal,
        Dict(:Lambda_tilde => -0.5);
        label="Lambda_tilde^(-1/2)",
    ),
)

method = LinearBasisRegression(basis)
model = fit(dataset, method)

formula(model)
coefficients(model)
metrics(model, dataset)

validation = validate(method, dataset; strategy=LeaveOneEOSOut())
validation.aggregate_metrics
```

For equation discovery, define the admissible grammar and retain the complete Pareto
frontier rather than selecting a formula from training loss alone:

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

`best_model` is a numerical convenience, not a scientific model-selection rule.
Candidate structure and hyperparameters should be chosen with training and validation
data; an untouched test set should be reserved for the final performance estimate.

## Scientific conventions

Observation weights are interpreted explicitly: direct weights minimize
`sum(w_i r_i^2)`, while inverse-variance weighting uses `w_i = 1/sigma_i^2`. Target
transformations record whether the optimized residuals belong to transformed or
original physical space. Units and provenance are retained as metadata; dimensional
consistency is not yet enforced automatically.

Group-aware validation is essential when simulations share an EOS, numerical sequence,
or other correlated origin. `LeaveOneEOSOut()` holds out a complete EOS and repeats all
fitted preprocessing within each fold. Random row splits should only be used when rows
are scientifically independent.

## Vretinaris2020 benchmark

The repository contains a reproducible benchmark against the published
Vretinaris et al. (2020) relations. It exercises direct least-squares reproduction and
symbolic equation discovery using data from the companion repository. The benchmark
documents which published coefficients can be recovered from printed table values and
where unavailable or rounded source quantities prevent exact reproduction.

See [`benchmarks/vretinaris2020/`](benchmarks/vretinaris2020/README.md) and the
[benchmark report](https://svretina.github.io/EmpiricalRelationsForge.jl/vretinaris2020/).

## Reproducibility

Run the package tests and Aqua quality checks with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the documentation locally with:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
python3 -m http.server --directory docs/build 8000
```

The test suite covers data validation and splitting, classical and nonlinear fitting,
metrics, cross-validation, symbolic adapter behavior, and package-quality checks.
Continuous integration tests the minimum supported Julia release, the current stable
release, and the prerelease channel.

## Scope and citation

Candidate relations require scientific validation before publication or use outside
their calibration domain. Record the package revision, source dataset, feature and
target definitions, weighting convention, split strategy, random seed, and retained
candidate frontier for every analysis.

Cite the original publications for reused measurements and equations. Release-level
software citation metadata will be added when an archived release is available.

EmpiricalRelationsForge.jl is distributed under the [MIT License](LICENSE).
