# EmpiricalRelationsForge.jl

<p align="center">
  <img src="docs/src/assets/images/logo.png" alt="EmpiricalRelationsForge.jl — binary neutron-star empirical-relation fitting and discovery logo" width="720">
</p>

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

The package is intended for researchers who need to move from a documented numerical
dataset to a ranked, auditable collection of candidate relations. It separates four
questions that are often mixed together: which variables are available, which
mathematical structures are admissible, how coefficients are estimated, and which
generalization question a validation split actually tests.

The package complements
[EmpiricalRelations.jl](https://github.com/svretina/EmpiricalRelations.jl), which is the
reviewed catalogue of published measurements and relations. Forge constructs and tests
research candidates; it never inserts them into the published catalogue automatically.

**[Read the documentation](https://svretina.github.io/EmpiricalRelationsForge.jl/)**
· [Getting started](https://svretina.github.io/EmpiricalRelationsForge.jl/getting_started/)
· [Capability cookbook](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/)
· [Regression theory](https://svretina.github.io/EmpiricalRelationsForge.jl/regression/)
· [Relation discovery](https://svretina.github.io/EmpiricalRelationsForge.jl/discovery/)
· [Symbolic discovery](https://svretina.github.io/EmpiricalRelationsForge.jl/symbolic/)
· [Vretinaris2020 benchmark](https://svretina.github.io/EmpiricalRelationsForge.jl/vretinaris2020/)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/svretina/EmpiricalRelationsForge.jl")
```

Julia 1.10 or later is supported.

To use the optional read-only adapter for the companion catalogue, install and load
EmpiricalRelations.jl in the same environment:

```julia
using Pkg
Pkg.add(url="https://github.com/svretina/EmpiricalRelations.jl")

using EmpiricalRelations
using EmpiricalRelationsForge
```

## What the repository provides

The repository is organized as a scientific workflow rather than a collection of
unrelated fit functions:

```text
scientific table
    → validated RegressionDataset with provenance
    → explicit target, feature pool, transformations, and model families
    → coefficient fitting or symbolic structure search
    → common-fold comparison and leakage-free discovery validation
    → uncertainty, physical, equivalence, and domain diagnostics
    → versioned candidate and campaign artifacts
```

EmpiricalRelationsForge.jl constructs and evaluates research candidates.
EmpiricalRelations.jl remains the reviewed catalogue of published relations. No Forge
operation automatically registers or verifies a discovered equation.

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
- Joint exploration of explanatory-variable subsets and polynomial, transformed,
  named-basis, and symbolic model families through `RelationDiscoveryProblem`.
- Ranked DataFrame leaderboards, accuracy--complexity Pareto fronts, observational-
  accessibility filters, complete campaign artifacts, and leakage-free outer validation.
- Random, grouped, and leave-one-EOS-out validation in which preprocessing and model
  coefficients are refitted independently inside every training fold.
- Fixed-structure symbolic refitting with stable constant identities, automatic
  linear/nonlinear solver selection, bounds, scaling, and identifiability diagnostics.
- Fair multi-backend comparison on shared folds and nested grouped validation for
  symbolic structure selection without access to outer validation targets.
- Jacobian covariance, parameter-induced prediction uncertainty, and whole-group
  bootstrap intervals conditional on a fixed mathematical structure.
- Physical normalization, conservative dimensional checks, native dimensional search
  constraints, versioned model artifacts, and conservative symbolic deduplication.
- Read-only published-relation adapters and review-oriented candidate exports; candidate
  discovery never changes the companion catalogue.
- Global and group-resolved error metrics, design rank and condition diagnostics, and
  auditable model metadata export.

The principal entry points are summarized below. Every row links to a worked example.

| Scientific task | Main interface | Worked documentation |
| --- | --- | --- |
| Load tables and retain scientific provenance | `RegressionDataset`, `load_dataset`, `subset` | [Data construction](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Common-dataset) |
| Validate rows, identifiers, uncertainties, and dependencies | `validate_dataset`, `duplicate_identifiers`, `physical_dependency_warnings` | [Data inspection](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Validate,-inspect,-and-subset-data) |
| Fit linear and polynomial relations | `LinearRegression`, `PolynomialRegression` | [Classical regression](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Linear-regression) |
| Fit explicit physical combinations | `BasisSet`, `PowerProductTerm`, `LinearBasisRegression` | [Named bases](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Named-physical-bases) |
| Fit parameters entering nonlinearly | `NonlinearModelSpec` | [Nonlinear least squares](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Nonlinear-least-squares) |
| Search expression trees | `SymbolicSearch`, `discover`, `best_model` | [Symbolic discovery](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Native-symbolic-discovery) |
| Refit a fixed symbolic structure | `refit`, `parameter_vector`, `with_parameters` | [Structure refitting](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Construct-and-refit-a-symbolic-structure) |
| Search variables and model families jointly | `RelationDiscoveryProblem`, `discover_relations` | [Relation discovery](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Discover-variables-and-structures-jointly) |
| Rank and filter candidates | `leaderboard`, `pareto_frontier`, `filter_candidates`, `select_model` | [Candidate ranking](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Rank,-filter,-and-select-discovered-relations) |
| Validate a fixed method | `validate`, `RandomHoldout`, `KFold`, `GroupedKFold`, `LeaveOneGroupOut` | [Cross-validation](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Cross-validation) |
| Validate the complete search procedure | `validate_discovery`, `nested_validate` | [Nested discovery validation](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Validate-the-complete-discovery-procedure) |
| Compare heterogeneous model families | `compare` | [Common-fold comparison](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Compare-model-families-on-identical-folds) |
| Estimate fixed-structure uncertainty | `parameter_covariance`, `prediction_uncertainty`, `bootstrap` | [Uncertainty](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Parameter-covariance-and-propagated-uncertainty) |
| Normalize and screen physical expressions | `normalize_physical`, `check_dimensions`, `expression_validity` | [Physical screening](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Dimensional-and-domain-screening) |
| Group equivalent symbolic candidates | `equivalence`, `structural_fingerprint`, `deduplicate_candidates` | [Equivalence](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Candidate-equivalence) |
| Compare against published relations | `published_predictor`, `PublishedPredictor` | [Published baselines](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Published-relations) |
| Save models and complete campaigns | `save_model`, `load_model`, `save_discovery`, `load_discovery` | [Serialization](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Save-and-restore-a-model) |
| Prepare a candidate for scientific review | `model_result`, `save_metadata`, `export_candidate` | [Candidate export](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Export-a-relation-for-scientific-review) |

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
    q = [0.78, 0.91, 0.84, 1.0, 0.76, 0.95, 0.87, 0.98],
    Lambda_tilde = range(900.0, 300.0; length=8),
)
measurements.f_peak = @. 2.1 + 0.7 * measurements.M_chirp -
                           4 / sqrt(measurements.Lambda_tilde)

dataset = RegressionDataset(
    measurements;
    target=:f_peak,
    features=[:M_chirp, :q, :Lambda_tilde],
    row_id=:simulation,
    eos_id=:eos,
    units=Dict(
        :M_chirp => "M_sun",
        :q => "1",
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

To search over both variable subsets and mathematical structure, define a discovery
problem. Targets are unrestricted, so the same workflow applies to forward frequency
prediction and inverse inference of a radius or tidal quantity.

```julia
problem = RelationDiscoveryProblem(
    target=:f_peak,
    candidate_features=[:M_chirp, :q, :Lambda_tilde],
    mandatory_features=[:M_chirp],
    max_features=3,
    transformations=[
        FeatureTransformation(:Lambda_tilde, :inverse_sqrt;
            name=:inverse_sqrt_Lambda),
    ],
    max_expression_complexity=20,
)

results = discover_relations(
    problem,
    dataset;
    methods=[
        PolynomialSearch(max_degree=3, standardize=true),
        SymbolicSearch(problem; niterations=100, seed=42),
    ],
    validation=KFold(k=5, seed=42),
)

leaderboard(results)
best = select_model(results; criterion=:validation_rmse)
formula(best)
```

Use `validate_discovery` for a held-out performance estimate: it repeats feature and
structure selection inside every outer training partition. The validation metric stored
in a discovery leaderboard is an inner model-selection estimate, not an untouched test
result.

`best_model` is a numerical convenience, not a scientific model-selection rule.
Candidate structure and hyperparameters should be chosen with training and validation
data; an untouched test set should be reserved for the final performance estimate.

## Complete worked examples

- [`examples/capability_tour.jl`](examples/capability_tour.jl) exercises the public
  workflow from dataset construction through fitting, comparison, uncertainty,
  discovery, physical screening, serialization, and candidate export. It is designed
  to run quickly; set `ERF_RUN_SYMBOLIC_SEARCH=true` to include a stochastic symbolic
  search.
- [`examples/relation_discovery.jl`](examples/relation_discovery.jl) demonstrates
  forward frequency discovery, independent inverse-radius discovery, and nested
  validation of the complete selection procedure.
- [`examples/scientific_refinement.jl`](examples/scientific_refinement.jl) demonstrates
  fixed symbolic-structure refitting, common-fold comparison, covariance, grouped
  bootstrap, dimensional screening, and review export.
- [`examples/polynomial_regression.jl`](examples/polynomial_regression.jl) is a minimal
  multivariate polynomial fit.
- [`examples/symbolic_regression.jl`](examples/symbolic_regression.jl) is a minimal
  SymbolicRegression.jl search.
- [`benchmarks/vretinaris2020/`](benchmarks/vretinaris2020/README.md) reproduces the
  fixed-form relations and runs representative forward and inverse symbolic searches
  against data maintained by EmpiricalRelations.jl.

Run the complete capability tour with:

```sh
julia --project=. examples/capability_tour.jl
```

Run its optional symbolic-search section with an explicit budget using:

```sh
ERF_RUN_SYMBOLIC_SEARCH=true julia --project=. examples/capability_tour.jl
```

## Scientific conventions

Observation weights are interpreted explicitly: direct weights minimize
`sum(w_i r_i^2)`, while inverse-variance weighting uses `w_i = 1/sigma_i^2`. Target
transformations record whether the optimized residuals belong to transformed or
original physical space. Units and provenance are retained as metadata; dimensional
consistency can be checked conservatively and can be enforced during symbolic search.

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
metrics, cross-validation, symbolic discovery and refitting, model comparison,
uncertainty, physical checks, serialization, relation discovery, restoration,
type-inference guarantees, and package-quality checks.
Continuous integration tests the minimum supported Julia release, the current stable
release, and the prerelease channel.

## Scope and citation

Candidate relations require scientific validation before publication or use outside
their calibration domain. Record the package revision, source dataset, feature and
target definitions, weighting convention, split strategy, random seed, and retained
candidate frontier for every analysis.

Cite the original publications for reused measurements and equations. Release-level
software citation metadata will be added when an archived release is available.

For a publishable analysis, report at minimum the immutable data source, row selection,
target and feature definitions, units, transformations, weighting convention, search
grammar and budget, random seeds, validation groups, retained Pareto frontier,
conditioning and identifiability diagnostics, uncertainty assumptions, calibration
domain, and exact software revision. The
[capability cookbook](https://svretina.github.io/EmpiricalRelationsForge.jl/capability_cookbook/#Minimum-reporting-record)
contains a complete checklist.

Scientific bug reports, requests for additional benchmark datasets, and proposals for
new reconstructible model families are welcome through the
[GitHub issue tracker](https://github.com/svretina/EmpiricalRelationsForge.jl/issues).
Reports should include a minimal dataset or reproducible script, Julia version, package
revision, and the scientific expectation being tested.

EmpiricalRelationsForge.jl is distributed under the [MIT License](LICENSE).
