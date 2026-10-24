# Capability cookbook

This page is an operational guide to the public capabilities of
EmpiricalRelationsForge.jl. The examples use one synthetic BNS-like table so that
individual operations can be combined into a complete analysis. For a single
executable version, see
[`examples/capability_tour.jl`](https://github.com/svretina/EmpiricalRelationsForge.jl/blob/master/examples/capability_tour.jl).

The examples demonstrate software behavior. A scientific result additionally requires
documented data selection, uncertainty assumptions, validation design, domain checks,
and an untouched final assessment set.

## Common dataset

```julia
using DataFrames
using EmpiricalRelationsForge
using Random

rng = MersenneTwister(17)
n = 36
i = collect(1:n)
table = DataFrame(
    simulation=1:n,
    eos=repeat([:APR, :DD2, :SFHo, :SLy, :TM1, :WFF1], inner=6),
    source=repeat([:catalogue_a, :catalogue_b], inner=18),
    M_chirp=collect(range(1.08, 1.35; length=n)),
    q=(@. 0.86 + 0.12sin(0.71i)),
    Lambda_tilde=(@. 625 + 300cos(0.37i)),
    sigma=fill(0.02, n),
)
noise = 0.005randn(rng, n)
table.f_peak = @. 1.8 + 0.9table.M_chirp +
    18 / sqrt(table.Lambda_tilde) + 0.15(1 - table.q) + noise

data = RegressionDataset(table;
    target=:f_peak,
    features=[:M_chirp, :q, :Lambda_tilde],
    row_id=:simulation,
    eos_id=:eos,
    source_paper=:source,
    uncertainty=:sigma,
    uncertainty_kind=:numerical,
    units=Dict(
        :M_chirp=>"M_sun", :q=>"1", :Lambda_tilde=>"1", :f_peak=>"kHz"),
)
```

Matrix input is also supported:

```julia
matrix_data = RegressionDataset(
    hcat(table.M_chirp, table.q, table.Lambda_tilde), table.f_peak;
    features=[:M_chirp, :q, :Lambda_tilde], target=:f_peak,
    row_ids=table.simulation, eos_ids=table.eos,
)
```

`load_dataset(table; kwargs...)` is an equivalent entry point when data loading is
written as a separate analysis stage. An `observation_kind` column may be supplied to
exclude `:predicted` and `:unreviewed` rows by default. Use `include_kinds` to make a
different, explicit selection.

## Validate, inspect, and subset data

```julia
validate_dataset(data)
size(data)                       # observations by features
duplicate_identifiers(data; by=:simulation)
physical_dependency_warnings([:mass_1, :mass_2, :M_chirp])

calibration = EmpiricalRelationsForge.subset(data, collect(1:24))
```

`subset` preserves identifiers, groups, units, uncertainties, weights, and
transformation history. It is qualified above because DataFrames also exports a
function named `subset`.

## Learn and apply standardization

```julia
standardizer = fit_standardizer(calibration)
standardized_calibration = transform_features(calibration, standardizer)
```

Fit the standardizer on training data only. Validation functions do this automatically
when a regression method requests `standardize=true`.

## Create explicit splits

```julia
random = random_split(data; test_fraction=0.2, rng=MersenneTwister(4))
grouped = grouped_kfold(data; by=:eos, k=3, rng=MersenneTwister(4))
one_eos_at_a_time = leave_one_group_out(data; by=:eos)

training = EmpiricalRelationsForge.subset(data, random.train)
testing = EmpiricalRelationsForge.subset(data, random.test)
```

Grouping may use `:eos`, `:simulation`, or `:source_paper`. Missing group identifiers
are rejected rather than silently treated as one group.

## Linear regression

```julia
method = LinearRegression(
    intercept=true,
    weighting=:inverse_variance,
    standardize=true,
    solver=:qr,
)
model = fit(data, method)

coefficients(model)
formula(model)
predict(model, data)
metrics(model, data)
```

Use `weighting=:observation` for explicit `dataset.weights`,
`weighting=:inverse_variance` for ``1/\sigma_i^2``, and `weighting=:none` for
unweighted least squares.

For direct observation weights, construct the dataset with `weights=...` for matrix
input or `weight=:column_name` for table input:

```julia
weighted_table = copy(table)
weighted_table.importance = ones(n)
weighted_data = RegressionDataset(weighted_table; target=:f_peak,
    features=[:M_chirp, :q, :Lambda_tilde], weight=:importance)
weighted_model = fit(weighted_data, LinearRegression(weighting=:observation))
```

## Polynomial regression

```julia
quadratic = fit(data, PolynomialRegression(
    degree=2,
    cross_terms=true,
    standardize=true,
    solver=:svd,
))

quadratic.term_names
quadratic.rank
quadratic.condition_number
```

Set `cross_terms=false` to exclude products between distinct variables.

## Named physical bases

```julia
basis = BasisSet(
    PowerProductTerm(:constant),
    PowerProductTerm(:chirp_mass, Dict(:M_chirp=>1); label="M_chirp"),
    PowerProductTerm(:mass_ratio, Dict(:q=>1); label="q"),
    PowerProductTerm(:inverse_sqrt_tidal, Dict(:Lambda_tilde=>-0.5);
        label="Lambda_tilde^(-1/2)"),
)

basis_method = LinearBasisRegression(basis; weighting=:inverse_variance)
basis_model = fit(data, basis_method)
formula(basis_model)
```

Fixed fractional or negative powers make power laws and rational combinations linear
in their unknown coefficients while preserving an auditable basis definition.

## Target transformations

To fit a relation in ``f_{\rm peak}/M_{\rm chirp}`` while returning predictions in
frequency units:

```julia
transformed_method = PolynomialRegression(
    degree=2,
    target_transform=FeatureScaleTarget(:M_chirp; power=-1),
    loss_space=:transformed,
)
transformed_model = fit(data, transformed_method)
frequency_prediction = predict(transformed_model, data)
```

Set `loss_space=:original` when the mathematical transformation is useful but the
least-squares objective should remain in the original target space.

## Nonlinear least squares

```julia
power_law = (X, p) -> @. p[1] + p[2] * X[:, 1]^p[3] * X[:, 3]^p[4]

specification = NonlinearModelSpec(
    power_law,
    [:offset, :amplitude, :mass_power, :tidal_power],
    [[1.0, 1.0, 1.0, -0.5], [1.5, 10.0, 0.5, -0.4]];
    expression="f_peak = offset + amplitude*M_chirp^mass_power*Lambda_tilde^tidal_power",
    lower_bounds=[0.0, -100.0, -3.0, -3.0],
    upper_bounds=[5.0, 100.0, 3.0, 3.0],
    parameter_scale=[1.0, 10.0, 1.0, 1.0],
    solver_options=(abstol=1e-9, reltol=1e-9, maxiters=500),
)

nonlinear = fit(data, specification)
nonlinear.converged
nonlinear.termination_status
nonlinear.residual_norm
```

Every initial guess is solved and the smallest objective is retained. Bounds and a
converged solver do not establish global identifiability.

## Metrics and group-resolved residuals

```julia
prediction = predict(basis_model, data)
overall = metrics(prediction, data.y)
by_eos = grouped_metrics(prediction, data.y, data.eos_ids)

overall.rmse
overall.max_absolute_error
by_eos[:APR].rmse
```

Residuals use prediction minus observation. Relative error is reported as `missing`
when any observed target is zero.

## Cross-validation

```julia
random_report = validate(basis_method, data;
    strategy=RandomHoldout(test_fraction=0.2, seed=3))
kfold_report = validate(basis_method, data;
    strategy=KFold(k=5, seed=3))
grouped_report = validate(basis_method, data;
    strategy=GroupedKFold(k=3, by=:eos, seed=3))
eos_report = validate(basis_method, data;
    strategy=LeaveOneEOSOut())
source_report = validate(basis_method, data;
    strategy=LeaveOneGroupOut(by=:source_paper))

grouped_report.aggregate_metrics
grouped_report.fold_metrics
grouped_report.group_metrics
```

Pass an unfitted method when coefficients and preprocessing must be refitted per fold.
`validate(fitted_model, data)` computes metrics only; it does not refit that model.

## Compare model families on identical folds

```julia
comparison = compare(data, [
    :linear => LinearRegression(),
    :quadratic => PolynomialRegression(degree=2),
    :physical_basis => basis_method,
]; validation=GroupedKFold(k=3, by=:eos, seed=4))

best_error = select_model(comparison; policy=:minimum_rmse)
best_group = select_model(comparison; policy=:minimum_worst_group_rmse)
simple = select_model(comparison;
    policy=:lowest_complexity_below_error, target_error=0.05)
```

Failed fits remain in `comparison.models` with their error messages.

## Native symbolic discovery

```julia
search = SymbolicSearch(
    binary_operators=[+, -, *, /],
    unary_operators=[sqrt, log],
    max_complexity=20,
    niterations=100,
    parallelism=:multithreading,
    seed=42,
    weighting=:inverse_variance,
)

symbolic_result = discover(data, search)
symbolic_table = leaderboard(symbolic_result)
symbolic_candidate = best_model(symbolic_result; criterion=:loss)
```

Run several independent seeds before interpreting structural recurrence. `best_model`
is a numerical convenience, not a scientific acceptance rule.

## Construct and refit a symbolic structure

```julia
using SymbolicRegression

operators = SymbolicRegression.OperatorEnum(1 => (sqrt,), 2 => (+, -, *, /))
options = SymbolicRegression.Options(; operators,
    progress=false, save_to_file=false)
tree = SymbolicRegression.parse_expression(
    "1.0 + 1.0*M_chirp + 1.0/sqrt(Lambda_tilde)";
    operators, variable_names=["M_chirp", "q", "Lambda_tilde"])
candidate = SymbolicCandidate(tree, Inf, 8,
    copy(data.feature_names), data.target_name, options)

refined = refit(candidate, data; weighting=:inverse_variance)
parameter_names(refined)
parameter_vector(refined)
refined.original_parameters
refined.diagnostics
```

`with_parameters(refined, values)` returns a new model with the same mathematical
structure. `model_structure(refined)` exposes its fingerprint and variable identities.

## Discover variables and structures jointly

```julia
problem = RelationDiscoveryProblem(
    target=:f_peak,
    candidate_features=[:M_chirp, :q, :Lambda_tilde],
    mandatory_features=[:M_chirp],
    forbidden_features=Symbol[],
    min_features=1,
    max_features=3,
    transformations=[
        FeatureTransformation(:Lambda_tilde, :inverse_sqrt;
            name=:inverse_sqrt_Lambda),
    ],
    binary_operators=[:+, :-, :*, :/],
    unary_operators=[:sqrt],
    max_expression_complexity=20,
    domains=Dict(:Lambda_tilde=>(1.0, Inf)),
    units=Dict(:f_peak=>"kHz"),
    variables=[
        VariableDescriptor(:M_chirp; role=:inspiral, accessible=true),
        VariableDescriptor(:q; role=:inspiral, accessible=true),
        VariableDescriptor(:Lambda_tilde; role=:inspiral, accessible=true),
    ],
    objective=:penalized,
    complexity_penalty=1e-4,
    variable_penalty=1e-3,
)

results = discover_relations(problem, data;
    methods=[
        PolynomialSearch(min_degree=1, max_degree=3, standardize=true),
        SymbolicSearch(problem; niterations=100, seed=42),
    ],
    subset_strategy=:exhaustive,
    validation=GroupedKFold(k=3, by=:eos, seed=42),
    seed=42,
)
```

Available subset strategies are `:exhaustive`, `:forward`, `:backward`,
`:penalized`, and `:symbolic`.

Fixed transformations support `:identity`, `:power`, `:inverse`, `:sqrt`,
`:inverse_sqrt`, `:log`, and `:exp`. Transformation domains are checked before a fit is
accepted.

## Rank, filter, and select discovered relations

```julia
table = leaderboard(results)
frontier = pareto_frontier(results)
observable = filter_candidates(results; accessible_only=true)
polynomials = filter_candidates(results; families=[:polynomial])

best = select_model(results; criterion=:validation_rmse)
formula(best)
coefficients(best)
best.selected_features
best.evaluation.validation
```

The default leaderboard groups safely equivalent structures. Use
`leaderboard(results; unique=false)` to retain every representation.

## Validate the complete discovery procedure

```julia
assessment = validate_discovery(problem, data;
    methods=[PolynomialSearch(max_degree=3, standardize=true)],
    subset_strategy=:forward,
    outer=GroupedKFold(k=3, by=:eos, seed=8),
    inner=KFold(k=3, seed=9),
    criterion=:validation_rmse,
    seed=8,
)

assessment.aggregate_metrics
assessment.selected_formulas
assessment.selected_features
```

This repeats feature selection and model selection inside every outer training
partition. It is the appropriate report for the predictive performance of a search
procedure.

For native symbolic searches, `nested_validate` supplies the corresponding
discovery--inner-selection--outer-assessment sequence:

```julia
nested = nested_validate(
    SymbolicSearch(niterations=100, seed=42), data;
    outer=GroupedKFold(k=3, by=:eos, seed=8),
    inner=GroupedKFold(k=2, by=:eos, seed=9),
    policy=:minimum_rmse,
)
nested.aggregate_metrics
nested.selected_formulas
```

## Forward and inverse relations

Fit inverse targets independently rather than rearranging a forward equation:

```julia
table.R_1_6 = @. 14.5 - 1.15table.f_peak + 0.8table.M_chirp + 0.2(1-table.q)
inverse_data = RegressionDataset(table; target=:R_1_6,
    features=[:f_peak, :M_chirp, :q], row_id=:simulation, eos_id=:eos,
    units=Dict(:f_peak=>"kHz", :M_chirp=>"M_sun", :q=>"1", :R_1_6=>"km"))

inverse_problem = RelationDiscoveryProblem(target=:R_1_6,
    candidate_features=[:f_peak, :M_chirp, :q], max_features=3)
inverse_results = discover_relations(inverse_problem, inverse_data;
    methods=[PolynomialSearch(max_degree=2)], validation=KFold(k=5, seed=5))
```

Inverse residuals are now measured directly in kilometres.

## Parameter covariance and propagated uncertainty

```julia
covariance = parameter_covariance(refined, data)
if covariance.status == :success
    propagated = prediction_uncertainty(refined, covariance, data.X[1:3, :])
    propagated.prediction
    propagated.standard_error
end
```

This is local, fixed-structure, parameter-induced uncertainty. It is not a complete
prediction interval.

## Grouped bootstrap

```julia
stability = bootstrap(refined, data;
    group=:eos, n=1000, seed=11, confidence_level=0.95)

stability.parameter_intervals
stability.prediction_intervals
stability.failures
stability.diagnostics
```

The bootstrap refits one fixed structure. Repeat complete symbolic searches separately
to study structural stability.

## Physical normalization

```julia
normalized_data, normalization = normalize_physical(data)
normalized_model = fit(normalized_data, LinearRegression())
model_in_physical_units = physical_model(normalized_model, normalization)
predict(model_in_physical_units, data)
```

Custom quantities can be supplied through a `Dict{Symbol,PhysicalQuantity}` registry.

```julia
registry = Dict(
    :x => PhysicalQuantity(:x, PhysicalDimension(length=1), "km", 1.0, (0.0, Inf)),
    :y => PhysicalQuantity(:y, PhysicalDimension(time=-1), "kHz", 1.0, (0.0, Inf)),
)
```

## Dimensional and domain screening

```julia
dimension_check = check_dimensions(candidate)
domain_check = expression_validity(candidate, data)

dimension_check.valid
dimension_check.message
domain_check.valid
domain_check.message
```

The dimensional checker is conservative because free coefficients may carry units.
For native constraints during search:

```julia
dimensioned_search = SymbolicSearch(problem;
    dimensional_constraints=true, niterations=100, seed=42)
```

## Candidate equivalence

```julia
candidate_copy = with_parameters(candidate, parameter_vector(candidate))
assessment = equivalence(candidate, candidate_copy; X=data.X)
fingerprint = structural_fingerprint(candidate)
deduplicated = deduplicate_candidates([candidate, candidate_copy])
```

Numerical agreement on supplied samples is reported as numerical evidence, not an
algebraic proof.

## Published relations

With the companion package installed and loaded:

```julia
using EmpiricalRelations

spec = relation(:Vretinaris2020;
    quantity=:f_peak, inputs=(:M_chirp, :Lambda_tilde))
published = published_predictor(spec; strict_domain=true)
published_predictions = predict(published, compatible_dataset)
```

A fixed callable can also be wrapped directly:

```julia
published = PublishedPredictor("reference-id", copy(data.feature_names), :f_peak,
    values -> 1.8 + 0.9values.M_chirp + 18/sqrt(values.Lambda_tilde) +
        0.15(1-values.q);
    expression="f_peak = 1.8 + 0.9*M_chirp + 18/sqrt(Lambda_tilde) + 0.15*(1-q)")

comparison = compare(data,
    [:published=>published, :candidate=>basis_method];
    validation=GroupedKFold(k=3, by=:eos, seed=4))
```

The feature identities, order, units, and calibration domain must match the published
relation.

## Save and restore a model

```julia
save_model("candidate.toml", refined;
    metadata=Dict(:dataset_hash=>dataset_fingerprint(data), :seed=>17))
restored = load_model("candidate.toml")
@assert predict(restored, data) ≈ predict(refined, data)
```

Arbitrary Julia closures, including general nonlinear callables, are not portable model
artifacts.

## Save analysis metadata

```julia
record = model_result(refined, data;
    validation=grouped_report,
    provenance=Dict(:analysis=>"postmerger frequency study"),
    reproducibility=Dict(:seed=>17, :revision=>"git commit"))
save_metadata("metadata.toml", record)
```

## Save and restore a discovery campaign

```julia
save_discovery("campaign", results)
restored_results = load_discovery("campaign")
@assert restored_results.dataset_hash == results.dataset_hash
```

Campaign artifacts require every candidate model to have a portable representation.
Published predictor closures and arbitrary nonlinear closures are deliberately rejected.

## Export a relation for scientific review

```julia
export_candidate(refined, grouped_report;
    path="candidate_relation",
    overwrite=true,
    uncertainty=stability.parameter_intervals,
    applicability="Calibration domain documented by the accompanying analysis",
    notes="Requires independent data and repeated structural searches.")
```

The export is a research artifact. It does not register the relation in the companion
catalogue or assign a verification status.

## Reproduce the Vretinaris2020 benchmark

The benchmark uses source-normalized data from a sibling checkout of
EmpiricalRelations.jl:

```sh
julia --project=benchmarks/vretinaris2020 -e '
using Pkg
Pkg.develop([
    PackageSpec(path=pwd()),
    PackageSpec(path=joinpath(dirname(pwd()), "EmpiricalRelations.jl")),
])
Pkg.instantiate()'

julia --project=benchmarks/vretinaris2020 \
    benchmarks/vretinaris2020/run.jl --symbolic
```

Use `--quick` for a symbolic smoke test. The committed result tables distinguish
published calibration reproduction from new-data evaluation and document missing or
rounded source quantities.

## Minimum reporting record

For a scientific result, retain at least:

- dataset source and immutable revision;
- row selection and observation classification;
- target and predictor definitions with units;
- transformations and fitting space;
- weighting or uncertainty convention;
- search grammar, budgets, and random seeds;
- exact validation strategy and group definition;
- complete candidate frontier, not only the selected formula;
- coefficient, conditioning, and identifiability diagnostics;
- residuals in physical target units;
- uncertainty assumptions and failed resamples;
- calibration and extrapolation domains;
- package, Julia, and backend versions.
