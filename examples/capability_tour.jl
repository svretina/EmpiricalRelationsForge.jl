using DataFrames
using EmpiricalRelationsForge
using Random
using SymbolicRegression

# A small, deterministic BNS-like table used throughout the tour.
rng = MersenneTwister(17)
n = 36
index = collect(1:n)
table = DataFrame(
    simulation=1:n,
    eos=repeat([:APR, :DD2, :SFHo, :SLy, :TM1, :WFF1], inner=6),
    source=repeat([:catalogue_a, :catalogue_b], inner=18),
    M_chirp=collect(range(1.08, 1.35; length=n)),
    q=(@. 0.86 + 0.12 * sin(0.71 * index)),
    Lambda_tilde=(@. 625 + 300 * cos(0.37 * index)),
    sigma=fill(0.02, n),
)
noise = 0.005randn(rng, n)
table.f_peak = @. 1.8 + 0.9 * table.M_chirp +
    18 / sqrt(table.Lambda_tilde) + 0.15 * (1 - table.q) + noise

data = RegressionDataset(table; target=:f_peak,
    features=[:M_chirp, :q, :Lambda_tilde], row_id=:simulation,
    eos_id=:eos, source_paper=:source, uncertainty=:sigma,
    uncertainty_kind=:numerical,
    units=Dict(:M_chirp=>"M_sun", :q=>"1", :Lambda_tilde=>"1", :f_peak=>"kHz"))

validate_dataset(data)
training_rows = EmpiricalRelationsForge.subset(data, collect(1:24))
standardizer = fit_standardizer(training_rows)
standardized = transform_features(training_rows, standardizer)
@assert size(standardized) == size(training_rows)

# Classical regression families.
linear = fit(data, LinearRegression(weighting=:inverse_variance))
quadratic = fit(data, PolynomialRegression(degree=2, standardize=true))
basis = BasisSet(
    PowerProductTerm(:constant),
    PowerProductTerm(:chirp_mass, Dict(:M_chirp=>1); label="M_chirp"),
    PowerProductTerm(:mass_ratio, Dict(:q=>1); label="q"),
    PowerProductTerm(:inverse_sqrt_tidal, Dict(:Lambda_tilde=>-0.5);
        label="Lambda_tilde^(-1/2)"),
)
basis_method = LinearBasisRegression(basis; weighting=:inverse_variance)
basis_model = fit(data, basis_method)
println(formula(basis_model))
println(metrics(basis_model, data))

# A nonlinear-in-parameter power law with multistart fitting and bounds.
power_law = (X, p) -> @. p[1] + p[2] * X[:, 1]^p[3] * X[:, 3]^p[4]
nonlinear_specification = NonlinearModelSpec(power_law,
    [:offset, :amplitude, :mass_power, :tidal_power],
    [[1.0, 1.0, 1.0, -0.5], [1.5, 10.0, 0.5, -0.4]];
    expression="f_peak = offset + amplitude*M_chirp^mass_power*Lambda_tilde^tidal_power",
    lower_bounds=[0.0, -100.0, -3.0, -3.0],
    upper_bounds=[5.0, 100.0, 3.0, 3.0],
    parameter_scale=[1.0, 10.0, 1.0, 1.0],
    solver_options=(abstol=1e-9, reltol=1e-9, maxiters=500))
nonlinear_model = fit(data, nonlinear_specification)
@assert nonlinear_model.converged

# Metrics, group summaries, splitting, validation, and common-fold comparison.
predictions = predict(basis_model, data)
global_metrics = metrics(predictions, data.y)
eos_metrics = grouped_metrics(predictions, data.y, data.eos_ids)
holdout = random_split(data; test_fraction=0.2, rng=MersenneTwister(4))
folds = grouped_kfold(data; by=:eos, k=3, rng=MersenneTwister(4))
leave_one_eos = leave_one_group_out(data; by=:eos)
validation = validate(basis_method, data; strategy=GroupedKFold(k=3, by=:eos, seed=4))
comparison = compare(data,
    [:linear=>LinearRegression(), :quadratic=>PolynomialRegression(degree=2),
     :physical_basis=>basis_method];
    validation=GroupedKFold(k=3, by=:eos, seed=4))
selected_comparison = select_model(comparison; policy=:minimum_rmse)

# Construct and independently refit a symbolic structure.
operators = SymbolicRegression.OperatorEnum(1 => (sqrt,), 2 => (+, -, *, /))
options = SymbolicRegression.Options(; operators, progress=false, save_to_file=false)
tree = SymbolicRegression.parse_expression(
    "1.0 + 1.0*M_chirp + 1.0/sqrt(Lambda_tilde)";
    operators, variable_names=["M_chirp", "q", "Lambda_tilde"])
symbolic = SymbolicCandidate(tree, Inf, 8, copy(data.feature_names), data.target_name, options)
refined = refit(symbolic, data; weighting=:inverse_variance)
@assert refined.status == :success

# Local parameter uncertainty and fixed-structure grouped stability.
covariance = parameter_covariance(refined, data)
propagated = prediction_uncertainty(refined, covariance, data.X[1:3, :])
resampling = bootstrap(refined, data; group=:eos, n=20, seed=11)

# Physical normalization and conservative dimensional screening.
normalized_data, normalization = normalize_physical(data)
normalized_model = fit(normalized_data, LinearRegression())
physical = physical_model(normalized_model, normalization)
@assert length(predict(physical, data)) == length(data)
dimensional = check_dimensions(symbolic)
validity = expression_validity(symbolic, data)

# Conservative structural equivalence grouping.
symbolic_copy = with_parameters(symbolic, parameter_vector(symbolic))
assessment = equivalence(symbolic, symbolic_copy; X=data.X)
groups = deduplicate_candidates([symbolic, symbolic_copy])
@assert assessment.equivalent
@assert length(groups.representatives) == 1

# Joint feature-subset and mathematical-family discovery.
problem = RelationDiscoveryProblem(target=:f_peak,
    candidate_features=[:M_chirp, :q, :Lambda_tilde],
    mandatory_features=[:M_chirp], max_features=3,
    transformations=[FeatureTransformation(:Lambda_tilde, :inverse_sqrt;
        name=:inverse_sqrt_Lambda)],
    variables=[
        VariableDescriptor(:M_chirp; role=:inspiral, accessible=true),
        VariableDescriptor(:q; role=:inspiral, accessible=true),
        VariableDescriptor(:Lambda_tilde; role=:inspiral, accessible=true),
    ],
    objective=:penalized, complexity_penalty=1e-4, variable_penalty=1e-3)
discovery = discover_relations(problem, data;
    methods=[PolynomialSearch(min_degree=1, max_degree=2, standardize=true)],
    subset_strategy=:exhaustive,
    validation=GroupedKFold(k=3, by=:eos, seed=8), seed=8)
display(first(leaderboard(discovery), 5))
frontier = pareto_frontier(discovery)
accessible = filter_candidates(discovery; accessible_only=true)
selected = select_model(discovery; criterion=:validation_rmse)

# Repeat the complete selection procedure inside each outer training fold.
outer = validate_discovery(problem, data;
    methods=[PolynomialSearch(min_degree=1, max_degree=2, standardize=true)],
    outer=GroupedKFold(k=3, by=:eos, seed=9),
    inner=KFold(k=3, seed=10), seed=9)

# A published or externally defined relation can be compared as a fixed baseline.
published = PublishedPredictor("synthetic-reference", copy(data.feature_names), :f_peak,
    values -> 1.8 + 0.9values.M_chirp + 18 / sqrt(values.Lambda_tilde) +
        0.15 * (1 - values.q);
    expression="f_peak = 1.8 + 0.9*M_chirp + 18/sqrt(Lambda_tilde) + 0.15*(1-q)",
    provenance=Dict(:status=>:synthetic_reference))
baseline_comparison = compare(data,
    [:published=>published, :candidate=>basis_method];
    validation=GroupedKFold(k=3, by=:eos, seed=4))

# Portable model, metadata, discovery-campaign, and review artifacts.
mktempdir() do directory
    model_path = joinpath(directory, "model.toml")
    save_model(model_path, refined;
        metadata=Dict(:dataset_hash=>dataset_fingerprint(data), :seed=>17))
    restored = load_model(model_path)
    @assert predict(restored, data) ≈ predict(refined, data)

    result = model_result(refined, data; validation,
        provenance=Dict(:analysis=>"capability tour"),
        reproducibility=Dict(:seed=>17))
    save_metadata(joinpath(directory, "metadata.toml"), result)

    campaign_path = joinpath(directory, "campaign")
    save_discovery(campaign_path, discovery)
    restored_discovery = load_discovery(campaign_path)
    @assert restored_discovery.dataset_hash == discovery.dataset_hash

    export_candidate(refined, validation;
        path=joinpath(directory, "candidate"), overwrite=true,
        uncertainty=resampling.parameter_intervals,
        applicability="Synthetic demonstration domain only")
end

# Native symbolic search is optional in this fast tour because it is stochastic and
# intentionally receives an explicit computational budget.
if get(ENV, "ERF_RUN_SYMBOLIC_SEARCH", "false") == "true"
    search = SymbolicSearch(problem; niterations=20, seed=17,
        parallelism=:serial)
    symbolic_discovery = discover(data, search)
    display(leaderboard(symbolic_discovery))
end

println((global_metrics, eos_count=length(eos_metrics), holdout,
    grouped_folds=length(folds), leave_one_eos=length(leave_one_eos),
    validation_rmse=validation.aggregate_metrics.rmse,
    selected_comparison=selected_comparison.name,
    covariance_status=covariance.status,
    propagated_error=propagated.standard_error,
    bootstrap_failures=resampling.failures,
    dimensional=dimensional.valid, expression_valid=validity.valid,
    candidates=length(discovery.candidates), pareto=length(frontier),
    accessible=length(accessible), selected=formula(selected),
    outer_rmse=outer.aggregate_metrics.rmse,
    baseline_models=length(baseline_comparison.models)))
