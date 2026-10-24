using DataFrames
using EmpiricalRelationsForge

n = 36
index = collect(1:n)
measurements = DataFrame(
    simulation=1:n,
    eos=repeat([:APR, :DD2, :SFHo, :SLy, :TM1, :WFF1], inner=6),
    M_chirp=collect(range(1.08, 1.35; length=n)),
    q=(@. 0.86 + 0.12 * sin(0.71 * index)),
    Lambda_tilde=(@. 625 + 300 * cos(0.37 * index)),
)
measurements.f_peak = @. 1.8 + 0.9 * measurements.M_chirp +
    18 / sqrt(measurements.Lambda_tilde) + 0.15 * (1 - measurements.q)

forward_data = RegressionDataset(measurements; target=:f_peak,
    features=[:M_chirp, :q, :Lambda_tilde], row_id=:simulation, eos_id=:eos,
    units=Dict(:M_chirp=>"M_sun", :q=>"1", :Lambda_tilde=>"1", :f_peak=>"kHz"))

forward_problem = RelationDiscoveryProblem(target=:f_peak,
    candidate_features=[:M_chirp, :q, :Lambda_tilde], max_features=3,
    transformations=[FeatureTransformation(:Lambda_tilde, :inverse_sqrt;
        name=:inverse_sqrt_Lambda)],
    variables=[
        VariableDescriptor(:M_chirp; role=:inspiral, accessible=true),
        VariableDescriptor(:q; role=:inspiral, accessible=true),
        VariableDescriptor(:Lambda_tilde; role=:inspiral, accessible=true),
    ])

forward_results = discover_relations(forward_problem, forward_data;
    methods=[PolynomialSearch(max_degree=2, standardize=true)],
    validation=GroupedKFold(k=3, by=:eos, seed=4))

display(first(leaderboard(forward_results), 8))

# An inverse relation is fitted independently in the target quantity of interest.
measurements.R_1_6 = @. 14.5 - 1.15 * measurements.f_peak +
    0.8 * measurements.M_chirp + 0.2 * (1 - measurements.q)
inverse_data = RegressionDataset(measurements; target=:R_1_6,
    features=[:f_peak, :M_chirp, :q], row_id=:simulation, eos_id=:eos,
    units=Dict(:f_peak=>"kHz", :M_chirp=>"M_sun", :q=>"1", :R_1_6=>"km"))
inverse_problem = RelationDiscoveryProblem(target=:R_1_6,
    candidate_features=[:f_peak, :M_chirp, :q], max_features=3)

inverse_results = discover_relations(inverse_problem, inverse_data;
    methods=[PolynomialSearch(max_degree=2, standardize=true)],
    validation=KFold(k=5, seed=4))
inverse_model = select_model(inverse_results; criterion=:validation_rmse)

println(formula(inverse_model))
println(metrics(inverse_model, inverse_data))

# This assessment repeats the complete selection process inside every outer fold.
assessment = validate_discovery(inverse_problem, inverse_data;
    methods=[PolynomialSearch(max_degree=2, standardize=true)],
    outer=GroupedKFold(k=3, by=:eos, seed=5), inner=KFold(k=3, seed=6))
println(assessment.aggregate_metrics)
