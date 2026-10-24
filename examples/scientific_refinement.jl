using EmpiricalRelationsForge
using Random
using SymbolicRegression

rng = MersenneTwister(42)
groups = repeat(Symbol.('A':'F'); inner=12)
x1 = 0.8 .+ 0.8rand(rng, length(groups))
x2 = 20 .+ 80rand(rng, length(groups))
noise = 0.01randn(rng, length(groups))
y = @. 1.2 + 0.7x1 - 4 / sqrt(x2) + noise
data = RegressionDataset(hcat(x1, x2), y; features=[:x1, :x2], target=:y,
    eos_ids=groups, row_ids=1:length(y), units=Dict(:x1=>"1", :x2=>"1", :y=>"1"))

basis = BasisSet(
    PowerProductTerm(:constant),
    PowerProductTerm(:x1, Dict(:x1=>1.0); label="x1"),
    PowerProductTerm(:inverse_sqrt_x2, Dict(:x2=>-0.5); label="x2^(-1/2)"),
)
classical = fit(data, LinearBasisRegression(basis))

options = SymbolicRegression.Options(binary_operators=[+, -, *, /],
    unary_operators=[sqrt], progress=false, save_to_file=false)
expression = SymbolicRegression.parse_expression("1.0 + 1.0*x1 + 1.0/sqrt(x2)";
    operators=options.operators, variable_names=["x1", "x2"])
candidate = SymbolicCandidate(expression, Inf, 8, [:x1, :x2], :y, options)
refined = refit(candidate, data)

comparison = compare(data, [:basis=>LinearBasisRegression(basis), :symbolic=>candidate];
    validation=LeaveOneEOSOut())
covariance = parameter_covariance(refined, data)
resampling = bootstrap(refined, data; group=:eos, n=100, seed=42)

registry = Dict(name=>PhysicalQuantity(name, PhysicalDimension(), "1", 1.0, (0.0, Inf))
    for name in (:x1, :x2, :y))
dimension_check = check_dimensions(candidate, registry)

output = joinpath(@__DIR__, "scientific_refinement_output")
mkpath(output)
save_model(joinpath(output, "model.toml"), refined;
    metadata=Dict(:dataset_hash=>dataset_fingerprint(data), :seed=>42))
restored = load_model(joinpath(output, "model.toml"))
@assert predict(restored, data) ≈ predict(refined, data)
export_candidate(refined, comparison.models[2].validation; path=joinpath(output, "candidate"),
    overwrite=true, uncertainty=resampling.parameter_intervals,
    applicability="Synthetic domain: x1 in [0.8, 1.6], x2 in [20, 100]")

if get(ENV, "ERF_RUN_SYMBOLIC_SEARCH", "false") == "true"
    search = SymbolicSearch(binary_operators=[+, -, *, /], unary_operators=[sqrt],
        max_complexity=14, niterations=50, seed=42)
    discovery = discover(data, search)
    println(leaderboard(discovery))
end

println((classical=coefficients(classical), symbolic=coefficients(refined),
    cv_rmse=comparison.models[2].validation.aggregate_metrics.rmse,
    covariance_status=covariance.status, dimensional=dimension_check.valid))
