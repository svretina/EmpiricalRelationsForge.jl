using SymbolicRegression

@testset "type stability" begin
    x = collect(range(1.0, 2.0; length=12))
    data = @inferred RegressionDataset(reshape(x, :, 1), @. 1 + 2x;
        features=[:x], target=:y, row_ids=101:112)
    @test eltype(data.row_ids) === Int
    @test eltype(data.eos_ids) === Missing

    sliced = @inferred EmpiricalRelationsForge.subset(data, [1, 3, 5, 7])
    @test eltype(sliced.row_ids) === Int

    linear = @inferred fit(data, LinearRegression())
    @inferred predict(linear, data)
    @inferred coefficients(linear)
    @inferred metrics(linear, data)
    report = @inferred validate(LinearRegression(), data; strategy=KFold(k=3, seed=2))
    @test eltype(report.row_ids) === Int

    basis = BasisSet(
        PowerProductTerm(:constant),
        PowerProductTerm(:x, Dict(:x => 1)),
    )
    basis_model = @inferred fit(data, LinearBasisRegression(basis))
    @inferred predict(basis_model, data)

    nonlinear_function = (X, p) -> @. p[1] + p[2] * X[:, 1]
    nonlinear_specification = NonlinearModelSpec(
        nonlinear_function, [:intercept, :slope], [[0.0, 1.0], [1.0, 1.5]];
        expression="y = intercept + slope*x",
        solver_options=(abstol=1e-10, reltol=1e-10, maxiters=100),
    )
    nonlinear = @inferred fit(data, nonlinear_specification)
    @inferred predict(nonlinear, data)

    operators = SymbolicRegression.OperatorEnum(1 => (), 2 => (+, -, *))
    options = SymbolicRegression.Options(; operators, progress=false,
        save_to_file=false)
    expression = SymbolicRegression.parse_expression("1.0 + 1.0*x";
        operators, variable_names=["x"])
    symbolic = SymbolicCandidate(expression, 1.0, 3, [:x], :y, options)
    refined = @inferred refit(symbolic, data)
    @inferred predict(refined, data)
    @inferred parameter_vector(refined)
    @inferred with_parameters(refined, [1.0, 2.0])

    problem = RelationDiscoveryProblem(target=:y, candidate_features=[:x])
    discovery = @inferred discover_relations(problem, data;
        methods=[PolynomialSearch(min_degree=1, max_degree=1)],
        validation=KFold(k=2, seed=3))
    @inferred leaderboard(discovery)
    @inferred filter_candidates(discovery; families=[:polynomial])
    @inferred predict(first(discovery.candidates), data)
    @test eltype(discovery.training_row_ids) === Int
end
