@testset "relation discovery problems and subset search" begin
    x = collect(range(1.0, 3.0; length=30))
    z = collect(range(2.0, 5.0; length=30)) .+ 0.1 .* sin.(1:30)
    noise = cos.(1:30)
    y = @. 2 + 3x + 4 / sqrt(z)
    data = RegressionDataset(hcat(x, z, noise), y;
        features=[:x, :z, :noise], target=:response, row_ids=101:130,
        units=Dict(:x=>"1", :z=>"1", :noise=>"1", :response=>"kHz"))
    problem = RelationDiscoveryProblem(
        target=:response,
        candidate_features=[:x, :z, :noise],
        mandatory_features=[:x],
        max_features=2,
        transformations=[FeatureTransformation(:z, :inverse_sqrt; name=:z_inverse_sqrt)],
        domains=Dict(:z=>(1.0, 6.0)),
        variables=[
            VariableDescriptor(:x; role=:inspiral, accessible=true),
            VariableDescriptor(:z; role=:eos, accessible=false),
            VariableDescriptor(:noise; role=:inspiral, accessible=true),
        ],
    )

    result = discover_relations(problem, data;
        methods=[PolynomialSearch(min_degree=1, max_degree=2)],
        validation=KFold(k=3, seed=7))
    table = leaderboard(result)
    @test table isa DataFrame
    @test nrow(table) == length(result.candidates)
    @test all(:x in candidate.selected_features for candidate in result.candidates)
    @test first(result.candidates).selected_features == [:x, :z]
    @test first(result.candidates).evaluation.validation.rmse < 1e-10
    @test predict(first(result.candidates), data) ≈ data.y atol=1e-10
    @test !isempty(pareto_frontier(result))
    @test all(candidate.family == :polynomial for candidate in
        filter_candidates(result; families=[:polynomial]))
    @test all(:z ∉ candidate.selected_features for candidate in
        filter_candidates(result; accessible_only=true))

    forward = discover_relations(problem, data;
        methods=[PolynomialSearch(min_degree=1, max_degree=1)],
        subset_strategy=:forward, validation=KFold(k=2, seed=3))
    backward = discover_relations(problem, data;
        methods=[PolynomialSearch(min_degree=1, max_degree=1)],
        subset_strategy=:backward, validation=KFold(k=2, seed=3))
    @test !isempty(forward.candidates)
    @test !isempty(backward.candidates)
    @test_throws ArgumentError RelationDiscoveryProblem(target=:response,
        candidate_features=[:x], mandatory_features=[:x], forbidden_features=[:x])
    @test_throws ArgumentError RelationDiscoveryProblem(target=:response,
        candidate_features=[:x], transformations=[FeatureTransformation(:z, :log)])
    mismatched_units = RelationDiscoveryProblem(target=:response,
        candidate_features=[:x], units=Dict(:response=>"Hz"))
    @test_throws ArgumentError discover_relations(mismatched_units, data)
    @test SymbolicSearch(problem; niterations=1, seed=2).niterations == 1
    @test LeaveOneGroupOut(by=:simulation).by == :simulation
end

@testset "model families, inverse targets, and discovery validation" begin
    mass = collect(range(1.0, 2.0; length=24))
    radius = collect(range(10.0, 14.0; length=24)) .+ 0.05 .* sin.(1:24)
    frequency = @. 0.5 + 2mass / radius
    forward_data = RegressionDataset(hcat(mass, radius), frequency;
        features=[:mass, :radius], target=:frequency, row_ids=1:24)
    forward_problem = RelationDiscoveryProblem(target=:frequency,
        candidate_features=[:mass, :radius], min_features=2, max_features=2)
    basis = BasisSet(
        PowerProductTerm(:constant),
        PowerProductTerm(:compactness, Dict(:mass=>1, :radius=>-1);
            label="mass/radius"),
    )
    basis_result = discover_relations(forward_problem, forward_data;
        methods=[LinearBasisRegression(basis)], validation=KFold(k=3, seed=2))
    @test only(basis_result.candidates).family == :linear_basis
    @test only(basis_result.candidates).evaluation.validation.rmse < 1e-10

    inferred_radius = @. 2mass / (frequency - 0.5)
    inverse_data = RegressionDataset(hcat(frequency, mass), inferred_radius;
        features=[:frequency, :mass], target=:radius, row_ids=1:24)
    inverse_problem = RelationDiscoveryProblem(target=:radius,
        candidate_features=[:frequency, :mass], min_features=1, max_features=2)
    inverse_result = discover_relations(inverse_problem, inverse_data;
        methods=[PolynomialSearch(min_degree=1, max_degree=2)],
        validation=KFold(k=3, seed=4))
    @test inverse_result.problem.target == :radius
    @test all(candidate.model.target_name == :radius for candidate in inverse_result.candidates)

    nested = validate_discovery(forward_problem, forward_data;
        methods=[PolynomialSearch(min_degree=1, max_degree=2)],
        outer=KFold(k=3, seed=11), inner=KFold(k=2, seed=12))
    @test length(nested.predictions) == length(forward_data)
    @test length(nested.selected_formulas) == 3
    @test nested.aggregate_metrics.rmse >= 0

    symbolic_x = collect(range(1.0, 2.0; length=16))
    symbolic_data = RegressionDataset(reshape(symbolic_x, :, 1), symbolic_x;
        features=[:x], target=:y)
    symbolic_problem = RelationDiscoveryProblem(target=:y,
        candidate_features=[:x], max_expression_complexity=8)
    symbolic_result = discover_relations(symbolic_problem, symbolic_data;
        methods=[SymbolicSearch(symbolic_problem; niterations=2, seed=3,
            parallelism=:serial)], subset_strategy=:symbolic,
        validation=KFold(k=2, seed=3))
    @test !isempty(symbolic_result.candidates)
    @test all(candidate.family == :symbolic for candidate in symbolic_result.candidates)
end

@testset "discovery serialization" begin
    x = collect(1.0:12.0)
    data = RegressionDataset(reshape(x, :, 1), @. 1 + 2x;
        features=[:x], target=:y, row_ids=21:32)
    problem = RelationDiscoveryProblem(target=:y, candidate_features=[:x])
    result = discover_relations(problem, data;
        methods=[PolynomialSearch(min_degree=1, max_degree=2)],
        validation=KFold(k=3, seed=5), seed=5)
    published = PublishedPredictor("synthetic-linear", [:x], :y,
        values -> 1 + 2values.x; expression="y = 1 + 2x")
    comparison = discover_relations(problem, data;
        methods=[PolynomialSearch(min_degree=1, max_degree=1), published],
        validation=KFold(k=3, seed=5))
    @test any(candidate.family == :published for candidate in comparison.candidates)
    mktempdir() do directory
        path = joinpath(directory, "campaign")
        save_discovery(path, result)
        restored = load_discovery(path)
        @test restored.problem.target == problem.target
        @test restored.problem.candidate_features == problem.candidate_features
        @test restored.dataset_hash == result.dataset_hash
        @test restored.training_row_ids == result.training_row_ids
        @test restored.pareto_indices == result.pareto_indices
        @test length(restored.candidates) == length(result.candidates)
        @test predict(first(restored.candidates), data) ≈
              predict(first(result.candidates), data)
        @test_throws ArgumentError save_discovery(path, result)
    end
end
