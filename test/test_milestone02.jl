using SymbolicRegression

function symbolic_candidate(expression, features, target=:y;
                            binary=[+, -, *, /, ^], unary=[sqrt, exp, log])
    options = SymbolicRegression.Options(; binary_operators=binary,
        unary_operators=unary, progress=false, save_to_file=false)
    tree = SymbolicRegression.parse_expression(expression;
        operators=options.operators, variable_names=string.(features))
    SymbolicCandidate(tree, 1.0, 1, Symbol.(features), target, options)
end

@testset "model representation and symbolic refitting" begin
    x = collect(range(0.5, 2.0; length=40))
    groups = repeat(1:4; inner=10)
    data = RegressionDataset(reshape(x, :, 1), @. 2 + 3x;
        features=[:x], target=:y, eos_ids=groups, units=Dict(:x=>"1", :y=>"1"))

    candidate = symbolic_candidate("1.0 + 1.0*x", [:x])
    @test parameter_vector(candidate) == [1.0, 1.0]
    @test parameter_names(candidate) == [:p1, :p2]
    replaced = with_parameters(candidate, [4.0, 4.0])
    @test parameter_vector(replaced) == [4.0, 4.0]
    @test parameter_vector(candidate) == [1.0, 1.0]
    @test predict(with_parameters(candidate, parameter_vector(candidate)), data) == predict(candidate, data)

    refined = refit(candidate, data)
    @test refined.method == :linear
    @test refined.status == :success
    @test coefficients(refined) ≈ [2.0, 3.0] atol=1e-12
    @test refined.structure_fingerprint == model_structure(candidate).fingerprint
    @test refined.objective < refined.original_objective

    constant_free = symbolic_candidate("x", [:x])
    fixed = refit(constant_free, RegressionDataset(reshape(x, :, 1), x;
        features=[:x], target=:y, eos_ids=groups))
    @test fixed.status == :success
    @test fixed.method == :fixed
    @test isempty(parameter_vector(fixed))

    nonlinear = symbolic_candidate("1.0*x^1.0", [:x])
    power_data = RegressionDataset(reshape(x, :, 1), @. 2.5*x^1.4;
        features=[:x], target=:y, eos_ids=groups)
    power_fit = refit(nonlinear, power_data; lower_bounds=[0.1, 0.1],
        upper_bounds=[5.0, 3.0], parameter_scale=[2.0, 1.0])
    @test power_fit.method == :nonlinear
    @test power_fit.status == :success
    @test power_fit.parameters ≈ [2.5, 1.4] rtol=1e-5

    redundant = symbolic_candidate("1.0*x + 1.0*x", [:x])
    redundant_fit = refit(redundant, data)
    @test redundant_fit.diagnostics.jacobian_rank < length(redundant_fit.parameters)

    invalid = symbolic_candidate("1.0/x", [:x])
    invalid_data = RegressionDataset(reshape([0.0, 1.0], :, 1), [0.0, 1.0];
        features=[:x], target=:y)
    @test !expression_validity(invalid, invalid_data).valid
end

@testset "comparison, uncertainty, and grouped resampling" begin
    x = collect(range(1.0, 4.0; length=24))
    groups = repeat([:A, :B, :C, :D]; inner=6)
    y = @. 1.5 - 0.4x
    data = RegressionDataset(reshape(x, :, 1), y;
        features=[:x], target=:y, eos_ids=groups)
    candidate = symbolic_candidate("1.0 + 1.0*x", [:x])
    strategy = GroupedKFold(k=2, seed=9)
    report = compare(data, [:linear=>LinearRegression(), :symbolic=>candidate]; validation=strategy)
    @test all(entry -> entry.status == :success, report.models)
    @test report.models[1].validation.folds == report.models[2].validation.folds == report.folds
    @test select_model(report).validation.aggregate_metrics.rmse < 1e-10
    @test select_model(report; policy=entries -> last(entries)).name == :symbolic

    fitted = refit(candidate, data)
    covariance = parameter_covariance(fitted, data)
    @test covariance.status == :success
    @test covariance.rank == 2
    propagated = prediction_uncertainty(fitted, covariance, reshape([2.0], :, 1))
    @test propagated.kind == :parameter_induced
    @test length(propagated.standard_error) == 1

    first_bootstrap = bootstrap(fitted, data; n=12, seed=17)
    second_bootstrap = bootstrap(fitted, data; n=12, seed=17)
    @test first_bootstrap.parameter_samples == second_bootstrap.parameter_samples
    @test first_bootstrap.unique_group_counts == second_bootstrap.unique_group_counts
    @test size(first_bootstrap.parameter_intervals) == (2, 2)
end

@testset "physical normalization and dimensions" begin
    registry = Dict(
        :length => PhysicalQuantity(:length, PhysicalDimension(length=1), "m", 10.0, (0.0, Inf)),
        :frequency => PhysicalQuantity(:frequency, PhysicalDimension(time=-1), "Hz", 100.0, (0.0, Inf)),
    )
    data = RegressionDataset(reshape([10.0, 20.0, 30.0, 40.0], :, 1),
        [200.0, 300.0, 400.0, 500.0]; features=[:length], target=:frequency,
        units=Dict(:length=>"m", :frequency=>"Hz"), eos_ids=1:4)
    normalized, normalization = normalize_physical(data, registry)
    @test normalized.X[:, 1] == [1.0, 2.0, 3.0, 4.0]
    @test normalized.y == [2.0, 3.0, 4.0, 5.0]
    physical = physical_model(fit(normalized, LinearRegression()), normalization)
    @test predict(physical, data) ≈ data.y

    candidate = symbolic_candidate("1.0/Lambda_tilde", [:Lambda_tilde], :f_peak)
    check = check_dimensions(candidate)
    @test check.valid # the fitted coefficient may carry the required frequency dimension
    candidate_data = RegressionDataset(reshape([100.0, 200.0, 300.0, 400.0], :, 1),
        [0.01, 0.005, 1 / 300, 0.0025]; features=[:Lambda_tilde], target=:f_peak)
    validity = expression_validity(candidate, candidate_data)
    @test validity.valid
    @test validity.finite_count == length(candidate_data)
end

@testset "serialization, equivalence, and candidate export" begin
    x = collect(range(1.0, 3.0; length=12))
    data = RegressionDataset(reshape(x, :, 1), @. 1 + 2x + 0.5x^2;
        features=[:x], target=:y, eos_ids=repeat(1:3; inner=4),
        units=Dict(:x=>"m", :y=>"kHz"))
    polynomial = fit(data, PolynomialRegression(degree=2, standardize=true))
    mktempdir() do directory
        artifact = joinpath(directory, "polynomial.toml")
        save_model(artifact, polynomial; metadata=Dict(:dataset_hash=>dataset_fingerprint(data)))
        restored = load_model(artifact)
        @test parameter_names(restored) == parameter_names(polynomial)
        @test parameter_vector(restored) == parameter_vector(polynomial)
        @test predict(restored, data) ≈ predict(polynomial, data)
        @test restored.units == polynomial.units

        symbolic = refit(symbolic_candidate("1.0 + 1.0*x", [:x]), data)
        symbolic_artifact = joinpath(directory, "symbolic.toml")
        save_model(symbolic_artifact, symbolic)
        restored_symbolic = load_model(symbolic_artifact)
        @test predict(restored_symbolic, data) ≈ predict(symbolic, data)
        @test model_structure(restored_symbolic).fingerprint == model_structure(symbolic).fingerprint
        @test restored_symbolic.diagnostics.jacobian_rank == symbolic.diagnostics.jacobian_rank

        invalid = joinpath(directory, "invalid.toml")
        write(invalid, "schema_version = 99\n")
        @test_throws ArgumentError load_model(invalid)

        validation = validate(PolynomialRegression(degree=2), data;
            strategy=GroupedKFold(k=3, seed=2))
        exported = export_candidate(polynomial, validation;
            path=joinpath(directory, "candidate"), applicability="Synthetic calibration range")
        @test isfile(exported.model_file)
        @test occursin("research candidate", read(exported.review_file, String))
    end

    a = symbolic_candidate("1.0 + x", [:x]; binary=[+])
    b = symbolic_candidate("x + 2.0", [:x]; binary=[+])
    assessment = equivalence(a, b)
    @test assessment.equivalent
    @test assessment.level == :canonical_structure
    catalogue = deduplicate_candidates([a, b])
    @test length(catalogue.groups) == 1

    published = PublishedPredictor("synthetic", [:x], :y, values -> 1 + 2values.x;
        expression="y = 1 + 2x", provenance=Dict(:publication=>:synthetic))
    comparison = compare(data, [:published=>published, :polynomial=>PolynomialRegression(degree=2)];
        validation=GroupedKFold(k=3, seed=2))
    @test comparison.models[1].source == :fixed_predictor
    @test comparison.models[1].status == :success
end
