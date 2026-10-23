@testset "linear scientific bases" begin
    x1 = collect(range(1.0, 2.0; length=30))
    x2 = collect(range(2.0, 8.0; length=30))

    # Problem A: fixed nonlinear predictor transformation, linear coefficients.
    y = @. 1.5 + 2.0 * x1 - 0.7 * x2^(-0.5)
    data = RegressionDataset(DataFrame(id=1:30, x1=x1, x2=x2, y=y); target=:y,
                             features=[:x1, :x2], row_id=:id)
    basis = BasisSet(
        PowerProductTerm(:constant),
        PowerProductTerm(:x1, Dict(:x1 => 1); label="x1"),
        PowerProductTerm(:inverse_sqrt_x2, Dict(:x2 => -0.5); label="x2^(-1/2)"),
    )
    model = fit(data, LinearBasisRegression(basis; solver=:qr))
    direct = hcat(ones(30), x1, x2 .^ -0.5) \ y
    @test coefficients(model) ≈ direct atol=1e-11
    @test predict(model, data) ≈ y atol=1e-11
    @test model.rank == 3

    # Problem B: multivariate polynomial with deterministic mixed-term order.
    polynomial_x1 = repeat(collect(range(-1.0, 1.0; length=6)), inner=5)
    polynomial_x2 = repeat(collect(range(-0.8, 0.8; length=5)), outer=6)
    ypoly = @. 1 + 2polynomial_x1 + 3polynomial_x2 +
                4polynomial_x1*polynomial_x2 + 5polynomial_x2^2
    polynomial_data = RegressionDataset(hcat(polynomial_x1, polynomial_x2), ypoly;
                                        features=[:x1, :x2], target=:y)
    polynomial = fit(polynomial_data, PolynomialRegression(degree=2))
    @test polynomial.term_names == ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
    @test coefficients(polynomial) ≈ [1, 2, 3, 0, 4, 5] atol=1e-10
    @test predict(polynomial, polynomial_data) ≈ ypoly atol=1e-9

    # A transformed target can still optimize error in the original observable.
    transformed = fit(data, LinearBasisRegression(basis;
        target_transform=FeatureScaleTarget(:x1), loss_space=:original))
    @test transformed.method.loss_space == :original
end

@testset "weighted and ill-conditioned fitting" begin
    x = collect(range(-2.0, 2.0; length=40))
    sigma = @. 0.05 + 0.1 * (x + 2) / 4
    noise = @. sigma * sin(3x)
    y = @. 2 + 3x + noise
    data = RegressionDataset(reshape(x, :, 1), y; features=[:x], target=:y,
                             uncertainties=sigma)
    weighted = fit(data, LinearRegression(weighting=:inverse_variance))
    A = hcat(ones(length(x)), x)
    rootweights = 1 ./ sigma
    direct = (A .* rootweights) \ (y .* rootweights)
    @test coefficients(weighted) ≈ direct atol=1e-11
    @test weighted.objective ≈ sum(abs2, rootweights .* (predict(weighted, data) .- y)) atol=1e-9

    xlarge = 1e9 .+ collect(1.0:20.0)
    ill_data = RegressionDataset(hcat(xlarge, xlarge .+ 1e-5 .* x[1:20]), 2 .+ 0.5 .* xlarge;
                                 features=[:large, :dependent], target=:y)
    raw = @test_logs (:warn, r"ill-conditioned") match_mode=:any fit(
        ill_data, LinearRegression(condition_warning=1e5))
    scaled = fit(ill_data, LinearRegression(standardize=true, solver=:svd))
    @test scaled.condition_number < raw.condition_number
end

@testset "nonlinear least squares" begin
    x1 = collect(range(1.0, 2.0; length=50))
    x2 = collect(range(2.0, 5.0; length=50))
    truth = [0.8, 1.7, 1.2, -0.4]
    y = @. truth[1] + truth[2] * x1^truth[3] * x2^truth[4]
    data = RegressionDataset(hcat(x1, x2), y; features=[:x1, :x2], target=:y)
    model_function = (X, p) -> @. p[1] + p[2] * X[:, 1]^p[3] * X[:, 2]^p[4]
    specification = NonlinearModelSpec(
        model_function, [:a0, :a1, :a2, :a3],
        [[0.5, 1.5, 1.0, -0.2], [1.0, 2.0, 0.8, -0.6]];
        expression="y = a0 + a1*x1^a2*x2^a3",
        solver_options=(abstol=1e-12, reltol=1e-12, maxiters=500),
    )
    model = fit(data, specification)
    @test model.converged
    @test model.attempted_initial_guesses == 2
    @test predict(model, data) ≈ y atol=1e-7
    @test model.solver_version >= v"4"
end

@testset "metrics" begin
    report = metrics([1.0, 2.5, 2.5], [1.0, 2.0, 3.0])
    @test report.mae ≈ 1 / 3
    @test report.mean_signed_residual == 0
    @test ismissing(metrics([1.0, 1.0], [0.0, 1.0]).mean_absolute_relative_error)
end

@testset "portable metadata" begin
    data = RegressionDataset(reshape(collect(1.0:5.0), :, 1), collect(3.0:2.0:11.0);
                             features=[:x], target=:y, row_ids=11:15)
    fitted = fit(data, LinearRegression())
    record = model_result(fitted, data; reproducibility=Dict(:seed => 4))
    path = tempname() * ".toml"
    @test save_metadata(path, record) == path
    contents = read(path, String)
    @test occursin("julia_version", contents)
    @test occursin("coefficients", contents)
end
