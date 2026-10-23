@testset "cross-validation refits preprocessing" begin
    eos = repeat([:A, :B, :C, :D], inner=5)
    x = collect(1.0:20.0)
    y = @. 1 + 2x
    data = RegressionDataset(DataFrame(id=1:20, eos=eos, x=x, y=y); target=:y,
                             features=[:x], row_id=:id, eos_id=:eos)
    report = validate(LinearRegression(standardize=true), data;
                      strategy=LeaveOneEOSOut())
    @test length(report.folds) == 4
    @test report.aggregate_metrics.rmse < 1e-10
    @test Set(report.row_ids) == Set(data.row_ids)
    @test all(fold -> isempty(intersect(data.eos_ids[fold.train], data.eos_ids[fold.test])), report.folds)

    random_report = validate(LinearRegression(), data;
                             strategy=RandomHoldout(test_fraction=0.25, seed=3))
    @test length(random_report.predictions) == 5
end
