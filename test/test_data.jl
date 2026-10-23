@testset "dataset construction" begin
    table = DataFrame(
        id=["a", "b", "c"], eos=[:APR, :SLy, :APR], simulation=1:3,
        paper=fill(:example, 3), mass=[1.2, missing, 1.4], lambda=[400, 500, 600],
        f_peak=[2.8, 3.0, 3.2], sigma=fill(0.1, 3), weight=fill(2.0, 3),
    )
    dataset = RegressionDataset(
        table; target=:f_peak, features=[:mass, :lambda], row_id=:id,
        eos_id=:eos, simulation_id=:simulation, source_paper=:paper,
        uncertainty=:sigma, weight=:weight, units=Dict(:f_peak => "kHz"),
    )
    @test size(dataset) == (2, 2)
    @test dataset.row_ids == ["a", "c"]
    @test dataset.feature_names == [:mass, :lambda]
    @test dataset.units[:f_peak] == "kHz"
    @test_throws ArgumentError RegressionDataset(table; target=:absent, features=[:mass])

    standardizer = fit_standardizer(dataset)
    scaled = transform_features(dataset, standardizer)
    @test vec(mean(scaled.X; dims=1)) ≈ zeros(2) atol=1e-12
    @test length(scaled.transformations) == 1
    @test scaled.transformations[1].parameters[:means] == standardizer.means

    duplicate_ids = DataFrame(id=[1, 1], x=[1.0, 2.0], y=[2.0, 3.0])
    @test_throws ArgumentError RegressionDataset(duplicate_ids; target=:y, features=[:x], row_id=:id)
end

@testset "measurement provenance and dependency checks" begin
    table = DataFrame(id=1:3, x=1.0:3.0, y=2.0:4.0,
                      kind=[:measured, :predicted, :unreviewed], simulation=[:a, :a, :b])
    measured = RegressionDataset(table; target=:y, features=[:x], row_id=:id,
                                 simulation_id=:simulation, observation_kind=:kind)
    @test length(measured) == 1
    all_data = RegressionDataset(table; target=:y, features=[:x], row_id=:id,
                                 simulation_id=:simulation, observation_kind=:kind,
                                 include_kinds=(:measured, :predicted, :unreviewed))
    @test all_data.observation_kinds == [:measured, :predicted, :unreviewed]
    @test duplicate_identifiers(all_data) == Dict(:a => [1, 2])
    @test !isempty(physical_dependency_warnings([:mass_1, :mass_2, :M_chirp]))
end

@testset "provider boundary" begin
    @test_throws ArgumentError load_dataset(:not_a_frequency)
    @test_throws ArgumentError load_dataset(:f_peak; features=[:mass])
end
