@testset "symbolic adapter" begin
    x = collect(range(-1.0, 1.0; length=20))
    y = @. 1 + 2x
    data = RegressionDataset(reshape(x, :, 1), y; features=[:x], target=:y)
    search = SymbolicSearch(
        binary_operators=[+, -, *], max_complexity=8, niterations=1,
        parallelism=:serial, seed=12,
        backend_options=(populations=2, population_size=10,
                         ncycles_per_iteration=20, tournament_selection_n=5,
                         deterministic=true),
    )
    result = discover(data, search)
    @test !isempty(result.candidates)
    @test result.feature_names == [:x]
    @test result.backend_version >= v"2"
    candidate = best_model(result)
    @test length(predict(candidate, data)) == length(data)
    @test !isempty(formula(candidate))
    @test length(leaderboard(result)) == length(result.candidates)
end
