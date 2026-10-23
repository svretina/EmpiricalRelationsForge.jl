@testset "reproducible and grouped splitting" begin
    table = DataFrame(
        id=1:8, eos=repeat([:A, :B, :C, :D], inner=2),
        x=collect(1.0:8.0), y=collect(2.0:2.0:16.0),
    )
    dataset = RegressionDataset(table; target=:y, features=[:x], row_id=:id, eos_id=:eos)
    a = random_split(dataset; rng=MersenneTwister(42))
    b = random_split(dataset; rng=MersenneTwister(42))
    @test a.train == b.train
    @test a.test == b.test
    @test isempty(intersect(a.train, a.test))

    folds = grouped_kfold(dataset; by=:eos, k=2, rng=MersenneTwister(7))
    @test length(folds) == 2
    for fold in folds
        @test isempty(intersect(dataset.eos_ids[fold.train], dataset.eos_ids[fold.test]))
    end
    loo = leave_one_group_out(dataset; by=:eos)
    @test length(loo) == 4
    @test all(fold -> length(unique(dataset.eos_ids[fold.test])) == 1, loo)
end
