using EmpiricalRelationsForge

x = collect(range(-1.0, 1.0; length=40))
dataset = RegressionDataset(reshape(x, :, 1), @. 1 + 2x;
                            features=[:x], target=:y)
result = discover(dataset, SymbolicSearch(niterations=10, seed=42))
foreach(println, leaderboard(result))
