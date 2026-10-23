using DataFrames
using EmpiricalRelationsForge

x1 = repeat(collect(range(-1.0, 1.0; length=8)), inner=5)
x2 = repeat(collect(range(-0.8, 0.8; length=5)), outer=8)
y = @. 1 + 2x1 + 3x2 + 4x1*x2
dataset = RegressionDataset(DataFrame(x1=x1, x2=x2, y=y);
                            target=:y, features=[:x1, :x2])
model = fit(dataset, PolynomialRegression(degree=2))
println(formula(model))
println(metrics(model, dataset))
