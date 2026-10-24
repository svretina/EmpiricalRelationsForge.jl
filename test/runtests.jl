using Aqua
using DataFrames
using EmpiricalRelationsForge
using Random
using Statistics
using Test

@testset "EmpiricalRelationsForge" begin
    include("test_data.jl")
    include("test_splitting.jl")
    include("test_regression.jl")
    include("test_validation.jl")
    include("test_symbolic.jl")
    Aqua.test_all(EmpiricalRelationsForge)
end
