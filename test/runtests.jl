using Aqua
using DataFrames
using EmpiricalRelationsForge
using Random
using Statistics
using Test

@testset "EmpiricalRelationsForge" begin
    include("test_data.jl")
    include("test_splitting.jl")
    Aqua.test_all(EmpiricalRelationsForge; ambiguities=false)
end
