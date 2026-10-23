using Documenter
using EmpiricalRelationsForge

makedocs(
    sitename="EmpiricalRelationsForge.jl",
    modules=[EmpiricalRelationsForge],
    pages=["Home" => "index.md", "Architecture" => "architecture.md", "API" => "api.md"],
    checkdocs=:exports,
    remotes=nothing,
)
