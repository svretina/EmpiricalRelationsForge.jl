using Documenter
using EmpiricalRelationsForge

makedocs(
    sitename="EmpiricalRelationsForge.jl",
    format=Documenter.HTML(repolink="https://github.com/svretina/EmpiricalRelationsForge.jl"),
    modules=[EmpiricalRelationsForge],
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Scientific data" => "data.md",
        "Regression theory" => "regression.md",
        "Nonlinear fitting" => "nonlinear.md",
        "Symbolic discovery" => "symbolic.md",
        "Validation and metrics" => "validation.md",
        "Scientific conventions" => "scientific_conventions.md",
        "Backend decision" => "backend_decision.md",
        "Reproducibility" => "reproducibility.md",
        "Architecture" => "architecture.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
    remotes=nothing,
)

deploydocs(
    repo="github.com/svretina/EmpiricalRelationsForge.jl.git",
    devbranch="master",
    push_preview=true,
)
