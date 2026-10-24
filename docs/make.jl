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
        "Relation discovery" => "discovery.md",
        "Structure refitting" => "refinement.md",
        "Model comparison" => "comparison.md",
        "Uncertainty" => "uncertainty.md",
        "Physical constraints" => "physics.md",
        "Serialization" => "serialization.md",
        "Candidate equivalence" => "equivalence.md",
        "Published relations and export" => "integration.md",
        "Vretinaris2020 benchmark" => "vretinaris2020.md",
        "Validation and metrics" => "validation.md",
        "Scientific conventions" => "scientific_conventions.md",
        "Backend decision" => "backend_decision.md",
        "Reproducibility" => "reproducibility.md",
        "Architecture" => "architecture.md",
        "Milestone 02 report" => "milestone02.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
    remotes=nothing,
)
