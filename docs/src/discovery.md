# Relation discovery

## Scientific problem

An empirical-relation search has three independent choices: the target quantity, the
available explanatory variables, and the admissible mathematical structures.
`RelationDiscoveryProblem` records those choices without assuming that the target is a
frequency or that EOS generalization is the objective.

```julia
problem = RelationDiscoveryProblem(
    target=:f_peak,
    candidate_features=[:M_chirp, :q, :Lambda_tilde, :R_1_6],
    mandatory_features=[:M_chirp],
    forbidden_features=Symbol[],
    min_features=1,
    max_features=3,
    transformations=[
        FeatureTransformation(:Lambda_tilde, :inverse_sqrt;
            name=:inverse_sqrt_Lambda),
    ],
    binary_operators=[:+, :-, :*, :/],
    unary_operators=[:sqrt, :log, :exp],
    max_expression_complexity=25,
    domains=Dict(:Lambda_tilde=>(1.0, Inf)),
    units=Dict(:f_peak=>"kHz", :M_chirp=>"M_sun"),
    objective=:validation_rmse,
)
```

Mandatory and forbidden variables constrain the subset search; correlated physical
variables are not removed automatically. `physical_dependency_warnings` distinguishes
known deterministic dependencies from an empirical correlation and leaves the final
scientific choice to the analysis.

`VariableDescriptor` labels inputs by role and accessibility. Candidate lists can then
be restricted to relations using inspiral-accessible quantities, postmerger
observables, EOS properties, or derived quantities.

```julia
variables = [
    VariableDescriptor(:M_chirp; role=:inspiral, accessible=true),
    VariableDescriptor(:f_peak; role=:postmerger, accessible=true),
    VariableDescriptor(:R_1_6; role=:eos, accessible=false),
]

observable_candidates = filter_candidates(results; accessible_only=true)
```

## Subsets and model families

`discover_relations` supports exhaustive enumeration, forward selection, backward
elimination, penalized enumeration, and symbolic-regression-driven selection through
`subset_strategy=:exhaustive`, `:forward`, `:backward`, `:penalized`, or `:symbolic`.
Exhaustive enumeration is preferable when the feature pool is small because it exposes
the complete accuracy--variable-count tradeoff.

```julia
results = discover_relations(
    problem,
    dataset;
    methods=[
        PolynomialSearch(min_degree=1, max_degree=3),
        SymbolicSearch(problem; niterations=100, seed=17),
    ],
    subset_strategy=:exhaustive,
    validation=KFold(k=5, seed=17),
)
```

`PolynomialSearch` fits linear and interacting multivariate polynomials. Fixed
transformations provide polynomial models in inverse, fractional-power, logarithmic,
or exponential coordinates. `LinearBasisRegression` supplies auditable power products,
rational terms through negative powers, and dimensionless physical combinations. A
`SymbolicSearch(problem)` derives its operator grammar and complexity ceiling directly
from the problem and permits fitted exponents and unrestricted compositions.

For example, a dimensionless compactness-like family can be included explicitly:

```julia
basis = BasisSet(
    PowerProductTerm(:constant),
    PowerProductTerm(:compactness, Dict(:M_tot=>1, :R_1_6=>-1);
        label="M_tot/R_1_6"),
)

results = discover_relations(problem, dataset;
    methods=[PolynomialSearch(max_degree=2), LinearBasisRegression(basis)])
```

Named bases with fixed powers use QR or SVD. Symbolic candidates have their constants
refitted without changing the operator tree; affine parameterizations use QR/SVD and
nonlinear parameterizations use NonlinearSolve.jl. The original and refined constants,
solver status, Jacobian rank, and conditioning remain available through the fitted
model and its diagnostics.

## Ranking and Pareto selection

```julia
table = leaderboard(results)
frontier = pareto_frontier(results)
best = select_model(results; criterion=:validation_rmse)
```

The DataFrame leaderboard contains the target, selected variables, formula, parameter
count, expression complexity, training and validation errors, conditioning, and the
configured score. Safely equivalent structures are grouped in the default table; use
`leaderboard(results; unique=false)` to inspect every original candidate. The complete
candidate collection is retained in `results.candidates`. `pareto_frontier`
removes candidates dominated simultaneously in predictive error, complexity, and input
count; it does not assert a unique scientifically preferred relation.

For `objective=:penalized`, the ranking score is

```math
J(F)=\operatorname{RMSE}_{\mathrm{validation}}(F)
 + \lambda_C C(F) + \lambda_V N_{\mathrm{variables}}(F).
```

The penalties are optional. Report the unpenalized metrics and frontier alongside any
penalized ranking. Coefficient covariance, prediction uncertainty, grouped bootstrap,
symbolic equivalence, and dimensional checks operate on selected candidates through
the same interfaces described in the corresponding manual sections.

## Validation without selection leakage

The validation result attached to each candidate is suitable for comparing candidates
during a search. It is not an untouched estimate of the performance of the complete
selection procedure. Use `validate_discovery` when reporting held-out accuracy:

```julia
assessment = validate_discovery(
    problem,
    dataset;
    methods=[PolynomialSearch(max_degree=3)],
    outer=LeaveOneGroupOut(by=:simulation),
    inner=KFold(k=5, seed=8),
    subset_strategy=:forward,
    seed=8,
)

assessment.aggregate_metrics
assessment.selected_features
```

Every outer training partition repeats feature selection, structure search, coefficient
fitting, and inner validation. Available outer strategies include random holdout,
ordinary K-fold, grouped K-fold, leave-one-EOS-out, and generic leave-one-group-out by
EOS, simulation family, or source paper. The strategy must match the intended
generalization question; no EOS-specific split is imposed by the discovery engine.

## Forward and inverse relations

Forward and inverse relations are separate fits. If one dataset has target `:f_peak`
and another has target `:R_1_6`, define two problems and run the same pipeline on each.
Algebraic rearrangement of the selected forward equation is not used as the inverse
model. Consequently, inverse residuals and uncertainty are expressed directly in the
stellar or EOS target unit.

Multiple observed frequencies may be included as inverse inputs. Compare a restricted
input problem, such as `[:M_chirp, :f_peak]`, with an expanded problem containing
secondary frequencies to quantify their incremental predictive information.

## Reproducibility and published baselines

```julia
save_discovery("campaign", results)
restored = load_discovery("campaign")
```

The versioned campaign artifact stores the problem, target and feature pool,
transformations, candidate models and coefficients, exact folds and row identifiers,
metrics, seed, backend configuration, package versions, provenance, and dataset hash.
Models are reconstructed from explicit data structures rather than serialized code.
Arbitrary Julia closures remain intentionally non-portable.

A `PublishedPredictor` may be included in `methods` when its target and complete input
set match the problem. It is evaluated on the same rows and folds as new candidates,
but its published calibration and evaluation on additional simulations should be
reported as distinct exercises. Accessibility labels expose when a new candidate gains
accuracy by requiring an EOS-dependent quantity unavailable to an observational use
case.

The complete synthetic BNS workflow is available in
[`examples/relation_discovery.jl`](https://github.com/svretina/EmpiricalRelationsForge.jl/blob/master/examples/relation_discovery.jl).
