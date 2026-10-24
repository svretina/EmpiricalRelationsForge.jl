"""Scientific role and observational accessibility of one variable."""
struct VariableDescriptor
    name::Symbol
    role::Symbol
    accessible::Bool
    description::String
end

function VariableDescriptor(name::Symbol; role::Symbol=:unspecified,
                            accessible::Bool=false, description::AbstractString="")
    role in (:inspiral, :postmerger, :eos, :derived, :unspecified) ||
        throw(ArgumentError("unknown variable role :$role"))
    VariableDescriptor(name, role, accessible, String(description))
end

"""A fixed, non-statistical transformation available to discovery methods."""
struct FeatureTransformation
    name::Symbol
    source::Symbol
    kind::Symbol
    parameter::Float64
end

function FeatureTransformation(source::Symbol, kind::Symbol; name=nothing, parameter::Real=1)
    kind in (:identity, :power, :inverse, :sqrt, :inverse_sqrt, :log, :exp) ||
        throw(ArgumentError("unsupported feature transformation :$kind"))
    generated = isnothing(name) ? Symbol(source, "__", kind,
        kind == :power ? "_$(Float64(parameter))" : "") : Symbol(name)
    FeatureTransformation(generated, source, kind, Float64(parameter))
end

function _transform_values(transform::FeatureTransformation, values)
    output = if transform.kind == :identity
        values
    elseif transform.kind == :power
        values .^ transform.parameter
    elseif transform.kind == :inverse
        inv.(values)
    elseif transform.kind == :sqrt
        sqrt.(values)
    elseif transform.kind == :inverse_sqrt
        inv.(sqrt.(values))
    elseif transform.kind == :log
        log.(values)
    else
        exp.(values)
    end
    all(isfinite, output) || throw(DomainError(values,
        "transformation $(transform.name) is invalid on the supplied domain"))
    Float64.(output)
end

"""Specification of a target-independent empirical-relation discovery campaign."""
struct RelationDiscoveryProblem
    target::Symbol
    candidate_features::Vector{Symbol}
    mandatory_features::Vector{Symbol}
    forbidden_features::Vector{Symbol}
    min_features::Int
    max_features::Int
    transformations::Vector{FeatureTransformation}
    binary_operators::Vector{Symbol}
    unary_operators::Vector{Symbol}
    max_expression_complexity::Int
    domains::Dict{Symbol,Tuple{Float64,Float64}}
    units::Dict{Symbol,String}
    variables::Dict{Symbol,VariableDescriptor}
    objective::Symbol
    complexity_penalty::Float64
    variable_penalty::Float64
end

function RelationDiscoveryProblem(; target, candidate_features,
        mandatory_features=Symbol[], forbidden_features=Symbol[], min_features::Integer=1,
        max_features::Integer=length(candidate_features), transformations=FeatureTransformation[],
        binary_operators=[:+, :-, :*, :/], unary_operators=Symbol[],
        max_expression_complexity::Integer=20, domains=Dict(), units=Dict(),
        variables=VariableDescriptor[], objective::Symbol=:validation_rmse,
        complexity_penalty::Real=0, variable_penalty::Real=0)
    features = unique(Symbol.(candidate_features))
    mandatory = unique(Symbol.(mandatory_features))
    forbidden = unique(Symbol.(forbidden_features))
    isempty(features) && throw(ArgumentError("candidate_features cannot be empty"))
    all(in(features), mandatory) || throw(ArgumentError("mandatory features must belong to candidate_features"))
    all(in(features), forbidden) || throw(ArgumentError("forbidden features must belong to candidate_features"))
    isempty(intersect(mandatory, forbidden)) || throw(ArgumentError("a mandatory feature cannot be forbidden"))
    available = setdiff(features, forbidden)
    1 <= min_features <= max_features <= length(available) ||
        throw(ArgumentError("feature-count bounds are inconsistent with the available features"))
    length(mandatory) <= max_features || throw(ArgumentError("too many mandatory features"))
    max_expression_complexity > 0 || throw(ArgumentError("max_expression_complexity must be positive"))
    complexity_penalty >= 0 && variable_penalty >= 0 || throw(ArgumentError("penalties must be nonnegative"))
    objective in (:training_rmse, :validation_rmse, :penalized) ||
        throw(ArgumentError("unsupported discovery objective :$objective"))
    descriptors = Dict(descriptor.name => descriptor for descriptor in variables)
    domain_map = Dict(Symbol(k) => (Float64(v[1]), Float64(v[2])) for (k, v) in domains)
    unit_map = Dict(Symbol(k) => String(v) for (k, v) in units)
    all(bounds -> !any(isnan, bounds) && first(bounds) <= last(bounds), values(domain_map)) ||
        throw(ArgumentError("physical-domain bounds must be ordered and non-NaN"))
    transformation_names = getfield.(transformations, :name)
    allunique(transformation_names) || throw(ArgumentError("transformation names must be unique"))
    isempty(intersect(features, transformation_names)) || throw(ArgumentError(
        "transformation names must not duplicate candidate-feature names"))
    all(transform -> transform.source in features, transformations) || throw(ArgumentError(
        "every transformation source must belong to candidate_features"))
    RelationDiscoveryProblem(Symbol(target), features, mandatory, forbidden,
        Int(min_features), Int(max_features), collect(transformations),
        Symbol.(binary_operators), Symbol.(unary_operators), Int(max_expression_complexity),
        domain_map, unit_map, descriptors, objective, Float64(complexity_penalty),
        Float64(variable_penalty))
end

const _DISCOVERY_BINARY_OPERATORS = Dict{Symbol,Function}(
    :+ => +, :- => -, :* => *, :/ => /, :^ => ^)
const _DISCOVERY_UNARY_OPERATORS = Dict{Symbol,Function}(
    :sqrt => sqrt, :log => log, :exp => exp, :sin => sin, :cos => cos, :abs => abs)

"""Construct a symbolic search using the grammar and complexity bound of a discovery problem."""
function SymbolicSearch(problem::RelationDiscoveryProblem; niterations::Integer=100,
        parallelism::Symbol=:multithreading, seed::Union{Nothing,Integer}=nothing,
        weighting::Symbol=:none, dimensional_constraints::Bool=false,
        backend_options::NamedTuple=(;), backend_kwargs::NamedTuple=(;))
    unknown_binary = setdiff(problem.binary_operators, collect(keys(_DISCOVERY_BINARY_OPERATORS)))
    unknown_unary = setdiff(problem.unary_operators, collect(keys(_DISCOVERY_UNARY_OPERATORS)))
    isempty(unknown_binary) || throw(ArgumentError("unsupported binary operators: $unknown_binary"))
    isempty(unknown_unary) || throw(ArgumentError("unsupported unary operators: $unknown_unary"))
    SymbolicSearch(;
        binary_operators=[_DISCOVERY_BINARY_OPERATORS[op] for op in problem.binary_operators],
        unary_operators=[_DISCOVERY_UNARY_OPERATORS[op] for op in problem.unary_operators],
        max_complexity=problem.max_expression_complexity, niterations, parallelism, seed,
        weighting, dimensional_constraints, backend_options, backend_kwargs)
end

"""Polynomial-family search configuration used by `discover_relations`."""
struct PolynomialSearch
    min_degree::Int
    max_degree::Int
    interactions::Bool
    weighting::Symbol
    standardize::Bool
    solver::Symbol
    include_transformations::Bool
end

function PolynomialSearch(; min_degree::Integer=1, max_degree::Integer=3,
        interactions::Bool=true, weighting::Symbol=:none, standardize::Bool=false,
        solver::Symbol=:qr, include_transformations::Bool=true)
    1 <= min_degree <= max_degree || throw(ArgumentError("polynomial degree bounds are invalid"))
    _check_weighting(weighting)
    solver in (:qr, :svd) || throw(ArgumentError("solver must be :qr or :svd"))
    PolynomialSearch(Int(min_degree), Int(max_degree), interactions, weighting,
        standardize, solver, include_transformations)
end
