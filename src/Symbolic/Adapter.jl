"""Configuration for a thin SymbolicRegression.jl equation-search adapter."""
struct SymbolicSearch{O,K<:NamedTuple}
    options::O
    niterations::Int
    parallelism::Symbol
    seed::Union{Nothing,Int}
    weighting::Symbol
    backend_kwargs::K
end

function SymbolicSearch(; binary_operators=[+, -, *, /], unary_operators=[],
                        max_complexity::Integer=20, niterations::Integer=100,
                        parallelism::Symbol=:multithreading, seed::Union{Nothing,Integer}=nothing,
                        weighting::Symbol=:none, options=nothing,
                        backend_options::NamedTuple=(;), backend_kwargs::NamedTuple=(;))
    niterations >= 1 || throw(ArgumentError("niterations must be positive"))
    max_complexity >= 1 || throw(ArgumentError("max_complexity must be positive"))
    _check_weighting(weighting)
    configured = if isnothing(options)
        operators = SymbolicRegression.OperatorEnum(
            1 => Tuple(unary_operators), 2 => Tuple(binary_operators))
        option_keywords = merge((; operators, maxsize=Int(max_complexity), seed,
                                  progress=false, save_to_file=false), backend_options)
        SymbolicRegression.Options(; option_keywords...)
    else
        isempty(backend_options) || throw(ArgumentError("backend_options cannot accompany an existing options object"))
        options
    end
    SymbolicSearch(configured, Int(niterations), parallelism,
                   isnothing(seed) ? nothing : Int(seed), weighting, backend_kwargs)
end

struct SymbolicCandidate{E,T<:AbstractFloat,O}
    expression::E
    loss::T
    complexity::Int
    feature_names::Vector{Symbol}
    target_name::Symbol
    options::O
end

struct SymbolicSearchResult{H,O,C<:SymbolicCandidate}
    hall_of_fame::H
    options::O
    candidates::Vector{C}
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    training_row_ids::Vector{Any}
    search_configuration::NamedTuple
    backend_version::VersionNumber
end

function discover(dataset::RegressionDataset, search::SymbolicSearch)
    weights = _fit_weights(dataset, search.weighting)
    base = (; niterations=search.niterations, options=search.options,
            variable_names=string.(dataset.feature_names), parallelism=search.parallelism,
            weights, verbosity=0)
    kwargs = merge(base, search.backend_kwargs)
    hall = SymbolicRegression.equation_search(permutedims(dataset.X), dataset.y; kwargs...)
    frontier = SymbolicRegression.calculate_pareto_frontier(hall)
    candidates = [SymbolicCandidate(
        member.tree, Float64(member.loss),
        SymbolicRegression.compute_complexity(member, search.options),
        copy(dataset.feature_names), dataset.target_name, search.options,
    ) for member in frontier]
    SymbolicSearchResult(
        hall, search.options, candidates, copy(dataset.feature_names), dataset.target_name,
        copy(dataset.units), copy(dataset.row_ids),
        (; niterations=search.niterations, parallelism=search.parallelism,
           seed=search.seed, weighting=search.weighting),
        pkgversion(SymbolicRegression),
    )
end

function predict(candidate::SymbolicCandidate, X::AbstractMatrix)
    size(X, 2) == length(candidate.feature_names) || throw(DimensionMismatch("symbolic model feature count differs"))
    values, complete = SymbolicRegression.eval_tree_array(candidate.expression, permutedims(X), candidate.options)
    complete || throw(DomainError(X, "symbolic expression could not be evaluated for every observation"))
    collect(values)
end

function predict(candidate::SymbolicCandidate, dataset::RegressionDataset)
    dataset.feature_names == candidate.feature_names || throw(ArgumentError("dataset feature order does not match candidate"))
    predict(candidate, dataset.X)
end

formula(candidate::SymbolicCandidate) = SymbolicRegression.string_tree(
    candidate.expression, candidate.options; variable_names=string.(candidate.feature_names))

leaderboard(result::SymbolicSearchResult) = [
    (; complexity=c.complexity, loss=c.loss, formula=formula(c)) for c in result.candidates]

function best_model(result::SymbolicSearchResult; criterion::Symbol=:loss)
    isempty(result.candidates) && throw(ArgumentError("symbolic search returned no candidates"))
    criterion == :loss && return result.candidates[argmin(c.loss for c in result.candidates)]
    criterion == :complexity && return result.candidates[argmin(c.complexity for c in result.candidates)]
    throw(ArgumentError("criterion must be :loss or :complexity"))
end
