"""Scientific specification for nonlinear-in-parameter least squares."""
struct NonlinearModelSpec{F,A,K<:NamedTuple} <: AbstractRegressionMethod
    model::F
    parameter_names::Vector{Symbol}
    initial_guesses::Vector{Vector{Float64}}
    expression::String
    target_transform::AbstractTargetTransformation
    loss_space::Symbol
    weighting::Symbol
    lower_bounds::Union{Nothing,Vector{Float64}}
    upper_bounds::Union{Nothing,Vector{Float64}}
    parameter_scale::Union{Nothing,Vector{Float64}}
    algorithm::A
    solver_options::K
end

function NonlinearModelSpec(model, parameter_names, initial_guess;
                            expression::AbstractString,
                            target_transform::AbstractTargetTransformation=IdentityTarget(),
                            loss_space::Symbol=:original, weighting::Symbol=:none,
                            lower_bounds=nothing, upper_bounds=nothing,
                            parameter_scale=nothing,
                            algorithm=NonlinearSolve.FastShortcutNLLSPolyalg(),
                            solver_options::NamedTuple=(;))
    guesses = initial_guess isa AbstractVector{<:Real} ? [Float64.(initial_guess)] :
              [Float64.(guess) for guess in initial_guess]
    isempty(guesses) && throw(ArgumentError("at least one initial guess is required"))
    nparameters = length(parameter_names)
    all(guess -> length(guess) == nparameters, guesses) || throw(DimensionMismatch("initial guesses must match parameter names"))
    loss_space in (:original, :transformed) || throw(ArgumentError("loss_space must be :original or :transformed"))
    _check_weighting(weighting)
    lb = isnothing(lower_bounds) ? nothing : Float64.(lower_bounds)
    ub = isnothing(upper_bounds) ? nothing : Float64.(upper_bounds)
    xor(isnothing(lb), isnothing(ub)) && throw(ArgumentError("both lower and upper bounds must be supplied"))
    !isnothing(lb) && (length(lb) == nparameters == length(ub) || throw(DimensionMismatch("bounds must match parameters")))
    pscale = isnothing(parameter_scale) ? nothing : Float64.(parameter_scale)
    !isnothing(pscale) && (length(pscale) == nparameters || throw(DimensionMismatch("parameter_scale must match parameters")))
    !isnothing(pscale) && all(>(0), pscale) || isnothing(pscale) || throw(ArgumentError("parameter_scale must be positive"))
    NonlinearModelSpec(model, Symbol.(parameter_names), guesses, String(expression), target_transform,
                       loss_space, weighting, lb, ub, pscale, algorithm, solver_options)
end

struct FittedNonlinearModel{T<:AbstractFloat,S} <: AbstractFittedModel
    specification::S
    parameters::Vector{T}
    feature_names::Vector{Symbol}
    target_name::Symbol
    units::Dict{Symbol,String}
    objective::T
    residual_norm::T
    termination_status::String
    iterations::Int
    converged::Bool
    attempted_initial_guesses::Int
    solver_name::String
    solver_version::VersionNumber
    training_row_ids::Vector{Any}
end

function fit(dataset::RegressionDataset, specification::NonlinearModelSpec)
    scale = _target_scale(specification.target_transform, dataset.X, dataset.feature_names)
    transformed_y = scale .* dataset.y
    weights = _fit_weights(dataset, specification.weighting)
    rootweights = isnothing(weights) ? ones(eltype(dataset.y), length(dataset)) : sqrt.(weights)
    residual_scale = specification.loss_space == :original ? rootweights ./ scale : rootweights
    pscale = isnothing(specification.parameter_scale) ? ones(length(specification.parameter_names)) : specification.parameter_scale
    residual = (scaled_parameters, _) -> residual_scale .* (
        specification.model(dataset.X, scaled_parameters .* pscale) .- transformed_y)
    solutions = Any[]
    for guess in specification.initial_guesses
        scaled_guess = guess ./ pscale
        problem = if isnothing(specification.lower_bounds)
            NonlinearSolve.NonlinearLeastSquaresProblem(residual, scaled_guess)
        else
            NonlinearSolve.NonlinearLeastSquaresProblem(
                residual, scaled_guess; lb=specification.lower_bounds ./ pscale,
                ub=specification.upper_bounds ./ pscale)
        end
        push!(solutions, NonlinearSolve.solve(problem, specification.algorithm;
                                              specification.solver_options...))
    end
    objectives = [sum(abs2, solution.resid) for solution in solutions]
    solution = solutions[argmin(objectives)]
    converged = NonlinearSolve.SciMLBase.successful_retcode(solution)
    !converged && @warn "nonlinear least-squares solver did not report success" retcode=solution.retcode
    FittedNonlinearModel(
        specification, collect(solution.u .* pscale), copy(dataset.feature_names), dataset.target_name,
        copy(dataset.units), minimum(objectives), norm(solution.resid), string(solution.retcode),
        solution.stats.nsteps, converged, length(solutions), string(typeof(specification.algorithm)),
        pkgversion(NonlinearSolve), copy(dataset.row_ids),
    )
end

function predict(model::FittedNonlinearModel, X::AbstractMatrix)
    transformed = model.specification.model(X, model.parameters)
    transformed ./ _target_scale(model.specification.target_transform, X, model.feature_names)
end
predict(model::FittedNonlinearModel, dataset::RegressionDataset) = begin
    dataset.feature_names == model.feature_names || throw(ArgumentError("dataset feature order does not match model"))
    predict(model, dataset.X)
end
coefficients(model::FittedNonlinearModel) = copy(model.parameters)
formula(model::FittedNonlinearModel) = model.specification.expression
