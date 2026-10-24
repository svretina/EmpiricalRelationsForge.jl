"""Stable description of a fitted model's mathematical structure."""
struct ModelStructure
    kind::Symbol
    feature_names::Vector{Symbol}
    target_name::Symbol
    parameter_names::Vector{Symbol}
    expression::String
    fingerprint::String
end

parameter_names(model) = throw(ArgumentError("parameter identities are not available for $(typeof(model))"))
parameter_vector(model) = throw(ArgumentError("parameters are not available for $(typeof(model))"))
with_parameters(model, values) = throw(ArgumentError("parameter replacement is not supported for $(typeof(model))"))
model_structure(model) = throw(ArgumentError("model structure is not available for $(typeof(model))"))

_digest(text) = bytes2hex(SHA.sha256(codeunits(text)))
_node_children(node) = ntuple(i -> node.children[i].x, Int(node.degree))

parameter_names(model::FittedPolynomialModel) = Symbol.(model.term_names)
parameter_vector(model::FittedPolynomialModel) = coefficients(model)
function with_parameters(model::FittedPolynomialModel, values)
    length(values) == length(model.beta) || throw(DimensionMismatch("parameter vector length differs"))
    FittedPolynomialModel(
        model.method, Float64.(values), copy(model.exponents), copy(model.term_names),
        copy(model.feature_names), model.target_name, copy(model.units), model.standardizer,
        copy(model.transformations), model.rank, model.condition_number,
        copy(model.training_row_ids), model.target_transform, model.loss_space, model.objective)
end
function model_structure(model::FittedPolynomialModel)
    structural = join(model.term_names, "|") * ";features=" * join(string.(model.feature_names), ",")
    ModelStructure(:polynomial, copy(model.feature_names), model.target_name,
                   parameter_names(model), formula(model), _digest(structural))
end

parameter_names(model::FittedBasisModel) = getfield.(model.method.basis.terms, :id)
parameter_vector(model::FittedBasisModel) = coefficients(model)
function with_parameters(model::FittedBasisModel, values)
    length(values) == length(model.beta) || throw(DimensionMismatch("parameter vector length differs"))
    FittedBasisModel(model.method, Float64.(values), copy(model.feature_names), model.target_name,
                     copy(model.units), model.standardizer, model.rank, model.condition_number,
                     model.objective, copy(model.training_row_ids))
end
function model_structure(model::FittedBasisModel)
    structural = join(["$(t.id):$(sort!(collect(t.powers); by=first))" for t in model.method.basis.terms], "|")
    ModelStructure(:linear_basis, copy(model.feature_names), model.target_name,
                   parameter_names(model), formula(model), _digest(structural))
end

parameter_names(model::FittedNonlinearModel) = copy(model.specification.parameter_names)
parameter_vector(model::FittedNonlinearModel) = coefficients(model)
function with_parameters(model::FittedNonlinearModel, values)
    length(values) == length(model.parameters) || throw(DimensionMismatch("parameter vector length differs"))
    FittedNonlinearModel(model.specification, Float64.(values), copy(model.feature_names),
        model.target_name, copy(model.units), model.objective, model.residual_norm,
        model.termination_status, model.iterations, model.converged,
        model.attempted_initial_guesses, model.solver_name, model.solver_version,
        copy(model.training_row_ids))
end
function model_structure(model::FittedNonlinearModel)
    structural = model.specification.expression * ";features=" * join(string.(model.feature_names), ",")
    ModelStructure(:nonlinear, copy(model.feature_names), model.target_name,
                   parameter_names(model), formula(model), _digest(structural))
end

function _tree_signature(node, options; constants::Bool=false)
    if node.degree == 0
        return node.constant ? (constants ? "C($(repr(node.val)))" : "P") : "X$(node.feature)"
    end
    op = options.operators.ops[node.degree][node.op]
    children = join((_tree_signature(child, options; constants) for child in _node_children(node)), ",")
    "$(nameof(op))($children)"
end

_raw_tree(expression) = SymbolicRegression.get_tree(expression)

parameter_names(candidate::SymbolicCandidate) =
    [Symbol("p", i) for i in eachindex(first(SymbolicRegression.get_scalar_constants(candidate.expression)))]
parameter_vector(candidate::SymbolicCandidate) =
    Float64.(first(SymbolicRegression.get_scalar_constants(candidate.expression)))

function with_parameters(candidate::SymbolicCandidate, values)
    length(values) == length(parameter_vector(candidate)) || throw(DimensionMismatch("parameter vector length differs"))
    expression = SymbolicRegression.copy_node(candidate.expression; break_sharing=Val(true))
    _, refs = SymbolicRegression.get_scalar_constants(expression)
    SymbolicRegression.set_scalar_constants!(expression, Float64.(values), refs)
    SymbolicCandidate(expression, candidate.loss, candidate.complexity, copy(candidate.feature_names),
                      candidate.target_name, candidate.options)
end

function model_structure(candidate::SymbolicCandidate)
    signature = _tree_signature(_raw_tree(candidate.expression), candidate.options)
    ModelStructure(:symbolic, copy(candidate.feature_names), candidate.target_name,
                   parameter_names(candidate), formula(candidate), _digest(signature))
end

coefficients(candidate::SymbolicCandidate) = parameter_vector(candidate)
