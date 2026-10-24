"""Physical dimensions represented by mass, length, and time exponents."""
struct PhysicalDimension
    mass::Float64
    length::Float64
    time::Float64
end
PhysicalDimension(; mass=0, length=0, time=0) = PhysicalDimension(mass, length, time)
Base.:+(a::PhysicalDimension, b::PhysicalDimension) =
    PhysicalDimension(a.mass + b.mass, a.length + b.length, a.time + b.time)
Base.:-(a::PhysicalDimension, b::PhysicalDimension) =
    PhysicalDimension(a.mass - b.mass, a.length - b.length, a.time - b.time)
Base.:*(x::Real, a::PhysicalDimension) = PhysicalDimension(x*a.mass, x*a.length, x*a.time)
Base.iszero(a::PhysicalDimension) = iszero(a.mass) && iszero(a.length) && iszero(a.time)
Base.:(==)(a::PhysicalDimension, b::PhysicalDimension) =
    a.mass == b.mass && a.length == b.length && a.time == b.time

struct PhysicalQuantity
    name::Symbol
    dimension::PhysicalDimension
    canonical_unit::String
    scale::Float64
    valid_domain::Tuple{Float64,Float64}
end

const BNS_QUANTITIES = Dict{Symbol,PhysicalQuantity}(
    :M_chirp => PhysicalQuantity(:M_chirp, PhysicalDimension(mass=1), "M_sun", 1.0, (0.0, Inf)),
    :M_tot => PhysicalQuantity(:M_tot, PhysicalDimension(mass=1), "M_sun", 1.0, (0.0, Inf)),
    :R_1_2 => PhysicalQuantity(:R_1_2, PhysicalDimension(length=1), "km", 1.0, (0.0, Inf)),
    :R_1_4 => PhysicalQuantity(:R_1_4, PhysicalDimension(length=1), "km", 1.0, (0.0, Inf)),
    :R_1_6 => PhysicalQuantity(:R_1_6, PhysicalDimension(length=1), "km", 1.0, (0.0, Inf)),
    :R_1_8 => PhysicalQuantity(:R_1_8, PhysicalDimension(length=1), "km", 1.0, (0.0, Inf)),
    :f_peak => PhysicalQuantity(:f_peak, PhysicalDimension(time=-1), "kHz", 1.0, (0.0, Inf)),
    :f_2_0 => PhysicalQuantity(:f_2_0, PhysicalDimension(time=-1), "kHz", 1.0, (0.0, Inf)),
    :f_spiral => PhysicalQuantity(:f_spiral, PhysicalDimension(time=-1), "kHz", 1.0, (0.0, Inf)),
    :Lambda_tilde => PhysicalQuantity(:Lambda_tilde, PhysicalDimension(), "1", 1.0, (0.0, Inf)),
    :q => PhysicalQuantity(:q, PhysicalDimension(), "1", 1.0, (0.0, 1.0)),
    :compactness => PhysicalQuantity(:compactness, PhysicalDimension(), "1", 1.0, (0.0, 0.5)),
)

struct PhysicalNormalization
    feature_names::Vector{Symbol}
    feature_scales::Vector{Float64}
    target_name::Symbol
    target_scale::Float64
    original_units::Dict{Symbol,String}
end

function normalize_physical(dataset::RegressionDataset,
                            registry::AbstractDict{Symbol,PhysicalQuantity}=BNS_QUANTITIES)
    missing_names = filter(name -> !haskey(registry, name), [dataset.feature_names; dataset.target_name])
    isempty(missing_names) || throw(ArgumentError("physical quantities are not registered: $(join(missing_names, ", "))"))
    feature_scales = [registry[name].scale for name in dataset.feature_names]
    target_scale = registry[dataset.target_name].scale
    all(>(0), feature_scales) && target_scale > 0 || throw(ArgumentError("normalization scales must be positive"))
    for (column, name) in enumerate(dataset.feature_names)
        lo, hi = registry[name].valid_domain
        all(x -> lo <= x <= hi, view(dataset.X, :, column)) ||
            throw(DomainError(name, "feature values lie outside the registered physical domain"))
    end
    lo, hi = registry[dataset.target_name].valid_domain
    all(x -> lo <= x <= hi, dataset.y) || throw(DomainError(dataset.target_name,
        "target values lie outside the registered physical domain"))
    record = TransformationRecord(:physical_normalization,
        [dataset.feature_names; dataset.target_name],
        Dict(:feature_scales => feature_scales, :target_scale => target_scale))
    normalized = RegressionDataset(dataset.X ./ feature_scales', dataset.y ./ target_scale;
        features=dataset.feature_names, target=dataset.target_name,
        units=Dict(name => "1" for name in [dataset.feature_names; dataset.target_name]),
        row_ids=dataset.row_ids, eos_ids=dataset.eos_ids, simulation_ids=dataset.simulation_ids,
        source_papers=dataset.source_papers, observation_kinds=dataset.observation_kinds,
        uncertainties=isnothing(dataset.uncertainties) ? nothing : dataset.uncertainties ./ target_scale,
        uncertainty_kind=dataset.uncertainty_kind, weights=dataset.weights,
        transformations=vcat(dataset.transformations, [record]))
    normalization = PhysicalNormalization(copy(dataset.feature_names), feature_scales,
        dataset.target_name, target_scale, copy(dataset.units))
    return normalized, normalization
end

struct NormalizedModel{M} <: AbstractFittedModel
    model::M
    normalization::PhysicalNormalization
end
physical_model(model, normalization::PhysicalNormalization) = NormalizedModel(model, normalization)
function predict(model::NormalizedModel, X::AbstractMatrix)
    size(X, 2) == length(model.normalization.feature_scales) || throw(DimensionMismatch("feature count differs"))
    model.normalization.target_scale .* predict(model.model, X ./ model.normalization.feature_scales')
end
predict(model::NormalizedModel, data::RegressionDataset) = begin
    data.feature_names == model.normalization.feature_names || throw(ArgumentError("feature order differs"))
    predict(model, data.X)
end
formula(model::NormalizedModel) = formula(model.model)
coefficients(model::NormalizedModel) = coefficients(model.model)
parameter_names(model::NormalizedModel) = parameter_names(model.model)
parameter_vector(model::NormalizedModel) = parameter_vector(model.model)
with_parameters(model::NormalizedModel, values) = NormalizedModel(with_parameters(model.model, values), model.normalization)
model_structure(model::NormalizedModel) = model_structure(model.model)

struct DimensionCheck
    valid::Bool
    inferred::Union{Nothing,PhysicalDimension}
    expected::PhysicalDimension
    message::String
end

function _infer_dimension(node, options, feature_dimensions)
    if node.degree == 0
        return node.constant ? nothing : feature_dimensions[node.feature]
    end
    op = options.operators.ops[node.degree][node.op]
    node_children = _node_children(node)
    children = [_infer_dimension(child, options, feature_dimensions) for child in node_children]
    if node.degree == 2 && (op === (+) || op === (-))
        a, b = children
        isnothing(a) && return b
        isnothing(b) && return a
        a == b || throw(DimensionMismatch("addition or subtraction combines incompatible dimensions"))
        return a
    elseif node.degree == 2 && op === (*)
        any(isnothing, children) && return nothing
        return children[1] + children[2]
    elseif node.degree == 2 && op === (/)
        any(isnothing, children) && return nothing
        return children[1] - children[2]
    elseif node.degree == 2 && string(nameof(op)) in ("^", "safe_pow")
        exponent = node_children[2]
        exponent.degree == 0 && exponent.constant || throw(DimensionMismatch("dimensionful powers require a constant exponent"))
        isnothing(children[1]) && return nothing
        return Float64(exponent.val) * children[1]
    elseif node.degree == 1 && string(nameof(op)) in ("sqrt", "safe_sqrt")
        return isnothing(children[1]) ? nothing : 0.5 * children[1]
    elseif node.degree == 1 && string(nameof(op)) in ("exp", "log", "safe_log", "sin", "cos", "tan")
        (isnothing(children[1]) || iszero(children[1])) || throw(DimensionMismatch("transcendental function requires a dimensionless argument"))
        return PhysicalDimension()
    end
    all(iszero, filter(x -> !isnothing(x), children)) || throw(DimensionMismatch("operator $(nameof(op)) has no dimensional rule"))
    PhysicalDimension()
end

function check_dimensions(candidate::SymbolicCandidate,
                          quantities::AbstractDict{Symbol,PhysicalQuantity}=BNS_QUANTITIES)
    all(haskey(quantities, name) for name in [candidate.feature_names; candidate.target_name]) ||
        throw(ArgumentError("quantity registry does not cover the candidate variables"))
    feature_dimensions = [quantities[name].dimension for name in candidate.feature_names]
    expected = quantities[candidate.target_name].dimension
    try
        inferred = _infer_dimension(_raw_tree(candidate.expression), candidate.options, feature_dimensions)
        valid = isnothing(inferred) || inferred == expected
        DimensionCheck(valid, inferred, expected, valid ? "dimensionally admissible; free coefficients may carry dimensions" :
            "expression output dimension differs from target")
    catch error
        DimensionCheck(false, nothing, expected, sprint(showerror, error))
    end
end

struct ExpressionValidity
    valid::Bool
    finite_count::Int
    observation_count::Int
    message::String
end
function expression_validity(candidate::SymbolicCandidate, dataset::RegressionDataset)
    try
        values = predict(candidate, dataset)
        finite_count = count(isfinite, values)
        ExpressionValidity(finite_count == length(values), finite_count, length(values),
                           finite_count == length(values) ? "ok" : "non-finite predictions")
    catch error
        ExpressionValidity(false, 0, length(dataset), sprint(showerror, error))
    end
end
