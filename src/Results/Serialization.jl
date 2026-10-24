const MODEL_SCHEMA_VERSION = 1

_standardizer_payload(::Nothing) = Dict("present"=>false)
_standardizer_payload(s::Standardizer) = Dict("present"=>true,
    "features"=>string.(s.feature_names), "means"=>s.means, "scales"=>s.scales)
function _load_standardizer(payload)
    payload["present"] || return nothing
    Standardizer(Symbol.(payload["features"]), Float64.(payload["means"]), Float64.(payload["scales"]))
end

function _method_payload(method::Union{LinearRegression,PolynomialRegression})
    Dict("type"=>method isa LinearRegression ? "linear" : "polynomial",
         "degree"=>method isa LinearRegression ? 1 : method.degree,
         "intercept"=>method.intercept,
         "cross_terms"=>method isa LinearRegression ? false : method.cross_terms,
         "weighting"=>string(method.weighting), "standardize"=>method.standardize,
         "condition_warning"=>method.condition_warning,
         "target_transform"=>_transform_payload(method.target_transform),
         "loss_space"=>string(method.loss_space), "solver"=>string(method.solver))
end

function _load_polynomial_method(payload)
    common = (; intercept=Bool(payload["intercept"]), weighting=Symbol(payload["weighting"]),
        standardize=Bool(payload["standardize"]), condition_warning=Float64(payload["condition_warning"]),
        target_transform=_load_transform(payload["target_transform"]),
        loss_space=Symbol(payload["loss_space"]), solver=Symbol(payload["solver"]))
    payload["type"] == "linear" && return LinearRegression(; common...)
    payload["type"] == "polynomial" || throw(ArgumentError("unknown polynomial method type"))
    PolynomialRegression(; degree=Int(payload["degree"]),
        cross_terms=Bool(payload["cross_terms"]), common...)
end

_transform_payload(::IdentityTarget) = Dict("type" => "identity")
_transform_payload(t::FeatureScaleTarget) = Dict("type" => "feature_scale", "feature" => string(t.feature), "power" => t.power)
function _load_transform(payload)
    payload["type"] == "identity" && return IdentityTarget()
    payload["type"] == "feature_scale" && return FeatureScaleTarget(Symbol(payload["feature"]); power=payload["power"])
    throw(ArgumentError("unknown target transformation $(payload["type"])"))
end

function _tree_payload(node)
    node.degree == 0 && return node.constant ? Dict("kind"=>"constant", "value"=>Float64(node.val)) :
        Dict("kind"=>"feature", "index"=>Int(node.feature))
    Dict("kind"=>"operator", "degree"=>Int(node.degree), "operator"=>Int(node.op),
         "children"=>[_tree_payload(child) for child in _node_children(node)])
end

function _load_tree(payload)
    kind = payload["kind"]
    kind == "constant" && return SymbolicRegression.Node{Float64}(; val=Float64(payload["value"]))
    kind == "feature" && return SymbolicRegression.Node{Float64}(; feature=Int(payload["index"]))
    kind == "operator" || throw(ArgumentError("unknown symbolic node kind $kind"))
    children = Tuple(_load_tree(child) for child in payload["children"])
    SymbolicRegression.Node{Float64}(; op=Int(payload["operator"]), children)
end

function _operator_id(op)
    name = string(nameof(op))
    name == "safe_sqrt" && return "sqrt"
    name == "safe_log" && return "log"
    name == "safe_pow" && return "^"
    name
end
function _known_operator(name::AbstractString)
    table = Dict("+"=>+, "-"=>-, "*"=>*, "/"=>/, "^"=>^,
                 "sqrt"=>sqrt, "log"=>log, "exp"=>exp, "sin"=>sin, "cos"=>cos, "abs"=>abs)
    haskey(table, name) || throw(ArgumentError("unknown or unsupported serialized operator $name"))
    table[name]
end

function _candidate_payload(candidate::SymbolicCandidate)
    operators = candidate.options.operators.ops
    Dict("tree"=>_tree_payload(_raw_tree(candidate.expression)),
         "unary_operators"=>[_operator_id(op) for op in operators[1]],
         "binary_operators"=>[_operator_id(op) for op in operators[2]],
         "feature_names"=>string.(candidate.feature_names), "target_name"=>string(candidate.target_name),
         "loss"=>candidate.loss, "complexity"=>candidate.complexity)
end
function _load_candidate(payload)
    unary = [_known_operator(name) for name in payload["unary_operators"]]
    binary = [_known_operator(name) for name in payload["binary_operators"]]
    options = SymbolicRegression.Options(; binary_operators=binary, unary_operators=unary,
                                          progress=false, save_to_file=false)
    SymbolicCandidate(_load_tree(payload["tree"]), Float64(payload["loss"]),
        Int(payload["complexity"]), Symbol.(payload["feature_names"]),
        Symbol(payload["target_name"]), options)
end

function dataset_fingerprint(dataset::RegressionDataset)
    io = IOBuffer()
    print(io, dataset.feature_names, dataset.target_name, dataset.row_ids, dataset.X, dataset.y)
    bytes2hex(SHA.sha256(take!(io)))
end

function _model_payload(model::FittedBasisModel)
    Dict("type"=>"linear_basis", "parameters"=>model.beta,
         "features"=>string.(model.feature_names), "target"=>string(model.target_name),
         "units"=>Dict(string(k)=>v for (k,v) in model.units),
         "terms"=>[Dict("id"=>string(t.id), "label"=>t.label,
            "powers"=>Dict(string(k)=>v for (k,v) in t.powers), "coefficient"=>t.coefficient,
            "unit"=>t.unit) for t in model.method.basis.terms],
         "weighting"=>string(model.method.weighting), "solver"=>string(model.method.solver),
         "target_transform"=>_transform_payload(model.method.target_transform),
         "loss_space"=>string(model.method.loss_space), "rank"=>model.rank,
         "standardizer"=>_standardizer_payload(model.standardizer),
         "condition_number"=>model.condition_number, "objective"=>model.objective,
         "training_row_ids"=>_toml_safe(model.training_row_ids))
end
function _model_payload(model::FittedPolynomialModel)
    Dict("type"=>"polynomial", "method"=>_method_payload(model.method),
         "parameters"=>model.beta, "exponents"=>[collect(row) for row in eachrow(model.exponents)],
         "term_names"=>model.term_names, "features"=>string.(model.feature_names),
         "target"=>string(model.target_name), "units"=>Dict(string(k)=>v for (k,v) in model.units),
         "standardizer"=>_standardizer_payload(model.standardizer),
         "rank"=>model.rank, "condition_number"=>model.condition_number,
         "objective"=>model.objective, "target_transform"=>_transform_payload(model.target_transform),
         "loss_space"=>string(model.loss_space), "training_row_ids"=>_toml_safe(model.training_row_ids))
end
function _model_payload(model::RefittedSymbolicModel)
    Dict("type"=>"refitted_symbolic", "candidate"=>_candidate_payload(model.candidate),
         "original_candidate"=>_candidate_payload(model.original),
         "original_parameters"=>model.original_parameters, "parameters"=>model.parameters,
         "original_objective"=>model.original_objective, "objective"=>model.objective,
         "method"=>string(model.method), "status"=>string(model.status),
         "solver_name"=>model.solver_name, "weighting"=>string(model.weighting),
         "target_transform"=>_transform_payload(model.target_transform),
         "loss_space"=>string(model.loss_space),
         "lower_bounds"=>isnothing(model.lower_bounds) ? Any[] : model.lower_bounds,
         "upper_bounds"=>isnothing(model.upper_bounds) ? Any[] : model.upper_bounds,
         "diagnostics"=>Dict("valid"=>model.diagnostics.valid,
             "message"=>model.diagnostics.message,
             "jacobian_rank"=>model.diagnostics.jacobian_rank,
             "jacobian_condition"=>model.diagnostics.jacobian_condition,
             "unused_parameters"=>string.(model.diagnostics.unused_parameters)),
         "structure_fingerprint"=>model.structure_fingerprint,
         "training_row_ids"=>_toml_safe(model.training_row_ids))
end
function _model_payload(model::NormalizedModel)
    normalization = model.normalization
    Dict("type"=>"normalized", "inner"=>_model_payload(model.model),
         "normalization"=>Dict("features"=>string.(normalization.feature_names),
             "feature_scales"=>normalization.feature_scales,
             "target"=>string(normalization.target_name), "target_scale"=>normalization.target_scale,
             "original_units"=>Dict(string(k)=>v for (k,v) in normalization.original_units)))
end
_model_payload(model::FittedNonlinearModel) = throw(ArgumentError("arbitrary nonlinear Julia callables are not portably serializable"))
_model_payload(model) = throw(ArgumentError("save_model does not support $(typeof(model))"))

"""Save a reconstructible, versioned model artifact without serialized Julia code."""
function save_model(path::AbstractString, model; metadata=Dict{Symbol,Any}())
    payload = Dict("schema_version"=>MODEL_SCHEMA_VERSION, "model"=>_model_payload(model),
                   "parameter_names"=>string.(parameter_names(model)),
                   "metadata"=>_toml_safe(metadata), "julia_version"=>string(VERSION),
                   "package_version"=>"0.1.0", "symbolic_backend_version"=>string(pkgversion(SymbolicRegression)))
    open(path, "w") do io
        TOML.print(io, payload; sorted=true)
    end
    path
end

function _load_basis(payload)
    terms = [PowerProductTerm(Symbol(t["id"]), Dict(Symbol(k)=>Float64(v) for (k,v) in t["powers"]);
        label=t["label"], coefficient=t["coefficient"], unit=t["unit"]) for t in payload["terms"]]
    method = LinearBasisRegression(BasisSet(terms); weighting=Symbol(payload["weighting"]),
        solver=Symbol(payload["solver"]), target_transform=_load_transform(payload["target_transform"]),
        loss_space=Symbol(payload["loss_space"]))
    FittedBasisModel(method, Float64.(payload["parameters"]), Symbol.(payload["features"]),
        Symbol(payload["target"]), Dict(Symbol(k)=>String(v) for (k,v) in payload["units"]),
        _load_standardizer(payload["standardizer"]), Int(payload["rank"]), Float64(payload["condition_number"]),
        Float64(payload["objective"]), Any[payload["training_row_ids"]...])
end
function _load_polynomial(payload)
    method = _load_polynomial_method(payload["method"])
    exponent_rows = [Int.(row) for row in payload["exponents"]]
    exponents = reduce(vcat, permutedims.(exponent_rows))
    FittedPolynomialModel(method, Float64.(payload["parameters"]), exponents,
        String.(payload["term_names"]), Symbol.(payload["features"]), Symbol(payload["target"]),
        Dict(Symbol(k)=>String(v) for (k,v) in payload["units"]),
        _load_standardizer(payload["standardizer"]), TransformationRecord[], Int(payload["rank"]),
        Float64(payload["condition_number"]), Any[payload["training_row_ids"]...],
        _load_transform(payload["target_transform"]), Symbol(payload["loss_space"]),
        Float64(payload["objective"]))
end
function _load_refitted(payload)
    candidate = _load_candidate(payload["candidate"])
    original = _load_candidate(payload["original_candidate"])
    parameters = Float64.(payload["parameters"])
    d = payload["diagnostics"]
    diagnostics = RefitDiagnostics(Bool(d["valid"]), String(d["message"]),
        Int(d["jacobian_rank"]), Float64(d["jacobian_condition"]),
        Symbol.(d["unused_parameters"]))
    RefittedSymbolicModel(original, candidate, Float64.(payload["original_parameters"]), parameters,
        Float64(payload["original_objective"]), Float64(payload["objective"]),
        Symbol(payload["method"]), Symbol(payload["status"]), String(payload["solver_name"]),
        Symbol(payload["weighting"]), _load_transform(payload["target_transform"]),
        Symbol(payload["loss_space"]), isempty(payload["lower_bounds"]) ? nothing : Float64.(payload["lower_bounds"]),
        isempty(payload["upper_bounds"]) ? nothing : Float64.(payload["upper_bounds"]), diagnostics,
        String(payload["structure_fingerprint"]), Any[payload["training_row_ids"]...])
end

function _load_model_payload(model)
    model["type"] == "linear_basis" && return _load_basis(model)
    model["type"] == "polynomial" && return _load_polynomial(model)
    model["type"] == "refitted_symbolic" && return _load_refitted(model)
    if model["type"] == "normalized"
        n = model["normalization"]
        normalization = PhysicalNormalization(Symbol.(n["features"]), Float64.(n["feature_scales"]),
            Symbol(n["target"]), Float64(n["target_scale"]),
            Dict(Symbol(k)=>String(v) for (k,v) in n["original_units"]))
        return NormalizedModel(_load_model_payload(model["inner"]), normalization)
    end
    throw(ArgumentError("unknown serialized model type $(model["type"])"))
end

function load_model(path::AbstractString)
    isfile(path) || throw(ArgumentError("model artifact does not exist: $path"))
    payload = try TOML.parsefile(path) catch error
        throw(ArgumentError("invalid or truncated model artifact: $(sprint(showerror, error))"))
    end
    get(payload, "schema_version", nothing) == MODEL_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported model artifact schema version $(get(payload, "schema_version", "missing"))"))
    _load_model_payload(payload["model"])
end
