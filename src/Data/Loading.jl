function _column(columns, name::Symbol, required::Bool, default)
    if hasproperty(columns, name)
        return getproperty(columns, name)
    elseif required
        throw(ArgumentError("required column :$name is missing"))
    end
    return default
end

"""
    RegressionDataset(table; target, features, ...)

Construct a dataset from any Tables.jl-compatible source. Rows with missing
features or targets are removed when `dropmissing=true`.
"""
function RegressionDataset(
    table;
    target::Symbol,
    features::AbstractVector{Symbol},
    units=Dict{Symbol,String}(),
    row_id::Union{Nothing,Symbol}=nothing,
    eos_id::Union{Nothing,Symbol}=nothing,
    simulation_id::Union{Nothing,Symbol}=nothing,
    source_paper::Union{Nothing,Symbol}=nothing,
    observation_kind::Union{Nothing,Symbol}=nothing,
    include_kinds=(:measured,),
    uncertainty::Union{Nothing,Symbol}=nothing,
    uncertainty_kind::Symbol=:unspecified,
    weight::Union{Nothing,Symbol}=nothing,
    dropmissing::Bool=true,
)
    columns = Tables.columntable(table)
    names = Set(Symbol.(propertynames(columns)))
    absent = setdiff(vcat(features, [target]), names)
    isempty(absent) || throw(ArgumentError("missing required columns: $(join(string.(absent), ", "))"))
    n = length(getproperty(columns, target))
    selected = vcat(features, [target])
    mask = trues(n)
    kinds = isnothing(observation_kind) ? fill(:measured, n) :
            Symbol.(_column(columns, observation_kind, true, nothing))
    mask .&= in.(kinds, Ref(Set(Symbol.(include_kinds))))
    if dropmissing
        for name in selected
            mask .&= .!ismissing.(getproperty(columns, name))
        end
    elseif any(name -> any(ismissing, getproperty(columns, name)), selected)
        throw(ArgumentError("features and target contain missing values; use dropmissing=true"))
    end
    indices = findall(mask)
    X = hcat((Float64.(getproperty(columns, name)[indices]) for name in features)...)
    y = Float64.(getproperty(columns, target)[indices])
    optional(name, fallback) = isnothing(name) ? fallback : _column(columns, name, true, nothing)[indices]
    uncertainties = isnothing(uncertainty) ? nothing : Float64.(optional(uncertainty, nothing))
    weights = isnothing(weight) ? nothing : Float64.(optional(weight, nothing))
    return RegressionDataset(
        X, y; features, target, units,
        row_ids=optional(row_id, indices),
        eos_ids=optional(eos_id, fill(missing, length(indices))),
        simulation_ids=optional(simulation_id, fill(missing, length(indices))),
        source_papers=optional(source_paper, fill(missing, length(indices))),
        observation_kinds=kinds[indices], uncertainties, uncertainty_kind, weights,
    )
end

load_dataset(table; kwargs...) = RegressionDataset(table; kwargs...)

"""Load measurements from a named provider (provider integration is pending its public data API)."""
function load_dataset(observable::Symbol; source::Symbol=:EmpiricalRelations, kwargs...)
    observable in (:f_peak, :f_2_0, :f_spiral) || throw(ArgumentError("unsupported observable :$observable"))
    source == :EmpiricalRelations || throw(ArgumentError("unknown data source :$source"))
    throw(ArgumentError(
        "EmpiricalRelations.jl does not currently expose a public measurement-table API. " *
        "Pass a Tables.jl-compatible table to load_dataset instead; no private provider internals are used."
    ))
end
