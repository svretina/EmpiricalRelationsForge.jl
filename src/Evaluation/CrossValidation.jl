struct RandomHoldout <: AbstractValidationStrategy
    test_fraction::Float64
    seed::Int
end
RandomHoldout(; test_fraction::Real=0.2, seed::Integer=0) = RandomHoldout(Float64(test_fraction), Int(seed))

struct KFold <: AbstractValidationStrategy
    k::Int
    seed::Int
end
KFold(; k::Integer=5, seed::Integer=0) = KFold(Int(k), Int(seed))

struct GroupedKFold <: AbstractValidationStrategy
    k::Int
    by::Symbol
    seed::Int
end
GroupedKFold(; k::Integer=5, by::Symbol=:eos, seed::Integer=0) = GroupedKFold(Int(k), by, Int(seed))

struct LeaveOneEOSOut <: AbstractValidationStrategy end

"""Hold out each EOS, simulation family, or source paper in turn."""
struct LeaveOneGroupOut <: AbstractValidationStrategy
    by::Symbol
    function LeaveOneGroupOut(; by::Symbol=:eos)
        by in (:eos, :simulation, :source_paper) || throw(ArgumentError(
            "grouping must be :eos, :simulation, or :source_paper"))
        new(by)
    end
end

struct ValidationReport{T<:AbstractFloat,S<:AbstractValidationStrategy}
    strategy::S
    folds::Vector{DataSplit}
    fold_metrics::Vector{MetricReport{T}}
    aggregate_metrics::MetricReport{T}
    predictions::Vector{T}
    observed::Vector{T}
    row_ids::Vector{Any}
    group_metrics::Dict{Any,MetricReport{T}}
end

_splits(strategy::RandomHoldout, dataset) = [random_split(dataset; test_fraction=strategy.test_fraction,
                                                         rng=MersenneTwister(strategy.seed))]

function _splits(strategy::KFold, dataset)
    n = length(dataset)
    2 <= strategy.k <= n || throw(ArgumentError("k must be between 2 and the number of observations"))
    order = randperm(MersenneTwister(strategy.seed), n)
    [begin
        test = sort(order[fold:strategy.k:end])
        DataSplit(setdiff(collect(1:n), test), test)
    end for fold in 1:strategy.k]
end

_splits(strategy::GroupedKFold, dataset) = grouped_kfold(
    dataset; by=strategy.by, k=strategy.k, rng=MersenneTwister(strategy.seed))
_splits(::LeaveOneEOSOut, dataset) = leave_one_group_out(dataset; by=:eos)
_splits(strategy::LeaveOneGroupOut, dataset) = leave_one_group_out(dataset; by=strategy.by)

function validate(method::AbstractRegressionMethod, dataset::RegressionDataset;
                  strategy::AbstractValidationStrategy=LeaveOneEOSOut())
    folds = _splits(strategy, dataset)
    predictions = eltype(dataset.y)[]
    observed = eltype(dataset.y)[]
    row_ids = Any[]
    reports = MetricReport{eltype(dataset.y)}[]
    prediction_groups = Any[]
    for partition in folds
        training = subset(dataset, partition.train)
        testing = subset(dataset, partition.test)
        model = fit(training, method)
        fold_predictions = predict(model, testing)
        append!(predictions, fold_predictions)
        append!(observed, testing.y)
        append!(row_ids, testing.row_ids)
        append!(prediction_groups, testing.eos_ids)
        push!(reports, metrics(fold_predictions, testing.y))
    end
    grouped = if any(ismissing, prediction_groups)
        Dict{Any,MetricReport{eltype(dataset.y)}}()
    else
        Dict{Any,MetricReport{eltype(dataset.y)}}(
            group => report for (group, report) in
            grouped_metrics(predictions, observed, prediction_groups))
    end
    ValidationReport(strategy, folds, reports, metrics(predictions, observed), predictions,
                     observed, row_ids, grouped)
end

validate(model::AbstractFittedModel, dataset::RegressionDataset) = metrics(model, dataset)
