"""Indices for one train/test partition."""
struct DataSplit
    train::Vector{Int}
    test::Vector{Int}
end

function random_split(dataset::RegressionDataset; test_fraction::Real=0.2, rng=Random.default_rng())
    0 < test_fraction < 1 || throw(ArgumentError("test_fraction must lie strictly between zero and one"))
    n = length(dataset)
    n >= 2 || throw(ArgumentError("at least two observations are required"))
    order = randperm(rng, n)
    ntest = clamp(round(Int, test_fraction * n), 1, n - 1)
    DataSplit(sort(order[(ntest + 1):end]), sort(order[1:ntest]))
end

function _groups(dataset::RegressionDataset, by::Symbol)
    by == :eos && return dataset.eos_ids
    by == :simulation && return dataset.simulation_ids
    by == :source_paper && return dataset.source_papers
    throw(ArgumentError("grouping must be :eos, :simulation, or :source_paper"))
end

function grouped_kfold(dataset::RegressionDataset; by::Symbol=:eos, k::Integer=5, rng=Random.default_rng())
    groups = _groups(dataset, by)
    any(ismissing, groups) && throw(ArgumentError("group identifiers cannot be missing"))
    unique_groups = unique(groups)
    2 <= k <= length(unique_groups) || throw(ArgumentError("k must be between 2 and the number of groups"))
    shuffled = unique_groups[randperm(rng, length(unique_groups))]
    [begin
        testgroups = Set(shuffled[fold:k:end])
        test = findall(in(testgroups), groups)
        train = setdiff(collect(eachindex(groups)), test)
        DataSplit(train, test)
    end for fold in 1:k]
end

leave_one_group_out(dataset::RegressionDataset; by::Symbol=:eos) =
    grouped_kfold(dataset; by, k=length(unique(_groups(dataset, by))), rng=MersenneTwister(0))
