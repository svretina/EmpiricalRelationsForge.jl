"""Standard residual summaries. Residuals are defined as prediction minus observation."""
struct MetricReport{T<:AbstractFloat}
    n::Int
    mae::T
    rmse::T
    max_absolute_error::T
    mean_signed_residual::T
    r_squared::Union{T,Missing}
    mean_absolute_relative_error::Union{T,Missing}
end

function metrics(predicted::AbstractVector, observed::AbstractVector)
    length(predicted) == length(observed) || throw(DimensionMismatch("predicted and observed lengths differ"))
    isempty(observed) && throw(ArgumentError("metrics require at least one observation"))
    all(isfinite, predicted) && all(isfinite, observed) || throw(ArgumentError("metrics require finite values"))
    T = promote_type(float(eltype(predicted)), float(eltype(observed)))
    residual = T.(predicted .- observed)
    observed_t = T.(observed)
    ss_res = sum(abs2, residual)
    ss_tot = sum(abs2, observed_t .- mean(observed_t))
    r_squared = iszero(ss_tot) ? missing : one(T) - ss_res / ss_tot
    relative = any(iszero, observed_t) ? missing : mean(abs.(residual ./ observed_t))
    MetricReport(length(observed), mean(abs, residual), sqrt(mean(abs2, residual)),
                 maximum(abs, residual), mean(residual), r_squared, relative)
end

metrics(model::AbstractFittedModel, dataset::RegressionDataset) = metrics(predict(model, dataset), dataset.y)

function grouped_metrics(predicted::AbstractVector, observed::AbstractVector, groups::AbstractVector)
    length(groups) == length(observed) || throw(DimensionMismatch("groups must match observations"))
    Dict(group => metrics(predicted[findall(==(group), groups)], observed[findall(==(group), groups)])
         for group in unique(groups))
end
