module EmpiricalRelationsForge

using DataFrames
using LinearAlgebra
using NonlinearSolve
using Random
using Statistics
using SymbolicRegression
using TOML
using Tables

include("Core/Types.jl")
include("Core/Interfaces.jl")
include("Core/Validation.jl")
include("Data/Loading.jl")
include("Data/Transformations.jl")
include("Data/Splitting.jl")
include("Regression/Classical.jl")
include("Regression/Basis.jl")
include("Regression/Nonlinear.jl")
include("Evaluation/Metrics.jl")
include("Evaluation/CrossValidation.jl")
include("Symbolic/Adapter.jl")
include("Results/Models.jl")

export RegressionDataset, TransformationRecord, Standardizer
export load_dataset, fit_standardizer, transform_features, subset
export DataSplit, random_split, grouped_kfold, leave_one_group_out
export validate_dataset
export duplicate_identifiers, physical_dependency_warnings
export AbstractRegressionMethod, LinearRegression, PolynomialRegression, FittedPolynomialModel
export AbstractTargetTransformation, IdentityTarget, FeatureScaleTarget
export PowerProductTerm, BasisSet, LinearBasisRegression, FittedBasisModel
export NonlinearModelSpec, FittedNonlinearModel
export fit, predict, formula, coefficients
export MetricReport, metrics, grouped_metrics
export AbstractValidationStrategy, RandomHoldout, KFold, GroupedKFold, LeaveOneEOSOut
export ValidationReport, validate
export SymbolicSearch, SymbolicCandidate, SymbolicSearchResult
export discover, leaderboard, best_model
export ModelResult, model_result, save_metadata

end
