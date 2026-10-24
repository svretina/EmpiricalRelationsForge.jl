module EmpiricalRelationsForge

using DataFrames
using DynamicQuantities
using LinearAlgebra
using NonlinearSolve
using Random
using SHA
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
include("Refinement/ModelInterface.jl")
include("Refinement/Refit.jl")
include("Evaluation/Comparison.jl")
include("Evaluation/Uncertainty.jl")
include("Physics/Dimensions.jl")
include("Results/Serialization.jl")
include("Symbolic/Equivalence.jl")
include("Integration/Candidates.jl")

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
export ModelStructure, parameter_names, parameter_vector, with_parameters, model_structure
export RefitDiagnostics, RefittedSymbolicModel, refit
export ModelComparison, ComparisonReport, compare, select_model
export NestedValidationReport, nested_validate
export CovarianceEstimate, parameter_covariance, prediction_uncertainty
export BootstrapResult, bootstrap
export PhysicalDimension, PhysicalQuantity, PhysicalNormalization, BNS_QUANTITIES
export normalize_physical, NormalizedModel, physical_model, check_dimensions
export ExpressionValidity, expression_validity
export save_model, load_model, dataset_fingerprint
export EquivalenceAssessment, equivalence, structural_fingerprint, deduplicate_candidates
export PublishedPredictor, published_predictor, CandidateExport, export_candidate

end
