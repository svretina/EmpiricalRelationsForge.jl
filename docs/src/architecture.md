# Architecture

`RegressionDataset` fixes the internal numerical convention to observations by
features and keeps row, EOS, simulation, paper, observation class, unit,
uncertainty, weighting, and transformation metadata beside the numbers.

Preprocessing is split into a learn/apply pair. For example,
`fit_standardizer(training_data)` learns statistics only from a training
partition, and `transform_features` applies them later. Grouped splitters keep
EOS, simulation, or paper groups from crossing partition boundaries.

EmpiricalRelations.jl currently has no public measurement-table API. The named
provider method therefore reports this limitation rather than depending on
private implementation details. Direct table construction is the supported
path until that provider contract exists.

The fitting layer separates scientific specification from numerical solution:

1. a polynomial, `BasisSet`, or nonlinear model defines the mathematical form;
2. a target transformation defines the fitted observable;
3. a loss-space and weighting convention define the objective;
4. a backend solves the resulting least-squares problem;
5. the fitted model stores feature order, units, transformations, training IDs,
   objective value, and numerical diagnostics.

Cross-validation receives a regression method rather than a pre-fitted model.
Every fold therefore learns standardization and coefficients from its training
partition only.

The refinement layer separates an expression tree from its ordered constants.
`ModelStructure` supplies a stable structural fingerprint, while
`parameter_vector` and `with_parameters` operate on numerical values without
parsing formula strings. This shared interface supports covariance propagation,
fixed-structure bootstrap, model artifacts, and comparison experiments.

Symbolic discovery and fixed-structure assessment are deliberately distinct.
`validate(candidate, data)` refits a structure in each fold. `nested_validate`
also repeats discovery and inner selection inside each outer training partition,
which is the appropriate protocol for assessing the complete discovery process.
