# Architecture

`RegressionDataset` fixes the internal numerical convention to observations by
features and keeps row, EOS, simulation, paper, unit, uncertainty, weighting,
and transformation metadata beside the numbers.

Preprocessing is split into a learn/apply pair. For example,
`fit_standardizer(training_data)` learns statistics only from a training
partition, and `transform_features` applies them later. Grouped splitters keep
EOS, simulation, or paper groups from crossing partition boundaries.

EmpiricalRelations.jl currently has no public measurement-table API. The named
provider method therefore reports this limitation rather than depending on
private implementation details. Direct table construction is the supported
path until that provider contract exists.
