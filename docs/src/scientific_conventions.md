# Scientific conventions

- Frequencies should use one canonical unit within a dataset, normally kHz.
- Feature and target units remain explicit metadata. Physical normalization and a
  conservative post-hoc dimensional checker are available, while native symbolic
  searches may use DynamicQuantities.jl dimensional constraints. These checks do not
  replace review of coefficient units and calibration conventions.
- Uncertainty values are positive standard deviations. Inverse-variance fitting
  uses ``1/\sigma^2``.
- Arbitrary importance weights and physical uncertainties are distinct fields.
- Predicted and unreviewed observations are excluded by default when an
  observation-kind column is supplied.
- A transformed target and the space in which residuals are minimized are
  separate choices.
- Standardization parameters are learned from training data only.
- Symbolic expressions remain research candidates until independently
  validated; training loss alone is insufficient evidence.
- Redundant physical variables are flagged, not silently removed.
- Forward and inverse relations are fitted independently in their respective target
  spaces.
- Parameter covariance and bootstrap intervals are conditional on the stated model
  structure unless a complete discovery procedure is explicitly repeated.
- Applicability outside the calibration domain is never implied by a successful fit.
