# Scientific conventions

- Frequencies should use one canonical unit within a dataset, normally kHz.
- Feature and target units are metadata; automatic dimensional algebra is not
  yet implemented. Basis terms provide a unit field for explicit review.
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
