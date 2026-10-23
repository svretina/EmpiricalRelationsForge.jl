# Vretinaris2020 reproduction benchmark

The repository includes a source-specific benchmark in
`benchmarks/vretinaris2020`. It exercises the fixed-form regression and
symbolic-discovery paths against the Vretinaris et al. (2020) catalogue stored
by EmpiricalRelations.jl.

## Fixed-form models

For each postmerger frequency $f_j\in\{f_{\rm peak},f_{2-0},f_{\rm spiral}\}$,
fiducial radius $R_x\in\{R_{1.2},R_{1.4},R_{1.6},R_{1.8}\}$, and published
population selection, the forward model is

```math
\frac{f_j}{M_{\rm chirp}} = b_0+b_1M_{\rm chirp}+b_2R_x
+b_3M_{\rm chirp}^2+b_4R_xM_{\rm chirp}+b_5R_x^2.
```

The inverse model is

```math
R_x=b_0+b_1M_{\rm chirp}+b_2\frac{f_j}{M_{\rm chirp}}
+b_3M_{\rm chirp}^2+b_4f_j
+b_5\left(\frac{f_j}{M_{\rm chirp}}\right)^2.
```

Both are linear least-squares problems in their coefficients. The benchmark
uses an explicit `BasisSet`; the forward problem applies
`FeatureScaleTarget(:M_chirp, power=-1)` so optimization occurs in the
published $f_j/M_{\rm chirp}$ space while predictions and residual summaries
remain in kHz.

The reproducibility fit uses the chirp masses printed in the publication's
source table. They are rounded to two decimals, but they recover the published
inverse-fit coefficients more closely than chirp masses recomputed from the
component masses. The TeX archive does not contain the original fitting script
or state whether a higher-precision private table was used, so this choice is
the directly auditable interpretation of the published input.

The inverse $R_{1.2}$ and $R_{1.4}$ fits use
$M_{\rm chirp}<1.3\,M_\odot$, $R_{1.6}$ uses the full mass range, and
$R_{1.8}$ uses $M_{\rm chirp}>1.3\,M_\odot$. Equal-mass fits additionally
require $q=1$. Missing secondary-frequency measurements and missing EOS
radii are excluded relation by relation.

The waveform label `SLY4` is joined to the `SLy` row in the source TOV table;
these are the same EOS label under the capitalization used by the two source
tables. `SHEN` has no corresponding TOV row and is therefore excluded from
radius fits.

## Symbolic parameter study

The symbolic study uses representative forward and inverse all-mass
$f_{\rm peak}$-$R_{1.6}$ problems. Searches use addition, subtraction, and
multiplication. The inverse search receives
$f_{\rm peak}/M_{\rm chirp}$ as a predictor, keeping the published quadratic
family inside this operator grammar without unrestricted division. Multiple
independently evolving populations, explicit complexity bounds, a small
parsimony pressure, and fixed seeds control each run. The search target for
the forward direction is also $f_{\rm peak}/M_{\rm chirp}$, matching the
published regression space. The generated result table records the selected expression, loss,
complexity, residual statistics, seed, and search budget; it is intended as a
starting point for subsequent searches, not as evidence that one stochastic
run identifies a unique physical law.

The committed benchmark run shows the practical tradeoff. With 100
iterations and a maximum complexity of 25, the forward representative reaches
the fixed quadratic model's error scale while retaining recognizable
low-order structure. Increasing the budget to 250 iterations and complexity
35 lowers training error in both directions, but does so by admitting
higher-order terms outside the published family. The smaller setting is the
recommended structural-search starting point; the larger setting is a useful
predictive comparison. Multiple seeds and held-out validation remain
necessary before interpreting a discovered expression.

These published models are linear in the six coefficients, so their
fixed-form refits use the package's QR/SVD linear path. `NonlinearSolve.jl` is
the default backend for models that are nonlinear in their parameters; it is
not used merely because the basis contains powers or products.

## Reproducibility boundary

The benchmark reads only `arxiv-1910-10856` rows and only the CFC/SPH
calibration catalogue. The normalized waveform table currently has no
mass-weighted tidal-deformability values. Consequently, the published

```math
f_{\rm peak}M_{\rm chirp}=b_0+b_1M_{\rm chirp}
+b_2\widetilde{\Lambda}^{-1/2}
```

relation is retained in the one-row-per-relation output but marked
unavailable for independent refitting. Published input rounding and the
missing SHEN radius row also prevent exact
coefficient equality; residual and adjusted-(R^2) agreement are the more
useful reproduction checks.

Adjusted $R^2$ is computed in the space minimized by each published fit:
$f_j/M_{\rm chirp}$ for forward relations and $R_x$ for inverse
relations. Maximum residuals are reported in the physical output unit (kHz or
km), matching the catalogue metadata.
