# Vretinaris2020 reproduction benchmark

The benchmark reads only the `arxiv-1910-10856` rows from the sibling EmpiricalRelations.jl data directory and uses only the CFC/SPH catalogue used for calibration in the publication.

## Fixed-form regression

- Catalogue entries: 49
- Refit relations: 48
- Unavailable relations: 1
- Median absolute coefficient difference from the rounded published values: 0.098554
- Largest absolute coefficient difference: 38.02

The detailed coefficient and residual comparison is in `fixed_form_results.csv`. The adjusted coefficient of determination uses five non-intercept predictors. The refit uses the chirp masses and frequencies printed in the publication's source table. Those values are rounded, and one SHEN radius row is absent from the archived TOV table; exact coefficient equality is therefore not expected.

The mass-weighted tidal-deformability relation is marked unavailable because `lambda_tilde` is empty for every CFC/SPH row in `waveforms.csv`. Its formula can be evaluated through EmpiricalRelations.jl, but it cannot be independently refitted from the currently normalized data.

## Symbolic search

The search uses the two representative all-mass `f_peak`/`R_1.6` directions, the operator set `+`, `-`, `*`, 12 populations of 48 members, 300 cycles per iteration, and a parsimony coefficient of `1e-5`. The inverse search receives `f_peak/M_chirp` as a predictor, so the published quadratic family is expressible without an unrestricted division operator. The detailed Pareto-selected results are in `symbolic_results.csv`.

- arxiv-1910-10856-frequency-from-radius-f-peak-khz-r1-6-all-masses, 100 iterations, maximum complexity 25: `(M_chirp * 1.257680497791867) + ((((((M_chirp * ((M_chirp * -0.2075220189003385) + 0.8732666852958471)) + -1.3216213318325656) * M_chirp) + ((R_x * 0.04026509918623264) + -0.6805021394774373)) * R_x) + 11.690839843093636)` (R²=0.97259, RMSE=0.057006; fixed-form RMSE=0.058404).
- arxiv-1910-10856-frequency-from-radius-f-peak-khz-r1-6-all-masses, 250 iterations, maximum complexity 35: `R_x + ((M_chirp * M_chirp) + (((M_chirp * 2.3379305396552854) + ((R_x * ((M_chirp * -0.6184166886749513) + ((((((R_x * -0.0013574883620884918) + 0.05058202583700539) * M_chirp) + 0.008773777774970942) * R_x) + -0.26814619548455965))) + 1.305383028083996)) * (R_x + M_chirp)))` (R²=0.97331, RMSE=0.056254; fixed-form RMSE=0.058404).
- arxiv-1910-10856-radius-from-frequency-r1-6-f-peak-khz-all-masses, 100 iterations, maximum complexity 25: `(((frequency_over_chirp_mass * ((((((frequency_over_chirp_mass * -2.5865338127480073) + 24.608458900203278) * frequency_over_chirp_mass) + -85.74398067337809) * frequency_over_chirp_mass) + 124.48912478840421)) - M_chirp) + -50.377850443853326) + (frequency_over_chirp_mass + frequency_over_chirp_mass)` (R²=0.96592, RMSE=0.19478; fixed-form RMSE=0.1853).
- arxiv-1910-10856-radius-from-frequency-r1-6-f-peak-khz-all-masses, 250 iterations, maximum complexity 35: `((((((frequency_over_chirp_mass * 0.18129235203483535) - 0.41007647396622715) * 1.0563943189534657) + (((M_chirp * 0.5168374129441766) + (((M_chirp + (frequency_over_chirp_mass + -6.983247778031761)) * (frequency_over_chirp_mass * 1.053564957883441)) + 7.412600512353376)) * M_chirp)) * (frequency_over_chirp_mass + -0.10367744808046712)) + 23.458520569108227) - (M_chirp * 8.038772152004523)` (R²=0.97114, RMSE=0.17926; fixed-form RMSE=0.1853).

For these seeds, increasing the budget from 100 iterations/complexity 25 to 250 iterations/complexity 35 reduced RMSE, but the improvement came from higher-order terms outside the published family. For structural recovery, a useful starting policy is therefore 100 iterations, maximum complexity 25, the three polynomial operators, the dimensionless transformed predictor, and several independent seeds. The 250/35 setting is useful as a predictive comparison. Exact recovery of the published six-term family should use a quadratic grammar or the explicit `BasisSet`; an unrestricted tree search is not an identifiability proof.
