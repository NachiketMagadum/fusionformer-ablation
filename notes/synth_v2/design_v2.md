# v2 simulated benchmark: pre-registered design

Written on 30 September 2026, after the dissertation was submitted and before any v2 model was trained.
This file, `analyse_synth_v2.py` and `run_synth_v2.sh` are committed and pushed to GitHub before the runs start,
so the commit and push times are the external timestamp for this design.

## Why a v2

In the submitted study (v1) the parameter-matched control was added after the main simulated results were seen, and
it cut the coupled-condition MSWEA gain from +0.017 to +0.011 (not significant). v2 tests the same question with the
matched control fixed in advance, on new data, with more series.

## Data

`synth_generator.py`, unchanged from v1, with generator seeds 100-109, which are new series not used in v1:
10 coupled and 10 independent series (8 channels, 2,000 training rows, 2,000 test rows, 5 anomaly segments).
The simulated data is labelled as simulated everywhere.

## Models

The same code and settings as v1 (T = 96, tau = 24, 15 epochs, forecast-error AUROC), with model seeds 0, 1 and 42:

- `full`: Fusionformer, d_model 252
- `no_mswea`: MSWEA removed, d_model 252
- `no_mswea_pm`: MSWEA removed, d_model chosen by `match_params.py` so the parameter count matches `full`
  (expected 354, as in v1)

This gives n = 30 pairs per comparison.

## Tests (all fixed now)

- **Tests:** four paired comparisons (full vs no_mswea, full vs no_mswea_pm, in each condition), each a two-sided exact
  Wilcoxon signed-rank test on the paired AUROC differences, Holm-corrected together.
- **H1 (primary):** MSWEA beats the parameter-matched control on coupled data. This requires all three of:
  - Holm p < 0.05
  - a positive mean difference
  - a positive seed-averaged difference on at least 7 of the 10 series
- **H2 (specificity):** the matched gain is larger on coupled than on independent data. This requires a two-sided
  Mann-Whitney U test on the 10 + 10 seed-averaged matched differences with p < 0.05, and a higher coupled median.
- **Secondary:** per-series Wilcoxon tests (n = 10) and the reference detectors (`synth_check.py`) for the new series.

Whatever the verdicts are, they are reported. No other test is added to the primary set after the results are seen.
