# v2 results: pre-registered test of MSWEA against a parameter-matched control

The design, analysis script and runner were committed and pushed before any v2 model was trained
(commit `1450506`, file `notes/synth_v2/design_v2.md`).

## Setup

- **Data:** 10 new simulated series per condition (generator seeds 100-109, not used in the dissertation).
- **Runs:** 3 model seeds per series, so 30 pairs per comparison and 180 training runs (219 minutes on an Apple M-series Mac).
- **Parameter-matched MSWEA-off:** d_model 354, giving 3.29M parameters against 3.26M for the full model.
- **Benchmark checks:** the reference detectors behave as intended.
  - Coupled series: a cross-channel regression scores 0.995 AUROC, while mean and persistence forecasts sit near chance.
  - Independent series: the cross-channel regression scores 0.48, and a per-channel AR(20) model scores 0.83.

## Pre-registered tests (two-sided exact Wilcoxon, Holm over four tests)

| Condition | Comparison | Mean AUROC difference (full - variant) | 95% CI | Pairs full higher | Holm p |
|---|---|---|---|---|---|
| Coupled | vs MSWEA-off | -0.0200 | [-0.0361, -0.0038] | 9 / 30 | 0.022 |
| Coupled | vs parameter-matched MSWEA-off | -0.0216 | [-0.0368, -0.0063] | 9 / 30 | 0.022 |
| Independent | vs MSWEA-off | +0.0051 | [+0.0034, +0.0068] | 25 / 30 | 0.0001 |
| Independent | vs parameter-matched MSWEA-off | +0.0040 | [+0.0021, +0.0059] | 23 / 30 | 0.0015 |

**H1** (MSWEA beats the parameter-matched control on coupled data): **not supported**.
- The effect goes the other way: the full model is lower on 21 of 30 pairs, and higher on only 3 of the 10 series.

**H2** (the matched gain is larger on coupled than on independent data): **not supported**.
- The difference is significant in the *opposite* direction: the coupled median is -0.014 and the independent median is +0.004 (Mann-Whitney p = 0.038).

## What this means

- **The v1 finding did not replicate.** In the dissertation, MSWEA appeared to help most when anomalies break cross-channel relationships (+0.017 on 5 series).
  - On 10 new series, with the matched control fixed in advance, the coupled-condition effect is negative.
  - It is also highly variable across series: one series (101) shows -0.124, and the rest range from -0.036 to +0.010.
  - The per-series test does not reach significance (n = 10, p = 0.08).
- **What does replicate is a small, consistent gain on independent-channel data.**
  - The gain is +0.004 to +0.005 AUROC, on 9 of 10 series, and it survives parameter matching.
- **Overall:** on this simulated benchmark, the variable-axis attention branch does **not** specifically help with cross-channel anomalies. The coupling hypothesis from the dissertation should be treated as refuted here.
- **Lesson:** five series per condition were too few to judge an effect of about 0.01-0.02 AUROC, given how much it varies between series.

Full output: `notes/synth_v2/synth_v2_stats.txt`, per-run results `notes/synth_v2/all_runs_synth_v2.csv`.
