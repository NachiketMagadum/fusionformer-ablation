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

---

# Cluster-aware re-analysis (all ablations)

Seeds on the same machine, file or series are not independent. `analyse_mixed.py` re-estimates every comparison in
two ways: per-unit means (seeds averaged first) and a mixed model with a random intercept per unit. Full table:
`notes/mixed_model_stats.txt`.

- **SMD and SKAB (both designs):** no component effect is distinguishable from zero. Every per-unit interval and
  every mixed-model interval includes 0. With 3 or 5 units these analyses have little power, which is itself the point.
- **Simulated v1 coupled:** the mixed model gives +0.017 (p = 0.017), but the per-unit test does not (p = 0.125, n = 5).
  The v2 replication then reversed it.
- **Simulated v2 coupled:** the negative MSWEA effect (-0.020 / -0.022 against the matched control) is **not**
  significant once series are treated as the unit (mixed p = 0.13 / 0.08, per-series p = 0.23 / 0.08).
  - The pre-registered pair-level test (Holm p = 0.022) is driven by between-series variation, mostly series 101.
- **Independent-channel data:** the one effect that holds everywhere is a small gain. It is +0.0035 in v1 and
  +0.0051 in v2, and it survives parameter matching (+0.0031 / +0.0040).
  - v2: 9 of 10 series positive, mixed-model p = 0.004 against the matched control.

**Summary:** the most defensible statement is that MSWEA gives a small, consistent gain of about 0.004 AUROC on the
independent-channel simulated series. There is no reliable cross-channel-specific effect, in either direction.

# Tuned LSTM baseline (pre-registered, commit `d97806c`)

- **Grid:** hidden size {64, 128, 256} x learning rate {1.5e-4, 1e-3, 3e-3}, selected on validation MSE from the last 20%
  of the training rows only (`notes/lstm_tune/tuning_grid.txt`).
- **Chosen configurations:**
  - coupled: hidden 64, lr 1.5e-4
  - independent: hidden 256, lr 1e-3

**Results on the v2 series (30 pairs per condition):**

| Condition | Fusionformer AUROC | Tuned LSTM AUROC | Holm p | Fusionformer MAE | Tuned LSTM MAE |
|---|---|---|---|---|---|
| Coupled | 0.788 | 0.487 | < 0.001 | 0.356 | 0.781 |
| Independent | 0.946 | 0.454 | < 0.001 | 0.310 | 0.818 |

**Tuning did not rescue the LSTM.**
- Every configuration in the grid had a validation MSE of 0.87-1.61 on standardised data. That is no better than
  forecasting the mean, whose MSE is about 1.
- So within this grid the LSTM never learns the simulated dynamics, and its AUROC stays at chance.
- The limitation looks like the baseline's design and training budget (last-state linear head, 15 epochs) rather
  than its hidden size or learning rate. A stronger recurrent baseline would need a different decoder or longer
  training, which was not tested here.
