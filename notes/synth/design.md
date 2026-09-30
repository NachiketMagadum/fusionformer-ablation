# Simulated benchmark: design fixed before any model was trained

Written on 26 September 2026, before `run_synth.sh` was run. The only thing checked beforehand was
that the generator behaves as intended: the reference detectors in `synth_check.py` (mean,
persistence, cross-channel regression, own-history AR). No Fusionformer or LSTM result had been
produced on this data when this note was written.

## Purpose

SMD and SKAB could not answer RQ1: on SMD a persistence forecast almost matches the trained models, and on
held-out SKAB every model is near chance. This experiment builds data where the answer is observable,
in order to test the variable-coupling hypothesis from the earlier draft directly.

## Data (synth_generator.py, clearly simulated)

- 8 channels, 2000 normal training rows, 2000 test rows with 5 labelled anomaly segments (30-60 rows, about 11% of rows).
- **Coupled condition:** every channel is a noisy, lagged mix of two shared latent signals (periods 50 and 130).
  An anomaly replaces 2-3 channels with the same channel's own signal from a different time (raised-cosine
  crossfade at the edges). Level, amplitude and rhythm are unchanged; only the relationship between channels breaks.
- **Independent condition:** every channel has its own sinusoid. An anomaly speeds up the rhythm of 2-3 channels
  (period x 0.6) at the same amplitude.
- 5 simulated series per condition (generator seeds 0-4) are the units.

## Models and protocol

Full Fusionformer and MSWEA-off, 3 seeds each (0, 1, 42), plus the LSTM baseline, all with exactly the same
settings, training loop and scoring as the SMD runs (T = 96, tau = 24, forecast-error AUROC, separate test set).

## Pre-specified test and decision rule

- **Primary test:** MSWEA on the coupled condition. The paired AUROC difference is full minus MSWEA-off
  (n = 15), tested with a two-sided exact Wilcoxon test, with Holm correction over the two MSWEA tests
  (coupled and independent).
- **"MSWEA helps"** requires all three of: Holm p < 0.05, a positive mean difference, and a positive
  seed-averaged difference on at least 4 of the 5 series.
- **Prediction under the coupling hypothesis:** MSWEA helps in the coupled condition and not in the
  independent one. Any other outcome is reported as it is.
