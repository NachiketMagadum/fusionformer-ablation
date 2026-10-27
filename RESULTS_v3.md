# v3 results (pre-registered)

The design, decision rules, analysis scripts and runners were committed and pushed before any v3 run (commit
`468b2ca`, see [notes/v3/design_v3.md](notes/v3/design_v3.md)). Everything below comes from
`python3 analyse_v3.py` (full output: [notes/v3/v3_stats.txt](notes/v3/v3_stats.txt)).

## What ran

- **Mac block (`run_v3_mac.sh`), no failed runs:**
  - 375 Fusionformer runs:
    - variants no_mswaa, no_mswaa_pm, no_fam, no_fam_pm and adv_only
    - data: 20 simulated v2 series and 5 SKAB files
    - 3 seeds each
  - 99 seq2seq-LSTM tuning runs and 75 final seq2seq runs
- **Parameter-matched sizes:**
  - no_mswaa_pm: d_model 354 (3.29M parameters)
  - no_fam_pm: d_model 426 (3.24M parameters)
  - full model: 3.26M parameters
- **SMD block (`run_v3_smd.sh`):** not run. As the design note specifies, item B (per-channel scoring), the SMD
  ablations and the SMD seq2seq runs are reported as **not done**.

Units: simulated series are labelled *coupled* or *independent* (simulated data); SKAB uses held-out scoring.
Differences are in AUROC, full minus ablated. A negative value means the model **without** the component scored
higher.

## A. MSWAA-off and FAM-off

### Primary family (Holm over 6 tests): no attention component helps

| Group | Full vs MSWAA-off (matched) | Full vs FAM-off (matched) |
|---|---|---|
| coupled | −0.025, Holm p = 0.098, 4/10 series positive | **−0.040, Holm p = 0.0005, 2/10 positive** |
| independent | −0.001, Holm p = 0.77, 5/10 | **−0.008, Holm p = 0.006, 3/10** |
| skab | −0.077, Holm p = 0.060, 1/5 | −0.018, Holm p = 0.46, 1/5 |

- **All six pre-registered verdicts are False.**
- **The attention-free model does better:** with no attention at all (SWSE embedding plus feed-forward blocks only),
  the model scores *higher* than the full Fusionformer on both simulated conditions.
- **Cluster-aware mixed models agree for FAM-off:**
  - coupled: −0.040 [−0.066, −0.014], p = 0.003
  - independent: −0.008 [−0.014, −0.001], p = 0.017
- **MSWAA-off:** no difference from the full model survives Holm or the mixed model.

### Secondary family (Holm within the family)

- **Unmatched FAM-off:** also beats the full model on coupled data (−0.043, Holm p < 0.001, mixed p = 0.001).
- **Which single branch is worth more (MSWEA-only minus MSWAA-only):** on SKAB the MSWEA-only model scored 0.113
  higher (Holm p = 0.007).
  - Per file, the direction was the same on all 5 files, but the mixed model is not significant (p = 0.11).
  - SKAB AUROC is near chance for every variant (0.46-0.57), so this is not read as a meaningful advantage.
- **Simulated data:** the branch comparisons are small and not significant after correction.

### Taken together with v2

- In v2, MSWEA gave a small, consistent gain on independent-channel data against its matched control (+0.004).
- In v3, removing **both** attention branches gives a larger gain in the other direction.
- On these benchmarks, Fusionformer's fusion attention module does not add detection performance over an
  attention-free model of the same size.

## C. Stronger recurrent baseline (seq2seq LSTM with early stopping)

### Tuning (validation only)

| Group | Chosen config | Validation MSE | Mean forecast | Persistence |
|---|---|---|---|---|
| coupled | hidden 256, lr 3e-3 | 0.67 | 0.99 | 1.30 |
| independent | hidden 128, lr 3e-4 | 0.90 | 1.01 | 1.25 |
| skab | hidden 64, lr 3e-3 | 2.44 | 2.77 | 1.35 |

- **Simulated series:** the LSTM now beats both naive forecasters on validation. The v2 LSTM did not.
- **SKAB:** it still does not beat persistence.
- **Early stopping:** the best epoch was usually early (median 5 on the simulated series, range 2-100; median 22 on SKAB).

### Test results

| Group | AUROC, full minus s2s | MAE, s2s minus full |
|---|---|---|
| coupled | +0.204 (30/30 pairs, Holm p < 0.001) | +0.347 |
| independent | +0.434 (30/30 pairs, Holm p < 0.001) | +0.530 |
| skab | +0.013 (n.s.) | +0.245 |

### Conclusion

Fusionformer forecasts and detects far better than this recurrent baseline on the simulated data, and both are near
chance on SKAB.

The stronger LSTM learned only modestly: its validation MSE was well above Fusionformer's forecast error. So the
comparison shows Fusionformer beats *this* LSTM. It does not show that recurrent models in general are weaker.

The new LSTM uses Fusionformer's scoring code, which removes the 23-row alignment offset of the original LSTM script.

## D. The paper's adversarial-only objective

| Group | AUROC, full minus adv_only | Test MAE: adv_only vs full vs mean forecast |
|---|---|---|
| coupled | +0.242 (30/30, Holm p < 0.001) | 3.25 vs 0.36 vs 0.83 |
| independent | +0.296 (30/30, Holm p < 0.001) | 2.58 vs 0.31 vs 0.88 |
| skab | −0.066 (n.s.; all near chance) | 8.94 vs 1.14 vs 1.26 |

- **The forecaster does not learn to forecast.** Trained on the adversarial loss alone (my reading of the paper's
  eq. 14 and Algorithm 1), its errors are 3-7 times worse than predicting the training mean, on every unit.
- **Detection drops too:** on simulated data, AUROC falls from 0.79/0.95 to 0.55/0.65.
- **This supports the deviation in the submitted study:** training on a prediction (Huber) loss was necessary to
  get a working model.
- **Caveat:** if the paper's training schedule differs from my reading, this result applies to my implementation of
  it.

## B. Per-channel scoring on SMD

Not run. This needs the SMD block (`run_v3_smd.sh`, a GPU is recommended).

## Practical notes

- The Mac block took about 107 hours of wall-clock time, far more than the 10-12 hours estimated in the design note.
- Most Fusionformer runs on the simulated series took 1-3 minutes, but a minority took hours (up to about 7). This
  was probably the Mac sleeping or under memory pressure.
- Wall-clock time is not used in any result.
