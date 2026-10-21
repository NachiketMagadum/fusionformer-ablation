# v3: pre-registered design

Written on 21 October 2026, before any v3 model was trained. This note, the analysis scripts (`analyse_v3.py`,
`smd_perchannel.py`, `select_s2s_config.py`) and the runners (`run_v3_mac.sh`, `run_v3_smd.sh`) are committed and
pushed to GitHub before the runs start, so the commit and push times are the external timestamp for this design.
A smoke test (`run_v3_smoke.sh`: 1-2 epochs on the v1 series `coupled_0`, which is not a v3 unit) was run only to
check that the code paths work; its outputs are not analysed.

v3 covers the four items still open after v2:

- **A.** The remaining attention ablations (MSWAA-off and FAM-off).
- **B.** Per-channel scoring on SMD.
- **C.** A stronger recurrent baseline.
- **D.** The paper's adversarial-only objective.

## Shared settings

- **Settings:** the same code and settings as the submitted study and v2. The full settings are:
  - T = 96, tau = 24, L_seg = 4, d_model = 252, 6 heads, one encoder and one decoder block, 15 epochs.
  - Huber loss.
  - Forecast-error anomaly score with the same row alignment.
- **Model seeds:** 0, 1 and 42.
- **Data groups:**

| Group | Units | Pairs per comparison |
|---|---|---|
| coupled | simulated v2 series `coupled_100` to `coupled_109` (simulated data, labelled as such) | 30 |
| independent | simulated v2 series `independent_100` to `independent_109` | 30 |
| skab | SKAB valve1 files 0-4, held-out scoring (train before the first anomaly, score after it) | 15 |
| smd | SMD machine-1-1, machine-1-4, machine-2-1 | 9 (3 units, descriptive only) |

- **Reference runs (reused, not re-run):**
  - full, no_mswea and no_mswea_pm on the simulated series (`notes/synth_v2/`)
  - full and no_mswea on SKAB (`notes/`, held-out design)
- **Code changes:**
  - The code for the full and MSWEA-off models is unchanged in effect: modules are created in the same order, so
    initialisation for a given seed is the same.
  - The only edits to the training script are new flags, plus printing the batch size actually used.
- **SMD:** SMD comparisons use the new v3 SMD full runs from item B.
- **Statistics:**
  - The paired difference is in AUROC unless stated otherwise.
  - Each comparison gets a two-sided exact Wilcoxon signed-rank test with Holm correction within its family, a
    95% t-interval and d_z.
  - Each comparison also gets a per-unit summary (seed-averaged differences) and a random-intercept mixed model
    (statsmodels).
- **Reporting:** every verdict is reported whatever it is. No test is added to a primary family after results are
  seen.

## A. MSWAA-off and FAM-off

### Variants

All variants are in `fusionformer_true.py` and `train_ff_forecast_and_score_anomaly.py`:

| Variant | Description |
|---|---|
| `no_mswaa` | Intravariable branch (MSWAA and its FFN) removed: MSWEA → FFN. Mirrors the existing `no_mswea`. |
| `no_fam` | Both attention branches removed. One FFN is kept per block, so each block is FFN only. |
| `no_mswaa_pm`, `no_fam_pm` | The same, with d_model raised so the parameter count matches the full model. The d_model comes from `match_params.py --variant ...` (8 channels) and is recorded in `notes/v3/param_match_*.txt`. |

### Primary family (Holm over 6 tests)

- **Tests:** full vs `no_mswaa_pm` and full vs `no_fam_pm`, in each of coupled, independent and skab.
- **Decision rule:** a component helps in a group if all three hold:
  - Holm p < 0.05
  - mean difference > 0
  - a positive seed-averaged difference on at least 70% of units (7 of 10 series, 4 of 5 SKAB files)

### Secondary family (Holm within the family)

- full vs `no_mswaa` (unmatched)
- full vs `no_fam` (unmatched)
- `no_mswea` vs `no_mswaa`, which asks which single branch is worth more

Each is tested in coupled, independent and skab.

### SMD

`no_mswaa` and `no_fam` are compared against the v3 SMD full runs. With 3 machines this is descriptive only.

## B. Per-channel scoring on SMD

The submitted score pooled the squared forecast error over all 38 channels. Channels with large natural error can
then hide an anomaly in a quiet channel.

### Scores

`smd_perchannel.py` computes the following, where e[w, d] is the squared error of channel d in window w, averaged
over the horizon:

| Score | Definition | Role |
|---|---|---|
| pooled | Mean over channels of e[w, d] | The submitted score, reproduced |
| pc_max | Max over channels of z[w, d] | Primary |
| pc_mean | Mean over channels of z[w, d] | Secondary |

- **The robust z:** z[w, d] = (e[w, d] - median_d) / IQR_d.
- **Medians and IQRs:** both are taken over the model's training windows.
- **IQR floor:** IQR_d is floored at 1% of the median IQR across channels, because some SMD channels are constant in
  training.
- **Runs:** the full model is re-run on the 3 machines x 3 seeds with `--save_errors`, which saves the per-channel
  errors.
- **Naive floor:** the same three scores are computed for the naive mean and persistence forecasters. The naive
  floor is also computed on all 28 SMD machines, descriptively.
- **Pre-run check:** before this note was written, the pooled naive AUROC from `smd_perchannel.py` was checked
  against `notes/naive_baselines.csv` for the 3 machines and matches exactly. Per-channel scores were not computed
  before this commit.

### Tests

- **B1:** pc_max minus pooled for Fusionformer. Paired over 9 runs, exact Wilcoxon, two-sided.
- **B2 (primary verdict):** "per-channel scoring widens Fusionformer's margin over persistence". True only if the
  seed-averaged margin (Fusionformer minus persistence) under pc_max is larger than under pooled on every one of the
  3 machines.
- **Interpretation:** with 3 machines, B is reported as descriptive evidence, not as a general claim about SMD.

## C. Stronger recurrent baseline (`train_lstm_s2s.py`)

### Why

The original LSTM (last hidden state into a linear head, 15 epochs) never beat the mean forecast on validation in
the v2 tuning (validation MSE 0.87-1.61).

### Model

- **Architecture:** a 2-layer LSTM encoder-decoder.
  - The decoder starts from the encoder state.
  - It rolls out 24 steps, feeding back its own prediction. The first input is the last observed row.
- **Training:**
  - Huber loss, AdamW, batch 64.
  - Up to 100 epochs, with early stopping (patience 10) on validation MSE. The best weights are kept.
- **Validation split:** the last max(20%, 170) training rows. The model is fitted on the rows before them.
  - On SKAB this leaves about 400 fit rows.
  - This also means the LSTM sees fewer training rows than Fusionformer, which uses all of them.

### Scoring

- s2s uses exactly the Fusionformer scoring code (`broadcast_scores_to_rows` and the same eval_start cut), so the
  scores are directly comparable.
- **Alignment difference found while writing v3:** the original LSTM script used a different convention.
  - It gave row t the score of the window whose horizon ends at t; Fusionformer uses the window whose horizon starts
    at t.
  - It also dropped the first 119 rows.
  - Both conventions score each row with a forecast that covers it. The 23-row offset was not mentioned in the
    dissertation.

### Tuning (validation only, seed 0)

- **Grid:** hidden {64, 128, 256} x lr {3e-4, 1e-3, 3e-3}.
- **Tuning units:**
  - coupled_100-102
  - independent_100-102
  - SKAB files 0-4
  - the 3 SMD machines (only if the SMD block is run)
- **Selection:** the lowest mean validation MSE per group. Ties go to the smaller hidden size, then the smaller lr.
- **Learning check:** `select_s2s_config.py` also reports whether the chosen configuration's validation MSE is below
  both the mean forecast and persistence on the same split.

### Final runs and tests

- **Runs:** the chosen configuration, 3 seeds, all units of the group.
- **Tests:**
  - full vs s2s AUROC, Holm over the groups that have data.
  - Forecast MAE, s2s minus full.

## D. The paper's adversarial-only objective (`--variant adv_only`)

### Objective

As I read eq. 14 and Algorithm 1 of the paper:

- The forecaster is updated on the adversarial loss alone, BCE of D(history, forecast) against "real".
- There is no prediction loss and no warm-up.
- It alternates with one discriminator step per batch.

### Settings

- Learning rates and epochs are the same as the full model.
- This is my reading of the paper. If the paper's text specifies a different schedule, that difference is a
  limitation of this test.

### Tests

- **AUROC:** full vs adv_only, Holm over coupled, independent and skab.
- **MAE:** adv_only minus full.
- **Descriptive:** adv_only test MAE compared with the training-mean forecast's MAE on the same windows. This shows
  whether the adversarial-only model forecasts at all.

## Compute plan

- **`run_v3_mac.sh`:** SKAB, seq2seq and the simulated series. About 550 runs, roughly 10-12 hours on an M-series
  Mac.
- **`run_v3_smd.sh`:** items B, A-SMD and C-SMD. Best on a rented GPU.

If the SMD block is not run, items B, A-SMD and C-SMD are reported as not done. Nothing else changes.
