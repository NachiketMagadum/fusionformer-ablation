# Tuned LSTM baseline: pre-registered design

Written on 30 September 2026, before any tuning run. It is committed and pushed to GitHub before the runs start.

## Why

In the dissertation the LSTM used fixed settings (hidden 128, lr 1.5e-4, 15 epochs) and scored near chance on the
simulated series. An untuned baseline says little, so this gives it a fair tuning budget.

## Data and tuning

- **Data:** the v2 simulated series (seeds 100-109, 10 per condition), the same series as the v2 Fusionformer runs.
- **Grid:** hidden size {64, 128, 256} x learning rate {1.5e-4, 1e-3, 3e-3}, with 2 layers and 15 epochs kept, as in the
  main study.
- **Selection, using training data only:** for every series, seed 0 is trained on the first 80% of the training rows
  and scored by forecast MSE on the last 20%. Test data is never used. For each condition, the configuration with
  the lowest mean validation MSE across its 10 series is chosen (`select_lstm_config.py`).

## Test (fixed now)

The chosen configuration is trained on the full training rows with seeds 0, 1 and 42 for all 20 series.

- **Primary:** paired AUROC, full Fusionformer (v2 runs) vs tuned LSTM, per condition. The test is a two-sided exact
  Wilcoxon on 30 pairs, Holm over the two conditions.
- **Secondary:** paired forecast MAE.

All results are reported as they come out.
