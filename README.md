# Fusionformer: independent reimplementation and ablation

Code and results for my MSc AI dissertation at Brunel University London (2025-26):
**"Independent Reimplementation and Ablation of Fusionformer for Multivariate Time Series Anomaly Detection: A Paired Multi-Seed Study on SKAB and SMD."**

Fusionformer (Wang et al., 2025, *IEEE TNNLS* 36(8), [doi:10.1109/TNNLS.2025.3542719](https://doi.org/10.1109/TNNLS.2025.3542719)) is a forecasting transformer for multivariate anomaly detection. It has three proposed components:

- **SWSE:** a segment-wise sequence embedding.
- **FAM:** a fusion attention module, whose variable-axis branch is **MSWEA**.
- **An adversarial training loop.**

The paper does not ablate these components or test on public benchmarks. This repository reimplements the architecture in PyTorch and ablates each component under a paired multi-seed design. Every result is kept.

## Main findings

- **No component gives a statistically supported improvement on SMD or held-out SKAB.**
  - The tests are exact Wilcoxon signed-rank tests with Holm correction.
  - On SMD every effect is below 0.005 AUROC.
- **An evaluation leak on SKAB, found and corrected.**
  - Scoring whole SKAB files included the rows the model was trained on. That inflated every model's AUROC by 0.35-0.41 and produced a spurious significant MSWEA effect (p = 0.008).
  - With held-out scoring, every model is near chance.
  - Both versions are archived in `notes/`.
- **Naive-forecast controls.**
  - On SMD a persistence forecast already reaches 0.870 AUROC, against 0.891 for Fusionformer. The benchmark barely rewards forecasting skill.
- **Forecasting.**
  - Fusionformer forecasts with lower MAE than an LSTM baseline on both benchmarks.
  - On SKAB, removing MSWEA or SWSE lowers forecast error further.
- **Simulated benchmark** (clearly labelled simulated data, generator in `synth_generator.py`):
  - In the dissertation (v1, 5 series), MSWEA passed its pre-specified test on cross-channel anomalies: +0.017 AUROC, Holm p = 0.005.
  - Against a parameter-matched control added afterwards, that gain fell to +0.011 and was not significant.
- **v2 (after submission, pre-registered before running; see [RESULTS_v2.md](RESULTS_v2.md)):**
  - 10 new series, with the matched control fixed in advance.
  - The coupled-condition gain did **not** replicate. MSWEA was worse than the matched control there (-0.022, Holm p = 0.022).
  - MSWEA gave a small, consistent gain on independent channels (+0.004, 9 of 10 series).
  - The coupling hypothesis is not supported.

## Repository layout

| Path | Contents |
|---|---|
| `fusionformer_true.py` | Fusionformer model (SWSE, MSWAA, MSWEA, discriminator) with ablation flags |
| `train_ff_forecast_and_score_anomaly.py` | Training and forecast-error anomaly scoring (SMD, SKAB, MSL, simulated) |
| `train_lstm_baseline.py` | LSTM forecasting baseline under the same protocol |
| `naive_baselines.py` | Mean and persistence forecasting controls |
| `synth_generator.py`, `synth_check.py`, `analyse_synth.py`, `match_params.py` | Simulated benchmark, reference detectors, analysis, parameter matching |
| `analyse_final.py` | All Chapter 4 statistics: exact Wilcoxon, t-intervals, d_z, Rosenthal r, Holm, per-unit tests |
| `generate_figures.py` | Figures 4.1-4.4 from the archived results |
| `run_*.sh` | The exact experiment runners used |
| `notes/` | Every per-run result file, `all_runs.csv`, `final_stats.txt`, `naive_baselines.csv`; `notes/synth/` holds the simulated-benchmark design note, addendum and results |
| `figures/` | The figures as submitted |
| `logs/` | Original Mac runs of the full model on SMD (compared with the GPU re-runs) |
| `prototype/`, `exploratory/` | Earlier reconstruction-based prototype and exploratory scripts (not used for any reported number) |

## Reproducing

```bash
pip install -r requirements.txt

# statistics and figures from the archived per-run results (no training needed)
bash reproduce.sh

# full re-runs (datasets needed, see datasets/README.md)
bash run_skab_heldout.sh           # SKAB, held-out scoring, all variants + LSTM + naive controls
bash run_skab_tail_holdout.sh      # second SKAB design
bash run_synth.sh                  # simulated benchmark
bash run_synth_pm.sh               # parameter-matched MSWEA-off control
bash run_synth_v2.sh               # v2: pre-registered replication (10 new series)
```

SMD runs are the slowest: roughly 10 minutes each on an RTX 4090 and up to 45 minutes on an Apple M-series Mac.

## Deviations from the paper

- **Training objective:** the model is trained on a Huber prediction loss, not the paper's adversarial-only objective. `adv_on` adds a small adversarial term (lambda = 0.01).
- **Anomaly scoring:** anomalies are scored by forecast error, not the paper's risk-assessment method.
- **Hyperparameters:** depth, segment length and epochs are reduced for compute.

The full list is in Section 3.6 of the dissertation.

## Citation

```
Magadum, N. (2026) Independent Reimplementation and Ablation of Fusionformer for Multivariate
Time Series Anomaly Detection: A Paired Multi-Seed Study on SKAB and SMD. MSc dissertation,
Brunel University London.
```

Supervised by Professor Xiaohui Liu, a co-author of the Fusionformer paper.

## Licence

MIT, see `LICENSE`. The SKAB, SMD and MSL datasets are not redistributed here and keep their own licences.
