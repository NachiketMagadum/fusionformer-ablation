"""
Naive forecasting controls for the anomaly-scoring pipeline.

Two forecasters that learn nothing beyond the training mean:
  mean        - forecast every future step as the training mean (0 after scaling)
  persistence - forecast every future step as the last observed input row

Both are scored exactly like Fusionformer: forecast MSE over the next
PRED_LEN steps, broadcast to rows with the same alignment, and (for SKAB)
evaluated only on rows after the training cut. They give a floor: if a
trained model barely beats the mean forecast in AUROC, the benchmark AUROC
is being driven by how far anomalies sit from normal, not by forecast skill.

Usage:
    python3 naive_baselines.py            # all SMD machines and SKAB files used in the study
    python3 naive_baselines.py --skab_train_frac 0.8 --out_dir notes/skab_tail_holdout

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from train_ff_forecast_and_score_anomaly import (
    load_smd_machine, load_skab_file, make_forecast_pairs,
    broadcast_scores_to_rows, SEQ_LEN, PRED_LEN,
)

SMD_MACHINES = ["machine-1-1", "machine-1-4", "machine-2-1"]
SKAB_FILES = [f"datasets/SKAB/data/valve1/{i}.csv" for i in range(5)]


def naive_window_errors(X_test_s, method):
    inputs, targets = make_forecast_pairs(X_test_s, SEQ_LEN, PRED_LEN)
    if method == "mean":
        pred = np.zeros_like(targets)
    else:  # persistence
        pred = np.repeat(inputs[:, -1:, :], PRED_LEN, axis=1)
    err = pred - targets
    return (err ** 2).mean(axis=(1, 2)), np.abs(err).mean(axis=(1, 2))


def evaluate(X_train, X_test, y_test, eval_start, method):
    scaler = StandardScaler().fit(X_train)
    X_test_s = scaler.transform(X_test)
    w_mse, w_mae = naive_window_errors(X_test_s, method)
    scores = broadcast_scores_to_rows(w_mse, len(X_test_s), SEQ_LEN, PRED_LEN)[eval_start:]
    y_eval = np.asarray(y_test)[eval_start:]
    first_win = max(0, eval_start - SEQ_LEN)
    p, r, _ = precision_recall_curve(y_eval, scores)
    return dict(auroc=roc_auc_score(y_eval, scores), pr_auc=auc(r, p),
                mse=float(w_mse[first_win:].mean()), mae=float(w_mae[first_win:].mean()),
                anom_rate=float(y_eval.mean()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skab_train_frac", type=float, default=1.0)
    ap.add_argument("--out_dir", default="notes")
    args = ap.parse_args()
    rows = []
    for m in SMD_MACHINES:
        X_train, X_test, y_test = load_smd_machine(m)
        for method in ("mean", "persistence"):
            rows.append(dict(method=method, dataset="smd", id=m,
                             **evaluate(X_train, X_test, y_test, 0, method)))
    for f in SKAB_FILES:
        X_train, X_test, y_test, eval_start = load_skab_file(f, args.skab_train_frac)
        for method in ("mean", "persistence"):
            rows.append(dict(method=method, dataset="skab", id=Path(f).stem,
                             **evaluate(X_train, X_test, y_test, eval_start, method)))

    out = Path(args.out_dir) / "naive_baselines.csv"
    out.parent.mkdir(exist_ok=True)
    keys = ["method", "dataset", "id", "auroc", "pr_auc", "mse", "mae", "anom_rate"]
    with open(out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(round(r[k], 4)) if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    for r in rows:
        print(f"{r['method']:12s} {r['dataset']:5s} {r['id']:12s} AUROC {r['auroc']:.4f}  "
              f"PR-AUC {r['pr_auc']:.4f}  MSE {r['mse']:.4f}  MAE {r['mae']:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
