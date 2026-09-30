"""
Reference detectors for the simulated benchmark (Section 4.9).

Checks that the simulated anomalies behave as designed before any
Fusionformer result is looked at. Four simple detectors, none of them trained
beyond least squares:
  mean          forecast the training mean (same pipeline as naive_baselines.py)
  persistence   repeat the last observed value (same pipeline)
  cross-channel predict each channel from the other channels at the same time
                step (linear regression fitted on training data); score = squared
                residual, smoothed over 24 rows
  own-history   predict each channel from its own previous 20 values (AR(20),
                one step ahead); score = squared residual, smoothed over 24 rows

Expected if the generator works: mean and persistence near 0.5 in both
conditions; cross-channel high in the coupled condition and near 0.5 in the
independent one; own-history clearly above 0.5 in the independent condition.

Writes notes/synth/reference_detectors.csv

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from naive_baselines import naive_window_errors
from train_ff_forecast_and_score_anomaly import broadcast_scores_to_rows, SEQ_LEN, PRED_LEN


def smooth(x, k=24):
    return pd.Series(x).rolling(k, center=True, min_periods=1).mean().values


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--first_seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out_dir", default="notes/synth")
    args = ap.parse_args()
    rows = []
    for cond in ("coupled", "independent"):
        for u in range(args.first_seed, args.first_seed + args.n):
            z = np.load(Path("datasets/SYNTH") / f"{cond}_{u}.npz")
            Xtr, Xte, y = z["X_train"], z["X_test"], z["y_test"]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            A, B = (Xtr - mu) / sd, (Xte - mu) / sd
            out = {}
            for method in ("mean", "persistence"):
                w, _ = naive_window_errors(B, method)
                out[method] = roc_auc_score(y, broadcast_scores_to_rows(w, len(B), SEQ_LEN, PRED_LEN))
            D = B.shape[1]
            sc = np.zeros(len(B))
            for d in range(D):
                o = [k for k in range(D) if k != d]
                coef, *_ = np.linalg.lstsq(np.c_[A[:, o], np.ones(len(A))], A[:, d], rcond=None)
                sc += (B[:, d] - np.c_[B[:, o], np.ones(len(B))] @ coef) ** 2
            out["cross_channel"] = roc_auc_score(y, smooth(sc))
            sc = np.zeros(len(B)); p = 20
            for d in range(D):
                Xa = np.stack([A[i:len(A) - p + i, d] for i in range(p)], 1)
                c, *_ = np.linalg.lstsq(Xa, A[p:, d], rcond=None)
                Xb = np.stack([B[i:len(B) - p + i, d] for i in range(p)], 1)
                sc += np.r_[np.zeros(p), (B[p:, d] - Xb @ c) ** 2]
            out["own_history"] = roc_auc_score(y, smooth(sc))
            rows.append(dict(condition=cond, unit=u, anomaly_rate=round(float(y.mean()), 3),
                             **{k: round(v, 4) for k, v in out.items()}))
    df = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out_dir) / "reference_detectors.csv", index=False)
    print(df.to_string(index=False))
    print(df.groupby("condition").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
