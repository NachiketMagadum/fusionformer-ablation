"""
v3, item B: per-channel normalised anomaly scoring on SMD, with a naive floor.
Design and decision rules: notes/v3/design_v3.md (section B), fixed before the runs.

The submitted study pooled the forecast error over all 38 SMD channels
(mean squared error over the window and channels). Channels with naturally
large errors can then swamp an anomaly in a quiet channel. Here every channel's
error is first put on its own scale:

    e[w, d]   = squared forecast error of channel d in window w, averaged over the 24-step horizon
    z[w, d]   = (e[w, d] - median_d) / iqr_d       median and IQR over the TRAINING windows
    iqr_d     floored at 1% of the median IQR across channels (SMD has channels that are constant in training)
    pc_max    = max over channels of z      (primary per-channel score)
    pc_mean   = mean over channels of z     (secondary)
    pooled    = mean over channels of e     (the submitted score, reproduced for comparison)

The same three scores are computed for the two naive forecasters (training
mean and persistence), so the gap between Fusionformer and the naive floor can
be compared under pooled and per-channel scoring.

Inputs : notes/v3/smd/ff_true_full_smd_<machine>_seed<s>.npz (written by
         train_ff_forecast_and_score_anomaly.py --save_errors)
Outputs: notes/v3/smd/perchannel_runs.csv, perchannel_naive.csv, perchannel_stats.txt
         notes/v3/smd/perchannel_naive_all_machines.csv (--all_machines: naive floor on all 28 SMD machines)

numpy / scikit-learn / scipy only (no torch), so it can be re-run anywhere.

Author: Nachiket Magadum
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEQ_LEN, PRED_LEN = 96, 24
MACHINES = ["machine-1-1", "machine-1-4", "machine-2-1"]
OUT = Path("notes/v3/smd")


def load_smd(machine):
    root = Path("datasets/SMD")
    X_tr = pd.read_csv(root / "train" / f"{machine}.txt", header=None).values
    X_te = pd.read_csv(root / "test" / f"{machine}.txt", header=None).values
    y = pd.read_csv(root / "test_label" / f"{machine}.txt", header=None).iloc[:, 0].values
    return X_tr, X_te, y


def broadcast(w, n_rows):
    """Same alignment as broadcast_scores_to_rows in the training script:
    rows < SEQ_LEN get window 0; row t >= SEQ_LEN gets window min(t - SEQ_LEN, last)."""
    idx = np.clip(np.arange(n_rows) - SEQ_LEN, 0, len(w) - 1)
    return w[idx]


def naive_errors(X, method):
    """Per-window, per-channel squared error of a naive forecast: (N_windows, D)."""
    n = len(X) - SEQ_LEN - PRED_LEN + 1
    tgt = sliding_window_view(X[SEQ_LEN:], PRED_LEN, axis=0)[:n]           # (n, D, PRED_LEN)
    if method == "mean":
        return (tgt ** 2).mean(axis=2)
    last = X[SEQ_LEN - 1:SEQ_LEN - 1 + n][:, :, None]                     # (n, D, 1)
    return ((tgt - last) ** 2).mean(axis=2)


def scores(E_tr, E_te):
    med = np.median(E_tr, axis=0)
    iqr = np.subtract(*np.percentile(E_tr, [75, 25], axis=0))
    iqr = np.maximum(iqr, 0.01 * np.median(iqr) + 1e-12)
    z = (E_te - med) / iqr
    return {"pooled": E_te.mean(axis=1), "pc_max": z.max(axis=1), "pc_mean": z.mean(axis=1)}


def aurocs(E_tr, E_te, y):
    return {k: roc_auc_score(y, broadcast(v, len(y))) for k, v in scores(E_tr, E_te).items()}


def naive_table(machines):
    rows = []
    for m in machines:
        X_tr, X_te, y = load_smd(m)
        sc = StandardScaler().fit(X_tr)
        A, B = sc.transform(X_tr), sc.transform(X_te)
        for method in ("mean", "persistence"):
            rows.append(dict(machine=m, method=method, **aurocs(naive_errors(A, method), naive_errors(B, method), y)))
        print(rows[-2]); print(rows[-1])
    return pd.DataFrame(rows)


def wil(d):
    d = np.round(np.asarray(d, float), 6); nz = d[d != 0]
    return stats.wilcoxon(nz).pvalue if len(nz) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_machines", action="store_true", help="also run the naive floor on all 28 machines")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    runs = []
    for f in sorted(OUT.glob("ff_true_full_smd_*_seed*.npz")):
        m, s = re.match(r"ff_true_full_smd_(.+)_seed(\d+)$", f.stem).groups()
        z = np.load(f)
        runs.append(dict(machine=m, seed=int(s), **aurocs(z["train"], z["test"], z["y_test"])))
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / "perchannel_runs.csv", index=False)
    naive = naive_table(MACHINES)
    naive.to_csv(OUT / "perchannel_naive.csv", index=False)

    out = ["V3 ITEM B: PER-CHANNEL SCORING ON SMD (pre-registered, notes/v3/design_v3.md)", "=" * 74,
           "Fusionformer (full) runs:", runs.round(4).to_string(index=False), "",
           "Naive forecasters:", naive.round(4).to_string(index=False), ""]
    if len(runs):
        # B1: per-channel max vs pooled, paired over the 9 runs
        d = runs.pc_max - runs.pooled
        out.append(f"B1  pc_max - pooled (Fusionformer, n={len(d)} runs): mean {d.mean():+.4f}, "
                   f"{int((d > 0).sum())}/{len(d)} positive, exact Wilcoxon p {wil(d):.4f}")
        # B2: margin over persistence, seed-averaged per machine, pooled vs per-channel
        ff = runs.groupby("machine")[["pooled", "pc_max", "pc_mean"]].mean()
        pers = naive[naive.method == "persistence"].set_index("machine")[["pooled", "pc_max", "pc_mean"]]
        marg = (ff - pers.loc[ff.index])
        out.append("B2  Fusionformer minus persistence, per machine (seed-averaged):")
        out.append(marg.round(4).to_string())
        verdict = bool((marg.pc_max > marg.pooled).all())
        out.append(f"    mean margin pooled {marg.pooled.mean():+.4f}   per-channel max {marg.pc_max.mean():+.4f}")
        out.append(f"VERDICT B2 (per-channel scoring widens the margin over persistence on every machine): {verdict}")
    text = "\n".join(out)
    (OUT / "perchannel_stats.txt").write_text(text + "\n"); print(text)

    if a.all_machines:
        allm = sorted(p.stem for p in Path("datasets/SMD/train").glob("machine-*.txt"))
        t = naive_table(allm)
        t.to_csv(OUT / "perchannel_naive_all_machines.csv", index=False)
        print(t.groupby("method")[["pooled", "pc_max", "pc_mean"]].agg(["mean", "median"]).round(4))


if __name__ == "__main__":
    main()
