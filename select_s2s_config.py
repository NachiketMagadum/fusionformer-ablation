"""
v3, item C: choose the seq2seq LSTM configuration per data group from validation MSE only
(design: notes/v3/design_v3.md, section C). No test data is read.

Groups and tuning units:
  coupled      synth coupled_100, 101, 102
  independent  synth independent_100, 101, 102
  skab         SKAB valve1 files 0-4
  smd          SMD machine-1-1, 1-4, 2-1   (only if the SMD tuning runs exist)
Grid: hidden {64, 128, 256} x lr {3e-4, 1e-3, 3e-3}, seed 0.
Rule: lowest mean validation MSE over the group's tuning units; ties -> smaller hidden, then smaller lr.

Also reports, on the same validation split, the MSE of the two naive forecasters
(training mean and persistence), so it is visible whether the LSTM learned anything.

Writes notes/v3/s2s/chosen_<group>.txt ("HIDDEN LR") and notes/v3/s2s/s2s_tuning.txt

Author: Nachiket Magadum
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("notes/v3/s2s")
SEQ_LEN, PRED_LEN = 96, 24


def group_of(dataset, run_id):
    if dataset == "synth":
        return run_id.split("_")[0]
    return dataset


def naive_val_mse(dataset, run_id):
    """MSE of mean and persistence forecasts on the seq2seq validation split (same split and scaling)."""
    if dataset == "synth":
        X = np.load(Path("datasets/SYNTH") / f"{run_id}.npz")["X_train"]
    elif dataset == "skab":
        df = pd.read_csv(f"datasets/SKAB/data/valve1/{run_id}.csv", sep=";")
        y = df["anomaly"].astype(int).values
        X = df[[c for c in df.columns if c not in ("datetime", "anomaly", "changepoint")]].values
        X = X[:int(np.argmax(y == 1)) if y.any() else len(y)]
    else:
        X = pd.read_csv(Path("datasets/SMD/train") / f"{run_id}.txt", header=None).values
    n_val = max(int(0.2 * len(X)), SEQ_LEN + PRED_LEN + 50)
    fit, val = X[:-n_val], X[-n_val:]
    mu, sd = fit.mean(0), fit.std(0); sd[sd == 0] = 1
    v = (val - mu) / sd
    n = len(v) - SEQ_LEN - PRED_LEN + 1
    tgt = np.stack([v[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN] for i in range(n)])
    last = np.stack([v[i + SEQ_LEN - 1] for i in range(n)])[:, None, :]
    return float((tgt ** 2).mean()), float(((tgt - last) ** 2).mean())


def main():
    rows = []
    for p in sorted(D.glob("s2s_val_*.txt")):
        m = re.match(r"s2s_val_(synth|skab|smd)_(.+)_h(\d+)_lr([\d.e-]+)_seed0$", p.stem)
        if not m:
            continue
        ds, rid, h, lr = m.groups()
        t = p.read_text()
        rows.append(dict(group=group_of(ds, rid), dataset=ds, id=rid, hidden=int(h), lr=float(lr),
                         val_mse=float(re.search(r"Validation forecast MSE:\s*([\d.]+)", t).group(1))))
    df = pd.DataFrame(rows)
    out = ["SEQ2SEQ LSTM TUNING (validation only)", "=" * 70]
    for g, d in df.groupby("group"):
        tab = d.groupby(["hidden", "lr"]).val_mse.agg(["mean", "count"]).reset_index()
        best = tab.sort_values(["mean", "hidden", "lr"]).iloc[0]
        naive = np.array([naive_val_mse(ds, rid) for ds, rid in d[["dataset", "id"]].drop_duplicates().values])
        out += [f"[{g}]", tab.round(5).to_string(index=False),
                f"  chosen: hidden={int(best.hidden)} lr={best.lr:g} (mean validation MSE {best['mean']:.5f})",
                f"  same split, naive forecasters: mean-forecast MSE {naive[:, 0].mean():.5f}, "
                f"persistence MSE {naive[:, 1].mean():.5f}",
                f"  chosen config beats both naive forecasters on validation: "
                f"{bool(best['mean'] < min(naive[:, 0].mean(), naive[:, 1].mean()))}", ""]
        (D / f"chosen_{g}.txt").write_text(f"{int(best.hidden)} {best.lr:g}\n")
    text = "\n".join(out)
    (D / "s2s_tuning.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
