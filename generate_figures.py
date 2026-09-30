"""
Regenerate the Chapter 4 figures from the archived per-run results.

Reads:  notes/all_runs.csv                         (analyse_final.py, held-out SKAB)
        notes/all_runs_tail_holdout.csv            (second SKAB design, optional)
        notes/skab_whole_file_eval/*.txt           (original whole-file SKAB scoring)
        notes/naive_baselines.csv                  (naive forecasting controls)
Writes: figures/fig_4_1_paired_diffs.png           paired AUROC differences, all six ablations
        figures/fig_4_2_skab_evaluation.png        SKAB AUROC under the three scoring designs
        figures/fig_4_3_naive_controls.png         trained models against naive forecasters
        figures/fig_4_4_synth.png                  simulated benchmark (Section 4.9)

Figures carry axis labels and panel labels only; the full description is in
the caption in the dissertation, so nothing is repeated inside the image.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTES = Path("notes")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 200,
                     "savefig.bbox": "tight"})

ABLATIONS = [("MSWEA", "full", "no_mswea"), ("SWSE", "full", "no_swse"),
             ("Adversarial", "adv_on", "full")]
UNIT_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def parse_dir(folder):
    rows = []
    for p in folder.glob("*.txt"):
        m = re.match(r"^ff_true_(full|no_mswea|no_swse|adv_on)_(smd|skab)_(.+)_seed(\d+)$", p.stem)
        if m:
            v, ds, i, s = m.groups()
        else:
            m = re.match(r"^lstm_baseline_(smd|skab)_(.+)_seed(\d+)$", p.stem)
            if not m:
                continue
            ds, i, s = m.groups(); v = "lstm"
        a = re.search(r"Test AUROC:\s*([\d.]+)", p.read_text())
        rows.append(dict(variant=v, dataset=ds, id=i, seed=int(s), auroc=float(a.group(1))))
    return pd.DataFrame(rows)


def paired(df, ds, a, b):
    A = df[(df.variant == a) & (df.dataset == ds)].set_index(["id", "seed"]).auroc
    B = df[(df.variant == b) & (df.dataset == ds)].set_index(["id", "seed"]).auroc
    return pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()


def fig_paired_diffs(df):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharey="row")
    for r, ds in enumerate(("smd", "skab")):
        for c, (name, a, b) in enumerate(ABLATIONS):
            ax = axes[r, c]
            j = paired(df, ds, a, b)
            if len(j) == 0:
                ax.set_visible(False); continue
            d = (j.a - j.b)
            units = sorted(d.index.get_level_values(0).unique())
            for k, u in enumerate(units):
                vals = d.xs(u, level=0).values
                x = np.full(len(vals), k) + np.linspace(-0.12, 0.12, len(vals))
                ax.scatter(x, vals, s=22, color=UNIT_COLOURS[k % 5], zorder=3)
            ax.axhline(0, color="black", lw=0.8)
            ax.axhline(d.mean(), color="grey", lw=1, ls="--")
            ax.set_xticks(range(len(units)))
            ax.set_xticklabels([u.replace("machine-", "m") if ds == "smd" else f"file {u}" for u in units],
                               fontsize=8)
            ax.set_title(f"{ds.upper()}: {name}", fontsize=10)
            if c == 0:
                ax.set_ylabel("AUROC difference\n(with - without)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_4_1_paired_diffs.png"); plt.close(fig)


def fig_skab_eval(df):
    old = parse_dir(NOTES / "skab_whole_file_eval")
    tail_csv = NOTES / "all_runs_tail_holdout.csv"
    tail = pd.read_csv(tail_csv) if tail_csv.exists() else pd.DataFrame(columns=df.columns)
    designs = [("Whole file\n(training rows scored)", old),
               ("Held out:\nrows after first anomaly", df),
               ("Held out:\nlast 20% of normal + after", tail)]
    variants = ["full", "no_mswea", "no_swse", "adv_on", "lstm"]
    labels = ["Full", "MSWEA off", "SWSE off", "adv_on", "LSTM"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.26
    for k, (lab, data) in enumerate(designs):
        means = []
        for v in variants:
            s = data[(data.dataset == "skab") & (data.variant == v)].auroc if len(data) else pd.Series(dtype=float)
            means.append(s.mean() if len(s) else np.nan)
            if len(s):
                ax.scatter(np.full(len(s), variants.index(v) + (k - 1) * w), s, s=6, color="black",
                           alpha=0.5, zorder=3)
        ax.bar(np.arange(len(variants)) + (k - 1) * w, means, w, label=lab,
               color=["#bbbbbb", "#1f77b4", "#ff7f0e"][k])
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xticks(range(len(variants))); ax.set_xticklabels(labels)
    ax.set_ylabel("SKAB AUROC"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_4_2_skab_evaluation.png"); plt.close(fig)


def fig_naive(df):
    nb = NOTES / "naive_baselines.csv"
    if not nb.exists():
        return
    n = pd.read_csv(nb)
    tail_nb = NOTES / "skab_tail_holdout" / "naive_baselines.csv"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, ds in zip(axes, ("smd", "skab")):
        units = sorted(n[n.dataset == ds].id.unique(), key=str)
        series = [("Mean forecast", n[(n.dataset == ds) & (n.method == "mean")].set_index("id").auroc),
                  ("Persistence", n[(n.dataset == ds) & (n.method == "persistence")].set_index("id").auroc),
                  ("LSTM", df[(df.dataset == ds) & (df.variant == "lstm")].groupby("id").auroc.mean()),
                  ("Fusionformer full", df[(df.dataset == ds) & (df.variant == "full")].groupby("id").auroc.mean())]
        w = 0.2
        for k, (lab, s) in enumerate(series):
            s.index = s.index.astype(str)
            vals = [s.get(str(u), np.nan) for u in units]
            ax.bar(np.arange(len(units)) + (k - 1.5) * w, vals, w, label=lab,
                   color=["#bbbbbb", "#888888", "#ff7f0e", "#1f77b4"][k])
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(len(units)))
        ax.set_xticklabels([str(u).replace("machine-", "m") if ds == "smd" else f"file {u}" for u in units],
                           fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(ds.upper(), fontsize=10)
        if ds == "smd":
            ax.set_ylabel("AUROC (seed mean)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGDIR / "fig_4_3_naive_controls.png"); plt.close(fig)


def fig_synth():
    runs_csv = NOTES / "synth" / "all_runs_synth.csv"
    ref_csv = NOTES / "synth" / "reference_detectors.csv"
    if not (runs_csv.exists() and ref_csv.exists()):
        return
    r = pd.read_csv(runs_csv); ref = pd.read_csv(ref_csv)
    labels = ["Mean forecast", "Persistence", "Own-history AR", "Cross-channel reg.",
              "FF, MSWEA off", "FF, MSWEA off\n(param.-matched)", "FF, full"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, cond in zip(axes, ("coupled", "independent")):
        rr = ref[ref.condition == cond]
        vals = [rr["mean"], rr["persistence"], rr["own_history"], rr["cross_channel"]]
        for v in ("no_mswea", "no_mswea_pm", "full"):
            vals.append(r[(r.condition == cond) & (r.variant == v)].groupby("id").auroc.mean())
        cols = ["#bbbbbb", "#888888", "#bbbbbb", "#888888", "#9ecae1", "#6baed6", "#1f77b4"]
        for k, v in enumerate(vals):
            if len(v) == 0:
                continue
            ax.bar(k, v.mean(), 0.7, color=cols[k])
            ax.scatter(np.full(len(v), k), v, s=8, color="black", zorder=3)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
        ax.set_title(f"Simulated, {cond}", fontsize=10); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("AUROC (mean over 5 series)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_4_4_synth.png"); plt.close(fig)


def main():
    df = pd.read_csv(NOTES / "all_runs.csv")
    df["id"] = df["id"].astype(str)
    fig_paired_diffs(df)
    fig_skab_eval(df)
    fig_naive(df)
    fig_synth()
    print("Saved figures to", FIGDIR)


if __name__ == "__main__":
    main()
