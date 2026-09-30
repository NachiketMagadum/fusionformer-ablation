"""
Final statistical analysis for Chapter 4.

Reads every per-run result file in notes/ and writes
  notes/all_runs.csv      one row per run (AUROC, PR-AUC, forecast MSE/MAE)
  notes/final_stats.txt   every number quoted in Chapter 4

For each paired comparison it reports the mean paired difference with a 95%
t-interval, paired Cohen's d_z, win counts, the exact Wilcoxon signed-rank p,
and Rosenthal's r computed from the normal-approximation Z,
    Z = (W+ - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24),   r = |Z| / sqrt(n)
with n the number of non-zero differences. Holm's correction is applied over
the six ablation tests and separately over the four LSTM tests.

Sign convention: every difference is (with component) - (without component),
so a positive AUROC difference always means the component helped. For the
adversarial loop that is adv_on - full.

Because seeds on the same machine or file are not independent, every
comparison is also repeated on per-unit means (3 SMD machines, 5 SKAB files).

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NOTES = Path("notes")
FF_RE = re.compile(r"^ff_true_(full|no_mswea|no_swse|adv_on)_(smd|skab)_(.+)_seed(\d+)$")
LSTM_RE = re.compile(r"^lstm_baseline_(smd|skab)_(.+)_seed(\d+)$")


def _get(text, pat):
    m = re.search(pat, text)
    return float(m.group(1)) if m else np.nan


def parse_dir(folder):
    rows = []
    for p in sorted(folder.glob("*.txt")):
        m = FF_RE.match(p.stem)
        if m:
            variant, dataset, run_id, seed = m.groups()
        else:
            m = LSTM_RE.match(p.stem)
            if not m:
                continue
            dataset, run_id, seed = m.groups()
            variant = "lstm"
        t = p.read_text()
        rows.append(dict(variant=variant, dataset=dataset, id=run_id, seed=int(seed),
                         auroc=_get(t, r"Test AUROC:\s*([\d.]+)"),
                         pr_auc=_get(t, r"Test PR-AUC:\s*([\d.]+)"),
                         mse=_get(t, r"Test forecast MSE:\s*([\d.]+)"),
                         mae=_get(t, r"Test forecast MAE:\s*([\d.]+)"),
                         anom_rate=_get(t, r"Test anomaly rate:\s*([\d.]+)"),
                         wall_min=_get(t, r"Wall-clock:\s*([\d.]+) min"),
                         heldout="Scored rows" in t))
    return pd.DataFrame(rows)


def paired(df, dataset, a, b, metric="auroc"):
    """Return aligned arrays (a, b) over matching (id, seed) pairs."""
    A = df[(df.variant == a) & (df.dataset == dataset)].set_index(["id", "seed"])[metric]
    B = df[(df.variant == b) & (df.dataset == dataset)].set_index(["id", "seed"])[metric]
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()
    return j


def test(diff):
    diff = np.round(np.asarray(diff, dtype=float), 6)
    nz = diff[diff != 0]
    n = len(nz)
    out = dict(n=len(diff), mean=diff.mean(), wins=int((diff > 0).sum()),
               losses=int((diff < 0).sum()), ties=int((diff == 0).sum()))
    sd = diff.std(ddof=1)
    half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
    out.update(ci_lo=out["mean"] - half, ci_hi=out["mean"] + half,
               dz=out["mean"] / sd if sd > 0 else np.nan)
    if n == 0:
        out.update(W=np.nan, p=1.0, Z=0.0, r=0.0)
        return out
    res = stats.wilcoxon(nz)
    ranks = stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    Z = (w_plus - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    out.update(W=res.statistic, p=res.pvalue, Z=Z, r=abs(Z) / np.sqrt(n))
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def fmt(name, a_lab, b_lab, j, s, padj=None):
    lines = [f"--- {name}",
             f"  {a_lab} mean {j.a.mean():.4f}   {b_lab} mean {j.b.mean():.4f}",
             f"  mean diff ({a_lab} - {b_lab}) {s['mean']:+.4f}  95% CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  d_z {s['dz']:+.2f}",
             f"  n {s['n']}  {a_lab} higher {s['wins']}  {b_lab} higher {s['losses']}  ties {s['ties']}",
             f"  Wilcoxon W {s['W']:.1f}  p {s['p']:.4f}" + (f"  Holm p {padj:.4f}" if padj is not None else "")
             + f"  Z {s['Z']:+.3f}  r {s['r']:.3f}"]
    return "\n".join(lines)


def unit_level(j):
    u = j.groupby(level=0).mean()
    d = (u.a - u.b).values
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return f"  per-unit (n={len(d)}): mean diff {d.mean():+.4f}, {int((d > 0).sum())}/{len(d)} units positive, Wilcoxon p {p:.4f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skab_dir", default="notes", help="folder holding the SKAB result files")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    skab_dir = Path(args.skab_dir)
    df = parse_dir(NOTES)
    if skab_dir != NOTES:
        df = pd.concat([df[df.dataset == "smd"], parse_dir(skab_dir).query("dataset == 'skab'")])
    df.drop(columns="heldout").to_csv(NOTES / f"all_runs{args.tag}.csv", index=False)
    out = ["FINAL STATISTICS (generated by analyse_final.py)", "=" * 74]

    skab_flags = df[df.dataset == "skab"].heldout
    out.append(f"SKAB runs parsed: {len(skab_flags)}, all held-out evaluation: {bool(skab_flags.all())}")
    out.append("")
    out.append("Mean AUROC / PR-AUC per variant")
    g = df.groupby(["dataset", "variant"])[["auroc", "pr_auc", "mse", "mae"]].agg(["mean", "std", "count"])
    out.append(g.round(4).to_string())
    out.append("")

    # ---- six ablation tests ----
    ablations = [("MSWEA", "full", "no_mswea"), ("SWSE", "full", "no_swse"),
                 ("Adversarial", "adv_on", "full")]
    res = []
    for ds in ("smd", "skab"):
        for name, a, b in ablations:
            j = paired(df, ds, a, b)
            res.append((f"{name} on {ds.upper()}", a, b, j, test(j.a - j.b)))
    padj = holm([r[4]["p"] for r in res])
    out.append("ABLATIONS (diff = with component - without; positive = component helps)")
    for (name, a, b, j, s), pa in zip(res, padj):
        out.append(fmt(name, a, b, j, s, pa))
        out.append(unit_level(j))
    out.append("")

    # ---- LSTM comparisons ----
    lres = []
    for ds in ("smd", "skab"):
        j = paired(df, ds, "full", "lstm", "auroc")
        lres.append((f"AUROC FF vs LSTM on {ds.upper()}", "full", "lstm", j, test(j.a - j.b)))
        j = paired(df, ds, "lstm", "full", "mae")
        lres.append((f"MAE LSTM vs FF on {ds.upper()} (positive = FF lower error)", "lstm", "full", j, test(j.a - j.b)))
    ladj = holm([r[4]["p"] for r in lres])
    out.append("FUSIONFORMER vs LSTM")
    for (name, a, b, j, s), pa in zip(lres, ladj):
        out.append(fmt(name, a, b, j, s, pa))
        out.append(unit_level(j))
    out.append("")

    # ---- naive controls ----
    nb = NOTES / "naive_baselines.csv"
    if nb.exists():
        n = pd.read_csv(nb)
        if skab_dir != NOTES and (skab_dir / "naive_baselines.csv").exists():
            n2 = pd.read_csv(skab_dir / "naive_baselines.csv")
            n = pd.concat([n[n.dataset == "smd"], n2[n2.dataset == "skab"]])
        out.append("NAIVE CONTROLS vs trained models (mean over units; FF/LSTM averaged over seeds first)")
        for ds in ("smd", "skab"):
            ff = df[(df.dataset == ds) & (df.variant == "full")].groupby("id").auroc.mean()
            ls = df[(df.dataset == ds) & (df.variant == "lstm")].groupby("id").auroc.mean()
            for method in ("mean", "persistence"):
                sub = n[(n.dataset == ds) & (n.method == method)].set_index("id").auroc
                out.append(f"  {ds.upper():4s} {method:11s} AUROC {sub.mean():.4f}   "
                           f"(FF full {ff.mean():.4f}, LSTM {ls.mean():.4f})")
            per = n[n.dataset == ds].pivot(index="id", columns="method", values="auroc")
            per["ff_full"] = ff
            per["lstm"] = ls
            out.append(per.round(4).to_string())
        out.append("")

    # ---- leakage post-mortem ----
    old = NOTES / "skab_whole_file_eval"
    if old.exists():
        o = parse_dir(old)
        if len(o):
            out.append("SKAB: WHOLE-FILE (old, training rows scored) vs HELD-OUT (new) evaluation")
            for v in ("full", "no_mswea", "no_swse", "adv_on", "lstm"):
                a = o[(o.dataset == "skab") & (o.variant == v)].auroc
                b = df[(df.dataset == "skab") & (df.variant == v)].auroc
                if len(a) and len(b):
                    out.append(f"  {v:9s} old mean {a.mean():.4f} (n={len(a)})   new mean {b.mean():.4f} (n={len(b)})")
            for name, a, b in ablations:
                j = paired(o, "skab", a, b)
                if len(j):
                    s = test(j.a - j.b)
                    out.append(f"  old {name}: mean diff {s['mean']:+.4f}, p {s['p']:.4f}, r {s['r']:.3f}")

    text = "\n".join(out)
    (NOTES / f"final_stats{args.tag}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
