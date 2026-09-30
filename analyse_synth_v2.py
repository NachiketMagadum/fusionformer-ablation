"""
Pre-registered analysis for the v2 simulated benchmark (notes/synth_v2/design_v2.md).

Written, committed and pushed to GitHub before any v2 model was trained. It
runs the tests exactly as the design note states and prints the verdicts.

Series: 10 new simulated series per condition (generator seeds 100-109),
3 model seeds each (0, 1, 42), so n = 30 pairs per comparison.

Four tests, Holm-corrected together:
  coupled      full vs MSWEA-off            full vs parameter-matched MSWEA-off
  independent  full vs MSWEA-off            full vs parameter-matched MSWEA-off

H1 (primary): in the coupled condition MSWEA beats the PARAMETER-MATCHED
    MSWEA-off: Holm p < 0.05, positive mean difference, and a positive
    seed-averaged difference on at least 7 of the 10 series.
H2 (specificity): the matched gain is larger in the coupled condition than in
    the independent one: two-sided Mann-Whitney U on the 10 + 10 seed-averaged
    per-series matched differences, p < 0.05, coupled median higher.
Both verdicts are reported whatever they are.

Writes notes/synth_v2/all_runs_synth_v2.csv and notes/synth_v2/synth_v2_stats.txt

Author: Nachiket Magadum
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyse_final import test, holm, fmt, _get

SYN = Path("notes/synth_v2")


def parse():
    rows = []
    for p in sorted(SYN.glob("*.txt")):
        m = re.match(r"^ff_true_(full|no_mswea_pm|no_mswea)_synth_(coupled|independent)_(\d+)_seed(\d+)$", p.stem)
        if not m:
            continue
        v, cond, u, s = m.groups()
        t = p.read_text()
        rows.append(dict(variant=v, condition=cond, id=u, seed=int(s),
                         auroc=_get(t, r"Test AUROC:\s*([\d.]+)"),
                         pr_auc=_get(t, r"Test PR-AUC:\s*([\d.]+)"),
                         mae=_get(t, r"Test forecast MAE:\s*([\d.]+)")))
    return pd.DataFrame(rows)


def paired(df, cond, a, b):
    A = df[(df.variant == a) & (df.condition == cond)].set_index(["id", "seed"]).auroc
    B = df[(df.variant == b) & (df.condition == cond)].set_index(["id", "seed"]).auroc
    return pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()


def main():
    df = parse()
    df.to_csv(SYN / "all_runs_synth_v2.csv", index=False)
    out = ["V2 SIMULATED BENCHMARK, PRE-REGISTERED ANALYSIS", "=" * 74,
           df.groupby(["condition", "variant"])[["auroc", "pr_auc", "mae"]].agg(["mean", "std", "count"]).round(4).to_string(), ""]
    tests = []
    for cond in ("coupled", "independent"):
        for b, lab in (("no_mswea", "MSWEA-off"), ("no_mswea_pm", "parameter-matched MSWEA-off")):
            j = paired(df, cond, "full", b)
            tests.append((cond, b, f"full vs {lab} on {cond}", j, test(j.a - j.b)))
    adj = holm([t[4]["p"] for t in tests])
    unit_diffs = {}
    for (cond, b, name, j, s), pa in zip(tests, adj):
        out.append(fmt(name, "full", b, j, s, pa))
        u = j.groupby(level=0).mean(); ud = (u.a - u.b)
        unit_diffs[(cond, b)] = ud
        wp = stats.wilcoxon(ud).pvalue if np.any(ud != 0) else 1.0
        out.append(f"  per-series (n={len(ud)}): mean {ud.mean():+.4f}, {int((ud > 0).sum())}/{len(ud)} positive, Wilcoxon p {wp:.4f}")
        if cond == "coupled" and b == "no_mswea_pm":
            h1 = pa < 0.05 and s["mean"] > 0 and int((ud > 0).sum()) >= 7
    out.append("")
    c, i = unit_diffs[("coupled", "no_mswea_pm")], unit_diffs[("independent", "no_mswea_pm")]
    mw = stats.mannwhitneyu(c, i, alternative="two-sided")
    h2 = mw.pvalue < 0.05 and np.median(c) > np.median(i)
    out.append(f"H2 specificity: matched gain coupled median {np.median(c):+.4f} vs independent {np.median(i):+.4f}, "
               f"Mann-Whitney U {mw.statistic:.1f}, p {mw.pvalue:.4f}")
    out.append("")
    out.append(f"VERDICT H1 (MSWEA beats the parameter-matched control on coupled data): {h1}")
    out.append(f"VERDICT H2 (the matched gain is larger on coupled than independent data): {h2}")
    text = "\n".join(out)
    (SYN / "synth_v2_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
