"""
Cluster-aware re-analysis of every paired ablation (reanalysis of existing results).

Seeds run on the same machine, file or simulated series are not independent, so
each comparison is re-estimated two ways:
  1. per-unit means: average the paired AUROC differences over seeds within each
     unit, then a one-sample t-interval and exact Wilcoxon test across units;
  2. a linear mixed model on the paired differences with a random intercept per
     unit:  diff_ij = mu + u_i + e_ij  (statsmodels MixedLM, REML).
     With only 3 (SMD) or 5 (SKAB) units the variance of u_i is poorly
     estimated, so the mixed-model p-values for those rows should be read as rough.

Writes notes/mixed_model_stats.txt and notes/mixed_model_stats.csv

Author: Nachiket Magadum
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise SystemExit("statsmodels is needed: pip install statsmodels")

warnings.filterwarnings("ignore")
NOTES = Path("notes")


def comparisons():
    main = pd.read_csv(NOTES / "all_runs.csv")
    tail = pd.read_csv(NOTES / "all_runs_tail_holdout.csv")
    for label, df in (("SMD", main[main.dataset == "smd"]), ("SKAB held out", main[main.dataset == "skab"]),
                      ("SKAB tail design", tail[tail.dataset == "skab"])):
        for comp, a, b in (("MSWEA", "full", "no_mswea"), ("SWSE", "full", "no_swse"), ("Adversarial", "adv_on", "full")):
            yield label, comp, df, a, b, "variant"
    for label, path in (("Simulated v1", NOTES / "synth" / "all_runs_synth.csv"),
                        ("Simulated v2", NOTES / "synth_v2" / "all_runs_synth_v2.csv")):
        if not path.exists():
            continue
        s = pd.read_csv(path)
        for cond in ("coupled", "independent"):
            d = s[s.condition == cond]
            for comp, b in (("MSWEA", "no_mswea"), ("MSWEA vs matched", "no_mswea_pm")):
                if (d.variant == b).any():
                    yield f"{label} {cond}", comp, d, "full", b, "variant"


def paired(df, a, b):
    df = df.copy(); df["id"] = df["id"].astype(str)
    A = df[df.variant == a].set_index(["id", "seed"]).auroc
    B = df[df.variant == b].set_index(["id", "seed"]).auroc
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna().reset_index()
    j["diff"] = j.a - j.b
    return j


rows = []
for label, comp, df, a, b, _ in comparisons():
    j = paired(df, a, b)
    unit = j.groupby("id")["diff"].mean()
    k = len(unit)
    se = unit.std(ddof=1) / np.sqrt(k)
    tcrit = stats.t.ppf(0.975, k - 1)
    unit_p_w = stats.wilcoxon(unit).pvalue if np.any(unit != 0) else 1.0
    try:
        fit = smf.mixedlm("diff ~ 1", j, groups=j["id"]).fit(reml=True)
        mm_est, mm_se, mm_p = fit.params["Intercept"], fit.bse["Intercept"], fit.pvalues["Intercept"]
        mm_lo, mm_hi = fit.conf_int().loc["Intercept"]
    except Exception:
        mm_est = mm_se = mm_p = mm_lo = mm_hi = np.nan
    rows.append(dict(data=label, comparison=comp, pairs=len(j), units=k,
                     mean_diff=j["diff"].mean(),
                     unit_ci_lo=unit.mean() - tcrit * se, unit_ci_hi=unit.mean() + tcrit * se,
                     units_positive=int((unit > 0).sum()), unit_wilcoxon_p=unit_p_w,
                     mixed_est=mm_est, mixed_ci_lo=mm_lo, mixed_ci_hi=mm_hi, mixed_p=mm_p))

res = pd.DataFrame(rows)
res.to_csv(NOTES / "mixed_model_stats.csv", index=False)
text = ("CLUSTER-AWARE RE-ANALYSIS (difference = with component - without; for Adversarial, adv_on - full)\n"
        + res.round(4).to_string(index=False)
        + "\n\nNote: with 3 or 5 units the unit-level Wilcoxon cannot go below p = 0.25 / 0.0625, and the mixed-model "
          "random-effect variance is poorly estimated; treat those rows as descriptive.\n")
(NOTES / "mixed_model_stats.txt").write_text(text)
print(text)
