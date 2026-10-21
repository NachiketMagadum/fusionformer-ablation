"""
Pre-registered analysis for v3 (notes/v3/design_v3.md). Written, committed and
pushed before any v3 model was trained; it runs the tests exactly as the design
note states and prints the verdicts, whatever they are.

Data groups (unit = simulated series, SKAB file or SMD machine; model seeds 0, 1, 42):
  coupled      simulated v2 series coupled_100..109       (10 units, n = 30 pairs)
  independent  simulated v2 series independent_100..109   (10 units, n = 30 pairs)
  skab         SKAB valve1 files 0-4, held-out scoring    (5 units,  n = 15 pairs)
  smd          SMD machine-1-1, 1-4, 2-1                  (3 units,  n = 9 pairs; only if the SMD runs exist)

Reference runs reused unchanged: full / no_mswea / no_mswea_pm on the simulated
series (notes/synth_v2/) and full / no_mswea on SKAB (notes/, held-out design).
SMD comparisons use the v3 SMD full runs (notes/v3/smd/).

A  attention ablation. PRIMARY family (Holm over 6 tests):
     full vs no_mswaa_pm and full vs no_fam_pm, in coupled, independent and skab.
   A component "helps" in a group if Holm p < 0.05, the mean AUROC difference is
   positive, and the seed-averaged difference is positive on >= 70% of units
   (7 of 10 series; 4 of 5 SKAB files).
   SECONDARY (Holm within the family): full vs no_mswaa, full vs no_fam (unmatched),
   and no_mswea vs no_mswaa (which single branch is worth more), in the same groups;
   smd descriptive only (3 units).
C  seq2seq LSTM: full vs s2s, AUROC (Holm over groups) and MAE.
D  adversarial-only objective: full vs adv_only, AUROC (Holm over coupled,
   independent, skab) and MAE; and (descriptive) adv_only MAE against the
   MAE of the training-mean forecast on the same test windows.
Every comparison also gets a per-unit summary and, if statsmodels is installed,
a random-intercept mixed model (as in analyse_mixed.py).

Writes notes/v3/v3_stats.txt and notes/v3/all_runs_v3.csv

Author: Nachiket Magadum
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyse_final import test, holm, fmt

warnings.filterwarnings("ignore")
V3 = Path("notes/v3")
FF_RE = re.compile(r"^ff_true_(.+)_(synth|skab|smd)_(.+)_seed(\d+)$")
S2S_RE = re.compile(r"^s2s_(synth|skab|smd)_(.+)_seed(\d+)$")
MIN_POS = {"coupled": 7, "independent": 7, "skab": 4, "smd": 3}
V2_IDS = {f"{c}_{u}" for c in ("coupled", "independent") for u in range(100, 110)}


def _get(t, pat):
    m = re.search(pat, t)
    return float(m.group(1)) if m else np.nan


def parse(folder, variants=None, need_heldout=False):
    rows = []
    for p in sorted(Path(folder).glob("*.txt")):
        m = FF_RE.match(p.stem)
        if m:
            v, ds, rid, s = m.groups()
        else:
            m = S2S_RE.match(p.stem)
            if not m:
                continue
            ds, rid, s = m.groups(); v = "s2s"
        if variants and v not in variants:
            continue
        t = p.read_text()
        if need_heldout and "Scored rows" not in t:
            continue
        g = rid.split("_")[0] if ds == "synth" else ds
        if ds == "synth" and rid not in V2_IDS:
            continue
        rows.append(dict(group=g, variant=v, id=rid, seed=int(s),
                         auroc=_get(t, r"Test AUROC:\s*([\d.]+)"), mae=_get(t, r"Test forecast MAE:\s*([\d.]+)"),
                         source=str(folder)))
    return pd.DataFrame(rows)


def load_all():
    parts = [parse("notes/synth_v2", {"full", "no_mswea", "no_mswea_pm"}),
             parse("notes", {"full", "no_mswea"}, need_heldout=True).query("group == 'skab'"),
             parse(V3 / "synth"), parse(V3 / "skab"), parse(V3 / "smd"), parse(V3 / "s2s")]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    dup = df.duplicated(["group", "variant", "id", "seed"], keep=False)
    if dup.any():
        raise SystemExit(f"duplicate runs found:\n{df[dup]}")
    return df


def paired(df, g, a, b, metric="auroc"):
    A = df[(df.group == g) & (df.variant == a)].set_index(["id", "seed"])[metric]
    B = df[(df.group == g) & (df.variant == b)].set_index(["id", "seed"])[metric]
    return pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()


def unit_summary(j):
    u = j.groupby(level=0).mean(); d = (u.a - u.b)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 1 else np.nan
    return d, f"  per-unit (n={len(d)}): mean {d.mean():+.4f}, {int((d > 0).sum())}/{len(d)} positive, Wilcoxon p {p:.4f}"


def mixed(j):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "  mixed model: statsmodels not installed"
    d = j.reset_index(); d["diff"] = d.a - d.b
    try:
        f = smf.mixedlm("diff ~ 1", d, groups=d["id"]).fit(reml=True)
        lo, hi = f.conf_int().loc["Intercept"]
        return f"  mixed model (random intercept per unit): {f.params['Intercept']:+.4f} [{lo:+.4f}, {hi:+.4f}], p {f.pvalues['Intercept']:.4f}"
    except Exception as e:  # noqa: BLE001
        return f"  mixed model failed: {e}"


def family(df, title, specs, out, verdict=False, metric="auroc"):
    """specs: list of (group, a, b, label). Holm over the tests that have data."""
    tests = []
    for g, a, b, lab in specs:
        j = paired(df, g, a, b, metric)
        if len(j) >= 2:
            tests.append((g, a, b, lab, j, test(j.a - j.b)))
    if not tests:
        out.append(f"{title}: no data yet\n"); return
    out.append(title)
    for (g, a, b, lab, j, s), pa in zip(tests, holm([t[5]["p"] for t in tests])):
        out.append(fmt(f"{lab} [{g}] ({metric})", a, b, j, s, pa))
        d, line = unit_summary(j); out.append(line); out.append(mixed(j))
        if verdict:
            ok = pa < 0.05 and s["mean"] > 0 and int((d > 0).sum()) >= MIN_POS[g]
            out.append(f"  VERDICT '{lab}' helps in {g}: {ok}")
    out.append("")


def mean_forecast_mae(g, rid):
    """Test MAE of the training-mean forecast (0 after scaling), same windows as the models."""
    if g in ("coupled", "independent"):
        z = np.load(Path("datasets/SYNTH") / f"{rid}.npz"); Xtr, Xte, start = z["X_train"], z["X_test"], 0
    elif g == "skab":
        d = pd.read_csv(f"datasets/SKAB/data/valve1/{rid}.csv", sep=";")
        y = d["anomaly"].astype(int).values
        X = d[[c for c in d.columns if c not in ("datetime", "anomaly", "changepoint")]].values
        start = int(np.argmax(y == 1)) if y.any() else len(y); Xtr, Xte = X[:start], X
    else:
        return np.nan
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    B = (Xte - mu) / sd
    n = len(B) - 96 - 24 + 1
    w = np.array([np.abs(B[i + 96:i + 120]).mean() for i in range(n)])
    return float(w[max(0, start - 96):].mean())


def main():
    df = load_all()
    V3.mkdir(parents=True, exist_ok=True)
    df.to_csv(V3 / "all_runs_v3.csv", index=False)
    out = ["V3 PRE-REGISTERED ANALYSIS (notes/v3/design_v3.md)", "=" * 74,
           df.groupby(["group", "variant"])[["auroc", "mae"]].agg(["mean", "std", "count"]).round(4).to_string(), ""]
    G3 = ("coupled", "independent", "skab")
    family(df, "A PRIMARY: attention branches vs parameter-matched controls (Holm over 6)",
           [(g, "full", b, lab) for g in G3 for b, lab in (("no_mswaa_pm", "MSWAA vs matched"),
                                                            ("no_fam_pm", "FAM vs matched"))], out, verdict=True)
    family(df, "A SECONDARY: unmatched ablations and branch comparison (Holm within family)",
           [(g, a, b, lab) for g in G3 for a, b, lab in (("full", "no_mswaa", "MSWAA unmatched"),
                                                         ("full", "no_fam", "FAM unmatched"),
                                                         ("no_mswea", "no_mswaa", "MSWAA-only minus MSWEA-only"))],
           out)
    family(df, "A SMD (descriptive, 3 machines)",
           [("smd", "full", b, lab) for b, lab in (("no_mswaa", "MSWAA"), ("no_fam", "FAM"))], out)
    groups_c = G3 + ("smd",)
    family(df, "C seq2seq LSTM: full vs s2s, AUROC (Holm over groups)",
           [(g, "full", "s2s", "Fusionformer vs seq2seq LSTM") for g in groups_c], out)
    family(df, "C seq2seq LSTM: forecast MAE, positive = Fusionformer lower error",
           [(g, "s2s", "full", "MAE s2s minus full") for g in groups_c], out, metric="mae")
    family(df, "D adversarial-only objective: full vs adv_only, AUROC (Holm over 3)",
           [(g, "full", "adv_only", "Huber objective vs adversarial-only") for g in G3], out)
    family(df, "D adversarial-only objective: forecast MAE, positive = Huber model lower error",
           [(g, "adv_only", "full", "MAE adv_only minus full") for g in G3], out, metric="mae")
    adv = df[df.variant == "adv_only"]
    if len(adv):
        rows = []
        for (g, rid), d in adv.groupby(["group", "id"]):
            try:
                rows.append(dict(group=g, id=rid, adv_only_mae=d.mae.mean(), mean_forecast_mae=mean_forecast_mae(g, rid)))
            except FileNotFoundError:
                pass
        if rows:
            t = pd.DataFrame(rows)
            t["adv_only_beats_mean_forecast"] = t.adv_only_mae < t.mean_forecast_mae
            out += ["D descriptive: adversarial-only test MAE vs the training-mean forecast (seed-averaged)",
                    t.round(4).to_string(index=False),
                    t.groupby("group")[["adv_only_mae", "mean_forecast_mae", "adv_only_beats_mean_forecast"]].mean()
                    .round(4).to_string(), ""]
    tuning = V3 / "s2s" / "s2s_tuning.txt"
    if tuning.exists():
        out += ["C tuning summary (validation only):", tuning.read_text()]
    text = "\n".join(out)
    (V3 / "v3_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
