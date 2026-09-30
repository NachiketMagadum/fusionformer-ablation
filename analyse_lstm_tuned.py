"""Pre-specified comparison: tuned LSTM vs full Fusionformer on the v2 simulated series.
Primary: two-sided exact Wilcoxon on paired AUROC (series x seed, n = 30 per condition), Holm over the two conditions.
Also reported: paired MAE, and the tuned LSTM's own AUROC level against the reference detectors.
Writes notes/lstm_tune/lstm_tuned_stats.txt
Author: Nachiket Magadum"""
import re
from pathlib import Path
import pandas as pd
from analyse_final import test, holm, fmt, _get

OUT = Path("notes/lstm_tune")
V2 = Path("notes/synth_v2")


def parse(folder, pattern, variant):
    rows = []
    for p in folder.glob("*.txt"):
        m = re.match(pattern, p.stem)
        if m:
            t = p.read_text()
            rows.append(dict(variant=variant, condition=m[1], id=m[2], seed=int(m[3]),
                             auroc=_get(t, r"Test AUROC:\s*([\d.]+)"), mae=_get(t, r"Test forecast MAE:\s*([\d.]+)")))
    return pd.DataFrame(rows)


ff = parse(V2, r"^ff_true_full_synth_(coupled|independent)_(\d+)_seed(\d+)$", "full")
ls = parse(OUT, r"^lstm_baseline_tuned_synth_(coupled|independent)_(\d+)_seed(\d+)$", "lstm_tuned")
df = pd.concat([ff, ls])
out = ["TUNED LSTM vs FUSIONFORMER (v2 simulated series)", "=" * 74,
       (OUT / "tuning_grid.txt").read_text() if (OUT / "tuning_grid.txt").exists() else "",
       df.groupby(["condition", "variant"])[["auroc", "mae"]].agg(["mean", "std", "count"]).round(4).to_string(), ""]
res = []
for cond in ("coupled", "independent"):
    A = df[(df.variant == "full") & (df.condition == cond)].set_index(["id", "seed"])
    B = df[(df.variant == "lstm_tuned") & (df.condition == cond)].set_index(["id", "seed"])
    j = pd.concat([A.auroc, B.auroc], axis=1, keys=["a", "b"]).dropna()
    res.append((f"AUROC Fusionformer vs tuned LSTM on {cond}", j, test(j.a - j.b)))
    jm = pd.concat([B.mae, A.mae], axis=1, keys=["a", "b"]).dropna()
    out.append(fmt(f"MAE tuned LSTM vs Fusionformer on {cond} (positive = Fusionformer lower error)", "lstm_tuned", "full", jm, test(jm.a - jm.b)))
for (name, j, s), pa in zip(res, holm([r[2]["p"] for r in res])):
    out.append(fmt(name, "full", "lstm_tuned", j, s, pa))
text = "\n".join(out)
(OUT / "lstm_tuned_stats.txt").write_text(text + "\n")
print(text)
