"""Pick one LSTM config per condition by mean validation MSE (training data only; no test data).
Writes notes/lstm_tune/chosen_{condition}.txt as 'hidden lr' and prints the full grid.
Author: Nachiket Magadum"""
import re
from pathlib import Path
import pandas as pd

OUT = Path("notes/lstm_tune")
rows = []
for p in OUT.glob("lstm_val_synth_*.txt"):
    m = re.match(r"lstm_val_synth_(coupled|independent)_(\d+)_h(\d+)_lr([\d.e-]+)_seed0", p.stem)
    if not m:
        continue
    mse = float(re.search(r"Validation forecast MSE:\s*([\d.]+)", p.read_text()).group(1))
    rows.append(dict(condition=m[1], series=m[2], hidden=int(m[3]), lr=float(m[4]), val_mse=mse))
df = pd.DataFrame(rows)
grid = df.groupby(["condition", "hidden", "lr"]).val_mse.agg(["mean", "count"]).reset_index()
lines = [grid.round(5).to_string(index=False), ""]
for cond, g in grid.groupby("condition"):
    best = g.sort_values("mean").iloc[0]
    (OUT / f"chosen_{cond}.txt").write_text(f"{int(best.hidden)} {best.lr:g}\n")
    lines.append(f"{cond}: chosen hidden={int(best.hidden)} lr={best.lr:g} (mean validation MSE {best['mean']:.5f})")
(OUT / "tuning_grid.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
