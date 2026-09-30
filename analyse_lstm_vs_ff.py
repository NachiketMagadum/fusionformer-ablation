"""
Head-to-head: LSTM baseline vs Fusionformer (full variant).

Reads notes/lstm_baseline_*.txt and notes/ff_true_full_*.txt,
pairs them by (dataset, run_id, seed), computes paired Wilcoxon
signed-rank on AUROC, and prints per-benchmark summaries.
"""
import re
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, norm

NOTES = Path("notes")


def parse_result(path):
    text = path.read_text()
    m_auroc = re.search(r"Test AUROC:\s*([0-9.]+)", text)
    m_prauc = re.search(r"Test PR-AUC:\s*([0-9.]+)", text)
    if not (m_auroc and m_prauc):
        return None
    return float(m_auroc.group(1)), float(m_prauc.group(1))


def parse_lstm_files():
    rows = []
    for p in sorted(NOTES.glob("lstm_baseline_*.txt")):
        # lstm_baseline_{dataset}_{run_id}_seed{N}.txt
        stem = p.stem[len("lstm_baseline_"):]
        m = re.match(r"(smd|skab)_(.+)_seed(\d+)", stem)
        if not m:
            continue
        dataset, run_id, seed = m.group(1), m.group(2), int(m.group(3))
        r = parse_result(p)
        if r is None:
            continue
        rows.append({"model": "lstm", "dataset": dataset, "run_id": run_id,
                     "seed": seed, "auroc": r[0], "pr_auc": r[1]})
    return rows


def parse_ff_full_files():
    rows = []
    for p in sorted(NOTES.glob("ff_true_full_*.txt")):
        stem = p.stem[len("ff_true_full_"):]
        m = re.match(r"(smd|skab)_(.+)_seed(\d+)", stem)
        if not m:
            continue
        dataset, run_id, seed = m.group(1), m.group(2), int(m.group(3))
        r = parse_result(p)
        if r is None:
            continue
        rows.append({"model": "ff_full", "dataset": dataset, "run_id": run_id,
                     "seed": seed, "auroc": r[0], "pr_auc": r[1]})
    return rows


def paired(lstm, ff):
    """Return per-pair AUROC diffs (ff - lstm) matched on (dataset, run_id, seed)."""
    key = lambda d: (d["dataset"], d["run_id"], d["seed"])
    ff_map = {key(r): r["auroc"] for r in ff}
    pairs = []
    for r in lstm:
        k = key(r)
        if k in ff_map:
            pairs.append((k, r["auroc"], ff_map[k], ff_map[k] - r["auroc"]))
    return pairs


def summarise(pairs, benchmark):
    subset = [p for p in pairs if p[0][0] == benchmark]
    if not subset:
        print(f"{benchmark.upper()}: no pairs")
        return
    lstm_vals = np.array([p[1] for p in subset])
    ff_vals = np.array([p[2] for p in subset])
    diffs = np.array([p[3] for p in subset])
    n = len(diffs)
    n_nonzero = int((diffs != 0).sum())

    print(f"\n{benchmark.upper()}  (n = {n})")
    print(f"  LSTM   AUROC mean = {lstm_vals.mean():.4f}   sd = {lstm_vals.std(ddof=1):.4f}")
    print(f"  FF_full AUROC mean = {ff_vals.mean():.4f}   sd = {ff_vals.std(ddof=1):.4f}")
    print(f"  Paired diff (FF - LSTM) mean = {diffs.mean():+.4f}   sd = {diffs.std(ddof=1):.4f}")
    if n_nonzero >= 1:
        try:
            stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            z = norm.isf(p / 2)
            r_eff = abs(z) / np.sqrt(n_nonzero)
            print(f"  Wilcoxon signed-rank: W = {stat:.2f}, p = {p:.4f}, r = {r_eff:.3f}")
        except ValueError as e:
            print(f"  Wilcoxon failed: {e}")
    print(f"  Per-run pairs:")
    for (k, lv, fv, d) in subset:
        print(f"    {k[1]:20s} seed={k[2]:2d}   LSTM={lv:.4f}  FF={fv:.4f}  diff={d:+.4f}")


def main():
    lstm = parse_lstm_files()
    ff = parse_ff_full_files()
    print(f"Found {len(lstm)} LSTM runs, {len(ff)} FF full runs")
    pairs = paired(lstm, ff)
    print(f"Matched {len(pairs)} pairs")
    for b in ["smd", "skab"]:
        summarise(pairs, b)


if __name__ == "__main__":
    main()
