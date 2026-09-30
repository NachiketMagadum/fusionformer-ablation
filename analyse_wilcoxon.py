"""
Wilcoxon signed-rank analysis of the paper-faithful Fusionformer MSWEA ablation.

Reads every ff_true_{variant}_{dataset}_{id}_seed{N}.txt in notes/, extracts
Test AUROC, and computes the paired Wilcoxon and Rosenthal effect size for:
  - SMD: full vs no_mswea (n=9 paired runs)
  - SKAB: full vs no_mswea (n=15 paired runs)

Also writes:
  - notes/all_runs.csv           per-run AUROC and metadata
  - notes/wilcoxon_summary.txt    paste-ready report

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""

import re
import glob
import csv
from pathlib import Path
import numpy as np
from scipy import stats


def parse_result(path: Path):
    text = path.read_text()
    def _get(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    stem = path.stem  # ff_true_full_smd_machine-1-1_seed42
    parts = stem.split('_')
    variant = parts[2] if parts[2] != 'no' else 'no_' + parts[3]
    offset = 3 if variant == 'full' else 4
    dataset = parts[offset]
    run_id = '_'.join(parts[offset+1:-1])
    seed = int(parts[-1].replace('seed', ''))
    return {
        'variant': variant,
        'dataset': dataset,
        'id': run_id,
        'seed': seed,
        'auroc': _get(r'Test AUROC:\s*([\d.]+)'),
        'pr_auc': _get(r'Test PR-AUC:\s*([\d.]+)'),
        'anom_rate': _get(r'Test anomaly rate:\s*([\d.]+)'),
        'wall_min': _get(r'Wall-clock:\s*([\d.]+) min'),
    }


def wilcoxon_report(name, full, ablated, label_a="full", label_b="no_mswea"):
    diffs = full - ablated
    n = len(diffs)
    n_nonzero = int((diffs != 0).sum())
    res = stats.wilcoxon(full, ablated, zero_method="wilcox", alternative="two-sided")
    z = abs(stats.norm.ppf(res.pvalue / 2))
    r = z / np.sqrt(n_nonzero) if n_nonzero else 0.0

    if r < 0.1: eff = "negligible"
    elif r < 0.3: eff = "small"
    elif r < 0.5: eff = "medium"
    else: eff = "large"

    sig = "SIGNIFICANT (p<0.05)" if res.pvalue < 0.05 else "not significant (p>=0.05)"

    lines = []
    lines.append("=" * 74)
    lines.append(name)
    lines.append("=" * 74)
    lines.append(f"  {label_a:15s} mean {full.mean():.4f}   std {full.std():.4f}")
    lines.append(f"  {label_b:15s} mean {ablated.mean():.4f}   std {ablated.std():.4f}")
    lines.append(f"  Mean paired diff ({label_a} - {label_b}): {diffs.mean():+.4f}")
    lines.append(f"  n pairs: {n}   {label_a} wins: {int((diffs > 0).sum())}   "
                 f"{label_b} wins: {int((diffs < 0).sum())}   ties: {n - n_nonzero}")
    lines.append(f"  Wilcoxon W: {res.statistic:.3f}")
    lines.append(f"  p-value (two-sided): {res.pvalue:.4f}")
    lines.append(f"  Rosenthal effect size r: {r:.3f}  ({eff})")
    lines.append(f"  Verdict: {sig}")
    return "\n".join(lines) + "\n", res.pvalue, r


def main():
    root = Path("notes")
    rows = []
    for f in sorted(root.glob("ff_true_*.txt")):
        try:
            rows.append(parse_result(f))
        except Exception as e:
            print(f"skip {f.name}: {e}")
            continue

    # Write per-run CSV
    csv_path = root / "all_runs.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")

    # Split into paired arrays per dataset
    def paired(dataset):
        full = {}
        ablated = {}
        for r in rows:
            if r['dataset'] != dataset:
                continue
            key = (r['id'], r['seed'])
            if r['variant'] == 'full':
                full[key] = r['auroc']
            elif r['variant'] == 'no_mswea':
                ablated[key] = r['auroc']
        keys = sorted(set(full) & set(ablated))
        return (np.array([full[k] for k in keys]),
                np.array([ablated[k] for k in keys]),
                keys)

    report_lines = []

    for ds in ["smd", "skab"]:
        full_arr, abl_arr, keys = paired(ds)
        if len(full_arr) == 0:
            continue
        header = f"{ds.upper()} paired MSWEA ablation (n = {len(full_arr)})"
        section, p, r = wilcoxon_report(header, full_arr, abl_arr)
        report_lines.append(section)

        # Per-pair breakdown
        breakdown = ["  --- per-pair breakdown ---"]
        for (rid, seed), fa, na in zip(keys, full_arr, abl_arr):
            breakdown.append(f"  {rid:20s} seed{seed:>2}   full {fa:.4f}   "
                             f"no_mswea {na:.4f}   diff {fa - na:+.4f}")
        report_lines.append("\n".join(breakdown) + "\n")

    # Cross-benchmark summary
    smd_full, smd_abl, _ = paired("smd")
    skab_full, skab_abl, _ = paired("skab")
    cross = [
        "=" * 74,
        "CROSS-BENCHMARK SUMMARY",
        "=" * 74,
        f"{'benchmark':<10} {'n':>3}  {'full':>7}  {'no_mswea':>10}  "
        f"{'diff':>7}  {'p':>7}  {'r':>6}",
    ]
    for name, full, abl in [("SMD", smd_full, smd_abl), ("SKAB", skab_full, skab_abl)]:
        res = stats.wilcoxon(full, abl, zero_method="wilcox", alternative="two-sided")
        z = abs(stats.norm.ppf(res.pvalue / 2))
        r = z / np.sqrt((full != abl).sum())
        cross.append(f"{name:<10} {len(full):>3}  {full.mean():>7.4f}  "
                     f"{abl.mean():>10.4f}  {full.mean() - abl.mean():>+7.4f}  "
                     f"{res.pvalue:>7.4f}  {r:>6.3f}")
    report_lines.append("\n".join(cross) + "\n")

    report = "\n".join(report_lines)
    print("\n" + report)

    out = root / "wilcoxon_summary.txt"
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
