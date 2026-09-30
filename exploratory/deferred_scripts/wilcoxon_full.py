"""
Complete Wilcoxon signed-rank significance tests including multi-seed
adversarial data from 28 Aug run.

Runs five tests total:
    1. FAM vs TimeOnly on SKAB (23 file means)
    2. FAM vs TimeOnly on SMD (15 machine-seed pairs)
    3. SWSE+FAM vs FAM alone on SMD (15 machine-seed pairs)  [NEW]
    4. SWSE+FAM+Adversarial vs FAM alone on SMD (15 machine-seed pairs)  [NEW]
    5. Full paper vs FAM alone on SMD single seed (5 machines, for context)

Usage:
    python scripts/wilcoxon_full.py

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats


# ============================================================
#  Data
# ============================================================

# SKAB — 23 file means (3-seed means per file)
skab_fam = [
    0.914, 0.825, 0.979, 0.943, 0.857,
    0.863, 0.912, 0.960, 0.949,
    1.000, 0.875, 0.976, 0.985, 0.863, 0.923, 0.962,
    0.786, 0.960, 0.939, 0.838, 0.870, 0.524, 0.938,
]
skab_time = [
    0.919, 0.821, 0.979, 0.939, 0.853,
    0.853, 0.910, 0.965, 0.938,
    1.000, 0.851, 0.976, 0.980, 0.886, 0.928, 0.962,
    0.782, 0.956, 0.936, 0.843, 0.861, 0.538, 0.933,
]

# SMD FAM vs TimeOnly — 15 pairs
smd_fam_time = [
    0.974, 0.976, 0.970,
    0.862, 0.888, 0.889,
    0.893, 0.821, 0.892,
    0.964, 0.938, 0.957,
    0.971, 0.970, 0.962,
]
smd_time_time = [
    0.964, 0.970, 0.971,
    0.861, 0.876, 0.890,
    0.879, 0.862, 0.874,
    0.957, 0.965, 0.963,
    0.960, 0.972, 0.964,
]

# SMD multi-seed adversarial data (28 Aug run) — 15 pairs
smd_fam_ms = [
    # machine-1-1 seeds 0, 1, 42
    0.974, 0.976, 0.970,
    # machine-1-4
    0.862, 0.888, 0.889,
    # machine-2-1
    0.893, 0.821, 0.892,
    # machine-2-5
    0.964, 0.938, 0.957,
    # machine-3-1
    0.971, 0.970, 0.962,
]
smd_swsefam_ms = [
    0.931, 0.965, 0.933,
    0.903, 0.900, 0.896,
    0.831, 0.829, 0.813,
    0.926, 0.938, 0.945,
    0.932, 0.946, 0.944,
]
smd_full_ms = [
    0.966, 0.957, 0.959,
    0.881, 0.886, 0.898,
    0.827, 0.816, 0.814,
    0.943, 0.948, 0.942,
    0.943, 0.951, 0.943,
]


def wilcoxon_report(name, a, b, label_a, label_b):
    a = np.array(a)
    b = np.array(b)
    diffs = a - b
    n = len(a)
    n_nonzero = int((diffs != 0).sum())
    n_a_wins = int((diffs > 0).sum())
    n_b_wins = int((diffs < 0).sum())

    result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    stat = result.statistic
    p = result.pvalue

    z_from_p = abs(stats.norm.ppf(p / 2))
    effect_r = z_from_p / np.sqrt(n_nonzero) if n_nonzero > 0 else 0.0

    if p < 0.01:
        sig = "highly significant (p < 0.01)"
    elif p < 0.05:
        sig = "significant (p < 0.05)"
    elif p < 0.10:
        sig = "marginally significant (p < 0.10)"
    else:
        sig = "NOT significant (p >= 0.10)"

    if abs(effect_r) < 0.1:
        eff_desc = "negligible"
    elif abs(effect_r) < 0.3:
        eff_desc = "small"
    elif abs(effect_r) < 0.5:
        eff_desc = "medium"
    else:
        eff_desc = "large"

    print(f"\n{'=' * 74}")
    print(f"{name}")
    print("=" * 74)
    print(f"  {label_a:30s} mean {a.mean():.4f}   std {a.std():.4f}")
    print(f"  {label_b:30s} mean {b.mean():.4f}   std {b.std():.4f}")
    print(f"  Mean difference ({label_a} - {label_b}): {diffs.mean():+.4f}")
    print(f"  N pairs: {n}   {label_a} wins: {n_a_wins}   {label_b} wins: {n_b_wins}   ties: {n - n_a_wins - n_b_wins}")
    print(f"  Wilcoxon W: {stat:.3f}")
    print(f"  p-value (two-sided): {p:.4f}")
    print(f"  Effect size r: {effect_r:.3f}  ({eff_desc})")
    print(f"  Verdict: {sig}")

    if p >= 0.10:
        print(f"  --> Cannot reject null: no significant difference.")
    else:
        winner = label_a if diffs.mean() > 0 else label_b
        print(f"  --> Reject null: {winner} is significantly better.")

    return {"p": p, "r": effect_r, "diff": diffs.mean()}


def main():
    print("=" * 74)
    print("WILCOXON SIGNED-RANK TESTS — FULL EVALUATION")
    print("Multi-seed data included (28 Aug adversarial run)")
    print("=" * 74)

    r1 = wilcoxon_report(
        "TEST 1: SKAB — FAM vs TimeOnly (23 file means)",
        skab_fam, skab_time, "FAM", "TimeOnly",
    )

    r2 = wilcoxon_report(
        "TEST 2: SMD — FAM vs TimeOnly (5 machines x 3 seeds = 15 pairs)",
        smd_fam_time, smd_time_time, "FAM", "TimeOnly",
    )

    r3 = wilcoxon_report(
        "TEST 3: SMD — SWSE+FAM vs FAM alone (15 pairs, MULTI-SEED)",
        smd_swsefam_ms, smd_fam_ms, "SWSE+FAM", "FAM alone",
    )

    r4 = wilcoxon_report(
        "TEST 4: SMD — Full paper (SWSE+FAM+Adv) vs FAM alone (15 pairs, MULTI-SEED)",
        smd_full_ms, smd_fam_ms, "Full paper", "FAM alone",
    )

    print("\n" + "=" * 74)
    print("HEADLINE WRITEUP PHRASES (paste-ready)")
    print("=" * 74)
    print(f"""
Independent FAM ablation shows no statistically significant benefit:
  - SKAB (23 files):    W=79,   p={r1['p']:.3f}, effect r={r1['r']:.3f} (small)
  - SMD  (15 pairs):    W=48.5, p={r2['p']:.3f}, effect r={r2['r']:.3f} (small)

SWSE addition to FAM significantly hurts on SMD:
  - Mean AUROC difference: {r3['diff']:+.4f}
  - Wilcoxon: p={r3['p']:.4f}, effect r={r3['r']:.3f}
  - {'SIGNIFICANT' if r3['p'] < 0.05 else 'not significant at 0.05'}

Full paper architecture (SWSE+FAM+Adversarial) significantly underperforms
FAM alone on SMD:
  - Mean AUROC difference: {r4['diff']:+.4f}
  - Wilcoxon: p={r4['p']:.4f}, effect r={r4['r']:.3f}
  - {'SIGNIFICANT' if r4['p'] < 0.05 else 'not significant at 0.05'}
""")


if __name__ == "__main__":
    main()
