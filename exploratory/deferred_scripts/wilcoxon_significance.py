"""
Wilcoxon signed-rank significance tests on the ablation findings.

Runs three paired non-parametric tests:
    1. FAM vs TimeOnly on SKAB (23 files, 3-seed means)
    2. FAM vs TimeOnly on SMD (5 machines x 3 seeds = 15 paired samples)
    3. Full paper (SWSE+FAM+Adv) vs FAM alone on SMD (5 machines)

Wilcoxon signed-rank is a non-parametric test for paired samples that
does not assume normality — appropriate for small AUROC datasets where
distribution shape is not guaranteed.

Reports statistic, p-value, and Rosenthal effect size r = Z / sqrt(N).

Usage:
    python scripts/wilcoxon_significance.py

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats


# ============================================================
#  Data (aggregated from experiment logs)
# ============================================================

# SKAB — per-file mean AUROC across 3 seeds
skab_fam = [
    0.914, 0.825, 0.979, 0.943, 0.857,      # valve1 0..4
    0.863, 0.912, 0.960, 0.949,             # valve2 0..3
    1.000, 0.875, 0.976, 0.985, 0.863, 0.923, 0.962,  # other 1..7
    0.786, 0.960, 0.939, 0.838, 0.870, 0.524, 0.938,  # other 8..14
]
skab_time = [
    0.919, 0.821, 0.979, 0.939, 0.853,
    0.853, 0.910, 0.965, 0.938,
    1.000, 0.851, 0.976, 0.980, 0.886, 0.928, 0.962,
    0.782, 0.956, 0.936, 0.843, 0.861, 0.538, 0.933,
]

# SMD — per-machine per-seed AUROC (15 paired samples)
smd_fam_all_seeds = [
    # machine-1-1: seed 0, 1, 42
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
smd_time_all_seeds = [
    0.964, 0.970, 0.971,
    0.861, 0.876, 0.890,
    0.879, 0.862, 0.874,
    0.957, 0.965, 0.963,
    0.960, 0.972, 0.964,
]

# SMD full 3-component test (single seed, 5 machines)
smd_fam_single = [0.970, 0.889, 0.892, 0.957, 0.962]
smd_full_single = [0.959, 0.898, 0.814, 0.942, 0.943]


# ============================================================
#  Helpers
# ============================================================

def wilcoxon_report(name, group_a, group_b, label_a, label_b):
    """Run Wilcoxon signed-rank test and print interpretation."""
    a = np.array(group_a)
    b = np.array(group_b)
    diffs = a - b
    n = len(a)
    n_nonzero = (diffs != 0).sum()

    # Two-sided Wilcoxon test
    result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    stat = result.statistic
    p = result.pvalue

    # Approximate Z from p-value for effect size estimate
    # For n large, Z ~ inverse-normal-cdf(p/2)
    z_from_p = abs(stats.norm.ppf(p / 2))
    effect_r = z_from_p / np.sqrt(n_nonzero) if n_nonzero > 0 else 0.0

    # Interpretation
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

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print("=" * 70)
    print(f"  {label_a:35s} mean {a.mean():.4f}   std {a.std():.4f}")
    print(f"  {label_b:35s} mean {b.mean():.4f}   std {b.std():.4f}")
    print(f"  Mean difference ({label_a} - {label_b}): {diffs.mean():+.4f}")
    print(f"  N pairs: {n}   Non-zero pairs: {n_nonzero}")
    print(f"  Wilcoxon statistic: {stat:.3f}")
    print(f"  p-value (two-sided): {p:.4f}")
    print(f"  Effect size (r): {effect_r:.3f}  ({eff_desc})")
    print(f"  Interpretation: {sig}")

    if p >= 0.10:
        print(f"  --> CANNOT REJECT null hypothesis: no significant difference.")
    else:
        winner = label_a if diffs.mean() > 0 else label_b
        print(f"  --> Reject null: {winner} is significantly better.")

    return {
        "n": n,
        "n_nonzero": int(n_nonzero),
        "mean_diff": float(diffs.mean()),
        "statistic": float(stat),
        "p_value": float(p),
        "effect_r": float(effect_r),
    }


def main():
    print("=" * 70)
    print("WILCOXON SIGNED-RANK SIGNIFICANCE TESTS")
    print("Non-parametric paired test for the FAM ablation finding")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1: FAM vs TimeOnly on SKAB (23 file-level means)
    # ------------------------------------------------------------
    r1 = wilcoxon_report(
        "TEST 1: SKAB — FAM vs TimeOnly (23 files, 3-seed means)",
        skab_fam, skab_time,
        "FAM (full)", "TimeOnly (ablated)",
    )

    # ------------------------------------------------------------
    # Test 2: FAM vs TimeOnly on SMD (15 machine-seed pairs)
    # ------------------------------------------------------------
    r2 = wilcoxon_report(
        "TEST 2: SMD — FAM vs TimeOnly (5 machines x 3 seeds = 15 pairs)",
        smd_fam_all_seeds, smd_time_all_seeds,
        "FAM (full)", "TimeOnly (ablated)",
    )

    # ------------------------------------------------------------
    # Test 3: Full paper vs FAM alone on SMD (5 machines, single seed)
    # ------------------------------------------------------------
    r3 = wilcoxon_report(
        "TEST 3: SMD — Full paper (SWSE+FAM+Adv) vs FAM alone (5 machines)",
        smd_full_single, smd_fam_single,
        "SWSE+FAM+Adversarial", "FAM alone (baseline)",
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — write-up ready phrases")
    print("=" * 70)

    print(f"""
FAM channel-axis attention does not provide statistically significant
improvement on either benchmark:

  SKAB: Wilcoxon signed-rank test on paired file means across 23 SKAB files
  yields W = {r1['statistic']:.1f}, p = {r1['p_value']:.3f} (n = {r1['n_nonzero']} non-zero pairs,
  effect size r = {r1['effect_r']:.3f}). Mean AUROC difference of {r1['mean_diff']:+.4f}
  is not statistically significant at the 0.05 level.

  SMD: Wilcoxon signed-rank test on 15 paired machine-seed AUROC values
  yields W = {r2['statistic']:.1f}, p = {r2['p_value']:.3f} (effect size r = {r2['effect_r']:.3f}).
  Mean AUROC difference of {r2['mean_diff']:+.4f} is not statistically significant.

Full paper architecture (SWSE + FAM + Adversarial) tested against
FAM alone on 5 SMD machines: W = {r3['statistic']:.1f}, p = {r3['p_value']:.3f},
mean AUROC difference of {r3['mean_diff']:+.4f}.
""")


if __name__ == "__main__":
    main()
