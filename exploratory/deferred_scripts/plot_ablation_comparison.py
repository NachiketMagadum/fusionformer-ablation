"""
Generate SKAB vs SMD ablation comparison plots.

Produces:
    figures/ablation_skab_vs_smd.png     - Per-file/machine bar chart, both datasets
    figures/ablation_grand_mean.png      - Grand mean comparison with error bars
    figures/seed_variance_comparison.png - Seed stability comparison

Uses the data from the multi-seed variance sweeps on both benchmarks.

Usage:
    python scripts/plot_ablation_comparison.py

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
#  Data (aggregated from experiment logs)
# ============================================================

# SKAB — grand mean across 3 seeds per file, 23 files total
skab_files = [
    "v1-0", "v1-1", "v1-2", "v1-3", "v1-4",
    "v2-0", "v2-1", "v2-2", "v2-3",
    "o-1", "o-2", "o-3", "o-4", "o-5", "o-6", "o-7",
    "o-8", "o-9", "o-10", "o-11", "o-12", "o-13", "o-14",
]

skab_fam_means = [
    0.914, 0.825, 0.979, 0.943, 0.857,
    0.863, 0.912, 0.960, 0.949,
    1.000, 0.875, 0.976, 0.985, 0.863, 0.923, 0.962,
    0.786, 0.960, 0.939, 0.838, 0.870, 0.524, 0.938,
]
skab_time_means = [
    0.919, 0.821, 0.979, 0.939, 0.853,
    0.853, 0.910, 0.965, 0.938,
    1.000, 0.851, 0.976, 0.980, 0.886, 0.928, 0.962,
    0.782, 0.956, 0.936, 0.843, 0.861, 0.538, 0.933,
]

# SMD — per-machine means across 3 seeds each
smd_machines = ["m1-1", "m1-4", "m2-1", "m2-5", "m3-1"]
smd_fam_means = [0.973, 0.880, 0.869, 0.953, 0.968]
smd_time_means = [0.968, 0.876, 0.872, 0.962, 0.965]

# Per-seed values for seed-variance plot
smd_fam_by_seed = {
    "m1-1": [0.974, 0.976, 0.970],
    "m1-4": [0.862, 0.888, 0.889],
    "m2-1": [0.893, 0.821, 0.892],
    "m2-5": [0.964, 0.938, 0.957],
    "m3-1": [0.971, 0.970, 0.962],
}
smd_time_by_seed = {
    "m1-1": [0.964, 0.970, 0.971],
    "m1-4": [0.861, 0.876, 0.890],
    "m2-1": [0.879, 0.862, 0.874],
    "m2-5": [0.957, 0.965, 0.963],
    "m3-1": [0.960, 0.972, 0.964],
}


# ============================================================
#  Plot 1: Side-by-side per-file / per-machine diff
# ============================================================

def plot_per_item_diff():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5),
                                    gridspec_kw={"width_ratios": [23, 5]})

    # SKAB panel
    skab_diffs = np.array(skab_fam_means) - np.array(skab_time_means)
    x_skab = np.arange(len(skab_files))
    colors_skab = ["#2E7D32" if d > 0.005 else ("#C62828" if d < -0.005 else "#9E9E9E") for d in skab_diffs]
    ax1.bar(x_skab, skab_diffs, color=colors_skab, edgecolor="black", linewidth=0.4)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(0.02, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.axhline(-0.02, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.set_xticks(x_skab)
    ax1.set_xticklabels(skab_files, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("FAM AUROC minus TimeOnly AUROC")
    ax1.set_title(f"SKAB — 23 files (grand mean diff: {np.mean(skab_diffs):+.4f})")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(-0.06, 0.06)

    # SMD panel
    smd_diffs = np.array(smd_fam_means) - np.array(smd_time_means)
    x_smd = np.arange(len(smd_machines))
    colors_smd = ["#2E7D32" if d > 0.005 else ("#C62828" if d < -0.005 else "#9E9E9E") for d in smd_diffs]
    ax2.bar(x_smd, smd_diffs, color=colors_smd, edgecolor="black", linewidth=0.4)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0.02, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.axhline(-0.02, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_xticks(x_smd)
    ax2.set_xticklabels(smd_machines, rotation=45, ha="right", fontsize=9)
    ax2.set_title(f"SMD — 5 machines (grand mean diff: {np.mean(smd_diffs):+.4f})")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(-0.06, 0.06)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#2E7D32", label="FAM wins by >0.005"),
        Patch(facecolor="#C62828", label="TimeOnly wins by >0.005"),
        Patch(facecolor="#9E9E9E", label="Tied (within ±0.005)"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("FAM vs TimeOnly per-file/machine — both datasets, 3-seed means")
    plt.tight_layout()
    out = FIGURES_DIR / "ablation_skab_vs_smd.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
#  Plot 2: Grand mean comparison
# ============================================================

def plot_grand_mean():
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = ["SKAB\n(23 files × 3 seeds)", "SMD\n(5 machines × 3 seeds)"]
    fam_means = [np.mean(skab_fam_means), np.mean(smd_fam_means)]
    time_means = [np.mean(skab_time_means), np.mean(smd_time_means)]
    fam_stds = [np.std(skab_fam_means), np.std(smd_fam_means)]
    time_stds = [np.std(skab_time_means), np.std(smd_time_means)]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, fam_means, width, yerr=fam_stds, capsize=8,
                   label="Fusionformer FAM (dual-axis)", color="#1F4E79",
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, time_means, width, yerr=time_stds, capsize=8,
                   label="TimeOnly (ablated)", color="#F4A261",
                   edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Grand mean AUROC (± std across files/machines)")
    ax.set_title("Grand mean AUROC — FAM vs TimeOnly on both datasets")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "ablation_grand_mean.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
#  Plot 3: Seed variance comparison (SMD)
# ============================================================

def plot_seed_variance():
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(smd_machines))
    width = 0.35

    fam_vals = [smd_fam_by_seed[m] for m in smd_machines]
    time_vals = [smd_time_by_seed[m] for m in smd_machines]

    # Box-and-whisker for both
    positions_fam = x - width/2
    positions_time = x + width/2

    bp1 = ax.boxplot(fam_vals, positions=positions_fam, widths=0.3,
                     patch_artist=True, medianprops={"color": "black"})
    for patch in bp1["boxes"]:
        patch.set_facecolor("#1F4E79")

    bp2 = ax.boxplot(time_vals, positions=positions_time, widths=0.3,
                     patch_artist=True, medianprops={"color": "black"})
    for patch in bp2["boxes"]:
        patch.set_facecolor("#F4A261")

    ax.set_xticks(x)
    ax.set_xticklabels(smd_machines)
    ax.set_xlabel("SMD machine")
    ax.set_ylabel("AUROC across 3 seeds")
    ax.set_title("Seed variance — FAM (blue) vs TimeOnly (orange)\n"
                 "TimeOnly is measurably more stable on some machines (e.g. m2-1)")
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1F4E79", label="FAM"),
        Patch(facecolor="#F4A261", label="TimeOnly"),
    ], loc="lower right")

    plt.tight_layout()
    out = FIGURES_DIR / "seed_variance_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_per_item_diff()
    plot_grand_mean()
    plot_seed_variance()
    print("\nAll comparison plots saved to figures/")
