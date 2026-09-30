"""
Generate SWSE ablation visualisation plots.

Produces:
    figures/swse_ablation_valve1.png    - Per-file AUROC for 3 variants
    figures/swse_config_comparison.png  - Effect of segment_len and epochs
    figures/fam_vs_timeonly_23files.png - Cross-folder ablation (FAM vs TimeOnly)

All plots saved as PNGs at 300 DPI, publication quality.

Usage:
    python scripts/plot_swse_results.py

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
#  Data (from tonight's experiments)
# ============================================================

# SWSE ablation on valve1 5 files at 50 epochs, seed 42
valve1_files = ["0.csv", "1.csv", "2.csv", "3.csv", "4.csv"]

fam_alone_50ep = [0.941, 0.825, 0.975, 0.944, 0.839]
swse_fam_50ep = [0.737, 0.694, 0.639, 0.684, 0.516]
swse_timeonly_50ep = [0.651, 0.636, 0.568, 0.636, 0.516]

# Same at 20 epochs
fam_alone_20ep = [0.926, 0.691, 0.862, 0.752, 0.845]
swse_fam_20ep = [0.565, 0.681, 0.579, 0.702, 0.699]
swse_timeonly_20ep = [0.496, 0.639, 0.562, 0.703, 0.569]

# At segment_len=2, 50 epochs
swse_fam_seg2 = [0.727, 0.607, 0.384, 0.616, 0.601]
swse_timeonly_seg2 = [0.569, 0.767, 0.409, 0.636, 0.635]


# ============================================================
#  Plot 1: Per-file bar chart, 3 variants at 50 epochs
# ============================================================

def plot_ablation_per_file():
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(valve1_files))
    width = 0.27

    bars1 = ax.bar(x - width, fam_alone_50ep, width,
                   label="FAM alone (no SWSE)", color="#2E7D32", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, swse_fam_50ep, width,
                   label="SWSE + FAM", color="#C62828", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, swse_timeonly_50ep, width,
                   label="SWSE + TimeOnly", color="#EF6C00", edgecolor="black", linewidth=0.5)

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # Random baseline reference
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random baseline")

    ax.set_xlabel("SKAB valve1 file")
    ax.set_ylabel("AUROC")
    ax.set_title("SWSE ablation on SKAB valve1 (50 epochs, seed 42)")
    ax.set_xticks(x)
    ax.set_xticklabels(valve1_files)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "swse_ablation_valve1.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
#  Plot 2: Effect of segment_len and epochs
# ============================================================

def plot_config_comparison():
    fig, ax = plt.subplots(figsize=(9, 5))

    configs = ["FAM alone\n(baseline)", "SWSE+FAM\nseg=5, 50ep",
               "SWSE+FAM\nseg=2, 50ep", "SWSE+FAM\nseg=5, 20ep"]

    means = [
        np.mean(fam_alone_50ep),
        np.mean(swse_fam_50ep),
        np.mean(swse_fam_seg2),
        np.mean(swse_fam_20ep),
    ]
    stds = [
        np.std(fam_alone_50ep),
        np.std(swse_fam_50ep),
        np.std(swse_fam_seg2),
        np.std(swse_fam_20ep),
    ]

    colors = ["#2E7D32", "#C62828", "#C62828", "#C62828"]

    bars = ax.bar(configs, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.5,
                  error_kw={"linewidth": 1.5, "ecolor": "black"})

    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(stds) * 0.3,
                f"{m:.2f}", ha="center", va="bottom", fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(3.5, 0.51, "random", color="gray", fontsize=8, ha="right", va="bottom")

    ax.set_ylabel("Mean AUROC across 5 valve1 files")
    ax.set_title("SWSE hurts across all tested configurations")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "swse_config_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
#  Plot 3: FAM vs TimeOnly across 23 SKAB files
#  (from multi-seed variance sweep, 3 seeds averaged)
# ============================================================

def plot_fam_vs_timeonly():
    # From experiment_variance_sweep.py output
    files_by_folder = {
        "valve1": ["0", "1", "2", "3", "4"],
        "valve2": ["0", "1", "2", "3"],
        "other": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"],
    }

    all_labels, fam_means, time_means = [], [], []
    for folder, ids in files_by_folder.items():
        for f in ids:
            all_labels.append(f"{folder[:2]}-{f}")

    # Mean AUROC across 3 seeds, from tonight's variance sweep
    # (Manually transcribed from log — could load from file too)
    fam_means_full = [
        # valve1
        0.914, 0.825, 0.979, 0.943, 0.857,
        # valve2
        0.863, 0.912, 0.960, 0.949,
        # other
        1.000, 0.875, 0.976, 0.985, 0.863, 0.923, 0.962,
        0.786, 0.960, 0.939, 0.838, 0.870, 0.524, 0.938,
    ]
    time_means_full = [
        # valve1
        0.919, 0.821, 0.979, 0.939, 0.853,
        # valve2
        0.853, 0.910, 0.965, 0.938,
        # other
        1.000, 0.851, 0.976, 0.980, 0.886, 0.928, 0.962,
        0.782, 0.956, 0.936, 0.843, 0.861, 0.538, 0.933,
    ]

    diffs = np.array(fam_means_full) - np.array(time_means_full)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(all_labels))
    colors = ["#2E7D32" if d > 0 else "#C62828" for d in diffs]
    bars = ax.bar(x, diffs, color=colors, edgecolor="black", linewidth=0.4)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.02, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(y=-0.02, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("FAM AUROC minus TimeOnly AUROC")
    ax.set_title("FAM vs TimeOnly per file (3-seed mean; grey bands = ±0.02 noise band)")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#2E7D32", label="FAM wins"),
        Patch(facecolor="#C62828", label="TimeOnly wins"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate grand mean
    grand = np.mean(diffs)
    ax.text(0.5, 0.95, f"Grand mean advantage: {grand:+.4f}",
            transform=ax.transAxes, ha="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    plt.tight_layout()
    out = FIGURES_DIR / "fam_vs_timeonly_23files.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_ablation_per_file()
    plot_config_comparison()
    plot_fam_vs_timeonly()
    print("\nAll plots saved to figures/")
