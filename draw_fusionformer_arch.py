"""Draw the Fusionformer architecture diagram for Figure 2.1 with strict
non-overlapping layout."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

FIG = Path("figures/fig_2_1_fusionformer_architecture.png")
FIG.parent.mkdir(exist_ok=True)

# Colors
SWSE_COL = "#c8d9f0"
MSWAA_COL = "#7ba7d4"
MSWEA_COL = "#4c85c4"
FFN_COL = "#e8eef7"
DISC_COL = "#f8d0c4"
REAL_COL = "#c8e6c8"
FAKE_COL = "#f4d8c8"
LOSS_COL = "#f4dfc8"
BOX_EDGE = "#333"
ARROW_COL = "#222"
FAM_COL = "#c0392b"
ADV_COL = "#a03020"

# Canvas
fig, ax = plt.subplots(figsize=(11, 15))
ax.set_xlim(0, 11)
ax.set_ylim(0, 20)
ax.set_axis_off()

def box(x, y, w, h, label, color=FFN_COL, fontsize=9, weight="normal"):
    """Draw a rounded rectangle with centered text."""
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.08",
                       linewidth=1.0, edgecolor=BOX_EDGE, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=fontsize, weight=weight)

def arrow(x1, y1, x2, y2, color=ARROW_COL, style="-", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                linestyle=style, lw=lw,
                                shrinkA=2, shrinkB=2))

# Title
ax.text(5.5, 19.4, "Figure 2.1: Fusionformer architecture (Wang et al., 2025, Section III)",
        ha="center", fontsize=13)

# =============================================================
# LEFT COLUMN: main forecasting pipeline (x = 0.6 to 5.6)
# =============================================================

# Historical window
box(1.6, 17.6, 4.0, 0.9, "Historical window\n$X_{1:T} \\in \\mathbb{R}^{B \\times T \\times D}$", color="#dbe4f2", fontsize=10)
arrow(3.6, 17.55, 3.6, 16.85)

# SWSE
box(1.2, 15.9, 4.8, 0.9, "Component 1 – SWSE\nsegment $\\to$ linear proj. + positional encoding\n$U \\in \\mathbb{R}^{B \\times D \\times L_{sn} \\times d_{model}}$", color=SWSE_COL, fontsize=9)
arrow(3.6, 15.85, 3.6, 15.15)

# Encoder header
ax.text(0.6, 15.05, "• Encoder block $\\times$ $N_0$", fontsize=10.5, weight="bold", ha="left")

# MSWAA
box(1.2, 13.7, 4.8, 1.0, "MSWAA — multi-head time attention\nattends over $L_{sn}$ segments per variable\nresidual + LayerNorm", color=MSWAA_COL, fontsize=9)
arrow(3.6, 13.65, 3.6, 12.95)

# FFN 1
box(1.2, 12.05, 4.8, 0.85, "Position-wise FFN + residual + LayerNorm", color=FFN_COL, fontsize=9)
arrow(3.6, 12.0, 3.6, 11.35)

# MSWEA
box(1.2, 10.3, 4.8, 1.0, "MSWEA — multi-head variable attention\nattends across $D$ variables at each segment\nresidual + LayerNorm", color=MSWEA_COL, fontsize=9)
arrow(3.6, 10.25, 3.6, 9.55)

# FFN 2
box(1.2, 8.65, 4.8, 0.85, "Position-wise FFN + residual + LayerNorm", color=FFN_COL, fontsize=9)
arrow(3.6, 8.6, 3.6, 7.95)

# Decoder header
ax.text(0.6, 7.85, "• Decoder block $\\times$ $M_1$   (same MSWAA $\\to$ FFN $\\to$ MSWEA $\\to$ FFN structure)",
        fontsize=9.5, weight="bold", ha="left")

# Decoder block
box(1.2, 6.35, 4.8, 1.15, "Decoder block\nMSWAA $\\to$ FFN $\\to$ MSWEA $\\to$ FFN\n(each with residual + LayerNorm)", color=FFN_COL, fontsize=9)
arrow(3.6, 6.3, 3.6, 5.6)

# Output projection
box(1.2, 4.45, 4.8, 1.05, "Output projection\nflatten per-variable ($L_{sn} \\times d_{model}$) $\\to$ Linear $\\to$ $\\tau$ future values\n$\\hat{Y}_{T+1:T+\\tau} \\in \\mathbb{R}^{B \\times \\tau \\times D}$", color="#dbe4f2", fontsize=9)

# =============================================================
# RIGHT COLUMN: adversarial branch (x = 6.8 to 10.7)
# =============================================================

# Header
ax.text(8.75, 15.95, "Component 3\nAdversarial training", ha="center",
        fontsize=10.5, color=ADV_COL, weight="bold")

# Y_real
box(7.4, 13.75, 3.0, 0.9, "$Y_{real}$\n$= [X_{1:T}, Y_{true}]$", color=REAL_COL, fontsize=9)

# Y_fake
box(7.4, 12.05, 3.0, 0.9, "$Y_{fake}$\n$= [X_{1:T}, \\hat{Y}]$", color=FAKE_COL, fontsize=9)

# Discriminator
box(7.0, 9.85, 3.8, 1.1, "Discriminator D\n3 fully-connected layers + Sigmoid\noutputs probability that input is real", color=DISC_COL, fontsize=9)

# Training loss
box(7.0, 7.55, 3.8, 1.35, "Training loss\n$\\mathcal{L} = $ Huber$(\\hat{Y}, Y_{true})$\n$+ \\lambda_{adv} \\cdot$ BCE$(D(Y_{fake}), 1)$", color=LOSS_COL, fontsize=9)

# Arrows in right column
arrow(8.9, 13.7, 8.9, 12.95)   # Y_real -> Y_fake area (both feed disc)
arrow(8.9, 12.0, 8.9, 10.95)   # Y_fake -> Discriminator
arrow(8.9, 9.8, 8.9, 8.9)      # Discriminator -> Training loss

# =============================================================
# CROSS-COLUMN CONNECTIONS (only in empty regions)
# =============================================================

# Historical window -> Y_real (true future, grey curve, through top-right whitespace)
ax.annotate("", xy=(7.5, 14.65), xytext=(5.5, 17.7),
            arrowprops=dict(arrowstyle="->", color="#7a7a7a",
                            connectionstyle="arc3,rad=-0.3",
                            lw=1.0))
ax.text(6.3, 16.2, "true future", fontsize=8.5, color="#5a5a5a", style="italic")

# Output projection -> Y_fake (predicted future, blue curve)
ax.annotate("", xy=(7.5, 12.35), xytext=(5.5, 5.0),
            arrowprops=dict(arrowstyle="->", color="#2960a8",
                            connectionstyle="arc3,rad=-0.55",
                            lw=1.0))
ax.text(6.4, 5.9, "predicted future", fontsize=8.5, color="#2960a8", style="italic")

# Training loss -> Output projection (gradient flow, red dashed)
ax.annotate("", xy=(5.7, 4.8), xytext=(7.0, 7.9),
            arrowprops=dict(arrowstyle="->", color="#c0392b",
                            connectionstyle="arc3,rad=0.25",
                            linestyle="dashed", lw=1.1))
ax.text(6.15, 6.3, "gradient flow", fontsize=8.5, color="#c0392b", style="italic")

# =============================================================
# FAM bracket (Component 2) — right of encoder, well clear of everything
# =============================================================
# Bracket runs from top of MSWAA (14.7) down to bottom of FFN2 (8.65).
# Placed at x = 6.15 so it does not touch the right column (starts x=7.0).
bracket_x = 6.15
ax.plot([bracket_x, bracket_x], [8.75, 14.6], color=FAM_COL, lw=1.5)
ax.plot([bracket_x - 0.10, bracket_x], [14.6, 14.6], color=FAM_COL, lw=1.5)
ax.plot([bracket_x - 0.10, bracket_x], [8.75, 8.75], color=FAM_COL, lw=1.5)
ax.text(bracket_x + 0.10, 11.7, "Component 2\nFAM =\nMSWAA + MSWEA",
        color=FAM_COL, fontsize=9, ha="left", va="center", weight="bold")

# =============================================================
# Footer note
# =============================================================
foot = FancyBboxPatch((0.5, 2.15), 10.0, 1.55,
                     boxstyle="round,pad=0.1,rounding_size=0.05",
                     linewidth=0.8, edgecolor="#999", facecolor="#f4f4f4")
ax.add_patch(foot)
ax.text(5.5, 3.15,
        r"$\bf{Anomaly\ scoring\ at\ inference}$: for each test window, compute the forecast MSE $\|\hat{Y} - Y_{true}\|^2$.  Higher MSE = more anomalous.",
        ha="center", fontsize=9.5, style="italic")
ax.text(5.5, 2.55,
        r"Three components ablated in this dissertation: (1) SWSE   (2) FAM (specifically the MSWEA branch)   (3) Adversarial training.",
        ha="center", fontsize=9.5, style="italic")

# =============================================================
# Legend (bottom-left)
# =============================================================
legend_items = [
    (SWSE_COL, "SWSE (Component 1)"),
    (MSWAA_COL, "MSWAA — time-axis attention"),
    (MSWEA_COL, "MSWEA — variable-axis attention"),
    (FFN_COL, "FFN + residual + LayerNorm"),
    (DISC_COL, "Discriminator (Component 3)"),
]
for i, (col, label) in enumerate(legend_items):
    y = 1.4 - i * 0.25
    r = Rectangle((0.6, y), 0.35, 0.18, facecolor=col, edgecolor=BOX_EDGE, linewidth=0.7)
    ax.add_patch(r)
    ax.text(1.05, y + 0.09, label, fontsize=9, va="center")

plt.tight_layout()
plt.savefig(FIG, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"wrote {FIG}")
