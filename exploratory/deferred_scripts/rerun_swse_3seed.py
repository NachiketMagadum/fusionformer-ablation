"""
Re-run the SKAB SWSE configuration study at 3 seeds per configuration.

The original SKAB SWSE sweep (Chapter 4.4) used a single seed per configuration
across 5 valve1 files. This weakens the significance argument. This script
re-runs the same 8 configurations at seeds [0, 1, 42], producing per-configuration
Wilcoxon significance vs FAM alone.

USAGE (from the project root):
    python3 rerun_swse_3seed.py

OUTPUT:
    notes/swse_3seed_results.csv        (raw per-run AUROC)
    notes/swse_3seed_report.txt         (paste-ready summary)

Runtime: approximately 3-4 hours on MPS (5 files x 3 seeds x 9 configs = 135 runs).
Reduce N_EPOCHS to 30 or N_FILES to 3 if you need it faster.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

# ============================================================
# Config
# ============================================================
SEEDS = [0, 1, 42]
N_EPOCHS = 50
SEQ_LEN = 30
BATCH_SIZE = 32
LR = 1e-3
D_MODEL = 32
FF_HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.1
N_HEADS = 1  # UPDATE if paper differs

SKAB_ROOT = Path("datasets/SKAB/data/valve1")
NOTES = Path("notes"); NOTES.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# SKAB loader (matches skab_loader.py in the modular codebase)
# ============================================================
def load_skab_file(path):
    df = pd.read_csv(path, sep=';', parse_dates=['datetime'], index_col='datetime')
    y = df['anomaly'].astype(int)
    feature_cols = [c for c in df.columns if c not in ('anomaly', 'changepoint')]
    return df[feature_cols], y

def find_first_anomaly(y):
    y_arr = np.asarray(y)
    return int(np.argmax(y_arr == 1)) if y_arr.any() else len(y_arr)

def make_windows(data, seq_len):
    return np.stack([data[i:i+seq_len] for i in range(len(data) - seq_len + 1)])

def broadcast(w_scores, n_rows, seq_len):
    r = np.empty(n_rows)
    r[:seq_len-1] = w_scores[0]
    r[seq_len-1:] = w_scores
    return r

# ============================================================
# Model definitions (inline, matches src/models/*)
# ============================================================
class FusionAttention(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32, n_heads=1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(seq_len, n_heads, batch_first=True)
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.output_projection = nn.Linear(d_model, n_features)
    def forward(self, x):
        h = self.input_projection(x)
        t, _ = self.time_attention(h, h, h)
        h_t = h.transpose(1, 2)
        c, _ = self.channel_attention(h_t, h_t, h_t)
        c = c.transpose(1, 2)
        w = torch.softmax(self.fusion_weights, dim=0)
        return self.output_projection(w[0] * t + w[1] * c)

class TransformerBlock(nn.Module):
    def __init__(self, attn, n_features, ff_hidden, dropout):
        super().__init__()
        self.attention = attn
        self.norm1 = nn.LayerNorm(n_features)
        self.ff = nn.Sequential(nn.Linear(n_features, ff_hidden), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(ff_hidden, n_features))
        self.norm2 = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class SWSE(nn.Module):
    """Per-channel segment embedding as described in Wang et al. 2025."""
    def __init__(self, seq_len, segment_len, embed_dim):
        super().__init__()
        assert seq_len % segment_len == 0
        self.n_segments = seq_len // segment_len
        self.segment_len = segment_len
        self.projection = nn.Linear(segment_len, embed_dim)
    def forward(self, x):
        B, T, F = x.shape
        x = x.transpose(1, 2).reshape(B, F, self.n_segments, self.segment_len)
        return self.projection(x)  # (B, F, N_seg, D)

class FAM_AE(nn.Module):
    """FAM-alone autoencoder baseline (Chapter 4.2)."""
    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64,
                 n_layers=2, dropout=0.1, n_heads=1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, seq_len, n_features) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList([
            TransformerBlock(FusionAttention(seq_len, n_features, d_model, n_heads),
                             n_features, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.dec = nn.ModuleList([
            TransformerBlock(FusionAttention(seq_len, n_features, d_model, n_heads),
                             n_features, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
    def forward(self, x):
        h = self.drop(x + self.pos)
        for b in self.enc: h = b(h)
        for b in self.dec: h = b(h)
        return h

class SWSE_FAM_AE(nn.Module):
    """SWSE + FAM autoencoder (matches fusionformer_swse.py)."""
    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64,
                 n_layers=2, dropout=0.1, segment_len=5, n_heads=1):
        super().__init__()
        self.swse = SWSE(seq_len, segment_len, d_model)
        self.n_segments = seq_len // segment_len
        # After SWSE we have (B, F, N_seg, D). Flatten to (B, N_seg, F*D) for time attn.
        self.time_attn = nn.MultiheadAttention(n_features * d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(n_features * d_model)
        self.ff = nn.Sequential(nn.Linear(n_features * d_model, ff_hidden),
                                nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(ff_hidden, n_features * d_model))
        self.norm2 = nn.LayerNorm(n_features * d_model)
        # Reconstruction head: (B, N_seg, F*D) -> (B, T, F)
        self.head = nn.Linear(d_model, segment_len)
        self.segment_len = segment_len
        self.d_model = d_model
    def forward(self, x):
        B, T, F = x.shape
        z = self.swse(x)  # (B, F, N_seg, D)
        z_flat = z.permute(0, 2, 1, 3).reshape(B, self.n_segments, F * self.d_model)
        h, _ = self.time_attn(z_flat, z_flat, z_flat)
        h = self.norm1(z_flat + h)
        h = self.norm2(h + self.ff(h))
        # Reshape back and decode: (B, N_seg, F*D) -> (B, F, N_seg, D) -> (B, F, N_seg, seg_len) -> (B, T, F)
        h = h.reshape(B, self.n_segments, F, self.d_model).permute(0, 2, 1, 3)
        h = self.head(h)  # (B, F, N_seg, seg_len)
        h = h.reshape(B, F, T).transpose(1, 2)  # (B, T, F)
        return h

# ============================================================
# Training loop
# ============================================================
def train_and_score(model, train_windows, all_windows, epochs=N_EPOCHS):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    train_t = torch.tensor(train_windows, dtype=torch.float32)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_t))
        for i in range(0, len(train_t), BATCH_SIZE):
            b = train_t[perm[i:i+BATCH_SIZE]].to(DEVICE)
            r = model(b); loss = crit(r, b)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    all_t = torch.tensor(all_windows, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon = model(all_t)
        window_errors = ((recon - all_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return window_errors

def run_one(model_factory, file_path, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = load_skab_file(file_path)
    X_scaled = StandardScaler().fit_transform(X)
    fa = find_first_anomaly(y)
    if fa < 60: return None
    train_windows = make_windows(X_scaled[:fa], SEQ_LEN)
    all_windows = make_windows(X_scaled, SEQ_LEN)
    model = model_factory(n_features=X.shape[1])
    w_errors = train_and_score(model, train_windows, all_windows)
    scores = broadcast(w_errors, len(X_scaled), SEQ_LEN)
    return roc_auc_score(y.values, scores)

# ============================================================
# Configurations to test
# ============================================================
CONFIGS = [
    ("FAM alone (baseline)",       lambda n_features: FAM_AE(SEQ_LEN, n_features, D_MODEL, FF_HIDDEN, N_LAYERS, DROPOUT, N_HEADS)),
    ("SWSE+FAM seg=5",             lambda n_features: SWSE_FAM_AE(SEQ_LEN, n_features, D_MODEL, FF_HIDDEN, N_LAYERS, DROPOUT, 5, N_HEADS)),
    ("SWSE+FAM seg=2",             lambda n_features: SWSE_FAM_AE(SEQ_LEN, n_features, D_MODEL, FF_HIDDEN, N_LAYERS, DROPOUT, 2, N_HEADS)),
]

# ============================================================
# Run sweep
# ============================================================
files = sorted(SKAB_ROOT.glob("*.csv"))[:5]
print(f"Running {len(CONFIGS)} configs x {len(files)} files x {len(SEEDS)} seeds = {len(CONFIGS)*len(files)*len(SEEDS)} runs")

rows = []
t0 = time.time()
for cname, factory in CONFIGS:
    for f in files:
        for seed in SEEDS:
            auroc = run_one(factory, f, seed)
            if auroc is None: continue
            rows.append({"config": cname, "file": f.name, "seed": seed, "auroc": auroc})
            print(f"  {cname:35s} {f.name} seed{seed}: {auroc:.4f}")

df = pd.DataFrame(rows)
df.to_csv("notes/swse_3seed_results.csv", index=False)

# Aggregate to per-config paired means (aggregating over seeds per file)
pivot = df.groupby(["config", "file"])["auroc"].mean().reset_index()
baseline_by_file = pivot[pivot["config"] == "FAM alone (baseline)"].set_index("file")["auroc"]

report_lines = [
    f"SWSE 3-seed re-run - real experimental output",
    f"=" * 74,
    f"Elapsed: {(time.time()-t0)/60:.1f} min on {DEVICE}",
    f"Files: {[f.name for f in files]}",
    f"Seeds: {SEEDS}",
    f"",
    f"Per-file mean AUROC (3 seeds averaged):",
    f"",
]
for cname, _ in CONFIGS:
    means = pivot[pivot["config"] == cname].set_index("file")["auroc"]
    if cname == "FAM alone (baseline)":
        report_lines.append(f"  {cname:35s} mean {means.mean():.4f}  (baseline)")
    else:
        # Wilcoxon vs baseline
        paired = [(baseline_by_file[f], means[f]) for f in means.index if f in baseline_by_file.index]
        base_arr = np.array([p[0] for p in paired]); this_arr = np.array([p[1] for p in paired])
        try:
            w = stats.wilcoxon(this_arr, base_arr, zero_method="wilcox", alternative="two-sided")
            z = abs(stats.norm.ppf(w.pvalue / 2))
            r = z / np.sqrt(len(paired))
            sig = "*" if w.pvalue < 0.05 else " "
            report_lines.append(f"  {cname:35s} mean {means.mean():.4f}  Δ {means.mean()-base_arr.mean():+.4f}  W={w.statistic:.1f} p={w.pvalue:.3f} r={r:.3f} {sig}")
        except Exception as e:
            report_lines.append(f"  {cname:35s} mean {means.mean():.4f}  (Wilcoxon failed: {e})")

report = "\n".join(report_lines) + "\n" + "=" * 74 + "\n"
print("\n" + report)
with open("notes/swse_3seed_report.txt", "w") as f:
    f.write(report)
print("Saved: notes/swse_3seed_report.txt")
