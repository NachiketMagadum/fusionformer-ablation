"""
Re-run F1 telemetry case study to produce the real reconstruction-error figure
for Chapter 4.6 of the dissertation.

Self-contained: no imports from `src.*`, everything defined inline so the
script runs from the flat project root.

USAGE (from the project root):
    python3 rerun_f1_case_study.py

OUTPUT:
    figures/fig_4_6_f1_reconstruction.png     (real plot)
    notes/f1_case_study_results.txt           (real numeric summary)

REQUIREMENTS:
    torch, fastf1, matplotlib, numpy, pandas, scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import fastf1

# ============================================================
# Config
# ============================================================
SEED = 42
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
D_MODEL = 32
FF_HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.1

FIGURES = Path("figures"); FIGURES.mkdir(exist_ok=True)
NOTES = Path("notes"); NOTES.mkdir(exist_ok=True)
CACHE = Path(".fastf1_cache"); CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Inline Fusionformer (FAM) — matches attention.py + blocks.py + fusionformer.py
# ============================================================
class FusionAttention(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(seq_len, 1, batch_first=True)
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
    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64, dropout=0.1):
        super().__init__()
        self.attention = FusionAttention(seq_len, n_features, d_model)
        self.norm1 = nn.LayerNorm(n_features)
        self.ff = nn.Sequential(nn.Linear(n_features, ff_hidden), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(ff_hidden, n_features))
        self.norm2 = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class Fusionformer(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, seq_len, n_features) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList([TransformerBlock(seq_len, n_features, d_model, ff_hidden, dropout)
                                  for _ in range(n_layers)])
        self.dec = nn.ModuleList([TransformerBlock(seq_len, n_features, d_model, ff_hidden, dropout)
                                  for _ in range(n_layers)])
    def forward(self, x):
        h = self.drop(x + self.pos)
        for b in self.enc: h = b(h)
        for b in self.dec: h = b(h)
        return h

# ============================================================
# Load F1 telemetry
# ============================================================
print("\nLoading Hamilton 2023 British GP telemetry via fastf1...")
session = fastf1.get_session(2023, 'British Grand Prix', 'R')
session.load(telemetry=True, laps=True, weather=False, messages=True)
hamilton = session.laps.pick_driver('HAM')
tel = hamilton.get_telemetry()

# Resample to 10 Hz uniform grid
tel = tel.set_index('Time').sort_index()
start = tel.index.min(); end = tel.index.max()
uniform_idx = pd.timedelta_range(start=start, end=end, freq='100ms')
features = ['Speed', 'RPM', 'Throttle', 'nGear', 'DRS', 'Brake']
tel_num = tel[features].apply(pd.to_numeric, errors='coerce').astype(float)
tel_resampled = tel_num.reindex(uniform_idx, method='nearest').ffill().bfill()

X = tel_resampled.values
race_minutes = (uniform_idx - uniform_idx[0]).total_seconds() / 60.0
print(f"  Loaded {len(X)} samples across {race_minutes[-1]:.1f} minutes, {X.shape[1]} channels")

# Standardise on first 25% (normal driving reference)
n_train = int(len(X) * 0.25)
scaler = StandardScaler().fit(X[:n_train])
X_scaled = scaler.transform(X)

# ============================================================
# Sliding windows
# ============================================================
def make_windows(data, seq_len):
    return np.stack([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])

train_windows = make_windows(X_scaled[:n_train], SEQ_LEN)
all_windows = make_windows(X_scaled, SEQ_LEN)
print(f"  Train windows: {len(train_windows)}, Score windows: {len(all_windows)}")

# ============================================================
# Train Fusionformer FAM
# ============================================================
model = Fusionformer(seq_len=SEQ_LEN, n_features=X.shape[1],
                     d_model=D_MODEL, ff_hidden=FF_HIDDEN,
                     n_layers=N_LAYERS, dropout=DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
train_t = torch.tensor(train_windows, dtype=torch.float32)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {n_params:,}")

print("\nTraining...")
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(train_t))
    losses = []
    for i in range(0, len(train_t), BATCH_SIZE):
        b = train_t[perm[i:i+BATCH_SIZE]].to(DEVICE)
        r = model(b); loss = criterion(r, b)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss = {np.mean(losses):.5f}")

# ============================================================
# Score every window
# ============================================================
print("\nScoring full race...")
model.eval()
all_t = torch.tensor(all_windows, dtype=torch.float32)
with torch.no_grad():
    errors = []
    for i in range(0, len(all_t), 512):
        chunk = all_t[i:i+512].to(DEVICE)
        recon = model(chunk)
        e = ((recon - chunk) ** 2).mean(dim=(1, 2)).cpu().numpy()
        errors.append(e)
    window_errors = np.concatenate(errors)

# Broadcast to per-sample
scores = np.empty(len(X_scaled))
scores[:SEQ_LEN - 1] = window_errors[0]
scores[SEQ_LEN - 1:] = window_errors

# ============================================================
# Plot
# ============================================================
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
p95 = np.percentile(scores, 95)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(race_minutes, scores, color="#2b6cb0", lw=0.8, alpha=0.85)
ax.axhline(p95, ls="--", c="grey", alpha=0.7, lw=1, label=f"95th percentile threshold ({p95:.3f})")
ax.axvspan(race_minutes[n_train], race_minutes[n_train], alpha=0)
ax.set_xlabel("Race time (minutes from race start)")
ax.set_ylabel("Reconstruction error")
ax.set_title("Figure 4.6: Fusionformer FAM reconstruction error on Hamilton's 2023 British GP telemetry\n"
             "Trained on first 25% of race, scored across full race")
ax.legend(loc="upper left", fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig("figures/fig_4_6_f1_reconstruction.png")
print(f"\nSaved figure to figures/fig_4_6_f1_reconstruction.png")

# ============================================================
# Numeric summary
# ============================================================
high = scores > p95
print(f"\nSamples above 95th percentile: {high.sum()} of {len(scores)} ({high.mean()*100:.1f}%)")
print(f"Max reconstruction error: {scores.max():.4f} at minute {race_minutes[np.argmax(scores)]:.1f}")
print(f"Baseline (mean of first 25%): {scores[:n_train].mean():.5f}")

with open("notes/f1_case_study_results.txt", "w") as f:
    f.write(f"F1 CASE STUDY RESULTS - Real experiment run\n")
    f.write(f"=" * 50 + "\n")
    f.write(f"Session: Hamilton, 2023 British Grand Prix (race)\n")
    f.write(f"Samples: {len(X)} (~10 Hz), Race length: {race_minutes[-1]:.1f} min\n")
    f.write(f"Features: {features}\n")
    f.write(f"Model: Fusionformer FAM, {n_params:,} params, {EPOCHS} epochs on {DEVICE}\n")
    f.write(f"Trained on first 25% ({n_train} samples), scored full race\n\n")
    f.write(f"95th percentile threshold: {p95:.5f}\n")
    f.write(f"Samples above threshold: {high.sum()} ({high.mean()*100:.1f}%)\n")
    f.write(f"Max error: {scores.max():.5f} at minute {race_minutes[np.argmax(scores)]:.1f}\n")
    f.write(f"Mean error, first 25%: {scores[:n_train].mean():.5f}\n")

print("\nDONE. Now re-run the stitch:")
print("  pandoc chapter_4_results.md -o chapter_4_results.docx")
print("  (then re-run master stitch)")
