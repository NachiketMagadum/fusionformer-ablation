"""
Cross-folder ablation: FAM vs TimeOnly on SKAB valve2 and 'other' folders.

Purpose: test whether the fusion attention mechanism helps on datasets
with different anomaly patterns. The valve1 ablation showed no benefit,
but valve1 anomalies may be predominantly temporal. valve2 (different
valve location) and 'other' (rotor imbalance, cavitation, water surge)
may exhibit stronger cross-channel signatures.

Test set:
    - 3 files from valve2 (inlet-valve closure): 1.csv, 8.csv, 15.csv
    - 3 files from other (diverse anomaly types):
        - 13.csv: Sharp rotor imbalance
        - 18.csv: Slow water increase
        - 21.csv: Two-phase flow / cavitation

Same 50-epoch config as valve1 ablation for direct comparability.

Usage:
    python scripts/experiment_ablation_crossfolder.py

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.data.skab_loader import load_skab_file
from src.eval.metrics import compute_all_metrics
from src.models.fusionformer import Fusionformer


# ---------- Config (matched to earlier ablation) ----------
SEQ_LEN     = 30
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SEED        = 42

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1

TEST_FILES = [
    ("valve2/1.csv",  "valve2 file 1"),
    ("valve2/8.csv",  "valve2 file 8"),
    ("valve2/15.csv", "valve2 file 15"),
    ("other/13.csv",  "other: sharp rotor imbalance"),
    ("other/18.csv",  "other: slow water increase"),
    ("other/21.csv",  "other: two-phase / cavitation"),
]

DATA_ROOT = "datasets/SKAB/data"


# ============================================================
#  Ablated modules (same as experiment_ablation.py)
# ============================================================

class TimeOnlyAttention(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x):
        h = self.input_projection(x)
        time_out, _ = self.time_attention(h, h, h)
        return self.output_projection(time_out)


class TimeOnlyBlock(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64, dropout=0.1):
        super().__init__()
        self.attention = TimeOnlyAttention(seq_len, n_features, d_model)
        self.norm1 = nn.LayerNorm(n_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_features, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, n_features),
        )
        self.norm2 = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class FusionformerTimeOnly(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, n_features) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TimeOnlyBlock(seq_len, n_features, d_model, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            TimeOnlyBlock(seq_len, n_features, d_model, ff_hidden, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = x + self.pos_encoding
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        for block in self.decoder_blocks:
            x = block(x)
        return x


# ============================================================
#  Training and evaluation
# ============================================================

def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_and_eval(model_cls, file_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = load_skab_file(file_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_arr = y.values
    first_anom = int(np.argmax(y_arr == 1)) if y_arr.any() else len(y_arr)

    if first_anom < SEQ_LEN + 5:
        # Not enough normal data to train — skip and mark
        return {"auroc": None, "skip_reason": "not enough normal data"}

    X_train_scaled = X_scaled[:first_anom]
    train_windows = make_windows(X_train_scaled, SEQ_LEN)
    train_windows_t = torch.tensor(train_windows, dtype=torch.float32)

    all_windows = make_windows(X_scaled, SEQ_LEN)
    all_windows_t = torch.tensor(all_windows, dtype=torch.float32)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = model_cls(
        seq_len=SEQ_LEN,
        n_features=X.shape[1],
        d_model=D_MODEL,
        ff_hidden=FF_HIDDEN,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_windows_t))
        for i in range(0, len(train_windows_t), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            batch = train_windows_t[batch_idx].to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_batch = all_windows_t.to(device)
        reconstruction = model(eval_batch)
        window_errors = ((reconstruction - eval_batch) ** 2).mean(
            dim=(1, 2)
        ).cpu().numpy()

    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    return compute_all_metrics(y.values, scores)


def main():
    print("=" * 78)
    print("CROSS-FOLDER ABLATION: FAM (dual-axis) vs TimeOnly (single-axis)")
    print("=" * 78)

    fam_scores, time_scores = [], []

    for rel_path, label in TEST_FILES:
        full_path = f"{DATA_ROOT}/{rel_path}"
        print(f"\n{label}")
        print(f"  ({rel_path})")

        fam_m = train_and_eval(Fusionformer, full_path)
        time_m = train_and_eval(FusionformerTimeOnly, full_path)

        if fam_m["auroc"] is None:
            print(f"  SKIPPED: {fam_m.get('skip_reason', 'unknown')}")
            continue

        fam_auroc = fam_m["auroc"]
        time_auroc = time_m["auroc"]
        fam_scores.append(fam_auroc)
        time_scores.append(time_auroc)

        diff = fam_auroc - time_auroc
        winner = "FAM" if diff > 0.01 else ("TimeOnly" if diff < -0.01 else "TIE")
        print(f"  FAM:      AUROC {fam_auroc:.3f}")
        print(f"  TimeOnly: AUROC {time_auroc:.3f}")
        print(f"  Diff:     {diff:+.3f}   ({winner})")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if fam_scores:
        fam_mean = np.mean(fam_scores)
        time_mean = np.mean(time_scores)
        diff = fam_mean - time_mean
        print(f"  FAM      mean AUROC:  {fam_mean:.3f}  (n={len(fam_scores)})")
        print(f"  TimeOnly mean AUROC:  {time_mean:.3f}  (n={len(time_scores)})")
        print(f"  FAM advantage over TimeOnly: {diff:+.3f}")
        print()
        if diff > 0.02:
            print("Interpretation: Channel attention DOES help on these folders.")
        elif diff > 0:
            print("Interpretation: Channel attention gives modest lift on these folders.")
        elif diff > -0.02:
            print("Interpretation: Channel attention still doesn't help (~tied).")
        else:
            print("Interpretation: Channel attention hurts performance on these folders.")


if __name__ == "__main__":
    main()
