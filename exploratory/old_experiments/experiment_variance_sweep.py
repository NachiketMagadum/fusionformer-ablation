"""
Multi-seed variance sweep: FAM vs TimeOnly across all 23 SKAB files.

Runs both models on all 23 files at 3 seeds each, giving proper error
bars on the FAM ablation finding. This tests whether the +0.005 FAM
advantage over TimeOnly is stable across random initialisations, or
whether it's within seed noise.

Total: 23 files x 2 models x 3 seeds = 138 training runs.
Estimated wall time: ~30 min on M5 with MPS.

Usage:
    python scripts/experiment_variance_sweep.py

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


# ---------- Config ----------
SEQ_LEN     = 30
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SEEDS       = [0, 1, 42]

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1

# All 23 SKAB files
FILES = (
    [(f"valve1/{i}.csv", f"valve1-{i}") for i in range(5)] +
    [(f"valve2/{i}.csv", f"valve2-{i}") for i in range(4)] +
    [(f"other/{i}.csv",  f"other-{i}")  for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
)

DATA_ROOT = "datasets/SKAB/data"


# ============================================================
#  Ablated modules (same as prior ablation script)
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


def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_and_eval(model_cls, file_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = load_skab_file(file_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_arr = y.values
    first_anom = int(np.argmax(y_arr == 1)) if y_arr.any() else len(y_arr)

    if first_anom < SEQ_LEN + 5:
        return None

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

    return compute_all_metrics(y.values, scores)["auroc"]


def main():
    print("=" * 78)
    print("MULTI-SEED VARIANCE SWEEP: FAM vs TimeOnly across all 23 SKAB files")
    print("=" * 78)
    print(f"Seeds: {SEEDS}   Epochs: {EPOCHS}")
    print("=" * 78)

    # Results dict: {label: {"FAM": {seed: auroc}, "TimeOnly": {seed: auroc}}}
    results = {}

    for rel_path, label in FILES:
        full_path = f"{DATA_ROOT}/{rel_path}"
        results[label] = {"FAM": {}, "TimeOnly": {}}
        print(f"\n{label}")
        for seed in SEEDS:
            fam_auroc = train_and_eval(Fusionformer, full_path, seed)
            time_auroc = train_and_eval(FusionformerTimeOnly, full_path, seed)
            if fam_auroc is None or time_auroc is None:
                print(f"  seed {seed}: SKIPPED (insufficient normal data)")
                continue
            results[label]["FAM"][seed] = fam_auroc
            results[label]["TimeOnly"][seed] = time_auroc
            print(f"  seed {seed}:  FAM {fam_auroc:.3f}   TimeOnly {time_auroc:.3f}   diff {fam_auroc - time_auroc:+.3f}")

    # ============================================================
    #  Per-file summary
    # ============================================================
    print("\n" + "=" * 78)
    print("PER-FILE SUMMARY (mean +/- std across seeds)")
    print("=" * 78)
    print(f"{'File':<15} {'FAM mean':>10} {'FAM std':>10} {'TO mean':>10} {'TO std':>10} {'Δ':>8}")
    print("-" * 78)

    fam_means, time_means = [], []
    for label, data in results.items():
        fam_vals = list(data["FAM"].values())
        time_vals = list(data["TimeOnly"].values())
        if not fam_vals:
            continue
        fm, fs = np.mean(fam_vals), np.std(fam_vals)
        tm, ts = np.mean(time_vals), np.std(time_vals)
        diff = fm - tm
        fam_means.append(fm)
        time_means.append(tm)
        print(f"{label:<15} {fm:>10.3f} {fs:>10.3f} {tm:>10.3f} {ts:>10.3f} {diff:>+8.3f}")

    # ============================================================
    #  Overall summary
    # ============================================================
    print("\n" + "=" * 78)
    print("OVERALL SUMMARY (across all 23 files, 3 seeds)")
    print("=" * 78)
    fam_grand = np.mean(fam_means)
    time_grand = np.mean(time_means)
    fam_grand_std = np.std(fam_means)
    time_grand_std = np.std(time_means)
    print(f"  FAM      grand mean AUROC: {fam_grand:.4f}   (across-file std {fam_grand_std:.3f})")
    print(f"  TimeOnly grand mean AUROC: {time_grand:.4f}   (across-file std {time_grand_std:.3f})")
    print(f"  FAM advantage:             {fam_grand - time_grand:+.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
