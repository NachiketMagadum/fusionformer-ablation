"""
Multi-seed SMD variance sweep: FAM vs TimeOnly across 5 machines, 3 seeds.

Solidifies the SMD ablation finding by adding proper error bars across
random initialisations. 30 training runs total.

Same 5 machines and hyperparameters as the single-seed SMD ablation.

Usage:
    python scripts/experiment_variance_smd.py

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.data.smd_loader import load_smd_file
from src.eval.metrics import compute_all_metrics
from src.models.fusionformer import Fusionformer


SEQ_LEN     = 30
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
SEEDS       = [0, 1, 42]

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1

MACHINES = [
    "machine-1-1",
    "machine-1-4",
    "machine-2-1",
    "machine-2-5",
    "machine-3-1",
]


# ============================================================
#  Ablated TimeOnly variant (same as SKAB ablation)
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


def train_and_eval(model_cls, machine, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = load_smd_file(machine)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_arr = y.values
    first_anom = int(np.argmax(y_arr == 1)) if y_arr.any() else len(y_arr)

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
    window_errors_list = []
    with torch.no_grad():
        for i in range(0, len(all_windows_t), 256):
            batch = all_windows_t[i:i + 256].to(device)
            reconstruction = model(batch)
            errors = ((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            window_errors_list.append(errors)
    window_errors = np.concatenate(window_errors_list)

    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    return compute_all_metrics(y.values, scores)["auroc"]


def main():
    print("=" * 78)
    print("SMD MULTI-SEED VARIANCE SWEEP: FAM vs TimeOnly")
    print("=" * 78)
    print(f"Seeds: {SEEDS}   Epochs: {EPOCHS}   Machines: {len(MACHINES)}")
    print(f"Total runs: {len(SEEDS)} x {len(MACHINES)} x 2 = {len(SEEDS) * len(MACHINES) * 2}")
    print("=" * 78)

    # {machine: {"FAM": {seed: auroc}, "TimeOnly": {seed: auroc}}}
    results = {}

    for machine in MACHINES:
        results[machine] = {"FAM": {}, "TimeOnly": {}}
        print(f"\n{machine}")
        for seed in SEEDS:
            fam_auroc = train_and_eval(Fusionformer, machine, seed)
            time_auroc = train_and_eval(FusionformerTimeOnly, machine, seed)
            results[machine]["FAM"][seed] = fam_auroc
            results[machine]["TimeOnly"][seed] = time_auroc
            print(f"  seed {seed}:  FAM {fam_auroc:.3f}   TimeOnly {time_auroc:.3f}   diff {fam_auroc - time_auroc:+.3f}")

    # ------------------------------------------------------------
    # Per-machine summary
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-MACHINE SUMMARY (mean +/- std across 3 seeds)")
    print("=" * 78)
    print(f"{'Machine':<15} {'FAM mean':>10} {'FAM std':>10} {'TO mean':>10} {'TO std':>10} {'Δ':>8}")
    print("-" * 78)

    fam_means, time_means = [], []
    for machine, data in results.items():
        fam_vals = list(data["FAM"].values())
        time_vals = list(data["TimeOnly"].values())
        fm, fs = np.mean(fam_vals), np.std(fam_vals)
        tm, ts = np.mean(time_vals), np.std(time_vals)
        diff = fm - tm
        fam_means.append(fm)
        time_means.append(tm)
        print(f"{machine:<15} {fm:>10.3f} {fs:>10.3f} {tm:>10.3f} {ts:>10.3f} {diff:>+8.3f}")

    # ------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OVERALL SUMMARY (across 5 machines, 3 seeds each)")
    print("=" * 78)
    fam_grand = np.mean(fam_means)
    time_grand = np.mean(time_means)
    fam_grand_std = np.std(fam_means)
    time_grand_std = np.std(time_means)
    print(f"  FAM      grand mean AUROC: {fam_grand:.4f}   (across-machine std {fam_grand_std:.3f})")
    print(f"  TimeOnly grand mean AUROC: {time_grand:.4f}   (across-machine std {time_grand_std:.3f})")
    print(f"  FAM advantage:             {fam_grand - time_grand:+.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
