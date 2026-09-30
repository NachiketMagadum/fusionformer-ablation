"""
Try pre-norm on SWSE variants to see if stability rescues the finding.

Pre-norm (LayerNorm BEFORE attention/FFN) is the modern transformer standard
and typically much more stable than post-norm. If SWSE was hurt by training
instability from post-norm, pre-norm should recover it.

Runs on SKAB valve1 5 files at 50 epochs, seed 42.
Compares:
    1. Fusionformer (FAM only, post-norm — existing baseline)
    2. FusionformerSWSE (FAM + SWSE, post-norm — original)
    3. FusionformerSWSE_PreNorm (FAM + SWSE, pre-norm — new)
    4. FusionformerSWSE_TimeOnly_PreNorm (SWSE + time attn only, pre-norm)

Usage:
    python scripts/experiment_swse_prenorm.py

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
from src.models.fusionformer_swse import (
    FusionformerSWSE, SWSE, FusionAttentionSWSE, ReconstructionHead,
)


SEQ_LEN     = 30
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SEED        = 42

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1
SEGMENT_LEN = 5

FILES = [f"datasets/SKAB/data/valve1/{i}.csv" for i in range(5)]


# ============================================================
#  Pre-norm SWSE transformer block
# ============================================================

class TransformerBlockSWSE_PreNorm(nn.Module):
    """
    Pre-norm variant: LayerNorm BEFORE attention and FFN.

    Modern standard for transformer stability. Difference from post-norm:
        Post-norm: y = LayerNorm(x + sublayer(x))
        Pre-norm:  y = x + sublayer(LayerNorm(x))
    """

    def __init__(self, embed_dim: int, ff_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attention = FusionAttentionSWSE(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention.
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        # Pre-norm FFN.
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class TimeOnlyAttentionSWSE(nn.Module):
    """SWSE time-only attention (no channel branch)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )

    def forward(self, x):
        B, F, N_seg, D = x.shape
        x_time = x.reshape(B * F, N_seg, D)
        out, _ = self.time_attention(x_time, x_time, x_time)
        return out.reshape(B, F, N_seg, D)


class TransformerBlockSWSE_TimeOnly_PreNorm(nn.Module):
    """Pre-norm block using TimeOnlyAttentionSWSE."""

    def __init__(self, embed_dim: int, ff_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attention = TimeOnlyAttentionSWSE(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


# ============================================================
#  Full pre-norm SWSE models
# ============================================================

class FusionformerSWSE_PreNorm(nn.Module):
    """SWSE + FAM with pre-norm transformer blocks."""

    def __init__(self, seq_len, n_features, segment_len=5, embed_dim=32,
                 ff_hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        n_segments = seq_len // segment_len
        self.swse = SWSE(seq_len, segment_len, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_features, n_segments, embed_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlockSWSE_PreNorm(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerBlockSWSE_PreNorm(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.reconstruction_head = ReconstructionHead(embed_dim, segment_len)

    def forward(self, x):
        h = self.swse(x)
        h = h + self.pos_encoding
        h = self.dropout(h)
        for block in self.encoder_blocks:
            h = block(h)
        for block in self.decoder_blocks:
            h = block(h)
        return self.reconstruction_head(h)


class FusionformerSWSE_TimeOnly_PreNorm(nn.Module):
    """SWSE + time-only attention with pre-norm."""

    def __init__(self, seq_len, n_features, segment_len=5, embed_dim=32,
                 ff_hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        n_segments = seq_len // segment_len
        self.swse = SWSE(seq_len, segment_len, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_features, n_segments, embed_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlockSWSE_TimeOnly_PreNorm(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerBlockSWSE_TimeOnly_PreNorm(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.reconstruction_head = ReconstructionHead(embed_dim, segment_len)

    def forward(self, x):
        h = self.swse(x)
        h = h + self.pos_encoding
        h = self.dropout(h)
        for block in self.encoder_blocks:
            h = block(h)
        for block in self.decoder_blocks:
            h = block(h)
        return self.reconstruction_head(h)


# ============================================================
#  Training + eval
# ============================================================

def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_and_eval(model, file_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = load_skab_file(file_path)
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
    model = model.to(device)
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
    print("PRE-NORM TEST: does pre-norm rescue SWSE variants?")
    print("=" * 78)
    print(f"5 files valve1, seed {SEED}, {EPOCHS} epochs")
    print("=" * 78)

    results = {
        "FAM alone (baseline)":          [],
        "SWSE+FAM (post-norm)":          [],
        "SWSE+FAM (PRE-NORM)":           [],
        "SWSE+TimeOnly (PRE-NORM)":      [],
    }

    for file_path in FILES:
        file_name = Path(file_path).name
        print(f"\nFile: {file_name}")

        variants = [
            ("FAM alone (baseline)",
             lambda: Fusionformer(seq_len=SEQ_LEN, n_features=8,
                 d_model=D_MODEL, ff_hidden=FF_HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT)),
            ("SWSE+FAM (post-norm)",
             lambda: FusionformerSWSE(seq_len=SEQ_LEN, n_features=8,
                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)),
            ("SWSE+FAM (PRE-NORM)",
             lambda: FusionformerSWSE_PreNorm(seq_len=SEQ_LEN, n_features=8,
                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)),
            ("SWSE+TimeOnly (PRE-NORM)",
             lambda: FusionformerSWSE_TimeOnly_PreNorm(seq_len=SEQ_LEN, n_features=8,
                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)),
        ]

        for name, factory in variants:
            model = factory()
            auroc = train_and_eval(model, file_path, SEED)
            n_params = sum(p.numel() for p in model.parameters())
            results[name].append(auroc)
            print(f"  {name:28s} AUROC {auroc:.3f}   params {n_params:,}")

    print("\n" + "=" * 78)
    print("SUMMARY across 5 valve1 files")
    print("=" * 78)
    for name, scores in results.items():
        m, s = np.mean(scores), np.std(scores)
        print(f"  {name:28s} mean {m:.3f}   std {s:.3f}")

    print("\nDIFFERENCES:")
    fam_mean = np.mean(results["FAM alone (baseline)"])
    for name in ["SWSE+FAM (post-norm)", "SWSE+FAM (PRE-NORM)", "SWSE+TimeOnly (PRE-NORM)"]:
        diff = np.mean(results[name]) - fam_mean
        print(f"  {name:28s} vs FAM alone: {diff:+.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
