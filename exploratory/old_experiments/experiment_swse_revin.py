"""
Try RevIN (Reversible Instance Normalization) on SWSE variants.

RevIN normalizes each input window per-feature at input, then denormalizes
the output before returning. This is a near-universal component in modern
patch-based time-series transformers (PatchTST, PatchTrAD, PatchAD).

Adapted from ts-kim/RevIN (https://github.com/ts-kim/RevIN).

Runs on SKAB valve1 5 files at 50 epochs, seed 42. Compares:
    1. FAM alone (baseline)
    2. SWSE+FAM (post-norm, original — fails)
    3. SWSE+FAM+RevIN (post-norm SWSE wrapped in RevIN)
    4. SWSE+TimeOnly+RevIN (time-only SWSE wrapped in RevIN)

Usage:
    python scripts/experiment_swse_revin.py

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
    FusionformerSWSE, SWSE, ReconstructionHead,
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
#  RevIN module (adapted from ts-kim/RevIN)
# ============================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time-series.

    Normalizes each input window per-feature (removes per-window mean,
    divides by per-window std). Stores stats for later denormalization.
    Learnable affine parameters (gamma, beta) applied after normalization.

    Args:
        num_features: Number of features (channels) in the input.
        eps:          Small value to avoid division by zero.
        affine:       Whether to include learnable affine transform.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # Stats buffers — set during 'norm' pass, used during 'denorm'.
        self.mean = None
        self.stdev = None

    def _get_statistics(self, x: torch.Tensor):
        # x: (B, T, F). Reduce over time axis, keep feature axis.
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode}")


# ============================================================
#  SWSE + RevIN variants
# ============================================================

class FusionformerSWSE_RevIN(nn.Module):
    """SWSE + FAM with RevIN normalization at input and output."""

    def __init__(self, seq_len, n_features, segment_len=5, embed_dim=32,
                 ff_hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.revin = RevIN(num_features=n_features)
        self.inner = FusionformerSWSE(
            seq_len=seq_len, n_features=n_features,
            segment_len=segment_len, embed_dim=embed_dim,
            ff_hidden=ff_hidden, n_layers=n_layers, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x = self.revin(x, mode="norm")
        # Run inner SWSE model
        out = self.inner(x)
        # Denormalize output back to original scale
        return self.revin(out, mode="denorm")


class TimeOnlyAttentionSWSE(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )

    def forward(self, x):
        B, F, N_seg, D = x.shape
        x_time = x.reshape(B * F, N_seg, D)
        out, _ = self.time_attention(x_time, x_time, x_time)
        return out.reshape(B, F, N_seg, D)


class TransformerBlockSWSE_TimeOnly(nn.Module):
    def __init__(self, embed_dim, ff_hidden=64, dropout=0.1):
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
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class FusionformerSWSE_TimeOnly_RevIN(nn.Module):
    """SWSE + time-only attention wrapped in RevIN."""

    def __init__(self, seq_len, n_features, segment_len=5, embed_dim=32,
                 ff_hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        n_segments = seq_len // segment_len
        self.revin = RevIN(num_features=n_features)
        self.swse = SWSE(seq_len, segment_len, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_features, n_segments, embed_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlockSWSE_TimeOnly(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerBlockSWSE_TimeOnly(embed_dim, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.reconstruction_head = ReconstructionHead(embed_dim, segment_len)

    def forward(self, x):
        x = self.revin(x, mode="norm")
        h = self.swse(x)
        h = h + self.pos_encoding
        h = self.dropout(h)
        for block in self.encoder_blocks:
            h = block(h)
        for block in self.decoder_blocks:
            h = block(h)
        out = self.reconstruction_head(h)
        return self.revin(out, mode="denorm")


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
    print("REVIN TEST: does RevIN rescue SWSE variants?")
    print("=" * 78)
    print(f"5 files valve1, seed {SEED}, {EPOCHS} epochs")
    print("=" * 78)

    results = {
        "FAM alone (baseline)":       [],
        "SWSE+FAM (no RevIN)":        [],
        "SWSE+FAM + RevIN":           [],
        "SWSE+TimeOnly + RevIN":      [],
    }

    for file_path in FILES:
        file_name = Path(file_path).name
        print(f"\nFile: {file_name}")

        variants = [
            ("FAM alone (baseline)",
             lambda: Fusionformer(seq_len=SEQ_LEN, n_features=8,
                 d_model=D_MODEL, ff_hidden=FF_HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT)),
            ("SWSE+FAM (no RevIN)",
             lambda: FusionformerSWSE(seq_len=SEQ_LEN, n_features=8,
                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)),
            ("SWSE+FAM + RevIN",
             lambda: FusionformerSWSE_RevIN(seq_len=SEQ_LEN, n_features=8,
                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)),
            ("SWSE+TimeOnly + RevIN",
             lambda: FusionformerSWSE_TimeOnly_RevIN(seq_len=SEQ_LEN, n_features=8,
                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT)),
        ]

        for name, factory in variants:
            model = factory()
            auroc = train_and_eval(model, file_path, SEED)
            n_params = sum(p.numel() for p in model.parameters())
            results[name].append(auroc)
            print(f"  {name:26s} AUROC {auroc:.3f}   params {n_params:,}")

    print("\n" + "=" * 78)
    print("SUMMARY across 5 valve1 files")
    print("=" * 78)
    for name, scores in results.items():
        m, s = np.mean(scores), np.std(scores)
        print(f"  {name:26s} mean {m:.3f}   std {s:.3f}")

    print("\nDIFFERENCES:")
    fam_mean = np.mean(results["FAM alone (baseline)"])
    for name in ["SWSE+FAM (no RevIN)", "SWSE+FAM + RevIN", "SWSE+TimeOnly + RevIN"]:
        diff = np.mean(results[name]) - fam_mean
        print(f"  {name:26s} vs FAM alone: {diff:+.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
