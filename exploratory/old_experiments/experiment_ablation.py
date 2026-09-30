"""
Ablation: Fusionformer FAM (dual-axis) vs time-only-attention variant.

Tests whether the fusion mechanism (dual-axis time + channel attention)
is actually what gives Fusionformer its edge over LSTM-AE, or whether
a standard time-only transformer at the same size would achieve
similar results.

Runs both variants on all 5 SKAB valve1 files, reports per-file and
mean AUROC comparison.

If FAM significantly beats TimeOnly: fusion mechanism is doing real work.
If they perform similarly: the improvement over LSTM-AE was mostly
architectural (positional encoding, feed-forward, layer norm), not
the channel attention specifically.

Usage:
    python scripts/experiment_ablation.py

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
from src.models.blocks import TransformerBlock


# ---------- Config (matched to main Fusionformer training) ----------
SEQ_LEN     = 30
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SEED        = 42

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1

FILES = [f"datasets/SKAB/data/valve1/{i}.csv" for i in range(5)]


# ============================================================
#  Ablated modules (time-only attention, no channel fusion)
# ============================================================

class TimeOnlyAttention(nn.Module):
    """
    Ablated attention with only the time-axis branch.

    Uses the SAME input projection, time-attention layer, and output
    projection as FusionAttention, but drops the channel-attention
    branch and the fusion weighting. This isolates the contribution of
    the channel-axis attention mechanism.
    """

    def __init__(self, seq_len: int, n_features: int, d_model: int = 32):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        time_out, _ = self.time_attention(h, h, h)
        return self.output_projection(time_out)


class TimeOnlyBlock(nn.Module):
    """Transformer block using TimeOnlyAttention instead of FusionAttention."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 32,
        ff_hidden: int = 64,
        dropout: float = 0.1,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class FusionformerTimeOnly(nn.Module):
    """
    Ablated Fusionformer with time-only attention throughout.

    Everything else (positional encoding, encoder-decoder structure,
    residual/norm pattern) is identical to Fusionformer. Only the
    attention mechanism differs.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 32,
        ff_hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, n_features) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([
            TimeOnlyBlock(seq_len, n_features, d_model, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            TimeOnlyBlock(seq_len, n_features, d_model, ff_hidden, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encoding
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        for block in self.decoder_blocks:
            x = block(x)
        return x


# ============================================================
#  Training and evaluation harness
# ============================================================

def make_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_and_eval(model_cls, file_path: str) -> dict:
    """Train fresh model on this file, return metrics."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    n_params = sum(p.numel() for p in model.parameters())
    metrics = compute_all_metrics(y.values, scores)
    metrics["n_params"] = n_params
    return metrics


# ============================================================
#  Main experiment
# ============================================================

def main() -> None:
    print("=" * 70)
    print("ABLATION: Fusionformer FAM (dual-axis) vs TimeOnly (single-axis)")
    print("=" * 70)
    print(f"Training: {EPOCHS} epochs, batch {BATCH_SIZE}, LR {LR}, seed {SEED}")
    print("=" * 70)

    results = {"Fusionformer (FAM)": [], "Fusionformer (TimeOnly)": []}
    params_recorded = {"Fusionformer (FAM)": None, "Fusionformer (TimeOnly)": None}

    for file_path in FILES:
        file_name = Path(file_path).name
        print(f"\nFile: {file_name}")

        for name, model_cls in [
            ("Fusionformer (FAM)", Fusionformer),
            ("Fusionformer (TimeOnly)", FusionformerTimeOnly),
        ]:
            metrics = train_and_eval(model_cls, file_path)
            params_recorded[name] = metrics["n_params"]
            results[name].append(metrics["auroc"])
            print(f"  {name:28s}  AUROC={metrics['auroc']:.3f}  "
                  f"params={metrics['n_params']:,}")

    print("\n" + "=" * 70)
    print("MEAN AUROC across 5 files")
    print("=" * 70)
    for name, auroc_list in results.items():
        mean_auroc = np.mean(auroc_list)
        std_auroc = np.std(auroc_list)
        n_params = params_recorded[name]
        print(f"  {name:28s}  mean {mean_auroc:.3f}  "
              f"std {std_auroc:.3f}  params {n_params:,}")

    diff = np.mean(results["Fusionformer (FAM)"]) - np.mean(results["Fusionformer (TimeOnly)"])
    print("=" * 70)
    print(f"FAM advantage over TimeOnly: {diff:+.3f} mean AUROC")
    print("=" * 70)
    if diff > 0.02:
        print("Interpretation: Channel-axis attention contributes meaningfully.")
    elif diff > 0:
        print("Interpretation: Channel-axis attention gives modest lift.")
    else:
        print("Interpretation: Channel-axis attention does not help on this dataset.")


if __name__ == "__main__":
    main()
