"""
SMD ablation: Fusionformer FAM (dual-axis) vs TimeOnly (single-axis).

Critical test of the "channel attention helps when cross-channel patterns
exist" hypothesis. On SKAB (temporal-dominant valve anomalies) FAM added
no benefit over TimeOnly. On SMD (server telemetry with genuinely
cross-channel fault signatures) we expect FAM to actually help.

Same 5 machines as the FAM sweep, same seed, same config, only the
attention mechanism differs.

If FAM > TimeOnly by more than +0.02 AUROC: hypothesis supported.
If FAM ≈ TimeOnly: hypothesis wrong; FAM lift is transformer architecture
generally, not the fusion mechanism specifically.

Usage:
    python scripts/experiment_ablation_smd.py

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
from src.models.blocks import TransformerBlock


SEQ_LEN     = 30
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
SEED        = 42

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
#  Ablated model — TimeOnly attention (drops channel branch)
# ============================================================

class TimeOnlyAttention(nn.Module):
    """Ablated attention: only time-axis, no channel-axis branch."""

    def __init__(self, seq_len: int, n_features: int, d_model: int = 32):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        time_out, _ = self.time_attention(h, h, h)
        return self.output_projection(time_out)


class TimeOnlyBlock(nn.Module):
    """Transformer block using TimeOnlyAttention."""

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
    """Ablated Fusionformer: time-only attention, no channel fusion."""

    def __init__(self, seq_len, n_features, d_model=32, ff_hidden=64, n_layers=2, dropout=0.1):
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

    def forward(self, x):
        x = x + self.pos_encoding
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        for block in self.decoder_blocks:
            x = block(x)
        return x


# ============================================================
#  Training and eval
# ============================================================

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

    n_params = sum(p.numel() for p in model.parameters())
    result = compute_all_metrics(y.values, scores)
    result["n_params"] = n_params
    return result


def main():
    print("=" * 70)
    print("SMD ABLATION: Fusionformer FAM (dual-axis) vs TimeOnly (single-axis)")
    print("=" * 70)
    print(f"Training: {EPOCHS} epochs, batch {BATCH_SIZE}, LR {LR}, seed {SEED}")
    print("=" * 70)

    results = {"FAM": [], "TimeOnly": []}
    params = {"FAM": None, "TimeOnly": None}

    for machine in MACHINES:
        print(f"\nMachine: {machine}")

        for name, cls in [("FAM", Fusionformer), ("TimeOnly", FusionformerTimeOnly)]:
            m = train_and_eval(cls, machine, SEED)
            params[name] = m["n_params"]
            results[name].append(m["auroc"])
            print(f"  {name:10s} AUROC {m['auroc']:.3f}   params {m['n_params']:,}")

        diff = results["FAM"][-1] - results["TimeOnly"][-1]
        winner = "FAM" if diff > 0.01 else ("TimeOnly" if diff < -0.01 else "TIE")
        print(f"  Diff:      {diff:+.3f}   ({winner})")

    print("\n" + "=" * 70)
    print("MEAN across 5 SMD machines")
    print("=" * 70)
    for name in ["FAM", "TimeOnly"]:
        m, s = np.mean(results[name]), np.std(results[name])
        print(f"  {name:10s} mean {m:.3f}   std {s:.3f}   params {params[name]:,}")

    diff = np.mean(results["FAM"]) - np.mean(results["TimeOnly"])
    print("=" * 70)
    print(f"FAM advantage over TimeOnly: {diff:+.3f}")
    print("=" * 70)
    if diff > 0.02:
        print("Interpretation: FAM channel attention CONTRIBUTES on SMD.")
        print("Hypothesis SUPPORTED: fusion attention helps when cross-channel signal exists.")
    elif diff > 0:
        print("Interpretation: FAM gives modest lift.")
    else:
        print("Interpretation: FAM channel attention does not help on SMD either.")
        print("Hypothesis WEAKER: transformer architecture is what helps, not the fusion mechanism.")


if __name__ == "__main__":
    main()
