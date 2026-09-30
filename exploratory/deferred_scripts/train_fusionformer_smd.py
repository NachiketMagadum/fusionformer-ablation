"""
Train Fusionformer on an SMD (Server Machine Dataset) machine.

Mirrors train_fusionformer.py pattern for direct comparability with the
SKAB results. Same architecture, same hyperparameters, same eval harness.

Differences from SKAB pipeline:
    - 38 features instead of 8 (server metrics)
    - Much longer sequences (~57k rows vs ~1k in SKAB valve1)
    - Semi-supervised split is explicit: SMD provides train (all normal) and
      test (with anomalies) separately, concatenated by the loader.

Usage:
    python scripts/train_fusionformer_smd.py

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


# ---------- Config (matched to SKAB training for comparability) ----------
SEQ_LEN     = 30
BATCH_SIZE  = 64          # slightly bigger — more training data than SKAB
EPOCHS      = 50
LR          = 1e-3
SEED        = 42
MACHINE     = "machine-1-1"

# Fusionformer hyperparameters
D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1


def make_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    """(N, F) → (N - seq_len + 1, seq_len, F) sliding windows."""
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------
    # 1. Load and standardize
    # ------------------------------------------------------------
    print(f"Loading SMD machine {MACHINE}...")
    X, y = load_smd_file(MACHINE)
    print(f"  {len(X)} rows, {X.shape[1]} features, anomaly rate {y.mean() * 100:.1f}%")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------
    # 2. Sliding windows.
    #    TRAIN only on initial contiguous NORMAL region (semi-supervised).
    #    SCORE on the full sequence.
    # ------------------------------------------------------------
    y_arr = y.values
    if y_arr.any():
        first_anom = int(np.argmax(y_arr == 1))
    else:
        first_anom = len(y_arr)

    X_train_scaled = X_scaled[:first_anom]
    print(f"  Training region: rows [0, {first_anom})  ({first_anom} normal rows)")

    train_windows = make_windows(X_train_scaled, SEQ_LEN)
    train_windows_t = torch.tensor(train_windows, dtype=torch.float32)

    all_windows = make_windows(X_scaled, SEQ_LEN)
    all_windows_t = torch.tensor(all_windows, dtype=torch.float32)

    print(f"  Train windows: {len(train_windows_t)}   Score windows: {len(all_windows_t)}")

    # ------------------------------------------------------------
    # 3. Device
    # ------------------------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Using device: {device}")

    # ------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------
    model = Fusionformer(
        seq_len=SEQ_LEN,
        n_features=X.shape[1],
        d_model=D_MODEL,
        ff_hidden=FF_HIDDEN,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # ------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------
    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_windows_t))
        epoch_losses = []
        for i in range(0, len(train_windows_t), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            batch = train_windows_t[batch_idx].to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  train_loss = {np.mean(epoch_losses):.4f}")

    # ------------------------------------------------------------
    # 6. Score every window (batched to avoid OOM on large SMD data)
    # ------------------------------------------------------------
    print("\nScoring...")
    model.eval()
    window_errors_list = []
    with torch.no_grad():
        for i in range(0, len(all_windows_t), 256):
            batch = all_windows_t[i:i + 256].to(device)
            reconstruction = model(batch)
            errors = ((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            window_errors_list.append(errors)
    window_errors = np.concatenate(window_errors_list)

    # ------------------------------------------------------------
    # 7. Broadcast window scores to per-row scores
    # ------------------------------------------------------------
    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    # ------------------------------------------------------------
    # 8. Evaluate
    # ------------------------------------------------------------
    metrics = compute_all_metrics(y.values, scores)

    print("\n" + "=" * 55)
    print(f"RESULTS — Fusionformer on SMD {MACHINE}")
    print("=" * 55)
    print(f"AUROC    (rank quality)                : {metrics['auroc']:.3f}")
    print(f"PR-AUC   (precision-recall AUC)        : {metrics['pr_auc']:.3f}")
    print(f"F1-PA    (point-adjusted, INFLATED)    : {metrics['f1_pa']:.3f}")
    print(f"Event-F1 (per-window, HONEST)          : {metrics['event_f1']:.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
