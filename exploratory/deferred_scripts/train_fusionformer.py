"""
Train a Fusionformer autoencoder on a SKAB file.

Semi-supervised anomaly detection using the Fusionformer architecture:
fit the model to reconstruct normal windows, then use per-window
reconstruction error as anomaly score.

Uses identical data pipeline, hyperparameters, and evaluation harness
as train_lstm_ae.py to make the comparison scientifically honest.

Usage:
    python scripts/train_fusionformer.py

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
# Kept identical to train_lstm_ae.py for direct comparability.
SEQ_LEN     = 30
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
SEED        = 42
SKAB_PATH   = "datasets/SKAB/data/valve1/0.csv"

# Fusionformer-specific hyperparameters.
D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1


def make_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Turn (N, F) into (N - seq_len + 1, seq_len, F) sliding windows."""
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # -------------------------------------------------------------
    # 1. Load and standardize
    # -------------------------------------------------------------
    print(f"Loading {SKAB_PATH}...")
    X, y = load_skab_file(SKAB_PATH)
    print(f"  {len(X)} rows, {X.shape[1]} features, anomaly rate {y.mean():.1%}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------------------
    # 2. Make sliding windows.
    #    TRAIN only on the initial contiguous NORMAL region so the
    #    model never sees anomalies during training (semi-supervised).
    #    SCORE on the full sequence.
    # -------------------------------------------------------------
    y_arr = y.values
    if y_arr.any():
        first_anom = int(np.argmax(y_arr == 1))
    else:
        first_anom = len(y_arr)

    X_train_scaled = X_scaled[:first_anom]
    print(f"  Training region: rows [0, {first_anom})  "
          f"({first_anom} normal rows)")

    train_windows = make_windows(X_train_scaled, SEQ_LEN)
    train_windows_t = torch.tensor(train_windows, dtype=torch.float32)

    # Full dataset windows for scoring
    all_windows = make_windows(X_scaled, SEQ_LEN)
    all_windows_t = torch.tensor(all_windows, dtype=torch.float32)

    print(f"  Train windows: {len(train_windows_t)}   "
          f"Score windows: {len(all_windows_t)}")

    # -------------------------------------------------------------
    # 3. Device (Apple Silicon MPS if available)
    # -------------------------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Using device: {device}")

    # -------------------------------------------------------------
    # 4. Build model, optimizer, loss
    # -------------------------------------------------------------
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
    print(f"  d_model={D_MODEL}  ff_hidden={FF_HIDDEN}  "
          f"n_layers={N_LAYERS}  dropout={DROPOUT}")

    # -------------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------------
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  "
                  f"train_loss = {np.mean(epoch_losses):.4f}")

    # -------------------------------------------------------------
    # 6. Score every window
    # -------------------------------------------------------------
    print("\nScoring...")
    model.eval()
    with torch.no_grad():
        eval_batch = all_windows_t.to(device)
        reconstruction = model(eval_batch)
        # Per-window mean squared error across seq_len and features.
        window_errors = ((reconstruction - eval_batch) ** 2).mean(
            dim=(1, 2)
        ).cpu().numpy()

    # -------------------------------------------------------------
    # 7. Broadcast window scores to per-row scores.
    #    Convention: row t gets the score of the window ENDING at row t.
    #    First (SEQ_LEN - 1) rows use the first window's score (padded).
    # -------------------------------------------------------------
    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    # -------------------------------------------------------------
    # 8. Evaluate with the shared harness
    # -------------------------------------------------------------
    metrics = compute_all_metrics(y.values, scores)

    print("\n" + "=" * 55)
    print("RESULTS — Fusionformer")
    print("=" * 55)
    print(f"AUROC    (rank quality)                : {metrics['auroc']:.3f}")
    print(f"PR-AUC   (precision-recall AUC)        : {metrics['pr_auc']:.3f}")
    print(f"F1-PA    (point-adjusted, INFLATED)    : {metrics['f1_pa']:.3f}")
    print(f"Event-F1 (per-window, HONEST)          : {metrics['event_f1']:.3f}")
    print(f"  k used: {metrics['k_used']}   threshold: {metrics['threshold']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
