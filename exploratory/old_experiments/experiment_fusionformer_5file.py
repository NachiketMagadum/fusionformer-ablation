"""
5-file Fusionformer sweep on SKAB valve1.

Mirrors experiment_skab_valve1.py for direct comparability with the
LSTM-AE 5-file sweep (from commit f5ec468 which reported LSTM-AE mean
AUROC 0.87 across the same 5 files).

Trains a fresh Fusionformer per file, reports per-file metrics and
5-file mean/std.

Usage:
    python scripts/experiment_fusionformer_5file.py

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


# ---------- Config (kept identical to train_fusionformer.py) ----------
SEQ_LEN     = 30
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
SEED        = 42

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1

FILES = [f"datasets/SKAB/data/valve1/{i}.csv" for i in range(5)]


def make_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_and_eval(file_path: str) -> dict:
    """Train a fresh Fusionformer on this file, return metrics."""
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

    # Training loop (quiet — no per-epoch prints).
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

    # Score every window on the full sequence.
    model.eval()
    with torch.no_grad():
        eval_batch = all_windows_t.to(device)
        reconstruction = model(eval_batch)
        window_errors = ((reconstruction - eval_batch) ** 2).mean(
            dim=(1, 2)
        ).cpu().numpy()

    # Broadcast window scores to per-row scores.
    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    return compute_all_metrics(y.values, scores)


def main() -> None:
    print("=" * 60)
    print("5-FILE FUSIONFORMER SWEEP — SKAB valve1")
    print("=" * 60)
    print(f"Model: Fusionformer  |  {N_LAYERS} layers, d_model={D_MODEL}")
    print(f"Training: {EPOCHS} epochs, batch {BATCH_SIZE}, LR {LR}, seed {SEED}")
    print("=" * 60)

    all_auroc, all_prauc, all_f1pa, all_eventf1 = [], [], [], []

    for file_path in FILES:
        file_name = Path(file_path).name
        print(f"\nFile: {file_name}")
        metrics = train_and_eval(file_path)
        print(f"  AUROC   : {metrics['auroc']:.3f}")
        print(f"  PR-AUC  : {metrics['pr_auc']:.3f}")
        print(f"  F1-PA   : {metrics['f1_pa']:.3f}")
        print(f"  Event-F1: {metrics['event_f1']:.3f}")

        all_auroc.append(metrics['auroc'])
        all_prauc.append(metrics['pr_auc'])
        all_f1pa.append(metrics['f1_pa'])
        all_eventf1.append(metrics['event_f1'])

    print("\n" + "=" * 60)
    print("MEAN RESULTS ACROSS 5 FILES")
    print("=" * 60)
    print(f"AUROC    : {np.mean(all_auroc):.3f}  (std {np.std(all_auroc):.3f})")
    print(f"PR-AUC   : {np.mean(all_prauc):.3f}  (std {np.std(all_prauc):.3f})")
    print(f"F1-PA    : {np.mean(all_f1pa):.3f}  (std {np.std(all_f1pa):.3f})")
    print(f"Event-F1 : {np.mean(all_eventf1):.3f}  (std {np.std(all_eventf1):.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
