"""
Quick single-seed comparison: Fusionformer (FAM only) vs FusionformerSWSE.

Runs on SKAB valve1 5 files at 50 epochs, seed 42.
Total: 10 training runs, ~2-3 min on M5.

Purpose: sanity check that SWSE trains well and produces reasonable AUROC
before scaling to full multi-seed 23-file comparison.

Usage:
    python scripts/experiment_swse_quicktest.py

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
from src.models.fusionformer_swse import FusionformerSWSE


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
    print("QUICK COMPARISON: Fusionformer (FAM) vs FusionformerSWSE")
    print("=" * 78)
    print(f"5 files valve1, seed {SEED}, {EPOCHS} epochs")
    print("=" * 78)

    fam_scores = []
    swse_scores = []

    for file_path in FILES:
        file_name = Path(file_path).name
        print(f"\nFile: {file_name}")

        # Reload model for each file to avoid parameter leakage.
        fam_model = Fusionformer(
            seq_len=SEQ_LEN, n_features=8,
            d_model=D_MODEL, ff_hidden=FF_HIDDEN,
            n_layers=N_LAYERS, dropout=DROPOUT,
        )
        swse_model = FusionformerSWSE(
            seq_len=SEQ_LEN, n_features=8,
            segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
            ff_hidden=FF_HIDDEN, n_layers=N_LAYERS,
            dropout=DROPOUT,
        )

        fam_auroc = train_and_eval(fam_model, file_path, SEED)
        swse_auroc = train_and_eval(swse_model, file_path, SEED)

        fam_scores.append(fam_auroc)
        swse_scores.append(swse_auroc)

        print(f"  FAM only:  AUROC {fam_auroc:.3f}   params 38,744")
        print(f"  FAM+SWSE:  AUROC {swse_auroc:.3f}   params 52,973")
        print(f"  Diff:      {swse_auroc - fam_auroc:+.3f}")

    print("\n" + "=" * 78)
    print("SUMMARY across 5 valve1 files")
    print("=" * 78)
    print(f"  FAM only  mean AUROC: {np.mean(fam_scores):.3f}   (std {np.std(fam_scores):.3f})")
    print(f"  FAM+SWSE  mean AUROC: {np.mean(swse_scores):.3f}   (std {np.std(swse_scores):.3f})")
    diff = np.mean(swse_scores) - np.mean(fam_scores)
    print(f"  SWSE advantage over FAM alone: {diff:+.3f}")
    print("=" * 78)
    if diff > 0.02:
        print("Interpretation: SWSE contributes meaningfully.")
    elif diff > 0:
        print("Interpretation: SWSE gives modest lift.")
    elif diff > -0.02:
        print("Interpretation: SWSE doesn't help much (~tied).")
    else:
        print("Interpretation: SWSE hurts performance.")


if __name__ == "__main__":
    main()
