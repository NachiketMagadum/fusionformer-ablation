"""
Fusionformer sweep across multiple SMD machines.

Runs Fusionformer FAM on 5 diverse SMD machines to establish a proper
baseline mean AUROC comparable to the SKAB 5-file sweep.

Machines chosen for diversity across the 3 SMD groups:
    - machine-1-1, machine-1-4  (group 1)
    - machine-2-1, machine-2-5  (group 2)
    - machine-3-1                (group 3)

Usage:
    python scripts/experiment_fusionformer_smd_sweep.py

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


def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_and_eval(machine, seed):
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

    return compute_all_metrics(y.values, scores)


def main():
    print("=" * 70)
    print("FUSIONFORMER FAM SWEEP — SMD 5 machines")
    print("=" * 70)
    print(f"Training: {EPOCHS} epochs, batch {BATCH_SIZE}, LR {LR}, seed {SEED}")
    print("=" * 70)

    all_auroc, all_prauc, all_f1pa, all_eventf1 = [], [], [], []

    for machine in MACHINES:
        print(f"\nMachine: {machine}")
        m = train_and_eval(machine, SEED)
        print(f"  AUROC   : {m['auroc']:.3f}")
        print(f"  PR-AUC  : {m['pr_auc']:.3f}")
        print(f"  F1-PA   : {m['f1_pa']:.3f}")
        print(f"  Event-F1: {m['event_f1']:.3f}")

        all_auroc.append(m["auroc"])
        all_prauc.append(m["pr_auc"])
        all_f1pa.append(m["f1_pa"])
        all_eventf1.append(m["event_f1"])

    print("\n" + "=" * 70)
    print("MEAN RESULTS ACROSS 5 SMD MACHINES")
    print("=" * 70)
    print(f"AUROC    : {np.mean(all_auroc):.3f}  (std {np.std(all_auroc):.3f})")
    print(f"PR-AUC   : {np.mean(all_prauc):.3f}  (std {np.std(all_prauc):.3f})")
    print(f"F1-PA    : {np.mean(all_f1pa):.3f}  (std {np.std(all_f1pa):.3f})")
    print(f"Event-F1 : {np.mean(all_eventf1):.3f}  (std {np.std(all_eventf1):.3f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
