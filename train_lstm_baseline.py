"""
LSTM forecasting baseline for the Fusionformer ablation study.

Same forecasting objective and anomaly-scoring pipeline as
train_ff_forecast_and_score_anomaly.py, so the AUROC / PR-AUC numbers
are directly comparable to my Fusionformer results.

Architecture: 2-layer LSTM encoder + linear head to (pred_len * D)
followed by reshape. No attention. No segment tokens. No adversarial.
Just a straight LSTM forecast + MSE-of-forecast anomaly score.

This is the non-transformer baseline referenced in Chapter 4 for
head-to-head comparison.

Usage:
    python3 train_lstm_baseline.py --dataset smd --machine machine-1-1 --seed 0
    python3 train_lstm_baseline.py --dataset skab --file datasets/SKAB/data/valve1/0.csv --seed 0
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ============================================================
# Config
# ============================================================
SEQ_LEN = 96
PRED_LEN = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
LR = 1.5e-4
EPOCHS = 15
GRAD_CLIP = 1.0
SEED = 42

DEVICE = ("mps" if torch.backends.mps.is_available()
          else ("cuda" if torch.cuda.is_available() else "cpu"))


# ============================================================
# Model
# ============================================================
class LSTMForecast(nn.Module):
    def __init__(self, d_in, hidden_size, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.d_in = d_in
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, pred_len * d_in)

    def forward(self, x):
        # x: (B, T, D)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, hidden). take top layer
        h_top = h_n[-1]                             # (B, hidden)
        y = self.head(h_top)                        # (B, pred_len * D)
        return y.reshape(-1, self.pred_len, self.d_in)


# ============================================================
# Data loaders (mirrored from Fusionformer training script)
# ============================================================
def load_smd_machine(machine: str):
    root = Path("datasets/SMD")
    X_train = pd.read_csv(root / "train" / f"{machine}.txt", header=None).values
    X_test = pd.read_csv(root / "test" / f"{machine}.txt", header=None).values
    y_test = pd.read_csv(root / "test_label" / f"{machine}.txt", header=None).iloc[:, 0].values
    return X_train, X_test, y_test


def load_skab_file(path: str, train_frac: float = 1.0):
    df = pd.read_csv(path, sep=';', parse_dates=['datetime'], index_col='datetime')
    y = df['anomaly'].astype(int).values
    feature_cols = [c for c in df.columns if c not in ('anomaly', 'changepoint')]
    X = df[feature_cols].values
    # Train on the rows before the first anomaly; only rows whose forecast
    # targets lie entirely after this cut are scored (see main).
    first_anom = int(np.argmax(y == 1)) if y.any() else len(y)
    # train_frac < 1 keeps the tail of the normal period out of training, so the
    # scored rows also include normal rows from the same regime as training.
    cut = int(first_anom * train_frac)
    return X[:cut], X, y, cut


def make_forecast_pairs(X, seq_len, pred_len):
    n = len(X) - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError(f"Series too short: len={len(X)}, need >= {seq_len + pred_len}")
    inputs = np.stack([X[i:i + seq_len] for i in range(n)])
    targets = np.stack([X[i + seq_len:i + seq_len + pred_len] for i in range(n)])
    return inputs, targets


# ============================================================
# Training
# ============================================================
def train(model, X_train, seq_len, pred_len, epochs, batch_size):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.HuberLoss(delta=1.0)

    inputs, targets = make_forecast_pairs(X_train, seq_len, pred_len)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    n = len(inputs)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            xb = inputs[b].to(DEVICE)
            yb = targets[b].to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch:2d}/{epochs}  loss={epoch_loss / n_batches:.4f}", flush=True)


# ============================================================
# Scoring
# ============================================================
def score_forecast_error_and_metrics(model, X, seq_len, pred_len, batch_size=128, eval_start=0):
    """Per-row anomaly scores plus test MSE and MAE over windows whose targets
    start at or after eval_start."""
    model.eval()
    inputs, targets = make_forecast_pairs(X, seq_len, pred_len)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    all_sq = []; all_ae = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            xb = inputs[start:start + batch_size].to(DEVICE)
            yb = targets[start:start + batch_size].to(DEVICE)
            pred = model(xb)
            all_sq.append(((pred - yb) ** 2).mean(dim=(1, 2)).cpu().numpy())
            all_ae.append((pred - yb).abs().mean(dim=(1, 2)).cpu().numpy())
    scores = np.concatenate(all_sq)
    first_win = max(0, eval_start - seq_len)
    total_mse = float(np.mean(scores[first_win:]))
    total_mae = float(np.mean(np.concatenate(all_ae)[first_win:]))
    padded = np.zeros(len(X))
    padded[seq_len + pred_len - 1 : seq_len + pred_len - 1 + len(scores)] = scores
    return padded, total_mse, total_mae


def score_forecast_error(model, X, seq_len, pred_len, batch_size=128):
    model.eval()
    inputs, targets = make_forecast_pairs(X, seq_len, pred_len)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    scores = np.zeros(len(inputs))
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            xb = inputs[start:start + batch_size].to(DEVICE)
            yb = targets[start:start + batch_size].to(DEVICE)
            pred = model(xb)
            err = ((pred - yb) ** 2).mean(dim=(1, 2)).cpu().numpy()
            scores[start:start + len(err)] = err
    # Align scores to timesteps (score at end of forecast horizon)
    padded = np.zeros(len(X))
    padded[seq_len + pred_len - 1 : seq_len + pred_len - 1 + len(scores)] = scores
    return padded


def compute_metrics(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    valid = scores > 0
    y_true_v = y_true[valid]
    scores_v = scores[valid]
    auroc = roc_auc_score(y_true_v, scores_v)
    precision, recall, _ = precision_recall_curve(y_true_v, scores_v)
    pr_auc = auc(recall, precision)
    return auroc, pr_auc


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smd", "skab", "msl", "synth"], required=True)
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--file", default=None)
    parser.add_argument("--channel", default=None, help="MSL channel id, e.g. M-1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skab_train_frac", type=float, default=1.0,
                        help="fraction of the pre-anomaly rows used for training (SKAB only)")
    parser.add_argument("--out_dir", default="notes")
    parser.add_argument("--synth", default=None, help="simulated series name, e.g. coupled_0")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {DEVICE}, seed: {args.seed}", flush=True)

    eval_start = 0   # SMD and MSL have separate train and test files
    if args.dataset == "smd":
        X_train, X_test, y_test = load_smd_machine(args.machine)
        run_id = args.machine
    elif args.dataset == "skab":
        if args.file is None:
            raise ValueError("--file required for skab")
        X_train, X_test, y_test, eval_start = load_skab_file(args.file, args.skab_train_frac)
        run_id = Path(args.file).stem
    elif args.dataset == "synth":
        # simulated benchmark (synth_generator.py); separate train and test like SMD
        z = np.load(Path("datasets/SYNTH") / f"{args.synth}.npz")
        X_train, X_test, y_test = z["X_train"], z["X_test"], z["y_test"]
        run_id = args.synth
    else:  # msl
        if args.channel is None:
            raise ValueError("--channel required for msl")
        from msl_loader import load_msl_channel
        X_train, X_test, y_test = load_msl_channel(args.channel)
        run_id = args.channel

    print(f"Dataset: {args.dataset} {run_id}", flush=True)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}", flush=True)
    print(f"  Test anomaly rate: {np.mean(y_test):.3f}", flush=True)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    d = X_train.shape[1]
    model = LSTMForecast(d_in=d, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                         pred_len=PRED_LEN, dropout=DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM params: {n_params:,}", flush=True)
    print(f"\nTraining {EPOCHS} epochs, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, batch={BATCH_SIZE}, lr={LR}", flush=True)

    t0 = time.time()
    train(model, X_train_s, SEQ_LEN, PRED_LEN, EPOCHS, BATCH_SIZE)
    wall = (time.time() - t0) / 60.0
    print(f"Wall-clock training: {wall:.1f} min", flush=True)

    scores, TEST_MSE, TEST_MAE = score_forecast_error_and_metrics(model, X_test_s, SEQ_LEN, PRED_LEN,
                                                                 eval_start=eval_start)
    # Score at index t covers targets t-PRED_LEN+1 .. t, so keep t >= eval_start + PRED_LEN - 1
    cut = eval_start + PRED_LEN - 1 if eval_start > 0 else 0
    y_eval = np.asarray(y_test)[cut:]
    auroc, pr_auc = compute_metrics(y_eval, scores[cut:])

    report = f"""LSTM FORECASTING BASELINE
==============================================================
Dataset: {args.dataset} {run_id}
Model: LSTM forecasting (T={SEQ_LEN}, tau={PRED_LEN}), seed={args.seed}
Hyperparameters: hidden={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, dropout={DROPOUT}
LSTM params: {n_params:,}
Training: {EPOCHS} epochs, batch {BATCH_SIZE}, lr={LR}
Wall-clock: {wall:.1f} min on {DEVICE}

Test forecast MSE: {TEST_MSE:.6f}
Test forecast MAE: {TEST_MAE:.6f}
Test AUROC: {auroc:.4f}
Test PR-AUC: {pr_auc:.4f}
Test anomaly rate: {np.mean(y_eval):.3f}
Scored rows: {len(y_eval)} of {len(y_test)} (evaluation starts at row {cut})
=============================================================="""
    print("\n" + report, flush=True)
    out = Path(args.out_dir) / f"lstm_baseline_{args.dataset}_{run_id}_seed{args.seed}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
