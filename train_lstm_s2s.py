"""
Stronger recurrent baseline (v3): sequence-to-sequence LSTM with early stopping.

Why: the original LSTM baseline (train_lstm_baseline.py; last hidden state ->
linear head, 15 epochs) never forecast better than the training mean on the
simulated series (validation MSE 0.87-1.61 for every tuned configuration), so
it was not a fair baseline. This script gives the recurrent model a fairer
chance:
  - encoder-decoder: a 2-layer LSTM encoder reads the 96-step history; a 2-layer
    LSTM decoder, started from the encoder state, rolls out the 24-step forecast
    one step at a time, feeding back its own previous prediction (the first
    input is the last observed row), so training and test behave the same;
  - up to 100 epochs with early stopping (patience 10) on a validation split
    taken from the end of the training rows, keeping the best weights;
  - anomaly scoring uses exactly the same code path as Fusionformer
    (broadcast_scores_to_rows and the same eval_start cut), so the AUROC values
    are directly comparable. (The original LSTM script aligned scores to the
    end of the forecast horizon instead, 23 rows later; see notes/v3/design_v3.md.)

Validation split: the last max(20% of training rows, SEQ_LEN + PRED_LEN + 50)
rows; the model is fitted on the rows before it. No test row is used for
tuning or early stopping.

Usage:
    python3 train_lstm_s2s.py --dataset synth --synth coupled_100 --seed 0 --hidden 128 --lr 0.001
    python3 train_lstm_s2s.py --dataset skab --file datasets/SKAB/data/valve1/0.csv --seed 0 --val_only ...
    --val_only: tuning mode, writes only the validation MSE (no test scoring)

Author: Nachiket Magadum
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from train_ff_forecast_and_score_anomaly import (load_smd_machine, load_skab_file, make_forecast_pairs,
                                                 broadcast_scores_to_rows, SEQ_LEN, PRED_LEN, DEVICE)

NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0


class LSTMSeq2Seq(nn.Module):
    def __init__(self, d_in, hidden, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.LSTM(d_in, hidden, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(d_in, hidden, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, d_in)

    def forward(self, x):                       # x: (B, T, D)
        _, state = self.encoder(x)
        step = x[:, -1:, :]                     # last observed row
        outs = []
        for _ in range(self.pred_len):
            h, state = self.decoder(step, state)
            step = self.head(h)                 # (B, 1, D)
            outs.append(step)
        return torch.cat(outs, dim=1)           # (B, pred_len, D)


def split_val(X_train):
    n_val = max(int(0.2 * len(X_train)), SEQ_LEN + PRED_LEN + 50)
    return X_train[:-n_val], X_train[-n_val:]


def window_errors(model, X, batch_size=256):
    model.eval()
    inputs, targets = make_forecast_pairs(X, SEQ_LEN, PRED_LEN)
    sq, ae = [], []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            xb = torch.tensor(inputs[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(targets[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            p = model(xb)
            sq.append(((p - yb) ** 2).mean(dim=(1, 2)).cpu().numpy())
            ae.append((p - yb).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(sq), np.concatenate(ae)


def fit(model, X_fit, X_val, lr):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.HuberLoss(delta=1.0)
    xi, yi = make_forecast_pairs(X_fit, SEQ_LEN, PRED_LEN)
    xi = torch.tensor(xi, dtype=torch.float32); yi = torch.tensor(yi, dtype=torch.float32)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xi)); tot = 0.0; nb = 0
        for s in range(0, len(xi), BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            loss = loss_fn(model(xi[b].to(DEVICE)), yi[b].to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); tot += loss.item(); nb += 1
        val = float(window_errors(model, X_val)[0].mean())
        print(f"  epoch {epoch:3d}  train={tot / nb:.4f}  val_MSE={val:.4f}", flush=True)
        if val < best - 1e-6:
            best, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:          # None only if validation MSE was never finite
        model.load_state_dict(best_state)
    return best, best_epoch, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["smd", "skab", "synth"], required=True)
    ap.add_argument("--machine", default="machine-1-1")
    ap.add_argument("--file", default=None)
    ap.add_argument("--synth", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_only", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--out_dir", default="notes/v3")
    a = ap.parse_args()
    global MAX_EPOCHS
    if a.max_epochs is not None:
        MAX_EPOCHS = a.max_epochs
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    eval_start = 0
    if a.dataset == "smd":
        X_train, X_test, y_test = load_smd_machine(a.machine); run_id = a.machine
    elif a.dataset == "skab":
        X_train, X_test, y_test, eval_start = load_skab_file(a.file); run_id = Path(a.file).stem
    else:
        z = np.load(Path("datasets/SYNTH") / f"{a.synth}.npz")
        X_train, X_test, y_test = z["X_train"], z["X_test"], z["y_test"]; run_id = a.synth

    X_fit, X_val = split_val(X_train)
    scaler = StandardScaler().fit(X_fit)
    X_fit_s, X_val_s, X_test_s = (scaler.transform(v) for v in (X_fit, X_val, X_test))
    model = LSTMSeq2Seq(X_train.shape[1], a.hidden, NUM_LAYERS, PRED_LEN, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device {DEVICE}  {a.dataset} {run_id}  seed {a.seed}  hidden {a.hidden}  lr {a.lr:g}  "
          f"fit rows {len(X_fit)}  val rows {len(X_val)}  params {n_params:,}", flush=True)
    t0 = time.time()
    val_mse, best_epoch, last_epoch = fit(model, X_fit_s, X_val_s, a.lr)
    wall = (time.time() - t0) / 60
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if a.val_only:
        rep = (f"LSTM SEQ2SEQ TUNING (validation only)\nDataset: {a.dataset} {run_id}\nseed={a.seed} hidden={a.hidden} "
               f"lr={a.lr:g}\nBest epoch: {best_epoch} of {last_epoch}\nValidation forecast MSE: {val_mse:.6f}\n")
        f = out_dir / f"s2s_val_{a.dataset}_{run_id}_h{a.hidden}_lr{a.lr:g}_seed{a.seed}.txt"
        f.write_text(rep); print(rep); return

    w_mse, w_mae = window_errors(model, X_test_s)
    scores = broadcast_scores_to_rows(w_mse, len(X_test_s), SEQ_LEN, PRED_LEN)[eval_start:]
    y_eval = np.asarray(y_test)[eval_start:]
    first_win = max(0, eval_start - SEQ_LEN)
    auroc = roc_auc_score(y_eval, scores)
    p, r, _ = precision_recall_curve(y_eval, scores)
    rep = f"""LSTM SEQ2SEQ BASELINE (v3)
==============================================================
Dataset: {a.dataset} {run_id}
Model: LSTM encoder-decoder (T={SEQ_LEN}, tau={PRED_LEN}), seed={a.seed}
Hyperparameters: hidden={a.hidden}, layers={NUM_LAYERS}, lr={a.lr:g}, batch={BATCH_SIZE}
Params: {n_params:,}
Training: early stopping, best epoch {best_epoch} of {last_epoch} (max {MAX_EPOCHS}, patience {PATIENCE})
Validation forecast MSE: {val_mse:.6f}
Wall-clock: {wall:.1f} min on {DEVICE}

Test forecast MSE: {float(w_mse[first_win:].mean()):.6f}
Test forecast MAE: {float(w_mae[first_win:].mean()):.6f}
Test AUROC: {auroc:.4f}
Test PR-AUC: {auc(r, p):.4f}
Test anomaly rate: {y_eval.mean():.3f}
Scored rows: {len(y_eval)} of {len(y_test)} (evaluation starts at row {eval_start})
==============================================================
"""
    f = out_dir / f"s2s_{a.dataset}_{run_id}_seed{a.seed}.txt"
    f.write_text(rep); print(rep)


if __name__ == "__main__":
    main()
