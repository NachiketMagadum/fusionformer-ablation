"""
Fusionformer at the paper's hyperparameters (d_model 256, L_seg 32, N_enc 4,
N_dec 3, batch 32, 30 epochs, lr 5e-4), used for the single sanity run on SMD
machine-1-1 in Section 3.6.1 and the MSL attempt. Same forecasting objective and
forecast-error anomaly score as train_ff_forecast_and_score_anomaly.py.

SKAB support was removed from this script: its old SKAB loader scored the
training rows (the problem described in Section 5.2). SKAB is only run through
train_ff_forecast_and_score_anomaly.py, which scores held-out rows only.

USAGE:
    python3 train_ff_paper_hparams.py --dataset smd --machine machine-1-1
    python3 train_ff_paper_hparams.py --dataset msl --channel M-1

OUTPUT:
    notes/ff_paperhp_<variant>_<dataset>_<id>_seed<seed>.txt
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from fusionformer_true import Fusionformer, FusionformerDiscriminator

# ============================================================
# Config (paper defaults)
# ============================================================
# ============================================================
# PAPER-HYPERPARAMETER SANITY RUN
# Values from Wang et al. (2025) Table III / Section IV.A
# One change: n_heads=8 instead of 6, because 256 % 6 != 0.
# The paper reports d_model=256 AND n_heads=6, which are incompatible
# without an extra projection. n_heads=8 is the closest even divisor.
# ============================================================
SEQ_LEN = 96          # historical window T (paper: 96)
PRED_LEN = 24         # forecast horizon tau (paper: variable; 24 keeps parity with main runs)
SEGMENT_LEN = 32      # L_seg (paper: 32 -> L_in = 96/32 = 3 segments)
D_MODEL = 256         # paper: 256
N_HEADS = 8           # paper says 6 but 256 is not divisible by 6; using nearest even divisor
N_ENC = 4             # paper: 4
N_DEC = 3             # paper: 3
BATCH_SIZE = 32       # paper: 32
LR_F = 5e-4           # paper LR range: {5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5} - middle value
LR_D = 2.5e-4         # discriminator LR (half of LR_F, standard practice)
EPOCHS = 30           # paper: 30
LAMBDA_ADV = 0.0      # keep off for sanity run - discriminator training separately documented
D_STEPS = 1
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
SEED = 42

DEVICE = ("mps" if torch.backends.mps.is_available()
          else ("cuda" if torch.cuda.is_available() else "cpu"))


# ============================================================
# Data loaders
# ============================================================

def load_smd_machine(machine: str):
    root = Path("datasets/SMD")
    X_train = pd.read_csv(root / "train" / f"{machine}.txt", header=None).values
    X_test = pd.read_csv(root / "test" / f"{machine}.txt", header=None).values
    y_test = pd.read_csv(root / "test_label" / f"{machine}.txt", header=None).iloc[:, 0].values
    return X_train, X_test, y_test


def make_forecast_pairs(X, seq_len, pred_len):
    """Build (input, target) forecasting pairs by sliding window.

    Returns:
        inputs: (N, seq_len, D)
        targets: (N, pred_len, D)
    """
    n = len(X) - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError(f"Series too short: len={len(X)}, need >= {seq_len + pred_len}")
    inputs = np.stack([X[i:i + seq_len] for i in range(n)])
    targets = np.stack([X[i + seq_len:i + seq_len + pred_len] for i in range(n)])
    return inputs, targets


# ============================================================
# Training loop
# ============================================================

def train_fusionformer_adversarial(model, disc, X_train, seq_len, pred_len,
                                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   verbose=True):
    """Train Fusionformer with combined prediction MSE + adversarial loss."""
    model = model.to(DEVICE)
    disc = disc.to(DEVICE)
    opt_F = torch.optim.AdamW(model.parameters(), lr=LR_F, weight_decay=1e-5)
    opt_D = torch.optim.AdamW(disc.parameters(), lr=LR_D, weight_decay=1e-5)
    mse = nn.HuberLoss(delta=1.0)   # Huber (smooth L1) is more stable than pure MSE for forecasting
    bce = nn.BCELoss()

    inputs, targets = make_forecast_pairs(X_train, seq_len, pred_len)
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    for epoch in range(epochs):
        model.train(); disc.train()
        perm = torch.randperm(len(inputs_t))
        f_losses = []; d_losses = []; pred_losses = []

        for i in range(0, len(inputs_t), batch_size):
            idx = perm[i:i + batch_size]
            x_batch = inputs_t[idx].to(DEVICE)
            y_batch = targets_t[idx].to(DEVICE)
            B = x_batch.size(0)
            ones = torch.ones(B, device=DEVICE)
            zeros = torch.zeros(B, device=DEVICE)

            # --- Update D only if adversarial is active and past warmup ---
            use_adv = LAMBDA_ADV > 0 and epoch >= WARMUP_EPOCHS
            if use_adv:
                with torch.no_grad():
                    y_pred = model(x_batch)
                y_real = torch.cat([x_batch, y_batch], dim=1)
                y_fake = torch.cat([x_batch, y_pred], dim=1)
                for _ in range(D_STEPS):
                    d_real = disc(y_real).clamp(1e-7, 1 - 1e-7)
                    d_fake = disc(y_fake.detach()).clamp(1e-7, 1 - 1e-7)
                    d_loss = bce(d_real, ones) + bce(d_fake, zeros)
                    opt_D.zero_grad()
                    d_loss.backward()
                    opt_D.step()
                    d_losses.append(d_loss.item())

            # --- Update F ---
            y_pred = model(x_batch)
            pred_loss = mse(y_pred, y_batch)
            if use_adv:
                y_fake = torch.cat([x_batch, y_pred], dim=1)
                d_fake_for_g = disc(y_fake).clamp(1e-7, 1 - 1e-7)
                adv_loss = bce(d_fake_for_g, ones)
                f_loss = pred_loss + LAMBDA_ADV * adv_loss
            else:
                f_loss = pred_loss
            opt_F.zero_grad()
            f_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt_F.step()
            f_losses.append(f_loss.item())
            pred_losses.append(pred_loss.item())

        if verbose:
            print(f"  epoch {epoch+1:2d}/{epochs}  "
                  f"pred_MSE={np.mean(pred_losses):.4f}  "
                  f"D_loss={np.mean(d_losses):.4f}  "
                  f"F_loss={np.mean(f_losses):.4f}")


# ============================================================
# Anomaly scoring via forecast error
# ============================================================

def score_forecast_error(model, X_test, seq_len, pred_len, batch_size=128):
    """
    For each valid starting position t in X_test, forecast the next pred_len
    values and compute MSE against actual. The score at position t is the
    forecast MSE for the window starting at t + seq_len (i.e., the score
    corresponds to the anomaly quality of the forecast horizon).
    """
    model.eval()
    inputs, targets = make_forecast_pairs(X_test, seq_len, pred_len)
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    errors = []
    with torch.no_grad():
        for i in range(0, len(inputs_t), batch_size):
            xb = inputs_t[i:i + batch_size].to(DEVICE)
            yb = targets_t[i:i + batch_size].to(DEVICE)
            yp = model(xb)
            e = ((yp - yb) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(e)
    return np.concatenate(errors)


def broadcast_scores_to_rows(window_errors, n_rows, seq_len, pred_len):
    """Assign each row a score based on the forecast that includes it.
    Row t gets the score of the window whose forecast horizon covers row t.
    Rows before the first valid forecast get the first window's score;
    rows after the last valid forecast get the last window's score."""
    scores = np.empty(n_rows)
    # First seq_len rows: use first window's error
    scores[:seq_len] = window_errors[0]
    # For row t in [seq_len, seq_len + len(window_errors) + pred_len - 1]:
    # the window forecasting this row is the one starting at max(0, t - seq_len - pred_len + 1)
    for t in range(seq_len, n_rows):
        window_idx = max(0, min(t - seq_len, len(window_errors) - 1))
        scores[t] = window_errors[window_idx]
    return scores


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smd", "msl"], required=True)  # used on SMD and MSL only
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--file", default=None)
    parser.add_argument("--channel", default=None, help="MSL channel id, e.g. M-1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variant", choices=["full", "no_mswea", "no_swse"],
                        default="full",
                        help="full = SWSE+MSWAA+MSWEA (paper); no_mswea = drop intervariable branch; no_swse = per-timestep tokens")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {DEVICE}, seed: {args.seed}")

    if args.dataset == "smd":
        X_train, X_test, y_test = load_smd_machine(args.machine)
        run_id = args.machine
    else:  # msl
        if args.channel is None:
            raise ValueError("--channel required for msl")
        from msl_loader import load_msl_channel
        X_train, X_test, y_test = load_msl_channel(args.channel)
        run_id = args.channel

    print(f"Dataset: {args.dataset} {run_id}")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Test anomaly rate: {y_test.mean():.3f}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    n_vars = X_train_s.shape[1]

    use_swse = args.variant != "no_swse"
    use_mswea = args.variant != "no_mswea"
    # Per-variant batch size override to prevent MPS OOM on larger variants
    variant_batch = {"full": 64, "no_mswea": 64, "no_swse": 24, "adversarial": 48}
    effective_batch = variant_batch.get(args.variant, BATCH_SIZE)
    print(f"Variant: {args.variant}  (use_swse={use_swse}, use_mswea={use_mswea})")
    print(f"Effective batch size: {effective_batch}")
    model = Fusionformer(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, n_vars=n_vars,
        segment_len=SEGMENT_LEN, d_model=D_MODEL, n_heads=N_HEADS,
        n_enc=N_ENC, n_dec=N_DEC, dropout=0.1,
        use_swse=use_swse, use_mswea=use_mswea,
    )
    disc = FusionformerDiscriminator(SEQ_LEN, PRED_LEN, n_vars)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_d = sum(p.numel() for p in disc.parameters())
    print(f"Fusionformer params: {n_params:,}")
    print(f"Discriminator params: {n_params_d:,}")

    print(f"\nTraining {EPOCHS} epochs with paper-matched hyperparameters:")
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, L_seg={SEGMENT_LEN}, "
          f"T={SEQ_LEN}, tau={PRED_LEN}")
    t0 = time.time()
    train_fusionformer_adversarial(model, disc, X_train_s, SEQ_LEN, PRED_LEN, batch_size=effective_batch)
    train_min = (time.time() - t0) / 60
    print(f"Training took {train_min:.1f} min")

    print("\nScoring test set via forecast error...")
    window_errors = score_forecast_error(model, X_test_s, SEQ_LEN, PRED_LEN)
    scores = broadcast_scores_to_rows(window_errors, len(X_test_s), SEQ_LEN, PRED_LEN)
    auroc = roc_auc_score(y_test, scores)
    p, r, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(r, p)

    report = f"""FAITHFUL FUSIONFORMER RESULTS (per Wang et al. 2025)
==============================================================
Dataset: {args.dataset} {run_id}
Model: Fusionformer forecasting (T={SEQ_LEN}, tau={PRED_LEN}), seed={args.seed}
Hyperparameters (paper-matched): d_model={D_MODEL}, n_heads={N_HEADS}, L_seg={SEGMENT_LEN}, N_enc={N_ENC}, N_dec={N_DEC}
Fusionformer params: {n_params:,}, Discriminator params: {n_params_d:,}
Training: {EPOCHS} epochs, batch {BATCH_SIZE}, lr_F={LR_F}, lr_D={LR_D}, adv weight={LAMBDA_ADV}
Wall-clock: {train_min:.1f} min on {DEVICE}

Test AUROC: {auroc:.4f}
Test PR-AUC: {pr_auc:.4f}
Test anomaly rate: {y_test.mean():.3f}
==============================================================
"""
    print("\n" + report)
    out = Path("notes") / f"ff_paperhp_{args.variant}_{args.dataset}_{run_id}_seed{args.seed}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
