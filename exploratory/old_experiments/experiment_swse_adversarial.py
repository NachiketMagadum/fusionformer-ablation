"""
Full adversarial training: SWSE + FAM + discriminator.

Adds the third core component from Wang et al. 2025 Fusionformer paper.
An auxiliary discriminator is trained alongside the SWSE autoencoder in
GAN-style adversarial dynamics. The autoencoder (generator) tries to
produce reconstructions the discriminator can't distinguish from real
inputs, while the discriminator learns to tell them apart.

Loss structure:
    L_D = BCE(D(x_real), 1) + BCE(D(x_recon.detach()), 0)
    L_G = L_recon + lambda_adv * BCE(D(x_recon), 1)

Compares:
    1. FAM alone (baseline, no SWSE, no adversarial)
    2. SWSE + FAM (no adversarial — fails)
    3. SWSE + FAM + Adversarial (full paper's three components)

Runs on SKAB valve1 5 files at 50 epochs, seed 42.

Usage:
    python scripts/experiment_swse_adversarial.py

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
LR_G        = 1e-3
LR_D        = 5e-4   # discriminator LR usually lower
SEED        = 42

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1
SEGMENT_LEN = 5

LAMBDA_ADV = 0.05  # weight on adversarial loss for generator (keep small)

FILES = [f"datasets/SKAB/data/valve1/{i}.csv" for i in range(5)]


# ============================================================
#  Simple discriminator
# ============================================================

class SimpleDiscriminator(nn.Module):
    """
    Small MLP discriminator over flattened windows.

    Input: (B, T, F) — either real or reconstructed
    Output: (B, 1) — logit for "is this real?"

    Kept small so it doesn't dominate the generator early in training.
    """

    def __init__(self, seq_len: int, n_features: int, hidden: int = 128):
        super().__init__()
        input_dim = seq_len * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
#  Adversarial training loop
# ============================================================

def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_adversarial(generator, discriminator, train_windows_t, device):
    """
    Train SWSE autoencoder (G) and discriminator (D) alternately.
    Standard GAN training pattern.
    """
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR_G)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR_D)
    recon_criterion = nn.MSELoss()
    adv_criterion = nn.BCEWithLogitsLoss()

    generator.to(device)
    discriminator.to(device)

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        perm = torch.randperm(len(train_windows_t))
        d_losses, g_losses, recon_losses = [], [], []

        for i in range(0, len(train_windows_t), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            real = train_windows_t[batch_idx].to(device)
            batch_size = real.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --- Train discriminator ---
            d_optimizer.zero_grad()
            real_pred = discriminator(real)
            d_loss_real = adv_criterion(real_pred, real_labels)

            with torch.no_grad():
                fake = generator(real)
            fake_pred = discriminator(fake)
            d_loss_fake = adv_criterion(fake_pred, fake_labels)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_optimizer.step()

            # --- Train generator ---
            g_optimizer.zero_grad()
            fake = generator(real)
            recon_loss = recon_criterion(fake, real)

            fake_pred_for_g = discriminator(fake)
            adv_loss = adv_criterion(fake_pred_for_g, real_labels)

            g_loss = recon_loss + LAMBDA_ADV * adv_loss
            g_loss.backward()
            g_optimizer.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            recon_losses.append(recon_loss.item())

        # Print occasionally for progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  "
                  f"D {np.mean(d_losses):.4f}  "
                  f"G {np.mean(g_losses):.4f}  "
                  f"recon {np.mean(recon_losses):.4f}")


def eval_recon(generator, all_windows_t, device):
    generator.eval()
    with torch.no_grad():
        eval_batch = all_windows_t.to(device)
        reconstruction = generator(eval_batch)
        window_errors = ((reconstruction - eval_batch) ** 2).mean(
            dim=(1, 2)
        ).cpu().numpy()
    return window_errors


def train_and_eval_standard(model, file_path, seed):
    """Standard (non-adversarial) training for baseline comparisons."""
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_G)
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

    window_errors = eval_recon(model, all_windows_t, device)
    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    return compute_all_metrics(y.values, scores)["auroc"]


def train_and_eval_adversarial(file_path, seed):
    """Adversarial training for SWSE+FAM+Discriminator."""
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

    generator = FusionformerSWSE(
        seq_len=SEQ_LEN, n_features=8,
        segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
        ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT,
    )
    discriminator = SimpleDiscriminator(SEQ_LEN, 8)

    print(f"    Adversarial training (LAMBDA_ADV={LAMBDA_ADV}, LR_G={LR_G}, LR_D={LR_D})")
    train_adversarial(generator, discriminator, train_windows_t, device)

    window_errors = eval_recon(generator, all_windows_t, device)
    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors

    return compute_all_metrics(y.values, scores)["auroc"]


def main():
    print("=" * 78)
    print("ADVERSARIAL TEST: does SWSE + FAM + Discriminator rescue SWSE?")
    print("=" * 78)
    print(f"5 files valve1, seed {SEED}, {EPOCHS} epochs")
    print("=" * 78)

    results = {
        "FAM alone (baseline)":       [],
        "SWSE+FAM (no adversarial)":  [],
        "SWSE+FAM+Adversarial":       [],
    }

    for file_path in FILES:
        file_name = Path(file_path).name
        print(f"\nFile: {file_name}")

        # 1. FAM alone
        fam_model = Fusionformer(
            seq_len=SEQ_LEN, n_features=8,
            d_model=D_MODEL, ff_hidden=FF_HIDDEN,
            n_layers=N_LAYERS, dropout=DROPOUT,
        )
        auroc = train_and_eval_standard(fam_model, file_path, SEED)
        results["FAM alone (baseline)"].append(auroc)
        print(f"  FAM alone (baseline)         AUROC {auroc:.3f}")

        # 2. SWSE+FAM (no adversarial)
        swse_model = FusionformerSWSE(
            seq_len=SEQ_LEN, n_features=8,
            segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
            ff_hidden=FF_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT,
        )
        auroc = train_and_eval_standard(swse_model, file_path, SEED)
        results["SWSE+FAM (no adversarial)"].append(auroc)
        print(f"  SWSE+FAM (no adversarial)    AUROC {auroc:.3f}")

        # 3. SWSE+FAM+Adversarial
        print(f"  SWSE+FAM+Adversarial:")
        auroc = train_and_eval_adversarial(file_path, SEED)
        results["SWSE+FAM+Adversarial"].append(auroc)
        print(f"  SWSE+FAM+Adversarial         AUROC {auroc:.3f}")

    print("\n" + "=" * 78)
    print("SUMMARY across 5 valve1 files")
    print("=" * 78)
    for name, scores in results.items():
        m, s = np.mean(scores), np.std(scores)
        print(f"  {name:30s} mean {m:.3f}   std {s:.3f}")

    print("\nDIFFERENCES vs FAM alone:")
    fam_mean = np.mean(results["FAM alone (baseline)"])
    for name in ["SWSE+FAM (no adversarial)", "SWSE+FAM+Adversarial"]:
        diff = np.mean(results[name]) - fam_mean
        print(f"  {name:30s} {diff:+.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
