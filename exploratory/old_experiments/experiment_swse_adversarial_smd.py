"""
Full 3-component test on SMD: SWSE + FAM + Adversarial training.

Tests the paper's full architecture (all three core components together)
on SMD to see whether the failure on SKAB was data-scale specific.
SMD has much more training data (~44k windows vs ~500) and richer
cross-channel signal (38 features vs 8).

Compares:
    1. FAM alone (baseline)
    2. SWSE + FAM (no adversarial)
    3. SWSE + FAM + Adversarial (full paper's three components)

Runs on 5 SMD machines at 50 epochs, seed 42.

Usage:
    python scripts/experiment_swse_adversarial_smd.py

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
from src.models.fusionformer_swse import FusionformerSWSE


SEQ_LEN     = 30
BATCH_SIZE  = 64
EPOCHS      = 50
LR_G        = 1e-3
LR_D        = 5e-4
SEED        = 42

D_MODEL   = 32
FF_HIDDEN = 64
N_LAYERS  = 2
DROPOUT   = 0.1
SEGMENT_LEN = 5

LAMBDA_ADV = 0.05

MACHINES = [
    "machine-1-1",
    "machine-1-4",
    "machine-2-1",
    "machine-2-5",
    "machine-3-1",
]


# ============================================================
#  Discriminator
# ============================================================

class SimpleDiscriminator(nn.Module):
    def __init__(self, seq_len, n_features, hidden=128):
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

    def forward(self, x):
        return self.net(x)


# ============================================================
#  Utility
# ============================================================

def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def batched_score(model, all_windows_t, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(all_windows_t), 256):
            batch = all_windows_t[i:i + 256].to(device)
            reconstruction = model(batch)
            e = ((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errs.append(e)
    return np.concatenate(errs)


# ============================================================
#  Standard training
# ============================================================

def train_standard(model, train_windows_t, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_G)
    criterion = nn.MSELoss()
    model.to(device)
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


# ============================================================
#  Adversarial training
# ============================================================

def train_adversarial(generator, discriminator, train_windows_t, device):
    g_opt = torch.optim.Adam(generator.parameters(), lr=LR_G)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=LR_D)
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
            bs = real.size(0)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # Train discriminator
            d_opt.zero_grad()
            real_pred = discriminator(real)
            d_loss_real = adv_criterion(real_pred, real_labels)
            with torch.no_grad():
                fake = generator(real)
            fake_pred = discriminator(fake)
            d_loss_fake = adv_criterion(fake_pred, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_opt.step()

            # Train generator
            g_opt.zero_grad()
            fake = generator(real)
            recon_loss = recon_criterion(fake, real)
            fake_pred_for_g = discriminator(fake)
            adv_loss = adv_criterion(fake_pred_for_g, real_labels)
            g_loss = recon_loss + LAMBDA_ADV * adv_loss
            g_loss.backward()
            g_opt.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            recon_losses.append(recon_loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  D {np.mean(d_losses):.4f}  "
                  f"G {np.mean(g_losses):.4f}  recon {np.mean(recon_losses):.4f}")


# ============================================================
#  Per-machine harness
# ============================================================

def prep_machine(machine, seed):
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
    return X_scaled, y, train_windows_t, all_windows_t, device


def score_to_metrics(window_errors, X_scaled, y):
    scores = np.empty(len(X_scaled))
    scores[:SEQ_LEN - 1] = window_errors[0]
    scores[SEQ_LEN - 1:] = window_errors
    return compute_all_metrics(y.values, scores)["auroc"]


def main():
    print("=" * 78)
    print("SMD FULL 3-COMPONENT TEST: FAM alone vs SWSE+FAM vs SWSE+FAM+Adversarial")
    print("=" * 78)
    print(f"5 SMD machines, seed {SEED}, {EPOCHS} epochs")
    print("=" * 78)

    results = {
        "FAM alone (baseline)":       [],
        "SWSE+FAM (no adversarial)":  [],
        "SWSE+FAM+Adversarial":       [],
    }

    for machine in MACHINES:
        print(f"\nMachine: {machine}")
        X_scaled, y, train_windows_t, all_windows_t, device = prep_machine(machine, SEED)
        n_features = X_scaled.shape[1]

        # 1. FAM alone
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        fam = Fusionformer(seq_len=SEQ_LEN, n_features=n_features,
                           d_model=D_MODEL, ff_hidden=FF_HIDDEN,
                           n_layers=N_LAYERS, dropout=DROPOUT)
        train_standard(fam, train_windows_t, device)
        auroc = score_to_metrics(batched_score(fam, all_windows_t, device), X_scaled, y)
        results["FAM alone (baseline)"].append(auroc)
        print(f"  FAM alone                    AUROC {auroc:.3f}")

        # 2. SWSE+FAM
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        swse = FusionformerSWSE(seq_len=SEQ_LEN, n_features=n_features,
                                 segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                                 ff_hidden=FF_HIDDEN, n_layers=N_LAYERS,
                                 dropout=DROPOUT)
        train_standard(swse, train_windows_t, device)
        auroc = score_to_metrics(batched_score(swse, all_windows_t, device), X_scaled, y)
        results["SWSE+FAM (no adversarial)"].append(auroc)
        print(f"  SWSE+FAM (no adversarial)    AUROC {auroc:.3f}")

        # 3. SWSE+FAM+Adversarial
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        gen = FusionformerSWSE(seq_len=SEQ_LEN, n_features=n_features,
                                segment_len=SEGMENT_LEN, embed_dim=D_MODEL,
                                ff_hidden=FF_HIDDEN, n_layers=N_LAYERS,
                                dropout=DROPOUT)
        disc = SimpleDiscriminator(SEQ_LEN, n_features)
        print("  SWSE+FAM+Adversarial:")
        train_adversarial(gen, disc, train_windows_t, device)
        auroc = score_to_metrics(batched_score(gen, all_windows_t, device), X_scaled, y)
        results["SWSE+FAM+Adversarial"].append(auroc)
        print(f"  SWSE+FAM+Adversarial         AUROC {auroc:.3f}")

    print("\n" + "=" * 78)
    print("SUMMARY across 5 SMD machines")
    print("=" * 78)
    for name, scores in results.items():
        m, s = np.mean(scores), np.std(scores)
        print(f"  {name:30s} mean {m:.3f}   std {s:.3f}")

    fam_mean = np.mean(results["FAM alone (baseline)"])
    print("\nDIFFERENCES vs FAM alone:")
    for name in ["SWSE+FAM (no adversarial)", "SWSE+FAM+Adversarial"]:
        diff = np.mean(results[name]) - fam_mean
        print(f"  {name:30s} {diff:+.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
