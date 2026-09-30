"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
np.random.seed(0)

# ---------- 1. Generate synthetic multivariate time-series ----------

def generate_normal(n=2000, seq_len=100, n_channels=3):
    """Normal = three sine waves with small noise."""
    t = np.linspace(0, 4 * np.pi, seq_len)
    data = []
    for _ in range(n):
        signal = np.stack([
            np.sin(t + np.random.rand()),
            np.sin(2 * t + np.random.rand()) * 0.7,
            np.cos(t + np.random.rand()) * 0.8,
        ], axis=1)
        signal += np.random.randn(*signal.shape) * 0.05
        data.append(signal)
    return np.stack(data)


def generate_anomalous(n=500, seq_len=100, n_channels=3):
    """Anomaly = normal signal with a random spike injected."""
    data = generate_normal(n, seq_len, n_channels)
    for i in range(n):
        spike_pos = np.random.randint(20, 80)
        spike_channel = np.random.randint(0, n_channels)
        data[i, spike_pos:spike_pos + 3, spike_channel] += np.random.choice([-3, 3])
    return data


print("Generating synthetic MTS data...")
train_data = generate_normal(2000)                # for training (normal only)
test_normal = generate_normal(500)                # for eval (normal)
test_anomaly = generate_anomalous(500)            # for eval (anomalous)
print(f"  train_normal: {train_data.shape}")
print(f"  test_normal:  {test_normal.shape}")
print(f"  test_anomaly: {test_anomaly.shape}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"  device: {device}\n")

train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

# ---------- 2. Define autoencoder architecture ----------

class AutoEncoder(nn.Module):
    """Simple MLP autoencoder.  seq_len * n_channels flattened."""
    def __init__(self, seq_len=100, n_channels=3, hidden=128, latent=32):
        super().__init__()
        input_dim = seq_len * n_channels
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )
        self.input_shape = (seq_len, n_channels)

    def forward(self, x):
        batch = x.shape[0]
        flat = x.view(batch, -1)
        z = self.encoder(flat)
        out = self.decoder(z)
        return out.view(batch, *self.input_shape), z


TEACHER_HIDDEN = 256
TEACHER_LATENT = 64
STUDENT_HIDDEN = 32       # small student
STUDENT_LATENT = 8


def train_model(model, name, epochs=20, teacher=None, alpha_feat=0.3, alpha_score=0.3):
    """Generic training loop.  If teacher is provided, do distillation."""
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        losses = []
        for (x,) in train_loader:
            optim.zero_grad()
            recon, z = model(x)
            loss_recon = F.mse_loss(recon, x)
            loss = loss_recon

            if teacher is not None:
                with torch.no_grad():
                    t_recon, t_z = teacher(x)
                # Feature-level distillation (only if student latent size == teacher latent)
                # Instead we project student latent to teacher size, or just skip if dims differ
                # For simplicity, distill on the RECON output (same shape) as a proxy
                loss_score = F.mse_loss(recon, t_recon)
                loss = (1 - alpha_score) * loss_recon + alpha_score * loss_score

            loss.backward()
            optim.step()
            losses.append(loss.item())
        if (epoch + 1) % 5 == 0:
            print(f"  [{name}] epoch {epoch+1}/{epochs}  loss={np.mean(losses):.5f}")
    return model


def evaluate(model, name):
    """Compute reconstruction error on test set, then AUROC for anomaly detection."""
    model.eval()
    with torch.no_grad():
        x_n = torch.tensor(test_normal, dtype=torch.float32).to(device)
        x_a = torch.tensor(test_anomaly, dtype=torch.float32).to(device)
        r_n, _ = model(x_n)
        r_a, _ = model(x_a)
        err_n = ((r_n - x_n) ** 2).mean(dim=(1, 2)).cpu().numpy()
        err_a = ((r_a - x_a) ** 2).mean(dim=(1, 2)).cpu().numpy()
    y_true = np.concatenate([np.zeros_like(err_n), np.ones_like(err_a)])
    y_score = np.concatenate([err_n, err_a])
    auroc = roc_auc_score(y_true, y_score)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] params={n_params:,}  normal_err={err_n.mean():.4f}  "
          f"anomaly_err={err_a.mean():.4f}  AUROC={auroc:.3f}\n")
    return auroc, n_params


# ---------- 3. Train teacher ----------

print("Training TEACHER (big autoencoder) ...")
teacher = AutoEncoder(hidden=TEACHER_HIDDEN, latent=TEACHER_LATENT).to(device)
train_model(teacher, "Teacher", epochs=20)
teacher.eval()

# ---------- 4. Train STUDENT from scratch ----------

print("\nTraining STUDENT FROM SCRATCH (small autoencoder, no teacher) ...")
student_scratch = AutoEncoder(hidden=STUDENT_HIDDEN, latent=STUDENT_LATENT).to(device)
train_model(student_scratch, "StudentScratch", epochs=20)

# ---------- 5. Train STUDENT with distillation from teacher ----------

print("\nTraining STUDENT WITH DISTILLATION (small autoencoder, guided by teacher) ...")
student_distilled = AutoEncoder(hidden=STUDENT_HIDDEN, latent=STUDENT_LATENT).to(device)
train_model(student_distilled, "StudentDistilled", epochs=20, teacher=teacher)

# ---------- 6. Evaluate all three ----------

print("\n" + "=" * 65)
print(f"{'Model':<25} params      normal_err  anomaly_err  AUROC")
print("=" * 65)

teacher_auroc, teacher_params = evaluate(teacher, "Teacher (big)")
scratch_auroc, scratch_params = evaluate(student_scratch, "Student from scratch")
distilled_auroc, distilled_params = evaluate(student_distilled, "Student distilled")

# ---------- 7. Summary ----------

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Teacher                    -> AUROC {teacher_auroc:.3f}    ({teacher_params:,} params)")
print(f"Student from scratch       -> AUROC {scratch_auroc:.3f}    ({scratch_params:,} params)")
print(f"Student with distillation  -> AUROC {distilled_auroc:.3f}    ({distilled_params:,} params)")

recovery = (distilled_auroc - scratch_auroc) / max(teacher_auroc - scratch_auroc, 1e-6)
size_ratio = teacher_params / max(student_params_for_ratio := distilled_params, 1)
print(f"\nDistilled student is {size_ratio:.0f}x smaller than teacher.")
print(f"Distillation closed {recovery*100:.1f}% of the accuracy gap between scratch and teacher.")
print("\nThat gain, at that size, is what distillation buys you.")
