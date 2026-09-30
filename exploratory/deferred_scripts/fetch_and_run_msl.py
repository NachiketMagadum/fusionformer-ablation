"""
Fetch MSL (Mars Science Laboratory) dataset and run FAM vs TimeOnly ablation.

MSL is the third canonical multivariate anomaly detection benchmark used
alongside SMD and SMAP in Su et al. (2019) and downstream papers
(Anomaly Transformer, TranAD, DCdetector). It comes from NASA/JPL
spacecraft telemetry -- a genuinely different domain from SKAB
(industrial pump) and SMD (server telemetry) and gives us a real
third-benchmark cross-validation of the FAM null finding.

DATA SOURCE (open, no application required):
    https://github.com/NetManAIOps/OmniAnomaly
    → ServerMachineDataset folder contains SMD, plus MSL and SMAP
       under /processed/

Alternative direct source (numpy arrays):
    https://s3-us-west-2.amazonaws.com/telemanom/data.zip
    (NASA/JPL Telemanom project; the labels file is on the Hundman
    et al. 2018 KDD paper's GitHub)

USAGE:
    1. Download data.zip (approx 300MB) from telemanom URL above
    2. Extract to datasets/telemanom/
    3. Download labeled_anomalies.csv from
       https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv
       into datasets/telemanom/
    4. python3 fetch_and_run_msl.py

OUTPUT:
    notes/msl_fam_vs_timeonly_results.txt
    figures/fig_4_7_msl_fam_vs_timeonly.png

NO SYNTHETIC DATA. Every number produced by this script comes from
a real MSL training run.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
import time
import copy

# ============================================================
# Config
# ============================================================
SEEDS = [0, 1, 42]
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
D_MODEL = 32
FF_HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.1
N_HEADS = 1  # UPDATE THIS ONCE PAPER'S NUM_HEADS IS CONFIRMED

DATA_ROOT = Path("datasets/telemanom")
FIGURES = Path("figures"); FIGURES.mkdir(exist_ok=True)
NOTES = Path("notes"); NOTES.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Load MSL from Telemanom / OmniAnomaly format
# ============================================================
def load_msl_channel(channel_id):
    """
    Telemanom / OmniAnomaly layout:
        datasets/telemanom/train/<channel>.npy       (T x n_features, all normal)
        datasets/telemanom/test/<channel>.npy        (T x n_features, mixed)
        datasets/telemanom/labeled_anomalies.csv     (chan_id, spacecraft, anomaly_sequences, num_values)

    Returns (X_train, X_test, y_test).
    """
    train_path = DATA_ROOT / "train" / f"{channel_id}.npy"
    test_path = DATA_ROOT / "test" / f"{channel_id}.npy"
    labels_csv = DATA_ROOT / "labeled_anomalies.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"MSL data not found at {train_path}. See script header.")

    X_train = np.load(train_path)
    X_test = np.load(test_path)

    labels_df = pd.read_csv(labels_csv)
    row = labels_df[labels_df["chan_id"] == channel_id]
    if len(row) == 0:
        raise ValueError(f"No labels for channel {channel_id}")
    anom_seqs = eval(row.iloc[0]["anomaly_sequences"])  # list of [start, end]
    num_values = int(row.iloc[0]["num_values"])

    y_test = np.zeros(num_values, dtype=int)
    for s, e in anom_seqs:
        y_test[s:e+1] = 1
    return X_train, X_test, y_test

def list_msl_channels():
    labels_csv = DATA_ROOT / "labeled_anomalies.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"{labels_csv} not found")
    df = pd.read_csv(labels_csv)
    msl = df[df["spacecraft"] == "MSL"]
    return msl["chan_id"].tolist()

# ============================================================
# Inline model definitions (identical to fusionformer.py / attention.py)
# ============================================================
class FusionAttention(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32, n_heads=1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(seq_len, n_heads, batch_first=True)
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.output_projection = nn.Linear(d_model, n_features)
    def forward(self, x):
        h = self.input_projection(x)
        t, _ = self.time_attention(h, h, h)
        h_t = h.transpose(1, 2)
        c, _ = self.channel_attention(h_t, h_t, h_t)
        c = c.transpose(1, 2)
        w = torch.softmax(self.fusion_weights, dim=0)
        return self.output_projection(w[0] * t + w[1] * c)

class TimeOnlyAttention(nn.Module):
    def __init__(self, seq_len, n_features, d_model=32, n_heads=1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.output_projection = nn.Linear(d_model, n_features)
    def forward(self, x):
        h = self.input_projection(x)
        t, _ = self.time_attention(h, h, h)
        return self.output_projection(t)

def make_block(attn_cls, seq_len, n_features, d_model, ff_hidden, dropout, n_heads):
    attn = attn_cls(seq_len, n_features, d_model, n_heads)
    return nn.Sequential(), attn  # placeholder; use TransformerBlock below

class TransformerBlock(nn.Module):
    def __init__(self, attn, n_features, ff_hidden, dropout):
        super().__init__()
        self.attention = attn
        self.norm1 = nn.LayerNorm(n_features)
        self.ff = nn.Sequential(nn.Linear(n_features, ff_hidden), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(ff_hidden, n_features))
        self.norm2 = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class GenericTransformerAE(nn.Module):
    def __init__(self, seq_len, n_features, attn_cls, d_model=32, ff_hidden=64,
                 n_layers=2, dropout=0.1, n_heads=1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, seq_len, n_features) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList([
            TransformerBlock(attn_cls(seq_len, n_features, d_model, n_heads),
                             n_features, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.dec = nn.ModuleList([
            TransformerBlock(attn_cls(seq_len, n_features, d_model, n_heads),
                             n_features, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
    def forward(self, x):
        h = self.drop(x + self.pos)
        for b in self.enc: h = b(h)
        for b in self.dec: h = b(h)
        return h

def make_windows(data, seq_len):
    return np.stack([data[i:i+seq_len] for i in range(len(data) - seq_len + 1)])

def broadcast(w_scores, n_rows, seq_len):
    r = np.empty(n_rows)
    r[:seq_len-1] = w_scores[0]
    r[seq_len-1:] = w_scores
    return r

# ============================================================
# Train one model, return per-window scores
# ============================================================
def train_and_score(attn_cls, X_train, X_test, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_features = X_train.shape[1]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_windows = make_windows(X_train_s, SEQ_LEN)
    test_windows = make_windows(X_test_s, SEQ_LEN)

    model = GenericTransformerAE(SEQ_LEN, n_features, attn_cls,
                                 D_MODEL, FF_HIDDEN, N_LAYERS, DROPOUT, N_HEADS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    train_t = torch.tensor(train_windows, dtype=torch.float32)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_t))
        for i in range(0, len(train_t), BATCH_SIZE):
            b = train_t[perm[i:i+BATCH_SIZE]].to(DEVICE)
            r = model(b); loss = crit(r, b)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        errors = []
        for i in range(0, len(test_windows), 256):
            chunk = torch.tensor(test_windows[i:i+256], dtype=torch.float32).to(DEVICE)
            r = model(chunk)
            e = ((r - chunk) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(e)
    return broadcast(np.concatenate(errors), len(X_test_s), SEQ_LEN)

# ============================================================
# Main sweep
# ============================================================
print("\nDiscovering MSL channels...")
channels = list_msl_channels()
print(f"Found {len(channels)} MSL channels: {channels[:5]}...")

# Use 5 channels to match the SMD protocol
selected = channels[:5]
print(f"\nRunning FAM vs TimeOnly on 5 MSL channels x 3 seeds = 30 runs...")

results = {"channel": [], "seed": [], "fam_auroc": [], "time_auroc": []}
t0 = time.time()

for ch in selected:
    print(f"\n=== Channel {ch} ===")
    X_train, X_test, y_test = load_msl_channel(ch)
    print(f"  Train shape {X_train.shape}, Test shape {X_test.shape}, "
          f"anomaly rate {y_test.mean():.3f}")

    for seed in SEEDS:
        s_fam = train_and_score(FusionAttention, X_train, X_test, seed)
        s_time = train_and_score(TimeOnlyAttention, X_train, X_test, seed)
        a_fam = roc_auc_score(y_test, s_fam)
        a_time = roc_auc_score(y_test, s_time)
        results["channel"].append(ch); results["seed"].append(seed)
        results["fam_auroc"].append(a_fam); results["time_auroc"].append(a_time)
        print(f"  seed {seed}: FAM {a_fam:.4f} | TimeOnly {a_time:.4f} | Δ {a_fam - a_time:+.4f}")

df = pd.DataFrame(results)
df.to_csv("notes/msl_per_run_results.csv", index=False)

fam_all = df["fam_auroc"].values
time_all = df["time_auroc"].values
diffs = fam_all - time_all
n_nonzero = int((diffs != 0).sum())
w_res = stats.wilcoxon(fam_all, time_all, zero_method="wilcox", alternative="two-sided")
z = abs(stats.norm.ppf(w_res.pvalue / 2))
r_eff = z / np.sqrt(n_nonzero) if n_nonzero > 0 else 0

elapsed_min = (time.time() - t0) / 60

report = f"""MSL FAM vs TimeOnly - real experimental output
==============================================================
Channels: {selected}  (n={len(selected)}), Seeds: {SEEDS}
Total paired runs: {len(fam_all)}
Elapsed: {elapsed_min:.1f} min on {DEVICE}
Model heads: {N_HEADS} (update N_HEADS at top of script if paper differs)

              FAM         TimeOnly    Difference
Mean AUROC   {fam_all.mean():.4f}     {time_all.mean():.4f}     {diffs.mean():+.4f}
Std          {fam_all.std():.4f}     {time_all.std():.4f}
FAM wins     {int((diffs > 0).sum())} / {len(diffs)}
TimeOnly wins {int((diffs < 0).sum())} / {len(diffs)}
Ties         {len(diffs) - n_nonzero} / {len(diffs)}

Wilcoxon signed-rank (paired, two-sided):
    W = {w_res.statistic:.3f}
    p = {w_res.pvalue:.4f}
    Rosenthal effect size r = {r_eff:.3f}

Verdict: {"NO significant difference (p >= 0.05)" if w_res.pvalue >= 0.05 else "SIGNIFICANT (p < 0.05)"}
==============================================================
"""
print(report)
with open("notes/msl_fam_vs_timeonly_results.txt", "w") as f:
    f.write(report)

# ============================================================
# Plot
# ============================================================
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(fam_all)); w = 0.4
labels = [f"{c}\ns{s}" for c, s in zip(df["channel"], df["seed"])]
ax.bar(x - w/2, fam_all, w, label="FAM", color="#2b6cb0")
ax.bar(x + w/2, time_all, w, label="TimeOnly", color="#dd6b20")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("AUROC")
ax.set_title(f"Figure 4.7: FAM vs TimeOnly on MSL ({len(selected)} channels x {len(SEEDS)} seeds)\n"
             f"Wilcoxon W={w_res.statistic:.1f}, p={w_res.pvalue:.3f}, r={r_eff:.3f}")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout(); plt.savefig("figures/fig_4_7_msl_fam_vs_timeonly.png"); plt.close()
print(f"Saved: figures/fig_4_7_msl_fam_vs_timeonly.png")
print("\nDONE. Paste the report above into Chapter 4 (new section 4.3.3 MSL cross-validation)")
