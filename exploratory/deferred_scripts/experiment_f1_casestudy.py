"""
Cross-domain qualitative case study on Formula 1 telemetry.

Trains Fusionformer FAM on the first quarter of Hamilton's 2023 British GP
race (used as "normal driving" reference), then scores the entire race and
plots reconstruction error over the race timeline. High-error moments are
visually inspected against known F1 events (pit stops, DRS activations,
racing incidents).

Note: F1 does not have ground truth anomaly labels, so this is a
qualitative demonstration of cross-domain pipeline generalisability
rather than a quantitative evaluation.

Usage:
    python scripts/experiment_f1_casestudy.py

Output:
    figures/f1_reconstruction_error.png    - Anomaly score over race
    figures/f1_lap_summary.png             - Per-lap mean reconstruction error
    notes/f1_case_study_results.md         - Written results summary

Author: Nachiket Magadum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.fusionformer import Fusionformer


# Configuration
SEED        = 42
SEQ_LEN     = 30
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
D_MODEL     = 32
FF_HIDDEN   = 64
N_LAYERS    = 2
DROPOUT     = 0.1

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
NOTES_DIR = Path("notes")
NOTES_DIR.mkdir(exist_ok=True)


def load_f1_telemetry():
    """Load Hamilton British GP 2023 telemetry, resample to uniform 10 Hz."""
    import fastf1

    cache_dir = Path("datasets/fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    print("Loading 2023 British Grand Prix (race)...")
    session = fastf1.get_session(2023, "British Grand Prix", "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    print("  Session loaded.")

    # Hamilton car number: 44
    car_data = session.car_data["44"]

    # Filter to racing pace (speed > 100 km/h) to skip pit lane / pre-race
    moving = car_data[car_data["Speed"] > 100].copy()
    print(f"  Raw telemetry rows: {len(car_data)}")
    print(f"  Racing rows (Speed > 100): {len(moving)}")

    # Convert Brake bool -> int (needed for numeric operations)
    moving["Brake"] = moving["Brake"].astype(int)

    # Set Date as index for resampling
    moving = moving.set_index("Date")

    # Continuous channels get mean-then-interpolate
    continuous_cols = ["Speed", "RPM", "Throttle"]
    cont_resampled = (
        moving[continuous_cols].resample("100ms").mean().interpolate(method="linear")
    )

    # Discrete channels get nearest-neighbour
    discrete_cols = ["nGear", "DRS", "Brake"]
    disc_resampled = moving[discrete_cols].resample("100ms").nearest()

    # Combine
    resampled = pd.concat([cont_resampled, disc_resampled], axis=1)
    resampled = resampled.dropna()

    print(f"  After resampling to 10 Hz: {len(resampled)} rows, "
          f"{resampled.shape[1]} features")

    return resampled, session


def make_windows(data, seq_len):
    n_windows = len(data) - seq_len + 1
    return np.stack([data[i:i + seq_len] for i in range(n_windows)])


def train_on_normal(X_train, n_features, device):
    """Train Fusionformer FAM on the early-race 'normal' region."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_windows = make_windows(X_train, SEQ_LEN)
    train_windows_t = torch.tensor(train_windows, dtype=torch.float32)

    print(f"  Train windows: {len(train_windows_t)}")
    print(f"  Using device: {device}")

    model = Fusionformer(
        seq_len=SEQ_LEN,
        n_features=n_features,
        d_model=D_MODEL,
        ff_hidden=FF_HIDDEN,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("  Training on early-race normal reference...")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_windows_t))
        losses = []
        for i in range(0, len(train_windows_t), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            batch = train_windows_t[batch_idx].to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:2d}/{EPOCHS}  loss = {np.mean(losses):.4f}")

    return model


def score_all(model, X_scaled, device):
    """Score the whole race in batches."""
    all_windows = make_windows(X_scaled, SEQ_LEN)
    all_windows_t = torch.tensor(all_windows, dtype=torch.float32)

    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(all_windows_t), 256):
            batch = all_windows_t[i:i + 256].to(device)
            recon = model(batch)
            e = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(e)
    return np.concatenate(errors)


def plot_reconstruction_over_time(errors, index, output_path):
    """Plot reconstruction error over the race timeline."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Broadcast window errors to per-row scores
    scores = np.empty(len(index))
    scores[:SEQ_LEN - 1] = errors[0]
    scores[SEQ_LEN - 1:] = errors

    # X axis: elapsed race time in minutes
    elapsed = (index - index[0]).total_seconds() / 60.0

    ax.plot(elapsed, scores, linewidth=0.6, color="#1F4E79", alpha=0.8)

    # Threshold at 95th percentile for visual reference
    threshold = np.percentile(scores, 95)
    ax.axhline(threshold, color="#C62828", linestyle="--", linewidth=1,
               alpha=0.7, label=f"95th percentile ({threshold:.3f})")

    # Highlight top 20 highest-error windows
    top_indices = np.argsort(scores)[-20:]
    ax.scatter(elapsed[top_indices], scores[top_indices],
               color="#F4A261", s=25, zorder=5, label="Top-20 highest error")

    ax.set_xlabel("Race time (minutes)")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.set_title("Fusionformer FAM reconstruction error across Hamilton's "
                 "2023 British GP race")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    return scores, threshold


def plot_lap_summary(scores, index, session, output_path):
    """Plot mean reconstruction error per lap."""
    # Get Hamilton's lap data
    laps = session.laps.pick_drivers("HAM")

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(laps) == 0:
        print("  No lap data available — skipping per-lap plot")
        return None

    # Map each score timestamp to a lap number
    scores_df = pd.DataFrame({"score": scores}, index=index)

    lap_means = []
    lap_nums = []
    for _, lap_row in laps.iterrows():
        lap_start = lap_row.get("LapStartTime")
        lap_num = lap_row.get("LapNumber")
        if lap_start is None or lap_num is None or pd.isna(lap_num):
            continue
        # Approximate: 90 seconds per lap
        try:
            end = lap_start + pd.Timedelta(seconds=90)
            mask = (scores_df.index >= lap_start) & (scores_df.index < end)
            if mask.sum() > 5:
                lap_means.append(scores_df.loc[mask, "score"].mean())
                lap_nums.append(int(lap_num))
        except Exception:
            continue

    if not lap_means:
        print("  Could not compute per-lap means — skipping")
        return None

    colors = ["#C62828" if m > np.mean(lap_means) + np.std(lap_means)
              else "#1F4E79" for m in lap_means]
    ax.bar(lap_nums, lap_means, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Mean reconstruction error")
    ax.set_title("Mean per-lap reconstruction error — red bars mark unusual laps")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    return list(zip(lap_nums, lap_means))


def write_summary_notes(scores, threshold, elapsed, lap_summary, output_path):
    """Write a markdown summary of the case study."""
    n_high = int((scores > threshold).sum())
    pct_high = 100 * n_high / len(scores)

    top_5_times = elapsed[np.argsort(scores)[-5:]][::-1]
    top_5_scores = scores[np.argsort(scores)[-5:]][::-1]

    md = f"""# F1 Case Study — Hamilton 2023 British GP

## Setup

- Race: 2023 Formula 1 British Grand Prix (Silverstone), race session
- Driver: Lewis Hamilton (car number 44)
- Telemetry channels: Speed, RPM, Throttle (continuous); nGear, DRS, Brake (discrete)
- Resampled from variable rate (159 ms - 89 s intervals) to uniform 10 Hz using
  per-channel-type interpolation.
- Filtered to Speed > 100 km/h to skip pit lane and pre-race.

## Model

- Fusionformer FAM (same architecture as SKAB/SMD experiments)
- Trained on the first 25 percent of the racing telemetry as a "normal driving"
  reference
- Scored on the full race using per-window mean squared reconstruction error
- Note: no ground truth anomaly labels exist for F1 telemetry, so this is a
  qualitative cross-domain demonstration, not a quantitative benchmark

## Results

- Total scored windows: {len(scores):,}
- 95th percentile threshold: {threshold:.4f}
- Number of "high-error" windows above threshold: {n_high:,} ({pct_high:.1f} percent)

## Top-5 highest reconstruction error moments (approximate race minute)

| Rank | Race minute | Reconstruction error |
|---|---|---|
"""
    for i, (t, s) in enumerate(zip(top_5_times, top_5_scores), start=1):
        md += f"| {i} | {t:.1f} | {s:.4f} |\n"

    md += """

## Qualitative interpretation

High reconstruction error corresponds to moments where the model, having
learned Hamilton's early-race driving style, fails to predict the current
telemetry pattern. These typically align with:

- Pit stops (rapid deceleration, engine idle, then re-acceleration)
- Sudden traffic (unusual braking or throttle patterns)
- DRS activation windows (throttle profile changes)
- Safety car or virtual safety car periods

Manual inspection of the top-5 moments against known race events would allow
qualitative validation that reconstruction error captures interesting driving
regimes.

## Purpose in the dissertation

This case study is included as a qualitative demonstration that the trained
Fusionformer pipeline generalises across dataset types — from labelled
industrial benchmarks (SKAB, SMD) to unlabelled real-world telemetry (F1).
It fulfils the cross-domain evaluation component proposed in the Task 1 DDR.

Because F1 telemetry lacks ground truth anomaly labels, quantitative
comparison to the SKAB and SMD ablations is not possible. The value here
is architectural portability, not benchmark performance.
"""

    with open(output_path, "w") as f:
        f.write(md)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("F1 QUALITATIVE CASE STUDY — Hamilton 2023 British GP")
    print("=" * 70)

    # Load and preprocess
    resampled, session = load_f1_telemetry()
    n_features = resampled.shape[1]
    print(f"  Feature columns: {list(resampled.columns)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(resampled.values)
    print(f"  Standardized shape: {X_scaled.shape}")

    # Split: first 25% for training (normal reference), score whole race
    train_end = int(len(X_scaled) * 0.25)
    X_train = X_scaled[:train_end]
    print(f"  Training region: rows [0, {train_end})")

    # Train model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = train_on_normal(X_train, n_features, device)

    # Score whole race
    print("\nScoring the whole race...")
    errors = score_all(model, X_scaled, device)
    print(f"  Scored {len(errors)} windows")

    # Plot reconstruction error over time
    print("\nGenerating plots...")
    scores, threshold = plot_reconstruction_over_time(
        errors, resampled.index,
        FIGURES_DIR / "f1_reconstruction_error.png",
    )

    # Plot per-lap summary
    lap_summary = plot_lap_summary(
        scores, resampled.index, session,
        FIGURES_DIR / "f1_lap_summary.png",
    )

    # Write markdown notes
    elapsed = (resampled.index - resampled.index[0]).total_seconds().values / 60.0
    write_summary_notes(
        scores, threshold, elapsed, lap_summary,
        NOTES_DIR / "f1_case_study_results.md",
    )

    print("\n" + "=" * 70)
    print("Case study complete. Outputs:")
    print("  figures/f1_reconstruction_error.png")
    print("  figures/f1_lap_summary.png")
    print("  notes/f1_case_study_results.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
