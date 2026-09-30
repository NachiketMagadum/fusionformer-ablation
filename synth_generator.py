"""
Simulated benchmark for the variable-coupling question (Section 4.9).

Generates clearly simulated multivariate series, not real sensor data. Two
conditions, both with 8 channels, 2000 training rows (all normal) and 2000
test rows with labelled anomaly segments:

  coupled      every channel is a noisy, lagged mix of two shared latent
               signals. An anomaly takes 2-3 channels in a segment and replaces
               each with the same channel's own signal from a different time.
               Each affected channel keeps its level, amplitude and rhythm; only
               its relationship with the other channels breaks.

  independent  every channel has its own latent signal. An anomaly takes 2-3
               channels in a segment and speeds up their rhythm (period x 0.6)
               at the same amplitude, so the change is visible from each
               channel's own history.

Neither anomaly type changes the size of the values, so a forecaster that has
learned nothing (mean or persistence) should score close to chance. In the
coupled condition only a model that uses cross-channel structure should be
able to spot the anomalies; in the independent condition cross-channel
structure carries no information.

Usage:
    python3 synth_generator.py          # writes datasets/SYNTH/{coupled,independent}_{0..4}.npz

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
from pathlib import Path

import numpy as np

D = 8
N_TRAIN = 2000
N_TEST = 2000
NOISE = 0.2
N_SEGMENTS = 5
SEG_LEN = (30, 60)
FADE = 10


def latent(t, period, rng):
    """A periodic signal with a harmonic, a random phase and slow AR(1) wander."""
    phase = rng.uniform(0, 2 * np.pi)
    s = np.sin(2 * np.pi * t / period + phase) + 0.4 * np.sin(6 * np.pi * t / period + 2 * phase)
    ar = np.zeros(len(t))
    for i in range(1, len(t)):
        ar[i] = 0.98 * ar[i - 1] + 0.05 * rng.standard_normal()
    return s + ar


def segments(rng, n, n_seg, lo_hi, margin=100):
    """Non-overlapping anomaly segments inside [margin, n - margin)."""
    starts = []
    while len(starts) < n_seg:
        s = int(rng.integers(margin, n - margin - lo_hi[1]))
        if all(abs(s - o) > lo_hi[1] + 20 for o in starts):
            starts.append(s)
    return [(s, s + int(rng.integers(*lo_hi))) for s in sorted(starts)]


def make_coupled(seed):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n + 400)
    f = np.stack([latent(t, 50.0, rng), latent(t, 130.0, rng)])          # 2 shared signals
    W = rng.uniform(0.5, 1.5, size=(D, 2)) * rng.choice([-1, 1], size=(D, 2))
    lags = rng.integers(0, 9, size=D)
    clean = np.stack([W[d] @ f[:, 200 - lags[d]: 200 - lags[d] + n] for d in range(D)], axis=1)
    X = clean + NOISE * rng.standard_normal(clean.shape)
    y = np.zeros(n, dtype=int)
    for a, b in segments(rng, N_TEST, N_SEGMENTS, SEG_LEN):
        a += N_TRAIN; b += N_TRAIN
        chans = rng.choice(D, size=int(rng.integers(2, 4)), replace=False)
        for d in chans:
            shift = int(rng.integers(300, 700)) * int(rng.choice([-1, 1]))
            src = np.arange(a, b) + shift
            src = np.clip(src, 0, n - 1)
            # raised-cosine crossfade over the first and last rows of the segment,
            # so the swap does not create a jump that a per-channel model could see
            w = np.ones(b - a)
            k = min(FADE, (b - a) // 2)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
            w[:k] = ramp; w[-k:] = ramp[::-1]
            X[a:b, d] = (1 - w) * X[a:b, d] + w * (clean[src, d] + NOISE * rng.standard_normal(b - a))
        y[a:b] = 1
    return X[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def make_independent(seed):
    rng = np.random.default_rng(10_000 + seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n)
    periods = rng.uniform(40, 140, size=D)
    amps = rng.uniform(0.5, 1.5, size=D)
    phases = rng.uniform(0, 2 * np.pi, size=D)
    X = np.stack([amps[d] * np.sin(2 * np.pi * t / periods[d] + phases[d]) for d in range(D)], axis=1)
    X += NOISE * rng.standard_normal(X.shape)
    y = np.zeros(n, dtype=int)
    for a, b in segments(rng, N_TEST, N_SEGMENTS, SEG_LEN):
        a += N_TRAIN; b += N_TRAIN
        chans = rng.choice(D, size=int(rng.integers(2, 4)), replace=False)
        for d in chans:
            # continue the phase at the segment start, then run 1/0.6 times faster
            p0 = 2 * np.pi * a / periods[d] + phases[d]
            tt = np.arange(b - a)
            X[a:b, d] = amps[d] * np.sin(p0 + 2 * np.pi * tt / (0.6 * periods[d])) \
                + NOISE * rng.standard_normal(b - a)
        y[a:b] = 1
    return X[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def main():
    out = Path("datasets/SYNTH")
    out.mkdir(parents=True, exist_ok=True)
    for cond, fn in (("coupled", make_coupled), ("independent", make_independent)):
        for seed in range(5):
            X_train, X_test, y_test = fn(seed)
            np.savez(out / f"{cond}_{seed}.npz", X_train=X_train, X_test=X_test, y_test=y_test)
            print(f"{cond}_{seed}: train {X_train.shape}, test {X_test.shape}, "
                  f"anomaly rate {y_test.mean():.3f}")


if __name__ == "__main__":
    main()
