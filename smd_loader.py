"""
SMD (Server Machine Dataset) loader.

Provides a load_smd_file() function that mirrors the SKAB loader signature:
    X, y = load_smd_file(machine_name)

Where X is a pandas DataFrame of features and y is a pandas Series of binary
anomaly labels.

SMD structure:
    datasets/SMD/train/{machine}.txt         — training data (all normal)
    datasets/SMD/test/{machine}.txt          — test data (contains anomalies)
    datasets/SMD/test_label/{machine}.txt    — binary labels for test
    datasets/SMD/interpretation_label/{machine}.txt — which features are anomalous

The loader concatenates train + test into one sequence so the existing
semi-supervised training pipeline (train on pre-anomaly region) works
without modification.

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""

from pathlib import Path
import numpy as np
import pandas as pd


SMD_ROOT = Path("datasets/SMD")


def list_smd_machines() -> list:
    """Return sorted list of available machine names, e.g. 'machine-1-1'."""
    train_dir = SMD_ROOT / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"SMD train folder not found at {train_dir}")
    files = sorted(train_dir.glob("*.txt"))
    return [f.stem for f in files]


def load_smd_file(machine: str) -> tuple:
    """
    Load one SMD machine's data.

    Args:
        machine: machine name, e.g. 'machine-1-1'.

    Returns:
        X: pandas DataFrame of shape (n_rows, 38) with feature columns.
        y: pandas Series of shape (n_rows,) with binary anomaly labels
           (0 = normal, 1 = anomaly). Train portion is all zeros.
    """
    train_path = SMD_ROOT / "train" / f"{machine}.txt"
    test_path = SMD_ROOT / "test" / f"{machine}.txt"
    label_path = SMD_ROOT / "test_label" / f"{machine}.txt"

    for p in (train_path, test_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"SMD file not found: {p}")

    # Load train + test features.
    X_train = pd.read_csv(train_path, header=None)
    X_test = pd.read_csv(test_path, header=None)
    y_test = pd.read_csv(label_path, header=None).iloc[:, 0]

    # Sanity check: same number of columns in train and test.
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, "
            f"test has {X_test.shape[1]}"
        )

    # Sanity check: test rows match label rows.
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Test rows ({len(X_test)}) does not match label rows ({len(y_test)})"
        )

    # Give columns names f0..f37 to mirror how SKAB features look.
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    X_train.columns = feature_names
    X_test.columns = feature_names

    # Concatenate train + test into one sequence.
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Labels: 0 for all train rows, real labels for test rows.
    y_train = pd.Series(np.zeros(len(X_train), dtype=int))
    y = pd.concat([y_train, y_test], axis=0, ignore_index=True)

    return X, y


if __name__ == "__main__":
    # Quick sanity check.
    machines = list_smd_machines()
    print(f"Found {len(machines)} SMD machines: {machines[:5]}...")

    m = machines[0]
    X, y = load_smd_file(m)
    print(f"\nLoaded {m}:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Feature range: {X.min().min():.2f} to {X.max().max():.2f}")
    print(f"  Total anomalies: {int(y.sum())} / {len(y)} ({y.mean() * 100:.1f}%)")
    first_anom = int(np.argmax(y.values == 1)) if y.any() else len(y)
    print(f"  First anomaly index: {first_anom} (train ends at {len(X) - len(y[y.index >= 0]) if False else 'end of train portion'})")
