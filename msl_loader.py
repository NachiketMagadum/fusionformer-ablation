"""
MSL (Mars Science Laboratory rover telemetry) dataset loader.

Source: Hundman et al. (2018) "Detecting Spacecraft Anomalies Using LSTMs
and Nonparametric Dynamic Thresholding", KDD 2018. Data released by NASA JPL
at https://github.com/khundman/telemanom (data.zip on their S3 bucket).

Directory layout expected:
    datasets/MSL/train/{chan_id}.npy      (T_train, 55) — 55 telemetry features
    datasets/MSL/test/{chan_id}.npy       (T_test,  55)
    datasets/MSL/labeled_anomalies.csv     with columns chan_id, spacecraft, anomaly_sequences, num_values

Use fetch_msl.sh to download and unpack.
"""
from pathlib import Path
import ast
import numpy as np
import pandas as pd

_ROOT = Path("datasets/MSL")


def load_msl_channel(chan_id: str):
    """Return (X_train, X_test, y_test) for one MSL channel.

    X_train, X_test: (T, 55) float arrays.
    y_test: (T_test,) int array with 1s inside labelled anomaly ranges.
    """
    train_path = _ROOT / "train" / f"{chan_id}.npy"
    test_path = _ROOT / "test" / f"{chan_id}.npy"
    labels_path = _ROOT / "labeled_anomalies.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"MSL channel {chan_id} not found under {_ROOT}. "
            f"Run bash fetch_msl.sh to download."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path}. Run bash fetch_msl.sh."
        )

    X_train = np.load(train_path).astype(np.float32)
    X_test = np.load(test_path).astype(np.float32)

    df = pd.read_csv(labels_path)
    row = df[(df.chan_id == chan_id) & (df.spacecraft == "MSL")]
    if len(row) == 0:
        raise ValueError(f"Channel {chan_id} not in labeled_anomalies.csv for MSL")
    row = row.iloc[0]
    seqs = ast.literal_eval(row.anomaly_sequences) if isinstance(row.anomaly_sequences, str) else row.anomaly_sequences

    y_test = np.zeros(len(X_test), dtype=np.int64)
    for span in seqs:
        a, b = int(span[0]), int(span[1])
        y_test[a:b + 1] = 1

    return X_train, X_test, y_test


def list_msl_channels():
    labels_path = _ROOT / "labeled_anomalies.csv"
    if not labels_path.exists():
        return []
    df = pd.read_csv(labels_path)
    return sorted(df[df.spacecraft == "MSL"]["chan_id"].tolist())


if __name__ == "__main__":
    chans = list_msl_channels()
    print(f"Found {len(chans)} MSL channels: {chans}")
    if chans:
        c = chans[0]
        X_tr, X_te, y_te = load_msl_channel(c)
        print(f"Channel {c}: train {X_tr.shape}, test {X_te.shape}, anomaly rate {y_te.mean():.3f}")
