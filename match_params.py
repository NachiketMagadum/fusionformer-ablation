"""
Find the d_model at which an ablated Fusionformer has the same number of
parameters as the full model (parameter-matched controls). Prints the chosen
d_model on the last line.

Usage:
    python3 match_params.py --n_vars 8                      # MSWEA-off (as used in Section 4.9 and v2)
    python3 match_params.py --n_vars 8 --variant no_mswaa   # MSWAA-off (v3)
    python3 match_params.py --n_vars 8 --variant no_fam     # FAM-off, no attention (v3)

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import argparse
from fusionformer_true import Fusionformer
from train_ff_forecast_and_score_anomaly import SEQ_LEN, PRED_LEN, SEGMENT_LEN, D_MODEL, N_HEADS, N_ENC, N_DEC

FLAGS = {"full": (True, True), "no_mswea": (False, True), "no_mswaa": (True, False), "no_fam": (False, False)}


def count(d_model, n_vars, variant):
    use_mswea, use_mswaa = FLAGS[variant]
    m = Fusionformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_vars=n_vars, segment_len=SEGMENT_LEN,
                     d_model=d_model, n_heads=N_HEADS, n_enc=N_ENC, n_dec=N_DEC, dropout=0.1,
                     use_swse=True, use_mswea=use_mswea, use_mswaa=use_mswaa)
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_vars", type=int, default=8)
    ap.add_argument("--variant", choices=["no_mswea", "no_mswaa", "no_fam"], default="no_mswea")
    a = ap.parse_args()
    target = count(D_MODEL, a.n_vars, "full")
    best = min(range(D_MODEL, 4 * D_MODEL, N_HEADS), key=lambda dm: abs(count(dm, a.n_vars, a.variant) - target))
    print(f"full model (d_model={D_MODEL}): {target:,} parameters")
    print(f"{a.variant} at d_model={best}: {count(best, a.n_vars, a.variant):,} parameters")
    print(best)


if __name__ == "__main__":
    main()
