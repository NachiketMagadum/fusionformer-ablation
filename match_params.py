"""
Find the d_model at which the MSWEA-off Fusionformer has the same number of
parameters as the full model (used for the parameter-matched control in
Section 4.9). Prints the chosen d_model on the last line.

Usage:  python3 match_params.py --n_vars 8

Author: Nachiket Magadum
MSc AI dissertation, Brunel University London, 2026.
"""
import argparse
from fusionformer_true import Fusionformer
from train_ff_forecast_and_score_anomaly import SEQ_LEN, PRED_LEN, SEGMENT_LEN, D_MODEL, N_HEADS, N_ENC, N_DEC


def count(d_model, n_vars, use_mswea):
    m = Fusionformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_vars=n_vars, segment_len=SEGMENT_LEN,
                     d_model=d_model, n_heads=N_HEADS, n_enc=N_ENC, n_dec=N_DEC, dropout=0.1,
                     use_swse=True, use_mswea=use_mswea)
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_vars", type=int, default=8)
    a = ap.parse_args()
    target = count(D_MODEL, a.n_vars, True)
    best = min(range(D_MODEL, 3 * D_MODEL, N_HEADS), key=lambda dm: abs(count(dm, a.n_vars, False) - target))
    print(f"full model (d_model={D_MODEL}): {target:,} parameters")
    print(f"MSWEA-off at d_model={best}: {count(best, a.n_vars, False):,} parameters")
    print(best)


if __name__ == "__main__":
    main()
