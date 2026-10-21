#!/bin/bash
# v3 smoke test: checks every new code path runs, on the v1 simulated series coupled_0
# (not a v3 test unit) with 1-2 epochs. Outputs go to notes/v3_smoke/ and are not analysed.
# Run this BEFORE committing the v3 pre-registration, then delete notes/v3_smoke/.
# USAGE (from the project root):  bash run_v3_smoke.sh
set -eu
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
OUT=notes/v3_smoke; mkdir -p "$OUT"
[ -f datasets/SYNTH/coupled_0.npz ] || python3 synth_generator.py
for v in no_mswea no_mswaa no_fam; do python3 match_params.py --n_vars 8 --variant $v | tail -2; done
for v in no_mswaa no_fam adv_only; do
  python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth coupled_0 --seed 0 --variant $v \
    --epochs 1 --out_dir "$OUT" | grep -E "Variant|params|Objective|AUROC"
done
python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth coupled_0 --seed 0 --variant no_fam_pm \
  --d_model "$(python3 match_params.py --n_vars 8 --variant no_fam | tail -1)" --epochs 1 --out_dir "$OUT" | grep -E "params"
python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth coupled_0 --seed 0 --variant full \
  --epochs 1 --save_errors --out_dir "$OUT" | grep -E "Saved"
python3 -c "import numpy as np; z=np.load('$OUT/ff_true_full_synth_coupled_0_seed0.npz'); print({k: z[k].shape for k in z})"
python3 train_lstm_s2s.py --dataset synth --synth coupled_0 --seed 0 --hidden 64 --lr 0.001 --max_epochs 2 --val_only --out_dir "$OUT" | tail -3
python3 train_lstm_s2s.py --dataset synth --synth coupled_0 --seed 0 --hidden 64 --lr 0.001 --max_epochs 2 --out_dir "$OUT" | grep -E "AUROC|epoch"
echo "SMOKE TEST PASSED. Now: rm -rf $OUT, then commit and push the v3 pre-registration."
