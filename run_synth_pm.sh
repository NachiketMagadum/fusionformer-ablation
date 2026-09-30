#!/bin/bash
# Parameter-matched control for the simulated benchmark (Section 4.9): MSWEA-off
# with d_model raised until its parameter count matches the full model.
# 2 conditions x 5 series x 3 seeds = 30 runs. Added after review; not part of
# the pre-specified design in notes/synth/design.md.
#
# USAGE (from the project root):  bash run_synth_pm.sh

set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
OUT=notes/synth
python3 match_params.py --n_vars 8 | tee "$OUT/param_match.txt"
DM=$(tail -1 "$OUT/param_match.txt")
START=$(date +%s)
for cond in coupled independent; do
  for u in 0 1 2 3 4; do
    for s in 0 1 42; do
      out="$OUT/ff_true_no_mswea_pm_synth_${cond}_${u}_seed${s}.txt"
      if [ -f "$out" ] && grep -q "Test AUROC" "$out"; then echo "skip $out"; continue; fi
      echo "==> no_mswea_pm (d_model=$DM) $cond $u seed=$s   ($(( ($(date +%s) - START) / 60 )) min elapsed)"
      python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth "${cond}_${u}" --seed "$s" \
        --variant no_mswea_pm --d_model "$DM" --out_dir "$OUT" || echo "FAILED: $cond $u $s"
    done
  done
done
python3 analyse_synth.py
echo "ALL DONE in $(( ($(date +%s) - START) / 60 )) min"
