#!/bin/bash
# v2 simulated benchmark (design in notes/synth_v2/design_v2.md, pushed before running).
# 2 conditions x 10 new series x 3 seeds x 3 variants = 180 runs (about 4-5 hours on an M-series Mac).
# USAGE (from the project root):  bash run_synth_v2.sh
set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
OUT=notes/synth_v2
mkdir -p "$OUT"
python3 synth_generator.py --first_seed 100 --n 10
python3 synth_check.py --first_seed 100 --n 10 --out_dir "$OUT"
python3 match_params.py --n_vars 8 | tee "$OUT/param_match.txt"
DM=$(tail -1 "$OUT/param_match.txt")
START=$(date +%s)
for cond in coupled independent; do
  for u in $(seq 100 109); do
    for s in 0 1 42; do
      for variant in full no_mswea no_mswea_pm; do
        out="$OUT/ff_true_${variant}_synth_${cond}_${u}_seed${s}.txt"
        if [ -f "$out" ] && grep -q "Test AUROC" "$out"; then echo "skip $out"; continue; fi
        extra=""; [ "$variant" = "no_mswea_pm" ] && extra="--d_model $DM"
        echo "==> $variant $cond $u seed=$s   ($(( ($(date +%s) - START) / 60 )) min elapsed)"
        python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth "${cond}_${u}" --seed "$s" \
          --variant "$variant" $extra --out_dir "$OUT" || echo "FAILED: $variant $cond $u $s"
      done
    done
  done
done
python3 analyse_synth_v2.py
echo "ALL DONE in $(( ($(date +%s) - START) / 60 )) min"
