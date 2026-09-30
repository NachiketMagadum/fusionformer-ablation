#!/bin/bash
# Simulated benchmark for the variable-coupling question (Section 4.9).
# Design and decision rule are fixed in notes/synth/design.md before this runs.
# 2 conditions x 5 simulated series x 3 seeds, for full, MSWEA-off and LSTM
# (90 training runs), plus the reference detectors and the analysis.
#
# USAGE (from the project root):  bash run_synth.sh

set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
OUT=notes/synth
mkdir -p "$OUT"
done_already() { [ -f "$1" ] && grep -q "Test AUROC" "$1"; }
START=$(date +%s)

python3 synth_generator.py
python3 synth_check.py

for cond in coupled independent; do
  for u in 0 1 2 3 4; do
    for s in 0 1 42; do
      for variant in full no_mswea; do
        out="$OUT/ff_true_${variant}_synth_${cond}_${u}_seed${s}.txt"
        if done_already "$out"; then echo "skip $out"; continue; fi
        echo "==> FF $variant $cond $u seed=$s   ($(( ($(date +%s) - START) / 60 )) min elapsed)"
        python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth "${cond}_${u}" \
          --seed "$s" --variant "$variant" --out_dir "$OUT" || echo "FAILED: $variant $cond $u $s"
      done
      out="$OUT/lstm_baseline_synth_${cond}_${u}_seed${s}.txt"
      if done_already "$out"; then echo "skip $out"; else
        echo "==> LSTM $cond $u seed=$s"
        python3 train_lstm_baseline.py --dataset synth --synth "${cond}_${u}" --seed "$s" --out_dir "$OUT" \
          || echo "FAILED: lstm $cond $u $s"
      fi
    done
  done
done

python3 analyse_synth.py
echo "ALL DONE in $(( ($(date +%s) - START) / 60 )) min"
