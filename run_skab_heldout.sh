#!/bin/bash
# SKAB re-run with held-out evaluation: models train on the rows before the
# first anomaly and are scored only on rows after that cut.
# 4 Fusionformer variants x 5 valve1 files x 3 seeds = 60 runs, plus 15 LSTM
# runs and the naive controls. The earlier SKAB result files (which were scored
# on the whole file, training rows included) are kept in notes/skab_whole_file_eval/.
#
# USAGE (from the project root):  bash run_skab_heldout.sh
# Resume-safe: a run is skipped if its result file already has the held-out marker.

set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
mkdir -p notes notes/skab_whole_file_eval

# Archive the old whole-file SKAB results once
for f in notes/ff_true_*_skab_*.txt notes/lstm_baseline_skab_*.txt; do
  [ -e "$f" ] || continue
  grep -q "Scored rows" "$f" || mv "$f" notes/skab_whole_file_eval/
done

done_already() { [ -f "$1" ] && grep -q "Scored rows" "$1"; }

SEEDS=(0 1 42)
FILES=(0 1 2 3 4)
START=$(date +%s)

for variant in full no_mswea no_swse adv_on; do
  extra=""
  [ "$variant" = "adv_on" ] && extra="--lambda_adv 0.01 --warmup_epochs 5"
  for f in "${FILES[@]}"; do
    for s in "${SEEDS[@]}"; do
      out="notes/ff_true_${variant}_skab_${f}_seed${s}.txt"
      if done_already "$out"; then echo "skip $out"; continue; fi
      echo "==> FF $variant SKAB file=$f seed=$s   ($(( ($(date +%s) - START) / 60 )) min elapsed)"
      python3 train_ff_forecast_and_score_anomaly.py --dataset skab \
        --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" --variant "$variant" $extra \
        || echo "FAILED: $variant $f $s"
    done
  done
done

for f in "${FILES[@]}"; do
  for s in "${SEEDS[@]}"; do
    out="notes/lstm_baseline_skab_${f}_seed${s}.txt"
    if done_already "$out"; then echo "skip $out"; continue; fi
    echo "==> LSTM SKAB file=$f seed=$s"
    python3 train_lstm_baseline.py --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" \
      || echo "FAILED: lstm $f $s"
  done
done

echo "==> naive controls (SMD and SKAB)"
python3 naive_baselines.py

echo "==> analysis"
python3 analyse_final.py

echo "ALL DONE in $(( ($(date +%s) - START) / 60 )) min"
