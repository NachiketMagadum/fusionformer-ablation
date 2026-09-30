#!/bin/bash
# Second held-out design for SKAB: train on the first 80% of the normal period
# before the first anomaly, and score everything after that cut. The scored rows
# then contain normal data from the same regime as training (the last 20% of
# the normal period) as well as the anomaly and the recovery period.
# Results go to notes/skab_tail_holdout/ so they do not overwrite the main run.
#
# USAGE (from the project root):  bash run_skab_tail_holdout.sh

set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
OUT=notes/skab_tail_holdout
mkdir -p "$OUT"
done_already() { [ -f "$1" ] && grep -q "Scored rows" "$1"; }
START=$(date +%s)

for variant in full no_mswea no_swse adv_on; do
  extra=""
  [ "$variant" = "adv_on" ] && extra="--lambda_adv 0.01 --warmup_epochs 5"
  for f in 0 1 2 3 4; do
    for s in 0 1 42; do
      out="$OUT/ff_true_${variant}_skab_${f}_seed${s}.txt"
      if done_already "$out"; then echo "skip $out"; continue; fi
      echo "==> FF $variant SKAB file=$f seed=$s   ($(( ($(date +%s) - START) / 60 )) min elapsed)"
      python3 train_ff_forecast_and_score_anomaly.py --dataset skab \
        --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" --variant "$variant" $extra \
        --skab_train_frac 0.8 --out_dir "$OUT" || echo "FAILED: $variant $f $s"
    done
  done
done
for f in 0 1 2 3 4; do
  for s in 0 1 42; do
    out="$OUT/lstm_baseline_skab_${f}_seed${s}.txt"
    if done_already "$out"; then echo "skip $out"; continue; fi
    echo "==> LSTM SKAB file=$f seed=$s"
    python3 train_lstm_baseline.py --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" \
      --skab_train_frac 0.8 --out_dir "$OUT" || echo "FAILED: lstm $f $s"
  done
done
python3 naive_baselines.py --skab_train_frac 0.8 --out_dir "$OUT"
python3 analyse_final.py --skab_dir "$OUT" --tag _tail_holdout
echo "ALL DONE in $(( ($(date +%s) - START) / 60 )) min"
