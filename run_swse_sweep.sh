#!/bin/bash
# SWSE ablation sweep (run on cloud GPU, RTX 4090): 9 SMD + 15 SKAB no_swse runs.
set -e
export PYTHONUNBUFFERED=1
mkdir -p notes

for m in machine-1-1 machine-1-4 machine-2-1; do
  for s in 0 1 42; do
    echo "==> SMD no_swse $m seed=$s"
    python3 train_ff_forecast_and_score_anomaly.py --dataset smd --machine "$m" --seed "$s" --variant no_swse
  done
done

for f in 0 1 2 3 4; do
  for s in 0 1 42; do
    echo "==> SKAB no_swse file=$f seed=$s"
    python3 train_ff_forecast_and_score_anomaly.py --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" --variant no_swse
  done
done

echo "==> DONE, taring results"
tar czf /workspace/swse_results.tar.gz notes/ff_true_no_swse_*.txt notes/swse_sweep.log
