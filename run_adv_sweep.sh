#!/bin/bash
# Adversarial ablation sweep (run on cloud GPU, RTX 4090): lambda_adv = 0.01, 5-epoch warmup.
set -e
export PYTHONUNBUFFERED=1
mkdir -p notes

for m in machine-1-1 machine-1-4 machine-2-1; do
  for s in 0 1 42; do
    echo "==> SMD adv_on $m seed=$s"
    python3 train_ff_forecast_and_score_anomaly.py \
      --dataset smd --machine "$m" --seed "$s" --variant adv_on \
      --lambda_adv 0.01 --warmup_epochs 5
  done
done

for f in 0 1 2 3 4; do
  for s in 0 1 42; do
    echo "==> SKAB adv_on file=$f seed=$s"
    python3 train_ff_forecast_and_score_anomaly.py \
      --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" \
      --seed "$s" --variant adv_on \
      --lambda_adv 0.01 --warmup_epochs 5
  done
done

echo "==> DONE, taring results"
tar czf /workspace/adv_results.tar.gz notes/ff_true_adv_on_*.txt notes/adv_sweep.log
