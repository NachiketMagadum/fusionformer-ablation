#!/bin/bash
# Forecast-quality sweep (run on cloud GPU, RTX 4090): full Fusionformer and LSTM
# re-run with test MSE / MAE reporting. 24 + 24 runs.
set -e
export PYTHONUNBUFFERED=1
mkdir -p notes

for m in machine-1-1 machine-1-4 machine-2-1; do
  for s in 0 1 42; do
    echo "==> FF full SMD $m seed=$s"
    python3 train_ff_forecast_and_score_anomaly.py --dataset smd --machine "$m" --seed "$s" --variant full
  done
done
for f in 0 1 2 3 4; do
  for s in 0 1 42; do
    echo "==> FF full SKAB file=$f seed=$s"
    python3 train_ff_forecast_and_score_anomaly.py --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" --variant full
  done
done

for m in machine-1-1 machine-1-4 machine-2-1; do
  for s in 0 1 42; do
    echo "==> LSTM SMD $m seed=$s"
    python3 train_lstm_baseline.py --dataset smd --machine "$m" --seed "$s"
  done
done
for f in 0 1 2 3 4; do
  for s in 0 1 42; do
    echo "==> LSTM SKAB file=$f seed=$s"
    python3 train_lstm_baseline.py --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s"
  done
done

echo "==> DONE, taring results"
tar czf /workspace/forecast_results.tar.gz notes/ff_true_full_*.txt notes/lstm_baseline_*.txt notes/forecast_sweep.log
