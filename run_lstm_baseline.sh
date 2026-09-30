#!/usr/bin/env bash
# LSTM forecasting baseline — matched coverage to my Fusionformer full runs.
# 3 SMD machines × 3 seeds + 5 SKAB files × 3 seeds = 24 runs total.
# Expected wall-clock on M5 MPS 16GB: ~2-4 hours (LSTM is much cheaper than Fusionformer).
#
# Launch:
#   nohup env PYTHONUNBUFFERED=1 bash run_lstm_baseline.sh > notes/lstm_bg.log 2>&1 &
#
# Monitor:
#   tail -f $(ls -t notes/lstm_baseline_run_*.log | head -1)

set -e
cd "$(dirname "$0")"
mkdir -p notes
export PYTHONUNBUFFERED=1

STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="notes/lstm_baseline_run_${STAMP}.log"

echo "[$(date)] Starting LSTM baseline sweep" | tee -a "$LOG"

# --- SMD: 3 machines × 3 seeds ---
for m in machine-1-1 machine-1-4 machine-2-1; do
  for s in 0 1 42; do
    echo "" | tee -a "$LOG"
    echo "[$(date)] SMD $m seed=$s" | tee -a "$LOG"
    python3 train_lstm_baseline.py --dataset smd --machine "$m" --seed "$s" 2>&1 | tee -a "$LOG"
  done
done

# --- SKAB: 5 files × 3 seeds ---
for f in 0 1 2 3 4; do
  for s in 0 1 42; do
    echo "" | tee -a "$LOG"
    echo "[$(date)] SKAB file=$f seed=$s" | tee -a "$LOG"
    python3 train_lstm_baseline.py --dataset skab --file "datasets/SKAB/data/valve1/${f}.csv" --seed "$s" 2>&1 | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "[$(date)] Done. Results in notes/lstm_baseline_*.txt" | tee -a "$LOG"
