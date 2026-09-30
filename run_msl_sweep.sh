#!/usr/bin/env bash
# MSL benchmark sweep: 3 channels × 3 seeds × 2 variants (full/no_mswea) = 18 FF runs
# plus 3 channels × 3 seeds LSTM baseline = 9 more runs. 27 runs total.
#
# MSL channels are small (train ~2-4k rows each), so each run is ~5-15 min.
# Expected total wall-clock: ~4-6 hours on M5 MPS 16GB.
#
# Prereq: bash fetch_msl.sh (downloads NASA MSL/SMAP data.zip)
#
# Launch:
#   nohup env PYTHONUNBUFFERED=1 bash run_msl_sweep.sh > notes/msl_bg.log 2>&1 &
#
# Monitor:
#   tail -f $(ls -t notes/msl_sweep_run_*.log | head -1)

set -e
cd "$(dirname "$0")"
mkdir -p notes
export PYTHONUNBUFFERED=1

STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="notes/msl_sweep_run_${STAMP}.log"

# Three representative MSL channels
CHANNELS=(M-1 M-2 M-3)
SEEDS=(0 1 42)

echo "[$(date)] Starting MSL sweep" | tee -a "$LOG"

# --- Fusionformer full + no_mswea ---
for v in full no_mswea; do
  for c in "${CHANNELS[@]}"; do
    for s in "${SEEDS[@]}"; do
      echo "" | tee -a "$LOG"
      echo "[$(date)] FF $v  channel=$c seed=$s" | tee -a "$LOG"
      python3 train_ff_forecast_and_score_anomaly.py \
        --dataset msl --channel "$c" --seed "$s" --variant "$v" 2>&1 | tee -a "$LOG"
    done
  done
done

# --- LSTM baseline ---
for c in "${CHANNELS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "" | tee -a "$LOG"
    echo "[$(date)] LSTM channel=$c seed=$s" | tee -a "$LOG"
    python3 train_lstm_baseline.py \
      --dataset msl --channel "$c" --seed "$s" 2>&1 | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "[$(date)] Done. FF results at notes/ff_true_*_msl_*.txt, LSTM at notes/lstm_baseline_msl_*.txt" | tee -a "$LOG"
