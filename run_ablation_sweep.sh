#!/bin/bash
# Ablation sweep: 3 variants x 3 machines x 3 seeds = 27 runs
# Expected total: ~43 hours at ~95 min per run
#
# USAGE (from the project root):
#   bash run_ablation_sweep.sh
#
# Resume-safe: skips any run whose output file already exists in notes/.
#
# Kill with: pkill -9 -f run_ablation_sweep.sh; pkill -9 -f train_ff_forecast

set -u
mkdir -p notes logs

VARIANTS=("full" "no_mswea" "no_swse")
MACHINES=("machine-1-1" "machine-1-4" "machine-2-1")
SEEDS=(0 1 42)

TOTAL=$(( ${#VARIANTS[@]} * ${#MACHINES[@]} * ${#SEEDS[@]} ))
COUNT=0
START_TIME=$(date +%s)

echo "==========================================================="
echo "Ablation sweep starting at $(date)"
echo "Total runs: $TOTAL (variants=${#VARIANTS[@]} x machines=${#MACHINES[@]} x seeds=${#SEEDS[@]})"
echo "Estimated total wall-clock at 95 min per run: ~$(( TOTAL * 95 / 60 )) hours"
echo "==========================================================="

for variant in "${VARIANTS[@]}"; do
  for machine in "${MACHINES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      OUT_FILE="notes/ff_true_${variant}_smd_${machine}_seed${seed}.txt"
      LOG_FILE="logs/${variant}_${machine}_seed${seed}.log"

      echo ""
      echo "----- [${COUNT}/${TOTAL}] variant=${variant} machine=${machine} seed=${seed} @ $(date +%H:%M) -----"

      if [ -f "$OUT_FILE" ]; then
        echo "  skipping (already completed): $OUT_FILE"
        continue
      fi

      python3 train_ff_forecast_and_score_anomaly.py \
        --dataset smd \
        --machine "$machine" \
        --seed "$seed" \
        --variant "$variant" 2>&1 | tee "$LOG_FILE"

      # After each run, print elapsed and estimated remaining
      NOW=$(date +%s)
      ELAPSED_MIN=$(( (NOW - START_TIME) / 60 ))
      DONE=$(ls notes/ff_true_*.txt 2>/dev/null | wc -l | xargs)
      if [ "$DONE" -gt 0 ]; then
        AVG_MIN=$(( ELAPSED_MIN / DONE ))
        REMAINING=$(( (TOTAL - DONE) * AVG_MIN ))
        echo "  progress: ${DONE}/${TOTAL} done, elapsed ${ELAPSED_MIN} min, estimated remaining ${REMAINING} min"
      fi
    done
  done
done

echo ""
echo "==========================================================="
echo "Ablation sweep finished at $(date)"
echo "Total wall-clock: $(( ($(date +%s) - START_TIME) / 60 )) min"
echo "Results in notes/ff_true_*.txt"
echo "==========================================================="
