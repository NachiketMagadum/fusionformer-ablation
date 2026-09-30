#!/bin/bash
# SKAB cross-benchmark sweep — paper-faithful Fusionformer, MSWEA ablation
# 2 variants (full, no_mswea) × 3 valve1 files × 3 seeds = 18 runs
# Expected: ~30-45 min per run × 18 = ~10-14 hours
# Resume-safe: skips completed runs.

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

set -u
mkdir -p notes logs

VARIANTS=("full" "no_mswea")
SKAB_FILES=(
    "datasets/SKAB/data/valve1/0.csv"
    "datasets/SKAB/data/valve1/1.csv"
    "datasets/SKAB/data/valve1/2.csv"
)
SEEDS=(0 1 42)

START_TIME=$(date +%s)
COUNT=0
TOTAL=$(( ${#VARIANTS[@]} * ${#SKAB_FILES[@]} * ${#SEEDS[@]} ))

echo "==========================================================="
echo "SKAB sweep started at $(date)"
echo "Total runs: $TOTAL"
echo "==========================================================="

for variant in "${VARIANTS[@]}"; do
  for f in "${SKAB_FILES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      BASENAME=$(basename "$f" .csv)
      OUT_FILE="notes/ff_true_${variant}_skab_${BASENAME}_seed${seed}.txt"
      LOG_FILE="logs/${variant}_skab_${BASENAME}_seed${seed}.log"

      echo ""
      echo "----- [${COUNT}/${TOTAL}] variant=${variant} file=${BASENAME} seed=${seed} @ $(date +%H:%M) -----"

      if [ -f "$OUT_FILE" ]; then
        echo "  skipping (already completed): $OUT_FILE"
        continue
      fi

      python3 train_ff_forecast_and_score_anomaly.py \
        --dataset skab --file "$f" --seed "$seed" --variant "$variant" 2>&1 | tee "$LOG_FILE"

      NOW=$(date +%s)
      ELAPSED_MIN=$(( (NOW - START_TIME) / 60 ))
      DONE=$(ls notes/ff_true_full_skab_*.txt notes/ff_true_no_mswea_skab_*.txt 2>/dev/null | wc -l | xargs)
      echo "  progress: ${DONE}/${TOTAL} done, elapsed ${ELAPSED_MIN} min"
    done
  done
done

echo ""
echo "==========================================================="
echo "SKAB sweep finished at $(date)"
echo "Total wall-clock: $(( ($(date +%s) - START_TIME) / 60 )) min"
echo "==========================================================="
