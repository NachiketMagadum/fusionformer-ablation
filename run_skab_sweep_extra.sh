#!/bin/bash
# Extension: SKAB valve1 files 3 and 4 for both variants
# 2 variants x 2 files x 3 seeds = 12 runs, ~4-6 minutes total

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
set -u
mkdir -p notes logs

VARIANTS=("full" "no_mswea")
SKAB_FILES=(
    "datasets/SKAB/data/valve1/3.csv"
    "datasets/SKAB/data/valve1/4.csv"
)
SEEDS=(0 1 42)

START_TIME=$(date +%s)
COUNT=0
TOTAL=12

echo "==========================================================="
echo "SKAB extension sweep started at $(date)"
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
        echo "  skipping (already completed)"
        continue
      fi

      python3 train_ff_forecast_and_score_anomaly.py \
        --dataset skab --file "$f" --seed "$seed" --variant "$variant" 2>&1 | tee "$LOG_FILE"
    done
  done
done

echo ""
echo "==========================================================="
echo "Extension finished at $(date)"
echo "==========================================================="
