#!/bin/bash
# Phase 2 sweep: complete the missing variants
# 1. no_swse on SMD (9 runs at batch=24 to avoid OOM)
# 2. SKAB ablation on faithful model (9 full + 9 no_mswea on valve1)
# 3. adversarial variant on SMD (9 runs at batch=48)
#
# USAGE:
#   bash run_ablation_phase2.sh
#
# Env var below tells MPS to allow more aggressive memory allocation.

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

set -u
mkdir -p notes logs

MACHINES=("machine-1-1" "machine-1-4" "machine-2-1")
SEEDS=(0 1 42)
SKAB_FILES=("datasets/SKAB/data/valve1/0.csv" "datasets/SKAB/data/valve1/1.csv" "datasets/SKAB/data/valve1/2.csv")

START_TIME=$(date +%s)
COUNT=0

run_one() {
    local variant=$1
    local dataset=$2
    local id=$3
    local seed=$4
    local extra=""

    COUNT=$((COUNT + 1))

    if [ "$dataset" = "smd" ]; then
        OUT_FILE="notes/ff_true_${variant}_smd_${id}_seed${seed}.txt"
        LOG_FILE="logs/${variant}_smd_${id}_seed${seed}.log"
        extra="--machine $id"
    else
        BASENAME=$(basename "$id" .csv)
        OUT_FILE="notes/ff_true_${variant}_skab_${BASENAME}_seed${seed}.txt"
        LOG_FILE="logs/${variant}_skab_${BASENAME}_seed${seed}.log"
        extra="--file $id"
    fi

    echo ""
    echo "----- [${COUNT}] variant=${variant} dataset=${dataset} id=${id} seed=${seed} @ $(date +%H:%M) -----"

    if [ -f "$OUT_FILE" ]; then
        echo "  skipping (already completed): $OUT_FILE"
        return
    fi

    # Wrap in retry: if MPS OOM crashes, retry once with fresh Python session
    for attempt in 1 2; do
        python3 train_ff_forecast_and_score_anomaly.py \
            --dataset "$dataset" $extra --seed "$seed" --variant "$variant" 2>&1 | tee "$LOG_FILE"

        if [ -f "$OUT_FILE" ]; then
            echo "  succeeded on attempt ${attempt}"
            break
        else
            echo "  attempt ${attempt} did not produce output — retrying"
            sleep 30
        fi
    done

    NOW=$(date +%s)
    ELAPSED_MIN=$(( (NOW - START_TIME) / 60 ))
    echo "  elapsed since sweep start: ${ELAPSED_MIN} min"
}

echo "==========================================================="
echo "Phase 2 ablation sweep started at $(date)"
echo "==========================================================="

# --- Priority 1: no_swse on SMD (biggest gap in the ablation) ---
echo ""
echo "===== PRIORITY 1: no_swse on SMD ====="
for machine in "${MACHINES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one "no_swse" "smd" "$machine" "$seed"
    done
done

# --- Priority 2: SKAB cross-benchmark on faithful model (full + no_mswea) ---
echo ""
echo "===== PRIORITY 2: SKAB cross-benchmark (full + no_mswea on valve1) ====="
for variant in "full" "no_mswea"; do
    for f in "${SKAB_FILES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$variant" "skab" "$f" "$seed"
        done
    done
done

# --- Priority 3: adversarial on SMD ---
echo ""
echo "===== PRIORITY 3: adversarial variant on SMD ====="
for machine in "${MACHINES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one "adversarial" "smd" "$machine" "$seed"
    done
done

echo ""
echo "==========================================================="
echo "Phase 2 sweep finished at $(date)"
echo "Total wall-clock: $(( ($(date +%s) - START_TIME) / 60 )) min"
echo "==========================================================="
