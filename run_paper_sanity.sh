#!/usr/bin/env bash
# Paper-hyperparameter sanity run.
# One SMD run (machine-1-1, seed 0), paper-close hyperparameters.
# Purpose: verify that Fusionformer at paper hyperparameters produces
# results roughly consistent with my compute-budget hyperparameters.
#
# Expected wall-clock on M5 MPS 16GB: ~8-10 hours (N_enc=4, N_dec=3, batch 32, 30 epochs).
# If MPS runs out of memory, drop BATCH_SIZE to 16 in train_ff_paper_hparams.py and rerun.
#
# Output: notes/paper_hparam_smd_machine-1-1_seed0.txt
# Launch:  bash run_paper_sanity.sh
# Launch in background (survives terminal close):
#   nohup bash run_paper_sanity.sh > notes/paper_hparam_run.log 2>&1 &

set -e
cd "$(dirname "$0")"
mkdir -p notes
export PYTHONUNBUFFERED=1   # ensure print() flushes to log in real time

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT="notes/ff_paperhp_full_smd_machine-1-1_seed0.txt"
LOG="notes/paper_hparam_run_${STAMP}.log"

echo "[$(date)] Starting paper-hparam sanity run" | tee -a "$LOG"
echo "  Dataset: SMD machine-1-1" | tee -a "$LOG"
echo "  Seed: 0" | tee -a "$LOG"
echo "  Hyperparameters: d_model=256, n_heads=8, L_seg=32, N_enc=4, N_dec=3, batch=32, epochs=30, lr=5e-4" | tee -a "$LOG"
echo "  Output file: $OUT" | tee -a "$LOG"
echo "  Full log:    $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python3 train_ff_paper_hparams.py \
  --dataset smd \
  --machine machine-1-1 \
  --seed 0 \
  --variant full 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date)] Done. Result file: $OUT" | tee -a "$LOG"
