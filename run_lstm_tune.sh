#!/bin/bash
# Tuned LSTM baseline on the v2 simulated series (design: notes/lstm_tune/design_lstm_tune.md).
# Stage 1: 9-config grid, seed 0, validation MSE on the last 20% of training rows (no test data).
# Stage 2: chosen config per condition, 3 seeds on all 20 series, then the pre-specified comparison.
# USAGE (from the project root):  bash run_lstm_tune.sh
set -u
export PYTHONUNBUFFERED=1
OUT=notes/lstm_tune
mkdir -p "$OUT"
for cond in coupled independent; do
  for u in $(seq 100 109); do
    for h in 64 128 256; do
      for lr in 0.00015 0.001 0.003; do
        f="$OUT/lstm_val_synth_${cond}_${u}_h${h}_lr$(python3 -c "print(f'{$lr:g}')")_seed0.txt"
        if [ -f "$f" ]; then echo "skip $f"; continue; fi
        echo "==> tune $cond $u h=$h lr=$lr"
        python3 train_lstm_baseline.py --dataset synth --synth "${cond}_${u}" --seed 0 \
          --hidden $h --lr $lr --val_only --out_dir "$OUT" || echo "FAILED tune $cond $u $h $lr"
      done
    done
  done
done
python3 select_lstm_config.py
for cond in coupled independent; do
  read H LR < "$OUT/chosen_${cond}.txt"
  for u in $(seq 100 109); do
    for s in 0 1 42; do
      f="$OUT/lstm_baseline_tuned_synth_${cond}_${u}_seed${s}.txt"
      if [ -f "$f" ] && grep -q "Test AUROC" "$f"; then echo "skip $f"; continue; fi
      echo "==> tuned LSTM $cond $u seed=$s (h=$H lr=$LR)"
      python3 train_lstm_baseline.py --dataset synth --synth "${cond}_${u}" --seed "$s" \
        --hidden $H --lr $LR --tag _tuned --out_dir "$OUT" || echo "FAILED tuned $cond $u $s"
    done
  done
done
python3 analyse_lstm_tuned.py
echo "ALL DONE"
