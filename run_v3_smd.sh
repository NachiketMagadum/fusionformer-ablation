#!/bin/bash
# v3 SMD runs (design: notes/v3/design_v3.md). Slow: ~9 min per Fusionformer run on an RTX 4090,
# up to ~45 min on an M-series Mac. Best on a rented GPU (about 6-8 hours in total).
#   B : full model x 3 machines x 3 seeds, saving per-channel errors, then smd_perchannel.py
#   A : no_mswaa and no_fam x 3 machines x 3 seeds (descriptive)
#   C : seq2seq LSTM tuning (27 runs) and final runs (9)
# USAGE (from the project root):  bash run_v3_smd.sh
set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
O=notes/v3/smd; S=notes/v3/s2s; mkdir -p $O $S
START=$(date +%s); el() { echo "($(( ($(date +%s) - START) / 60 )) min elapsed)"; }
done_ok() { [ -f "$1" ] && grep -q "Test AUROC" "$1"; }
M="machine-1-1 machine-1-4 machine-2-1"

for m in $M; do for s in 0 1 42; do
  done_ok $O/ff_true_full_smd_${m}_seed${s}.txt && [ -f $O/ff_true_full_smd_${m}_seed${s}.npz ] && continue
  echo "==> B full $m seed $s $(el)"
  python3 train_ff_forecast_and_score_anomaly.py --dataset smd --machine $m --seed $s --variant full \
    --save_errors --out_dir $O > /dev/null || echo "FAILED full $m $s"
done; done
python3 smd_perchannel.py --all_machines

for v in no_mswaa no_fam; do for m in $M; do for s in 0 1 42; do
  done_ok $O/ff_true_${v}_smd_${m}_seed${s}.txt && continue
  echo "==> A $v $m seed $s $(el)"
  python3 train_ff_forecast_and_score_anomaly.py --dataset smd --machine $m --seed $s --variant $v \
    --out_dir $O > /dev/null || echo "FAILED $v $m $s"
done; done; done

for h in 64 128 256; do for lr in 0.0003 0.001 0.003; do for m in $M; do
  [ -f $S/s2s_val_smd_${m}_h${h}_lr${lr}_seed0.txt ] && continue
  echo "==> s2s tune $m h$h lr$lr $(el)"
  python3 train_lstm_s2s.py --dataset smd --machine $m --seed 0 --hidden $h --lr $lr --val_only --out_dir $S > /dev/null \
    || echo "FAILED tune $m $h $lr"
done; done; done
python3 select_s2s_config.py
read H LR < $S/chosen_smd.txt
for m in $M; do for s in 0 1 42; do
  done_ok $S/s2s_smd_${m}_seed${s}.txt && continue
  echo "==> s2s $m seed $s $(el)"
  python3 train_lstm_s2s.py --dataset smd --machine $m --seed $s --hidden $H --lr $LR --out_dir $S > /dev/null \
    || echo "FAILED s2s $m $s"
done; done

python3 analyse_v3.py
echo "ALL DONE $(el)"
