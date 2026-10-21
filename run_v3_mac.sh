#!/bin/bash
# v3 runs that fit on an M-series Mac (design: notes/v3/design_v3.md, pushed before running).
#   SKAB : 5 new Fusionformer variants x 5 files x 3 seeds          =  75 runs (~0.3 min each)
#   s2s  : seq2seq LSTM tuning (SKAB 45, simulated 54) + final runs  = 174 runs
#   synth: 5 new Fusionformer variants x 20 series x 3 seeds        = 300 runs (~1.2 min each)
# Roughly 10-12 hours in total. Resume-safe: finished runs are skipped.
# USAGE (from the project root):  bash run_v3_mac.sh
set -u
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
mkdir -p notes/v3/synth notes/v3/skab notes/v3/s2s
START=$(date +%s); el() { echo "($(( ($(date +%s) - START) / 60 )) min elapsed)"; }
done_ok() { [ -f "$1" ] && grep -q "Test AUROC" "$1"; }

# --- SKAB data (public, https://github.com/waico/SKAB) ---
if [ ! -f datasets/SKAB/data/valve1/0.csv ]; then
  git clone --depth 1 https://github.com/waico/SKAB /tmp/SKAB_src && mkdir -p datasets/SKAB && cp -r /tmp/SKAB_src/data datasets/SKAB/
fi
# --- simulated v2 series (regenerated deterministically if missing) ---
[ -f datasets/SYNTH/coupled_109.npz ] || python3 synth_generator.py --first_seed 100 --n 10

# --- parameter-matched d_model (8 channels for both SKAB valve1 and the simulated series) ---
for v in no_mswaa no_fam; do python3 match_params.py --n_vars 8 --variant $v | tee notes/v3/param_match_$v.txt; done
DM_MSWAA=$(tail -1 notes/v3/param_match_no_mswaa.txt); DM_FAM=$(tail -1 notes/v3/param_match_no_fam.txt)
extra_for() { case $1 in no_mswaa_pm) echo "--d_model $DM_MSWAA";; no_fam_pm) echo "--d_model $DM_FAM";; *) echo "";; esac; }
VARIANTS="no_mswaa no_mswaa_pm no_fam no_fam_pm adv_only"

# --- 1. SKAB, held-out scoring ---
for v in $VARIANTS; do for f in 0 1 2 3 4; do for s in 0 1 42; do
  out=notes/v3/skab/ff_true_${v}_skab_${f}_seed${s}.txt; done_ok $out && continue
  echo "==> $v SKAB $f seed $s $(el)"
  python3 train_ff_forecast_and_score_anomaly.py --dataset skab --file datasets/SKAB/data/valve1/$f.csv --seed $s \
    --variant $v $(extra_for $v) --out_dir notes/v3/skab > /dev/null || echo "FAILED $v skab $f $s"
done; done; done

# --- 2. seq2seq LSTM: tuning on validation only, then final runs ---
for h in 64 128 256; do for lr in 0.0003 0.001 0.003; do
  for f in 0 1 2 3 4; do
    [ -f notes/v3/s2s/s2s_val_skab_${f}_h${h}_lr${lr}_seed0.txt ] && continue
    echo "==> s2s tune SKAB $f h$h lr$lr $(el)"
    python3 train_lstm_s2s.py --dataset skab --file datasets/SKAB/data/valve1/$f.csv --seed 0 --hidden $h --lr $lr \
      --val_only --out_dir notes/v3/s2s > /dev/null || echo "FAILED tune skab $f $h $lr"
  done
  for c in coupled independent; do for u in 100 101 102; do
    [ -f notes/v3/s2s/s2s_val_synth_${c}_${u}_h${h}_lr${lr}_seed0.txt ] && continue
    echo "==> s2s tune ${c}_$u h$h lr$lr $(el)"
    python3 train_lstm_s2s.py --dataset synth --synth ${c}_$u --seed 0 --hidden $h --lr $lr \
      --val_only --out_dir notes/v3/s2s > /dev/null || echo "FAILED tune ${c}_$u $h $lr"
  done; done
done; done
python3 select_s2s_config.py
read H LR < notes/v3/s2s/chosen_skab.txt
for f in 0 1 2 3 4; do for s in 0 1 42; do
  done_ok notes/v3/s2s/s2s_skab_${f}_seed${s}.txt && continue
  echo "==> s2s SKAB $f seed $s (h$H lr$LR) $(el)"
  python3 train_lstm_s2s.py --dataset skab --file datasets/SKAB/data/valve1/$f.csv --seed $s --hidden $H --lr $LR \
    --out_dir notes/v3/s2s > /dev/null || echo "FAILED s2s skab $f $s"
done; done
for c in coupled independent; do
  read H LR < notes/v3/s2s/chosen_${c}.txt
  for u in $(seq 100 109); do for s in 0 1 42; do
    done_ok notes/v3/s2s/s2s_synth_${c}_${u}_seed${s}.txt && continue
    echo "==> s2s ${c}_$u seed $s (h$H lr$LR) $(el)"
    python3 train_lstm_s2s.py --dataset synth --synth ${c}_$u --seed $s --hidden $H --lr $LR \
      --out_dir notes/v3/s2s > /dev/null || echo "FAILED s2s ${c}_$u $s"
  done; done
done

# --- 3. simulated v2 series ---
for c in coupled independent; do for u in $(seq 100 109); do for s in 0 1 42; do for v in $VARIANTS; do
  out=notes/v3/synth/ff_true_${v}_synth_${c}_${u}_seed${s}.txt; done_ok $out && continue
  echo "==> $v ${c}_$u seed $s $(el)"
  python3 train_ff_forecast_and_score_anomaly.py --dataset synth --synth ${c}_$u --seed $s \
    --variant $v $(extra_for $v) --out_dir notes/v3/synth > /dev/null || echo "FAILED $v ${c}_$u $s"
done; done; done; done

python3 analyse_v3.py
echo "ALL DONE $(el)"
