#!/bin/bash
# Rebuild every statistic and figure from the archived per-run results in notes/.
# No training and no datasets needed.
set -e
python3 analyse_final.py                                                        # notes/final_stats.txt, all_runs.csv
python3 analyse_final.py --skab_dir notes/skab_tail_holdout --tag _tail_holdout # second SKAB design
python3 analyse_synth.py                                                        # notes/synth/synth_stats.txt
python3 generate_figures.py                                                     # figures/
echo "Done: see notes/final_stats.txt, notes/final_stats_tail_holdout.txt, notes/synth/synth_stats.txt and figures/"
