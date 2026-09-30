# Datasets (not included)

Download the public datasets and place them here:

| Dataset | Source | Expected path |
|---|---|---|
| SKAB | https://github.com/waico/SKAB | `datasets/SKAB/data/valve1/0.csv` ... `4.csv` |
| SMD | https://github.com/NetManAIOps/OmniAnomaly (ServerMachineDataset) | `datasets/SMD/{train,test,test_label}/machine-*.txt` |
| MSL | https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl | `datasets/MSL/` (see `fetch_msl.sh`) |

The simulated benchmark is generated, not downloaded: `python3 synth_generator.py` writes `datasets/SYNTH/`.
