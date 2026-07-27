#!/bin/bash
set -euo pipefail
source ~/miniforge3/bin/activate myenv
cd ~/Htautau-mass-regression

# 1. Build the SM sample h5s (hadronic-tautau selection, no GhostHBosonsMass cut)
python -u ~/Htautau-mass-regression/python_scripts/prepare_sm.py

# 2. Evaluate the best checkpoint on each SM sample (full sample, no cap)
EPOCH=047     # <-- best epoch from the val-loss curve
RUN=~/Htautau-mass-regression/logs/htautau_mass_regression_20260629-T044141
CKPT=$RUN/ckpts/epoch=${EPOCH}-val_loss=0.00000.ckpt

for S in sm603700 sm603419; do
    echo "=== salt test on $S ==="
    ~/miniforge3/envs/myenv/bin/salt test \
        --config "$RUN/config.yaml" \
        --ckpt_path "$CKPT" \
        --data.test_file ~/Htautau-mass-regression/data/htautau_${S}.h5
done

echo "SM DONE -> $RUN/ckpts/epoch=${EPOCH}-val_loss=0.00000__test_htautau_sm603700.h5"
echo "          $RUN/ckpts/epoch=${EPOCH}-val_loss=0.00000__test_htautau_sm603419.h5"
