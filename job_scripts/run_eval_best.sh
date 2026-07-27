#!/bin/bash
set -euo pipefail
source ~/miniforge3/bin/activate myenv
cd ~/Htautau-mass-regression

# Definitive full-validation evaluation of the best checkpoint.
# No --data.num_test cap => uses ALL ~1.4M validation jets (run_score used 200k).
# Overwrites that epoch's __test_htautau_val.h5 with the full-stats version.

EPOCH=014     # <-- set to the best epoch from score_checkpoints.py
RUN=~/Htautau-mass-regression/logs/htautau_mass_regression_20260629-T044141
CKPT=$RUN/ckpts/epoch=${EPOCH}-val_loss=0.00000.ckpt
VAL=~/Htautau-mass-regression/data/htautau_val.h5

echo "Full-val eval of $CKPT"
~/miniforge3/envs/myenv/bin/salt test \
    --config "$RUN/config.yaml" \
    --ckpt_path "$CKPT" \
    --data.test_file "$VAL"

echo "EVAL DONE -> ${CKPT%.ckpt}__test_htautau_val.h5"
