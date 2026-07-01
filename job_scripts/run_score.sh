#!/bin/bash
set -euo pipefail
source ~/miniforge3/bin/activate myenv
cd ~/Htautau-mass-regression

RUN=~/Htautau-mass-regression/logs/htautau_mass_regression_20260629-T044141
CFG=$RUN/config.yaml
VAL=~/Htautau-mass-regression/data/htautau_val.h5

# Evaluate every checkpoint on a fixed 200k-jet val subset (GPU, ~2 min each).
# salt writes epoch=XXX-...__test_htautau_val.h5 next to each checkpoint.
for ckpt in "$RUN"/ckpts/epoch=*.ckpt; do
    echo "=== scoring $ckpt ==="
    ~/miniforge3/envs/myenv/bin/salt test \
        --config "$CFG" \
        --ckpt_path "$ckpt" \
        --data.test_file "$VAL" \
        --data.num_test 200000
done

echo "SCORE EVAL DONE — now run: python python_scripts/score_checkpoints.py $RUN/ckpts"
