#!/bin/bash
set -euo pipefail
source ~/miniforge3/bin/activate myenv
cd ~/Htautau-mass-regression
~/miniforge3/envs/myenv/bin/salt fit --config ~/Htautau-mass-regression/configs/htautau_regression.yaml
echo "TRAIN EXIT CODE: $?"
