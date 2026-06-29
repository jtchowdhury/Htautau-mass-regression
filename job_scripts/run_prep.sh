#!/bin/bash
set -euo pipefail
source ~/miniforge3/bin/activate myenv
python -u ~/Htautau-mass-regression/python_scripts/prepare_htautau.py
echo "PREP EXIT CODE: $?"
