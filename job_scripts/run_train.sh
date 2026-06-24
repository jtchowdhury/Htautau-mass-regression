#!/bin/bash
source ~/miniforge3/bin/activate myenv
cd ~/htautau_regression

# Run Salt training
salt train --config ~/htautau_regression/htautau_regression.yaml
