#!/bin/bash

python3 hypara.py --trials 1 --max_num_cases 1000 --data_dir ./data --num_epochs 1000 --train_fraction 0.8 --random_seed 42 --log_file optuna_history.tsv | tee log
