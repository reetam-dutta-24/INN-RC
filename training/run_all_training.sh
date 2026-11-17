#!/bin/bash
# Trains every model used in the paper (CPU, about 45 min in total on 2 cores).
set -e
cd "$(dirname "$0")"
# proposed IC-INN, quality-oriented (main model)
[ -f ../models/icinn.mat ] || python3 train_icinn.py --iters 12000 --w_msg 10 --w_of 5 --scales 2,4,8 --out ../models/icinn > ../models/icinn_train.txt 2>&1
# proposed IC-INN, robustness-oriented (higher message weight)
python3 train_icinn.py --iters 12000 --w_msg 30 --w_of 5 --scales 2,4,8 --out ../models/icinn_r > ../models/icinn_r_train.txt 2>&1
# baseline float INN + residual compensation, payload s = 4
python3 train_inn.py --mode ours --s 4 --iters 6000 --w_msg 0.02 --out ../models/finn_s4 > ../models/finn_s4_train.txt 2>&1
