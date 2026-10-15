"""Stage 3: LSTM / GRU grid on the v7 primary split (1 recurrent layer only).

Input (timesteps, channels) = block-mean of the 256x256 (time x freq) patch by
(256/T, 256/F) -- aggressive downsampling, frequency pooled first.
Grid: cell {lstm, gru} x units {4,8,16,32} x T {16,32,64} x F {16,32,64}; then a
small lr/dropout refinement of the best configs (stage3_refine).

Usage: python scripts/sweep_rnn.py grid|refine > cfgs.json  then run_configs.py
"""
import itertools
import json
import sys

if sys.argv[1] == "grid":
    cfgs = []
    for cell, units, T, F in itertools.product(["lstm", "gru"], [4, 8, 16, 32], [16, 32, 64], [16, 32, 64]):
        cfgs.append(dict(run_id=f"s3_{cell}_u{units}_T{T}_F{F}", kind="rnn", cell=cell, units=units,
                         t_pool=256 // T, f_pool=256 // F, lr=3e-3, dropout=0.1, dense=0, batch=128, epochs=30,
                         patience=8, train_pool="mix_primary", class_weight=False, train_snr_min=3.0, time_flip=False, seed=0,
                         split="primary", track="rnn", stage="stage3_grid"))
    json.dump(cfgs, sys.stdout)
