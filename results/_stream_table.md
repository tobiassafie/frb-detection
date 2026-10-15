**Threshold: val_fa<=10/h** (lowest threshold with ≤ 10 false alarms/h on the matching VAL-files stream)

| model | run_id | stream | thr | test FA/h | det S/N≥7 | det S/N≥15 | det all S/N | median time-to-detect (ms, S/N≥7) | p90 TTD | median frac. of band arrived at detection | detected before half the band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v7: s1mB_t0041_rfi | `s1mB_t0041_rfi` | v7 stored (as trained) | 0.761 | 80 | 0.827 | 0.962 | 0.593 | 58 | 104 | 1.00 | 0.22 |
| v7: s1mB_t0041_rfi | `s1mB_t0041_rfi` | v7 physical (high-f first) | 0.761 | 80 | 0.019 | 0.018 | 0.016 | 145 | 286 | 1.00 | 0.00 |
| v7: s1mB_t0041_rfi | `s1mB_t0041_rfi` | v6 tiled (comparison only) | 1.000 | 6960 | — | — | 0.987 | 42 | 42 | 1.00 | 0.00 |
| v7: loc_t0041_b32 | `loc_t0041_b32` | v7 stored (as trained) | 0.880 | 207 | 0.818 | 0.963 | 0.595 | 65 | 106 | 1.00 | 0.20 |
| v7: loc_t0041_b32 | `loc_t0041_b32` | v7 physical (high-f first) | 0.880 | 207 | 0.049 | 0.048 | 0.043 | 169 | 329 | 1.00 | 0.02 |
| v7: loc_t0041_b32 | `loc_t0041_b32` | v6 tiled (comparison only) | 1.000 | 8772 | — | — | 0.987 | 42 | 42 | 1.00 | 0.00 |
| v7: 68k BN CNN | `loc_t0041_A_w8k5bn` | v7 stored (as trained) | 0.784 | 82 | 0.774 | 0.931 | 0.553 | 71 | 127 | 1.00 | 0.11 |
| v7: 68k BN CNN | `loc_t0041_A_w8k5bn` | v7 physical (high-f first) | 0.784 | 82 | 0.012 | 0.008 | 0.012 | 176 | 349 | 1.00 | 0.00 |
| v7: 68k BN CNN | `loc_t0041_A_w8k5bn` | v6 tiled (comparison only) | 1.000 | 11664 | — | — | 0.922 | 42 | 210 | 1.00 | 0.00 |
| v7: LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v7 stored (as trained) | 0.999 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v7: LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v7 physical (high-f first) | 0.999 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v7: LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v6 tiled (comparison only) | 0.999 | 0 | — | — | 0.000 | — | — | — | — |
| v7: GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v7 stored (as trained) | 0.999 | 22 | 0.168 | 0.262 | 0.115 | 76 | 116 | 1.00 | 0.00 |
| v7: GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v7 physical (high-f first) | 0.999 | 22 | 0.050 | 0.082 | 0.036 | 65 | 86 | 1.00 | 0.00 |
| v7: GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v6 tiled (comparison only) | 1.000 | 0 | — | — | 0.000 | — | — | — | — |
| v7 stream-trained: t0041 cfg | `strm_t0041_s1` | v7 stored (as trained) | 0.784 | 494 | 0.782 | 0.926 | 0.571 | 48 | 98 | 0.81 | 0.30 |
| v7 stream-trained: t0041 cfg | `strm_t0041_s1` | v7 physical (high-f first) | 0.784 | 494 | 0.775 | 0.922 | 0.569 | 42 | 96 | 0.86 | 0.18 |
| v7 stream-trained: t0041 cfg | `strm_t0041_s1` | v6 tiled (comparison only) | 1.000 | 11394 | — | — | 0.974 | 42 | 210 | 1.00 | 0.00 |
| v6: t0041 cfg | `v6_s1mB_t0041_rfi_s2` | v7 stored (as trained) | 1.000 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: t0041 cfg | `v6_s1mB_t0041_rfi_s2` | v7 physical (high-f first) | 1.000 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: t0041 cfg | `v6_s1mB_t0041_rfi_s2` | v6 tiled (comparison only) | 1.000 | 0 | — | — | 0.578 | 126 | 210 | 1.00 | 0.00 |
| v6: b32 cfg | `v6_loc_t0041_b32_s1` | v7 stored (as trained) | 1.000 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: b32 cfg | `v6_loc_t0041_b32_s1` | v7 physical (high-f first) | 1.000 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: b32 cfg | `v6_loc_t0041_b32_s1` | v6 tiled (comparison only) | 1.000 | 6728 | — | — | 0.922 | 84 | 168 | 1.00 | 0.00 |
| v6: LSTM | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v7 stored (as trained) | 0.088 | 8 | 0.169 | 0.236 | 0.117 | 24 | 48 | 0.51 | 0.49 |
| v6: LSTM | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v7 physical (high-f first) | 0.088 | 8 | 0.153 | 0.208 | 0.107 | 26 | 66 | 0.83 | 0.17 |
| v6: LSTM | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v6 tiled (comparison only) | 1.000 | 0 | — | — | 0.149 | 168 | 168 | 1.00 | 0.00 |
| v6: GRU | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v7 stored (as trained) | 0.050 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: GRU | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v7 physical (high-f first) | 0.050 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: GRU | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v6 tiled (comparison only) | 1.000 | 7133 | — | — | 0.818 | 42 | 210 | 1.00 | 0.00 |
| v7: 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v7 stored (as trained) | 0.880 | 84 | 0.811 | 0.958 | 0.580 | 62 | 110 | 1.00 | 0.21 |
| v7: 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v7 physical (high-f first) | 0.880 | 85 | 0.024 | 0.023 | 0.019 | 145 | 308 | 1.00 | 0.04 |
| v7: 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v6 tiled (comparison only) | 1.000 | 9909 | — | — | 0.987 | 42 | 42 | 1.00 | 0.00 |
| v6: 8-bit QAT CNN | `qat_v6_s1mB_t0041_rfi_s2_8b` | v7 stored (as trained) | 1.000 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: 8-bit QAT CNN | `qat_v6_s1mB_t0041_rfi_s2_8b` | v7 physical (high-f first) | 1.000 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: 8-bit QAT CNN | `qat_v6_s1mB_t0041_rfi_s2_8b` | v6 tiled (comparison only) | 1.000 | 0 | — | — | 0.604 | 126 | 210 | 1.00 | 0.00 |
| v6: 68k BN CNN | `v6_loc_t0041_A_w8k5bn_s0` | v7 stored (as trained) | 1.000 | 2201 | 1.000 | 1.000 | 1.000 | 22 | 39 | 0.26 | 0.70 |
| v6: 68k BN CNN | `v6_loc_t0041_A_w8k5bn_s0` | v7 physical (high-f first) | 1.000 | 2201 | 1.000 | 1.000 | 1.000 | 22 | 39 | 0.46 | 0.53 |
| v6: 68k BN CNN | `v6_loc_t0041_A_w8k5bn_s0` | v6 tiled (comparison only) | 1.000 | 0 | — | — | 0.110 | 168 | 210 | 1.00 | 0.00 |
| v6: sweep winner (r128 avg flatten) | `v6_sw_r128_avg_flatten_d16` | v7 stored (as trained) | 1.000 | 292 | 0.181 | 0.180 | 0.170 | 20 | 38 | 0.60 | 0.42 |
| v6: sweep winner (r128 avg flatten) | `v6_sw_r128_avg_flatten_d16` | v7 physical (high-f first) | 1.000 | 292 | 0.181 | 0.180 | 0.170 | 20 | 38 | 0.69 | 0.33 |
| v6: sweep winner (r128 avg flatten) | `v6_sw_r128_avg_flatten_d16` | v6 tiled (comparison only) | 1.000 | 11028 | — | — | 0.994 | 42 | 168 | 1.00 | 0.00 |

**Threshold: fixed_0.5** 

| model | run_id | stream | thr | test FA/h | det S/N≥7 | det S/N≥15 | det all S/N | median time-to-detect (ms, S/N≥7) | p90 TTD | median frac. of band arrived at detection | detected before half the band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v7: s1mB_t0041_rfi | `s1mB_t0041_rfi` | v7 stored (as trained) | 0.500 | 453 | 0.891 | 0.987 | 0.668 | 53 | 103 | 0.99 | 0.25 |
| v7: s1mB_t0041_rfi | `s1mB_t0041_rfi` | v7 physical (high-f first) | 0.500 | 453 | 0.083 | 0.085 | 0.075 | 166 | 392 | 1.00 | 0.03 |
| v7: s1mB_t0041_rfi | `s1mB_t0041_rfi` | v6 tiled (comparison only) | 0.500 | 2988 | — | — | 1.000 | 42 | 42 | 1.00 | 0.00 |
| v7: loc_t0041_b32 | `loc_t0041_b32` | v7 stored (as trained) | 0.500 | 2566 | 0.920 | 0.992 | 0.751 | 52 | 101 | 0.99 | 0.28 |
| v7: loc_t0041_b32 | `loc_t0041_b32` | v7 physical (high-f first) | 0.500 | 2566 | 0.280 | 0.276 | 0.279 | 142 | 330 | 1.00 | 0.12 |
| v7: loc_t0041_b32 | `loc_t0041_b32` | v6 tiled (comparison only) | 0.500 | 2988 | — | — | 1.000 | 42 | 42 | 1.00 | 0.00 |
| v7: 68k BN CNN | `loc_t0041_A_w8k5bn` | v7 stored (as trained) | 0.500 | 678 | 0.853 | 0.965 | 0.645 | 66 | 119 | 1.00 | 0.14 |
| v7: 68k BN CNN | `loc_t0041_A_w8k5bn` | v7 physical (high-f first) | 0.500 | 678 | 0.089 | 0.088 | 0.089 | 179 | 362 | 1.00 | 0.09 |
| v7: 68k BN CNN | `loc_t0041_A_w8k5bn` | v6 tiled (comparison only) | 0.500 | 3605 | — | — | 1.000 | 42 | 42 | 1.00 | 0.00 |
| v7: LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v7 stored (as trained) | 0.500 | 4362 | 0.802 | 0.914 | 0.717 | 82 | 160 | 1.00 | 0.14 |
| v7: LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v7 physical (high-f first) | 0.500 | 4362 | 0.693 | 0.801 | 0.631 | 81 | 245 | 1.00 | 0.08 |
| v7: LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v6 tiled (comparison only) | 0.500 | 16696 | — | — | 0.987 | 84 | 247 | 1.00 | 0.00 |
| v7: GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v7 stored (as trained) | 0.500 | 5356 | 0.845 | 0.942 | 0.771 | 74 | 161 | 1.00 | 0.21 |
| v7: GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v7 physical (high-f first) | 0.500 | 5356 | 0.706 | 0.789 | 0.666 | 78 | 245 | 1.00 | 0.12 |
| v7: GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v6 tiled (comparison only) | 0.500 | 11471 | — | — | 0.987 | 42 | 168 | 1.00 | 0.00 |
| v7 stream-trained: t0041 cfg | `strm_t0041_s1` | v7 stored (as trained) | 0.500 | 1760 | 0.888 | 0.963 | 0.703 | 43 | 95 | 0.75 | 0.35 |
| v7 stream-trained: t0041 cfg | `strm_t0041_s1` | v7 physical (high-f first) | 0.500 | 1760 | 0.878 | 0.976 | 0.693 | 39 | 98 | 0.81 | 0.23 |
| v7 stream-trained: t0041 cfg | `strm_t0041_s1` | v6 tiled (comparison only) | 0.500 | 2988 | — | — | 1.000 | 42 | 42 | 1.00 | 0.00 |
| v6: t0041 cfg | `v6_s1mB_t0041_rfi_s2` | v7 stored (as trained) | 0.500 | 243 | 0.049 | 0.057 | 0.047 | 156 | 350 | 1.00 | 0.21 |
| v6: t0041 cfg | `v6_s1mB_t0041_rfi_s2` | v7 physical (high-f first) | 0.500 | 243 | 0.055 | 0.066 | 0.050 | 70 | 348 | 0.93 | 0.22 |
| v6: t0041 cfg | `v6_s1mB_t0041_rfi_s2` | v6 tiled (comparison only) | 0.500 | 9967 | — | — | 0.987 | 42 | 168 | 1.00 | 0.00 |
| v6: b32 cfg | `v6_loc_t0041_b32_s1` | v7 stored (as trained) | 0.500 | 1794 | 1.000 | 1.000 | 1.000 | 21 | 39 | 0.25 | 0.71 |
| v6: b32 cfg | `v6_loc_t0041_b32_s1` | v7 physical (high-f first) | 0.500 | 1794 | 1.000 | 1.000 | 1.000 | 21 | 39 | 0.45 | 0.55 |
| v6: b32 cfg | `v6_loc_t0041_b32_s1` | v6 tiled (comparison only) | 0.500 | 15231 | — | — | 1.000 | 42 | 126 | 1.00 | 0.00 |
| v6: LSTM | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v7 stored (as trained) | 0.500 | 0 | 0.001 | 0.001 | 0.001 | 75 | 75 | 1.00 | 0.00 |
| v6: LSTM | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v7 physical (high-f first) | 0.500 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: LSTM | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v6 tiled (comparison only) | 0.500 | 15500 | — | — | 1.000 | 42 | 42 | 1.00 | 0.00 |
| v6: GRU | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v7 stored (as trained) | 0.500 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: GRU | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v7 physical (high-f first) | 0.500 | 0 | 0.000 | 0.000 | 0.000 | — | — | — | — |
| v6: GRU | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v6 tiled (comparison only) | 0.500 | 14883 | — | — | 1.000 | 42 | 84 | 1.00 | 0.00 |
| v7: 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v7 stored (as trained) | 0.500 | 1460 | 0.923 | 0.994 | 0.735 | 50 | 103 | 0.93 | 0.28 |
| v7: 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v7 physical (high-f first) | 0.500 | 1461 | 0.202 | 0.202 | 0.193 | 126 | 365 | 1.00 | 0.09 |
| v7: 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v6 tiled (comparison only) | 0.500 | 2988 | — | — | 1.000 | 42 | 42 | 1.00 | 0.00 |
| v6: 8-bit QAT CNN | `qat_v6_s1mB_t0041_rfi_s2_8b` | v7 stored (as trained) | 0.500 | 202 | 0.041 | 0.045 | 0.039 | 149 | 355 | 1.00 | 0.21 |
| v6: 8-bit QAT CNN | `qat_v6_s1mB_t0041_rfi_s2_8b` | v7 physical (high-f first) | 0.500 | 202 | 0.049 | 0.058 | 0.044 | 58 | 351 | 0.93 | 0.23 |
| v6: 8-bit QAT CNN | `qat_v6_s1mB_t0041_rfi_s2_8b` | v6 tiled (comparison only) | 0.500 | 10006 | — | — | 0.987 | 42 | 168 | 1.00 | 0.00 |
| v6: 68k BN CNN | `v6_loc_t0041_A_w8k5bn_s0` | v7 stored (as trained) | 0.500 | 1812 | 1.000 | 1.000 | 1.000 | 21 | 39 | 0.25 | 0.71 |
| v6: 68k BN CNN | `v6_loc_t0041_A_w8k5bn_s0` | v7 physical (high-f first) | 0.500 | 1812 | 1.000 | 1.000 | 1.000 | 21 | 39 | 0.45 | 0.55 |
| v6: 68k BN CNN | `v6_loc_t0041_A_w8k5bn_s0` | v6 tiled (comparison only) | 0.500 | 8560 | — | — | 0.981 | 84 | 168 | 1.00 | 0.00 |
| v6: sweep winner (r128 avg flatten) | `v6_sw_r128_avg_flatten_d16` | v7 stored (as trained) | 0.500 | 292 | 0.181 | 0.180 | 0.170 | 20 | 38 | 0.60 | 0.42 |
| v6: sweep winner (r128 avg flatten) | `v6_sw_r128_avg_flatten_d16` | v7 physical (high-f first) | 0.500 | 292 | 0.181 | 0.180 | 0.170 | 20 | 38 | 0.69 | 0.33 |
| v6: sweep winner (r128 avg flatten) | `v6_sw_r128_avg_flatten_d16` | v6 tiled (comparison only) | 0.500 | 14633 | — | — | 1.000 | 42 | 84 | 1.00 | 0.00 |
