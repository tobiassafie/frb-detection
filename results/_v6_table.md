**v6 benchmark - leaky split, comparison only** (original `lstm_dataset_v6.npz` split, test = 154 FRB + 154 noise; mean ± std over the 3 seeds; *dead* = seed that never left loss ln 2)

| model (architecture source) | run_ids | params | acc | precision | recall | F1 | ROC-AUC | recall@1%FPR | dedup-test recall (149 FRB, 0 noise) | same model on **v7** test (acc) |
|---|---|---|---|---|---|---|---|---|---|---|
| **no-learning baseline: lookup_exact_noise** | — | 0 | 0.984 | 1.000 | 0.968 | 0.983 | 0.984 | 0.968 | 1.000 | — |
| **no-learning baseline: nearest_neighbour_1nn** | — | 0 | 0.899 | 1.000 | 0.799 | 0.888 | 0.899 | 0.799 | 0.826 | — |
| loc_t0041_A_w8k5bn (1 dead seed excluded from mean) | `v6_loc_t0041_A_w8k5bn_s0`, `v6_loc_t0041_A_w8k5bn_s1`, `v6_loc_t0041_A_w8k5bn_s2` | 68,425 | 0.963 ± 0.002 | 1.000 ± 0.000 | 0.925 ± 0.003 | 0.961 ± 0.002 | 0.978 ± 0.003 | 0.961 ± 0.000 | 0.956 ± 0.003 | 0.414 ± 0.000 |
| loc_t0041_b32 (1 dead seed excluded from mean) | `v6_loc_t0041_b32_s0`, `v6_loc_t0041_b32_s1`, `v6_loc_t0041_b32_s2` | 6,433 | 0.963 ± 0.011 | 1.000 ± 0.000 | 0.925 ± 0.023 | 0.961 ± 0.012 | 0.976 ± 0.000 | 0.958 ± 0.003 | 0.956 ± 0.023 | 0.412 ± 0.001 |
| s1mB_t0041_rfi | `v6_s1mB_t0041_rfi_s0`, `v6_s1mB_t0041_rfi_s1`, `v6_s1mB_t0041_rfi_s2` | 6,433 | 0.938 ± 0.005 | 1.000 ± 0.000 | 0.877 ± 0.009 | 0.934 ± 0.005 | 0.978 ± 0.002 | 0.946 ± 0.011 | 0.906 ± 0.009 | 0.559 ± 0.039 |
| s3r_gru_u16_T64_F8_lr0.003_do0.0_d16 | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s0`, `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1`, `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s2` | 1,537 | 0.943 ± 0.015 | 1.000 ± 0.000 | 0.885 ± 0.030 | 0.939 ± 0.017 | 0.966 ± 0.011 | 0.926 ± 0.011 | 0.915 ± 0.031 | 0.586 ± 0.000 |
| s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0 | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s0`, `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1`, `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s2` | 1,617 | 0.940 ± 0.002 | 1.000 ± 0.000 | 0.881 ± 0.003 | 0.937 ± 0.002 | 0.979 ± 0.009 | 0.952 ± 0.012 | 0.911 ± 0.003 | 0.586 ± 0.000 |
| sw_r128_avg_flatten_d16 | `v6_sw_r128_avg_flatten_d16`, `v6_sw_r128_avg_flatten_d16_seed1`, `v6_sw_r128_avg_flatten_d16_seed2` | 10,273 | 0.976 ± 0.002 | 1.000 ± 0.000 | 0.952 ± 0.003 | 0.976 ± 0.002 | 0.980 ± 0.002 | 0.963 ± 0.003 | 0.984 ± 0.003 | 0.563 ± 0.017 |
| sw_r128_avg_flatten_d32 | `v6_sw_r128_avg_flatten_d32` | 14,401 | 0.971 | 1.000 | 0.942 | 0.970 | 0.986 | 0.968 | 0.973 | 0.586 |
| sw_r128_avg_flatten_d8 | `v6_sw_r128_avg_flatten_d8` | 8,209 | 0.971 | 1.000 | 0.942 | 0.970 | 0.984 | 0.968 | 0.973 | 0.586 |
| sw_r128_avg_gap_d16 (all seeds dead) | `v6_sw_r128_avg_gap_d16` | 6,433 | 0.526 | 0.516 | 0.851 | 0.642 | 0.563 | 0.065 | 0.852 | 0.414 |
| sw_r128_max_flatten_d16 | `v6_sw_r128_max_flatten_d16`, `v6_sw_r128_max_flatten_d16_seed1`, `v6_sw_r128_max_flatten_d16_seed2` | 10,273 | 0.979 ± 0.003 | 1.000 ± 0.000 | 0.959 ± 0.006 | 0.979 ± 0.003 | 0.983 ± 0.001 | 0.968 ± 0.000 | 0.991 ± 0.006 | 0.511 ± 0.012 |
| sw_r128_max_gap_d16 | `v6_sw_r128_max_gap_d16` | 6,433 | 0.971 | 1.000 | 0.942 | 0.970 | 0.980 | 0.968 | 0.973 | 0.415 |
| sw_r64_avg_flatten_d16 | `v6_sw_r64_avg_flatten_d16`, `v6_sw_r64_avg_flatten_d16_seed1`, `v6_sw_r64_avg_flatten_d16_seed2` | 7,201 | 0.970 ± 0.002 | 1.000 ± 0.000 | 0.939 ± 0.003 | 0.969 ± 0.002 | 0.983 ± 0.004 | 0.965 ± 0.003 | 0.971 ± 0.003 | 0.591 ± 0.009 |
| sw_r64_avg_gap_d16 | `v6_sw_r64_avg_gap_d16` | 6,433 | 0.961 | 1.000 | 0.922 | 0.959 | 0.980 | 0.961 | 0.953 | 0.586 |
| sw_r64_max_flatten_d16 | `v6_sw_r64_max_flatten_d16` | 7,201 | 0.977 | 1.000 | 0.955 | 0.977 | 0.982 | 0.961 | 0.987 | 0.539 |
| sw_r64_max_gap_d16 | `v6_sw_r64_max_gap_d16` | 6,433 | 0.974 | 1.000 | 0.948 | 0.973 | 0.975 | 0.961 | 0.980 | 0.503 |
| QAT8 s1mB_t0041_rfi | `qat_v6_s1mB_t0041_rfi_s0_8b`, `qat_v6_s1mB_t0041_rfi_s1_8b`, `qat_v6_s1mB_t0041_rfi_s2_8b` | 6,433 | 0.944 ± 0.008 | 1.000 ± 0.000 | 0.887 ± 0.015 | 0.940 ± 0.009 | 0.968 ± 0.009 | 0.935 ± 0.019 | 0.917 ± 0.016 | 0.558 ± 0.040 |

**v7-trained models evaluated on the v6 test set** (cross-dataset; input renormalised per channel as in v7 training)

| v7 model | params | v6 acc | precision | recall | F1 | ROC-AUC | recall@1%FPR | dedup recall | its own v7 test acc |
|---|---|---|---|---|---|---|---|---|---|
| `s1mB_t0041_rfi` | 6,433 | 0.500 | 0.500 | 1.000 | 0.667 | 0.488 | 0.000 | 1.000 | 0.944 |
| `loc_t0041_b32` | 6,433 | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 | 0.000 | 1.000 | 0.915 |
| `loc_t0041_A_w8k5bn` | 68,425 | 0.500 | 0.500 | 1.000 | 0.667 | 0.571 | 0.000 | 1.000 | 0.924 |
| `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | 1,617 | 0.468 | 0.472 | 0.545 | 0.506 | 0.519 | 0.136 | 0.544 | 0.835 |
| `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | 1,537 | 0.526 | 0.516 | 0.844 | 0.640 | 0.608 | 0.247 | 0.846 | 0.781 |
| `qat_s1mB_t0041_rfi_8b` | 6,433 | 0.500 | 0.500 | 1.000 | 0.667 | 0.519 | 0.000 | 1.000 | 0.943 |