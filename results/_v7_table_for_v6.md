**v7 dataset (file-grouped, leak-free; test = 6 held-out source files, noise + FRB S/N ≥ 7)**

| model | run_id | trained on | params | acc | precision | recall | F1 | ROC-AUC | recall@1%FPR |
|---|---|---|---|---|---|---|---|---|---|
| v7 synthesis CNN | `s1mB_t0041_rfi` | v7 train | 6,433 | 0.944 | 0.994 | 0.871 | 0.928 | 0.972 | 0.888 |
| v7 accuracy-track CNN | `loc_t0041_b32` | v7 train | 6,433 | 0.915 | 0.893 | 0.905 | 0.899 | 0.962 | 0.853 |
| v7 68k BN CNN | `loc_t0041_A_w8k5bn` | v7 train | 68,425 | 0.924 | 0.974 | 0.838 | 0.901 | 0.968 | 0.831 |
| v7 8-bit QAT CNN | `qat_s1mB_t0041_rfi_8b` | v7 train | 6,433 | 0.943 | 0.963 | 0.895 | 0.928 | 0.971 | 0.870 |
| v7 LSTM | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | v7 train | 1,617 | 0.835 | 0.840 | 0.744 | 0.789 | 0.873 | 0.593 |
| v7 GRU | `s3r_gru_u16_T64_F8_lr0.003_do0.0_d16` | v7 train | 1,537 | 0.781 | 0.734 | 0.738 | 0.736 | 0.852 | 0.531 |
| stream-trained CNN (t0041 cfg) | `strm_t0041_s0` | v7 stream-sampled pool | 6,433 | 0.798 | 0.717 | 0.846 | 0.776 | 0.898 | 0.612 |
| stream-trained CNN (t0041 cfg) | `strm_t0041_s1` | v7 stream-sampled pool | 6,433 | 0.880 | 0.884 | 0.817 | 0.849 | 0.930 | 0.661 |
| stream-trained CNN (t0041 cfg) | `strm_t0041_s2` | v7 stream-sampled pool | 6,433 | 0.857 | 0.839 | 0.810 | 0.825 | 0.923 | 0.682 |
| v6-trained (cross) | `v6_loc_t0041_A_w8k5bn_s0` | v6 train | 68,425 | 0.414 | 0.414 | 1.000 | 0.585 | 0.494 | 0.000 |
| v6-trained (cross) | `v6_loc_t0041_A_w8k5bn_s1` | v6 train | 68,425 | 0.414 | 0.414 | 1.000 | 0.585 | 0.500 | 0.000 |
| v6-trained (cross) | `v6_loc_t0041_A_w8k5bn_s2` | v6 train | 68,425 | 0.557 | 0.413 | 0.166 | 0.237 | 0.413 | 0.002 |
| v6-trained (cross) | `v6_loc_t0041_b32_s0` | v6 train | 6,433 | 0.414 | 0.414 | 1.000 | 0.585 | 0.510 | 0.019 |
| v6-trained (cross) | `v6_loc_t0041_b32_s1` | v6 train | 6,433 | 0.411 | 0.412 | 0.988 | 0.581 | 0.441 | 0.002 |
| v6-trained (cross) | `v6_loc_t0041_b32_s2` | v6 train | 6,433 | 0.412 | 0.411 | 0.977 | 0.579 | 0.506 | 0.000 |
| v6-trained (cross) | `v6_s1mB_t0041_rfi_s0` | v6 train | 6,433 | 0.586 | 0.000 | 0.000 | 0.000 | 0.370 | 0.001 |
| v6-trained (cross) | `v6_s1mB_t0041_rfi_s1` | v6 train | 6,433 | 0.503 | 0.144 | 0.041 | 0.063 | 0.322 | 0.000 |
| v6-trained (cross) | `v6_s1mB_t0041_rfi_s2` | v6 train | 6,433 | 0.586 | 1.000 | 0.001 | 0.002 | 0.478 | 0.008 |
| v6-trained (cross) | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s0` | v6 train | 1,537 | 0.586 | 1.000 | 0.000 | 0.001 | 0.438 | 0.014 |
| v6-trained (cross) | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s1` | v6 train | 1,537 | 0.586 | 0.000 | 0.000 | 0.000 | 0.449 | 0.017 |
| v6-trained (cross) | `v6_s3r_gru_u16_T64_F8_lr0.003_do0.0_d16_s2` | v6 train | 1,537 | 0.586 | 0.000 | 0.000 | 0.000 | 0.430 | 0.012 |
| v6-trained (cross) | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s0` | v6 train | 1,617 | 0.586 | 0.000 | 0.000 | 0.000 | 0.420 | 0.004 |
| v6-trained (cross) | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s1` | v6 train | 1,617 | 0.586 | 0.000 | 0.000 | 0.000 | 0.388 | 0.003 |
| v6-trained (cross) | `v6_s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0_s2` | v6 train | 1,617 | 0.586 | 1.000 | 0.000 | 0.001 | 0.493 | 0.014 |
| v6-trained (cross) | `v6_sw_r128_avg_flatten_d16` | v6 train | 10,273 | 0.557 | 0.412 | 0.165 | 0.236 | 0.492 | 0.000 |
| v6-trained (cross) | `v6_sw_r128_avg_flatten_d16_seed1` | v6 train | 10,273 | 0.545 | 0.432 | 0.318 | 0.366 | 0.524 | 0.000 |
| v6-trained (cross) | `v6_sw_r128_avg_flatten_d16_seed2` | v6 train | 10,273 | 0.586 | 0.000 | 0.000 | 0.000 | 0.555 | 0.029 |
| v6-trained (cross) | `v6_sw_r128_avg_flatten_d32` | v6 train | 14,401 | 0.586 | 0.000 | 0.000 | 0.000 | 0.498 | 0.009 |
| v6-trained (cross) | `v6_sw_r128_avg_flatten_d8` | v6 train | 8,209 | 0.586 | 1.000 | 0.000 | 0.001 | 0.500 | 0.017 |
| v6-trained (cross) | `v6_sw_r128_avg_gap_d16` | v6 train | 6,433 | 0.414 | 0.414 | 1.000 | 0.585 | 0.623 | 0.071 |
| v6-trained (cross) | `v6_sw_r128_max_flatten_d16` | v6 train | 10,273 | 0.511 | 0.419 | 0.471 | 0.444 | 0.494 | 0.000 |
| v6-trained (cross) | `v6_sw_r128_max_flatten_d16_seed1` | v6 train | 10,273 | 0.527 | 0.407 | 0.316 | 0.356 | 0.462 | 0.000 |
| v6-trained (cross) | `v6_sw_r128_max_flatten_d16_seed2` | v6 train | 10,273 | 0.496 | 0.409 | 0.491 | 0.447 | 0.499 | 0.000 |
| v6-trained (cross) | `v6_sw_r128_max_gap_d16` | v6 train | 6,433 | 0.415 | 0.414 | 0.995 | 0.585 | 0.504 | 0.000 |
| v6-trained (cross) | `v6_sw_r64_avg_flatten_d16` | v6 train | 7,201 | 0.584 | 0.424 | 0.017 | 0.032 | 0.279 | 0.009 |
| v6-trained (cross) | `v6_sw_r64_avg_flatten_d16_seed1` | v6 train | 7,201 | 0.586 | 0.000 | 0.000 | 0.000 | 0.482 | 0.158 |
| v6-trained (cross) | `v6_sw_r64_avg_flatten_d16_seed2` | v6 train | 7,201 | 0.604 | 1.000 | 0.042 | 0.082 | 0.458 | 0.089 |
| v6-trained (cross) | `v6_sw_r64_avg_gap_d16` | v6 train | 6,433 | 0.586 | 0.000 | 0.000 | 0.000 | 0.391 | 0.015 |
| v6-trained (cross) | `v6_sw_r64_max_flatten_d16` | v6 train | 7,201 | 0.539 | 0.389 | 0.201 | 0.265 | 0.357 | 0.000 |
| v6-trained (cross) | `v6_sw_r64_max_gap_d16` | v6 train | 6,433 | 0.503 | 0.426 | 0.575 | 0.489 | 0.500 | 0.000 |
| v6-trained (cross) | `qat_v6_s1mB_t0041_rfi_s0_8b` | v6 train | 6,433 | 0.586 | 0.000 | 0.000 | 0.000 | 0.377 | 0.004 |
| v6-trained (cross) | `qat_v6_s1mB_t0041_rfi_s1_8b` | v6 train | 6,433 | 0.500 | 0.100 | 0.026 | 0.041 | 0.322 | 0.000 |
| v6-trained (cross) | `qat_v6_s1mB_t0041_rfi_s2_8b` | v6 train | 6,433 | 0.586 | 1.000 | 0.001 | 0.002 | 0.479 | 0.004 |