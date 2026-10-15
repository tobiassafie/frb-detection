**v6 benchmark - leaky split, comparison only** — hypotheses (one change off the base; base = 128² max/flatten CNN, 3 seeds test 0.977–0.984; RNN base = lstm_v3 recipe, test 0.963 ± 0.003)

| id | change | run_id | params | val | test | dedup FRB recall (149) | v7 test acc | v6 test time-reversed acc | noise acc, +0.05σ probe | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | flatten > GAP | `s1_H1_gap` | 6,433 | 0.942 | 0.968 | 0.966 | 0.443 | 0.688 | 0.182 | confirmed |
| H2 | max > avg pool | `s1_H2_avgpool` | 10,273 | 0.961 | 0.977 | 0.987 | 0.529 | 0.380 | 0.766 | inconclusive |
| H3 | 256 vs 128 | `s1_H3_res256` | 22,561 | 0.964 | 0.984 | 1.000 | 0.527 | 0.477 | 0.240 | weakly confirmed (3 seeds: 0.984 vs 0.979) |
| H3 | 64 vs 128 | `s1_H3_res64` | 7,201 | 0.958 | 0.981 | 0.993 | 0.549 | 0.351 | 0.500 | inconclusive |
| H4 | ½ filters/dense | `s1_H4_half` | 2,601 | 0.951 | 0.977 | 0.987 | 0.547 | 0.393 | 0.169 | confirmed: within 0.5 pt (smallest acceptable) |
| H4 | ¼ filters/dense | `s1_H4_quarter` | 667 | 0.925 | 0.951 | 0.933 | 0.554 | 0.571 | 0.312 | confirmed drop (−2.8 pt) |
| H5a | L2 1e-4 + dropout 0.3 | `s1_H5a_l2_drop` | 10,273 | 0.948 | 0.974 | 0.980 | 0.539 | 0.601 | 0.247 | rejected (no gain) |
| H5b | time-shift ±16 + noise 0.1 aug | `s1_H5b_aug` | 10,273 | 0.909 | 0.938 | 0.906 | 0.525 | 0.841 | 0.994 | rejected for v6 acc (−4 pt); only CNN on this recipe that stops memorising |
| H6 | 64 t × 64 f | `s2_H6_T64_F64` | 13,505 | 0.916 | 0.948 | 0.926 | 0.532 | 0.500 | 0.727 | rejected (−1.5 pt vs 256×256+1×1) |
| H6 | 32 t × 128 f | `s2_H6_T32_F128` | 21,697 | 0.912 | 0.942 | 0.913 | 0.542 | 0.581 | 0.760 | rejected |
| H6 | 64 t × 128 f | `s2_H6_T64_F128` | 21,697 | 0.938 | 0.945 | 0.919 | 0.520 | 0.584 | 0.773 | rejected (best H6) |
| H7 | LSTM64 vs 32 (64×128) | `s2_H7_lstm64` | 51,521 | 0.948 | 0.961 | 0.953 | 0.517 | 0.568 | 0.753 | weakly confirmed (+1.6 pt seed 42; 3-seed 0.953) |
| H7 | GRU32 vs LSTM32 | `s2_H7_gru32` | 16,641 | 0.942 | 0.955 | 0.940 | 0.525 | 0.500 | 0.786 | inconclusive (≈) |
| H8 | Conv1D front end | `s2_H8_conv1d_front` | 35,073 | 0.912 | 0.945 | 0.919 | 0.560 | 0.584 | 0.812 | rejected |