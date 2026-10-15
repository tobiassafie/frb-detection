**v6 benchmark - leaky split, comparison only** — streaming context (v6-trained models; causal 256-bin windows, stride 64; 1 h v7-style test stream, 1,716 bursts per orientation; threshold = lowest with ≤ 10 FA/h on the val stream; 1.000 = target not reachable)

| model | run_id | stream | thr | test FA/h | det S/N≥7 | det S/N≥15 | det all | median TTD ms |
|---|---|---|---|---|---|---|---|---|
| v6a CNN old recipe [2,4,4,4,4] s142 | `s0_baseline_2_4_4_4_4_s142` | v7 stored | 1.000 | 1 | 0.001 | 0.001 | 0.001 | 75 |
| v6a CNN old recipe [2,4,4,4,4] s142 | `s0_baseline_2_4_4_4_4_s142` | v7 physical (high-f first) | 1.000 | 1 | 0.001 | 0.001 | 0.001 | 75 |
| v6a CNN old recipe [2,4,4,4,4] s142 | `s0_baseline_2_4_4_4_4_s142` | v6 tiled | 1.000 | 0 | — | — | 0.701 | 126 |
| v6a CNN base 128 max flatten (seed2) | `v6_sw_r128_max_flatten_d16_seed2` | v7 stored | 1.000 | 561 | 0.325 | 0.344 | 0.326 | 20 |
| v6a CNN base 128 max flatten (seed2) | `v6_sw_r128_max_flatten_d16_seed2` | v7 physical (high-f first) | 1.000 | 561 | 0.325 | 0.344 | 0.326 | 20 |
| v6a CNN base 128 max flatten (seed2) | `v6_sw_r128_max_flatten_d16_seed2` | v6 tiled | 1.000 | 10006 | — | — | 0.994 | 42 |
| v6a CNN 256 input (seed0) | `s1_H3_res256` | v7 stored | 1.000 | 649 | 0.325 | 0.344 | 0.326 | 20 |
| v6a CNN 256 input (seed0) | `s1_H3_res256` | v7 physical (high-f first) | 1.000 | 649 | 0.325 | 0.344 | 0.326 | 20 |
| v6a CNN 256 input (seed0) | `s1_H3_res256` | v6 tiled | 1.000 | 7538 | — | — | 0.981 | 84 |
| v6a CNN half-size 2.6k | `s1_H4_half` | v7 stored | 1.000 | 3887 | 0.299 | 0.311 | 0.301 | 84 |
| v6a CNN half-size 2.6k | `s1_H4_half` | v7 physical (high-f first) | 1.000 | 3887 | 0.300 | 0.313 | 0.301 | 73 |
| v6a CNN half-size 2.6k | `s1_H4_half` | v6 tiled | 1.000 | 8676 | — | — | 0.942 | 42 |
| v6a CNN aug (H5b) | `s1_H5b_aug` | v7 stored | 1.000 | 3425 | 0.286 | 0.301 | 0.288 | 98 |
| v6a CNN aug (H5b) | `s1_H5b_aug` | v7 physical (high-f first) | 1.000 | 3425 | 0.286 | 0.301 | 0.288 | 84 |
| v6a CNN aug (H5b) | `s1_H5b_aug` | v6 tiled | 1.000 | 1388 | — | — | 0.539 | 126 |
| v6a LSTM v3 recipe (seed1) | `s0_lstm_v3_recipe_full150_seed1` | v7 stored | 0.923 | 10 | 0.001 | 0.001 | 0.001 | 169 |
| v6a LSTM v3 recipe (seed1) | `s0_lstm_v3_recipe_full150_seed1` | v7 physical (high-f first) | 0.923 | 10 | 0.002 | 0.001 | 0.002 | 89 |
| v6a LSTM v3 recipe (seed1) | `s0_lstm_v3_recipe_full150_seed1` | v6 tiled | 0.989 | 0 | — | — | 0.136 | 168 |