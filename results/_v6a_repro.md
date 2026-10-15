**v6 benchmark - leaky split, comparison only** — Stage 0: reproduced vs originally reported (v6 test, 308 samples; lookup baseline 0.984; max achievable 0.984)

| model | how | run_id | params | reported | reproduced | dedup FRB recall (149) | v7 test acc | v6 test time-reversed acc | noise acc, +0.05σ probe |
|---|---|---|---|---|---|---|---|---|---|
| cnn_v1 | saved weights | `saved__cnn_v1` | 8,797 | 0.9740 | 0.9740 | 0.980 | 0.586 | 0.558 | 0.994 |
| group1_f16 | saved weights | `saved__group1_f16` | 33,817 | 0.9740 | 0.9740 | 0.980 | 0.526 | 0.510 | 0.916 |
| group1_f4 | saved weights | `saved__group1_f4` | 8,797 | 0.9643 | 0.9643 | 0.960 | 0.586 | 0.565 | 0.922 |
| group1_f64 | saved weights | `saved__group1_f64` | 133,897 | 0.9740 | 0.9740 | 0.980 | 0.526 | 0.614 | 0.461 |
| group2_b_front_loaded | saved weights | `saved__group2_b_front_loaded` | 8,885 | 0.9643 | 0.9643 | 0.960 | 0.580 | 0.529 | 0.981 |
| lstm_v1 | saved weights | `saved__lstm_v1` | 38,081 | 0.9351 | 0.9351 | 0.899 | 0.488 | 0.500 | 0.935 |
| lstm_v2 | saved weights | `saved__lstm_v2` | 29,953 | 0.9578 | 0.9578 | 0.946 | 0.578 | 0.500 | 0.922 |
| lstm_v3 | saved weights | `saved__lstm_v3` | 29,953 | 0.9643 | 0.9643 | 0.960 | 0.417 | 0.500 | 0.948 |
| seedcheck_baseline_s142 | saved weights | `saved__seedcheck_baseline_s142` | 8,797 | 0.9675 | 0.9675 | 0.966 | 0.586 | 0.604 | 0.734 |
| v6_sw_r128_max_flatten_d16 | saved weights | `saved__v6_sw_r128_max_flatten_d16` | 10,273 | 0.9773 | 0.9773 | 0.987 | 0.511 | 0.513 | 0.084 |
| v6_sw_r128_max_flatten_d16_seed1 | saved weights | `saved__v6_sw_r128_max_flatten_d16_seed1` | 10,273 | 0.9838 | 0.9838 | 1.000 | 0.527 | 0.591 | 0.305 |
| v6_sw_r128_max_flatten_d16_seed2 | saved weights | `saved__v6_sw_r128_max_flatten_d16_seed2` | 10,273 | 0.9773 | 0.9773 | 0.987 | 0.496 | 0.591 | 0.247 |
| seedcheck_baseline_s142 | retrained, original recipe | `s0_baseline_2_4_4_4_4_s142` | 8,797 | 0.9675 | 0.9773 | 0.987 | 0.586 | 0.542 | 1.000 |
| train_cnn.py | retrained, original recipe | `s0_train_cnn_py_4_8_8` | 263,129 | — | 0.9773 | 0.987 | 0.532 | 0.831 | 0.935 |
| lstm_v3 (ES patience 10) | retrained, original recipe | `s0_lstm_v3_recipe` | 29,953 | 0.9643 | 0.7662 | 0.732 | 0.420 | 0.532 | 0.786 |
| lstm_v3 (all 150 epochs) | retrained, original recipe | `s0_lstm_v3_recipe_full150` | 29,953 | 0.9643 | 0.9610 | 0.953 | 0.490 | 0.532 | 0.870 |