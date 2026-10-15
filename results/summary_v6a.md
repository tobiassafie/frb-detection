# v6a — winning models on the ORIGINAL v6 split (2026-10-07)

All numbers are on `augmented_dataset/lstm_dataset_v6.npz` as on disk (byte-identical to
`fpga_handoff/X_test.npy`), original train/val/test split, **v6 benchmark - leaky split, comparison
only**. Every run is in `results/v6a_runs.csv` (`results/runs_v6a/*.json`); reasoning, pre-registered
hypotheses and incidents are in `results/NOTES_v6a.md`. 26 new training runs, ~1.5 h wall clock.

## TL;DR

1. **Every saved historical model reproduces exactly.** Retraining with the original recipes also
   reproduces, within run-to-run noise:
   * `cnn_v1` 0.9740, `seedcheck_baseline_s142` 0.9675, `group1_f64` 0.9740 and `lstm_v3` 0.9643
     all match bit-for-bit from their saved weights;
   * the [2,4,4,4,4] retrain scores 0.977 vs 0.9675;
   * the lstm_v3 retrain scores 0.961 vs 0.964, but only when trained for all 150 epochs as the
     original was. Its early stop would otherwise fire at epoch 49.

   No ~99 % v6 model exists anywhere in the repo or its history, and `train_cnn(2).py` does not
   exist.
2. **The v6 test set has a hard ceiling of 303/308 = 0.9838.** 5 test samples labelled FRB are
   *byte-identical* to noise images (the injected burst landed outside the patch). Identical
   inputs must get identical outputs, so no model can exceed 0.9838 on test (0.9675 on val; 36
   such samples are in train). **99 % is mathematically impossible on this split.**
3. **Best CNN: 0.984 ± 0.000 over 3 seeds = the ceiling.** It is `s1_H3_res256*`: 256 × 256 input,
   Conv[4,8,16,16,16] + MaxPool + Flatten + Dense16, 22,561 params, val 0.9675 = also the val
   ceiling. Every real FRB in the deduplicated test is detected (149/149), with precision 1.000.
4. **Smallest CNN within 0.5 pt:** `s1_H4_half`, 2,601 params, 0.977. Quartering the model (667
   params) drops it to 0.951.
5. **Best many-to-one RNN: lstm_v3 recipe**, 0.963 ± 0.003 over 3 seeds. It reads 256 time steps
   × 256 channels, applies a 1×1 Conv1D(64), then LSTM(32) → Dense32, for 29,953 params.
   * Feeding fewer time steps with 64–128 channels per step (H6) or a Conv1D front end (H8) did
     not beat it.
   * LSTM64 on 64 × 128 reaches 0.953.
   * GRU ≈ LSTM.
6. **The caveat beside every number.** The models that reach the ceiling do so partly by
   recognising the 5 memorised noise images:
   * adding invisible N(0, 0.05) noise to the test noise drops the best CNN's noise accuracy
     to 0.42;
   * the old-recipe CNN (0.966 ± 0.008) stays at 1.00, but gives up ~2 pt of v6 test;
   * on the v7 test set and on time-reversed v6 every v6 model is near chance.

   v6 accuracy shows how well a model fits *this* dataset, not how well it detects FRBs in
   new data.
7. **Synthesis.** No Vitis and no C++ compiler on this machine, so C-sim was not run.
   * hls4ml projects are written for the best CNN, the smallest CNN, the robust CNN and both
     RNNs (`hls_projects_v6a/`).
   * Fixed-point emulation at ap_fixed<16,6> matches float for the CNNs; it costs 0.3 pt (lstm_v3)
     and 1 pt (LSTM64).
   * All hardware numbers are `est_`.
   * The RNNs carry real synthesis risk: their io_parallel inputs are 65,536 and 8,192 values,
     ~7–13k multipliers at RF 4.


**v6 benchmark - leaky split, comparison only** — headline models (test metrics = mean ± std over the listed seeds; context columns for the exported run)

| model | run_ids | exported run | params | MACs | acc | precision | recall | F1 | ROC-AUC | recall@1%FPR | lookup baseline | max achievable | dedup FRB recall (149) | v7 test acc | v6 test time-reversed acc | noise acc, +0.05σ probe | fixed-point emu acc (16,6) | est. mults (RF4) | est. latency µs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Best CNN (by 3-seed val) | `s1_H3_res256`, `s1_H3_res256_seed1`, `s1_H3_res256_seed2` | `s1_H3_res256_seed2` | 22,561 | 14,762,000 | 0.984 ± 0.000 | 1.000 ± 0.000 | 0.968 ± 0.000 | 0.983 ± 0.000 | 0.985 ± 0.002 | 0.968 ± 0.000 | 0.984 | 0.9838 | 1.000 | 0.525 | 0.614 | 0.422 | 0.984 | 5624 | 1441.8 |
| Best smallest CNN within 0.5 pt (½ filters/dense) | `s1_H4_half` | `s1_H4_half` | 2,601 | 1,070,088 | 0.977 | 1.000 | 0.955 | 0.977 | 0.982 | 0.968 | 0.984 | 0.9838 | 0.987 | 0.547 | 0.393 | 0.169 | 0.977 | 648 | 360.4 |
| Most robust CNN (old v6 recipe, 3 seeds) | `s0_baseline_2_4_4_4_4_s142`, `s0_baseline_2_4_4_4_4_seed1`, `s0_baseline_2_4_4_4_4_seed2` | `s0_baseline_2_4_4_4_4_s142` | 8,797 | 3,141,664 | 0.966 ± 0.008 | 1.000 ± 0.000 | 0.933 ± 0.015 | 0.965 ± 0.008 | 0.982 ± 0.001 | 0.968 ± 0.000 | 0.984 | 0.9838 | 0.987 | 0.586 | 0.542 | 1.000 | 0.977 | 2194 | 1441.8 |
| Best RNN (lstm_v3 recipe, 3 seeds) | `s0_lstm_v3_recipe_full150`, `s0_lstm_v3_recipe_full150_seed1`, `s0_lstm_v3_recipe_full150_seed2` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | 0.963 ± 0.003 | 1.000 ± 0.000 | 0.926 ± 0.006 | 0.962 ± 0.003 | 0.979 ± 0.002 | 0.955 ± 0.005 | 0.984 | 0.9838 | 0.953 | 0.425 | 0.500 | 0.857 | 0.958 | 7432 | 20.5 |
| Smaller-input RNN (LSTM64, 64×128, 3 seeds) | `s2_H7_lstm64`, `s2_H7_lstm64_seed1`, `s2_H7_lstm64_seed2` | `s2_H7_lstm64` | 51,521 | 3,147,808 | 0.953 ± 0.007 | 1.000 ± 0.000 | 0.907 ± 0.013 | 0.951 ± 0.007 | 0.985 ± 0.003 | 0.965 ± 0.003 | 0.984 | 0.9838 | 0.953 | 0.517 | 0.568 | 0.753 | 0.951 | 12808 | 5.1 |

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

## Reading the streaming table

* Every v6-trained model is either saturated in a causal stream of real CHIME noise (threshold
  1.000: no threshold reaches ≤ 10 FA/h on the val stream) or fires independently of bursts.
  The "≈ 0.3 detected" rows have identical rates in both sweep orientations, at 560–3,900 FA/h.
* None is usable as a streaming trigger, including the 0.984 ceiling model.
* The v6 tiled stream (v6 test patches butted end-to-end; see NOTES_v6) is comparison only.

## Bottom line

* **CNN:** on this split, `s1_H3_res256*` (22.6k params) reaches the **mathematical ceiling of
  0.9838** on 3/3 seeds. `s1_H4_half` (2.6k params) is the smallest within 0.5 pt (0.977).
* **RNN:** the lstm_v3 recipe is best (0.963 ± 0.003, 30k params). Smaller-input RNNs are
  1–2 pt lower.
* The CNNs are hls4ml-ready (emulated at ap_fixed<16,6> without loss). The RNNs are written but
  large for io_parallel.
* None of these numbers transfers off v6. Use the v7 / streaming results (`summary.md`,
  `summary_v6.md`) for any claim about detecting new bursts.
