# v6 benchmark + streaming simulation — summary (2026-10-01)

Companion to `results/summary.md` (v7). Every number is next to the `run_id` it comes from.
Per-run rows: `results/v6_benchmark.csv` (v6 / dedup / cross-dataset) and
`results/stream_results.csv` (streaming). Audit, assumptions and incidents: `results/NOTES_v6.md`.

## TL;DR

1. **Audit (NOTES_v6 Stage 0): nothing in the repo evaluated a real stream.**
   * v6/v7 training and eval use independent, fixed-size patches containing the whole
     (residual-)dispersed burst. 97 % of each v7 sweep lies inside its patch.
   * `frb_streaming_pipeline.ipynb` only masks frequency channels of a fixed 0.66 ms patch, using
     a random, unrelated DM. Its extraction crashed on 29/31 files (2 positive patches), and its
     training/streaming cells have no recorded outputs.
   * No detection latency or false-alarm-per-time metric existed.
2. **v6 benchmark (leaky split, comparison only).** Retrained on the original v6 split, the
   v7 architectures score **0.94–0.98** on the v6 test set:
   * best: the sweep's 128² max-pool/flatten CNN, 0.979 ± 0.003 over 3 seeds;
   * the v7 synthesis CNN `s1mB_t0041_rfi`: 0.938 ± 0.005.
   * **None beats the no-learning exact-match lookup (0.984).** Every model has precision 1.000
     only because all 154 v6 test noise images are verbatim training images.
   * On the deduplicated test (149 FRBs, 0 noise) recall is 0.91–0.99. The old v6 "96–99 %"
     numbers measure memorisation of 5 noise images plus detection of FRBs on those same 5
     backgrounds.
3. **Cross-dataset: no transfer either way.**
   * v6-trained models on the v7 test: 0.41–0.59.
   * v7-trained models on the v6 test: 0.47–0.53.
   * The two datasets contain different "FRBs" (v6: DM 100–2500 slivers in the top channels;
     v7: real residual sweeps).
4. **Streaming (new): orientation is the dominant effect.**
   * In a causal 1-hour stream with the sweep in the orientation every training sample had,
     the v7 CNN `s1mB_t0041_rfi` detects 82.7 % of S/N ≥ 7 bursts (96.2 % at S/N ≥ 15).
   * With physical high-frequency-first arrival it detects **1.9 %**. The patch-trained models
     learned the sweep direction.
   * The **stream-trained CNN (`strm_t0041_s1`) detects 77.5 % / 78.2 % (physical / stored) at
     S/N ≥ 7** at the same val-chosen operating point, triggering after ~81–86 % of the band
     has arrived. Its false-alarm rate on the test files is higher, though: 494 vs 80 per hour.
   * At that operating point the RNNs and the v6-trained models detect **≤ 17 %** of S/N ≥ 7
     bursts. At a fixed 0.5 threshold the v7 RNNs detect 70–85 %, but at 4,000–5,000 FA/h.
5. **False-alarm rates do not transfer from val files to test files.** A threshold giving
   8 FA/h on the val stream gives 80–500 FA/h on the test stream. With only 5–6 files per split,
   RFI differs a lot between files. The FA/h column should be read as "order of magnitude".


## Table 1 — v6 dataset (original split)

Numbers likely inflated by leakage:
* **every v6 accuracy / precision / AUC in this table.** Compare each row with the lookup
  baseline (0.984), which scores higher without learning.
* **every precision of 1.000.** The test noise images are verbatim training images.

The dedup column is FRB-only (recall only). A constant "FRB" predictor scores 1.000 on it.

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

## Table 2 — v7 dataset (leak-free reference, plus v6-trained models cross-evaluated)

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

## Table 3 — streaming metrics (causal 256-bin window, stride 64; 1.0 h v7-style test stream, 1,716 bursts per orientation)

Rows with threshold **1.000** mean the model could not meet ≤ 10 FA/h on the val stream at any
threshold, because its output is saturated. Their detection rates are not an operating point,
e.g. `v6_loc_t0041_A_w8k5bn_s0` fires continuously on v7 data (2,201 FA/h, so "100 %"
detection). The v6 tiled stream is comparison only; see NOTES_v6 Stage 2.

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

## What changed vs the v7 results (`results/summary.md`)

* **The v6 numbers look better than v7 (0.94–0.98 vs 0.92–0.94), but only because of leakage.**
  * On v6 the same architectures score higher than on v7, yet still below a lookup table that
    has learned nothing (0.984).
  * Their precision is 1.000 for free, since all v6 test noise images are memorised training
    images.
  * They do not transfer to v7 or to any stream (≈ chance / ≈ 0 % detection at a usable
    threshold).
  * The v7 file-grouped numbers remain the ones to quote for patch classification.
* **v6 prefers different architectures.** The sweep on v6 prefers flatten heads (0.976–0.979)
  over the v7 winner's GAP head (0.938). Flatten keeps position, and position is what you
  need to memorise 5 fixed backgrounds. Avg-pool + GAP models sometimes fail to train at all
  on v6 (dead seeds).
* **Patch accuracy ≠ streaming performance.** v7 patch test accuracy (0.94) corresponds to
  83 % detection of S/N ≥ 7 bursts at ~80 FA/h in a causal stream — and only when the sweep
  direction matches training.
  * With physical high-frequency-first arrival, every v7 patch-trained CNN is ≤ 5 % at the
    val-chosen threshold (≤ 28 % at a fixed 0.5).
  * The stream-trained CNN fixes the orientation problem (77–78 % in both orientations, about
    5 points below the standard CNN in the stored orientation).
  * It triggers earlier: median time-to-detect 42–48 ms vs 58 ms, at ~81–86 % of the band.
  * It costs a higher test false-alarm rate (494 vs 80 /h at the same val-chosen operating
    point).
  * The RNNs, which looked acceptable on patches (0.78–0.84), are not usable as streaming
    triggers: they detect ≤ 17 % at ≤ 10 FA/h on val.
* **Open items:**
  * the FA/h calibration (val→test transfer is poor with 5–6 files per split);
  * whether the deployed system sees stored-orientation (baseband-dump) or
    physical-orientation data;
  * the 256×256 and filter-width v6 sweep configs that did not finish in the 2 h budget.
