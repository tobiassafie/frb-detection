# Best models (`models/best/`)

Each file is a copy of a trained Keras 3 model (`keras.models.load_model(path)`). The `run_id` in each
filename points to its full run record (config, metrics, seeds) under `results/runs*/`.

**v6 numbers are on the original `lstm_dataset_v6.npz` split.** That split leaks: every test noise image
is also in training, and a no-learning lookup scores 0.984. Its maximum achievable test accuracy is
0.9838. **v7 numbers are leak-free**: the split is by source burst, rebuilt from the raw CHIME files.

## CNNs — `models/best/cnn/`
| file | what | input | params | MACs | test acc |
|---|---|---|---|---|---|
| `cnn_v6_best_256in_22k_s1_H3_res256_seed2.keras` | best CNN on v6 (hits the 0.9838 ceiling on 3/3 seeds) | 256 × 256 | 22,561 | 14.8M | **0.984** (v6) |
| `cnn_v6_smallest_2p6k_s1_H4_half.keras` | smallest CNN within 0.5 pt | 128 × 128 (2×2 mean) | 2,601 | 1.07M | 0.977 (v6) |
| `cnn_v6_robust_oldrecipe_8p8k_s0_baseline_s142.keras` | original [2,4,4,4,4] recipe; does **not** memorise the noise images | 256 × 256 | 8,797 | 3.14M | 0.977 (v6) |
| `cnn_v7_leakfree_6p4k_s1mB_t0041_rfi.keras` | best synthesis-friendly CNN on leak-free v7 | 128 × 128 | 6,433 | 3.69M | 0.944 (v7, S/N ≥ 7) |

Details: `results/summary_v6a.md`, `results/summary.md`.

## LSTM / GRU (whole patch → one P) — `models/best/lstm/`
| file | what | input (time × freq) | params | MACs | v6 test acc |
|---|---|---|---|---|---|
| `lstm_v6_best_526kMAC_d_c32_p8_lstm32_d64_s42.keras` | best LSTM: max-pool ×8 → Conv1D 32 → LSTM 32 → Dense 64 | 32 × 256 | 18,721 | 526k | 0.975 ± 0.002 (3 seeds) |
| `gru_v6_cheapest_140kMAC_d_c16_p16_gru32_d32_s42.keras` | cheapest within 1 pt of lstm_v3: max-pool ×16 → Conv1D 16 → GRU 32 → Dense 32 | 16 × 256 | 10,001 | 140k | 0.965 ± 0.002 (3 seeds) |
| `lstm_v3_reference_30k_s0_lstm_v3_recipe_full150_seed1.keras` | historic lstm_v3 recipe, retrained (reference) | 256 × 256 | 29,953 | 7.34M | 0.963 ± 0.003 (3 seeds) |

Input is the v6 patch time-max-pooled by the stated factor, in the stored orientation.
Details: `results/summary_v6d.md`.

## Rolling (causal, P(FRB | data up to t) at every step) — `models/best/rolling/`
| file | what | params | MACs / step | v6 test acc @1 % FA | latency |
|---|---|---|---|---|---|
| `rolling_gru32_best_r_gru32_f16_p4_onset.keras` | **best rolling model**: Conv1D 16 → GRU 32 → Dense 16 → sigmoid, per step | 9,457 | 9,232 | 0.990 ± 0.005 (3 seeds) | ≈ 0 bins after onset |
| `rolling_gru32_front32_r_gru32_f32_p4_onset.keras` | same with a 32-channel front end | 15,105 | 14,864 | 0.993 (1 seed) | ≈ 0 bins |
| `rolling_lstm32_best_strict_lstm_r_lstm32_f32_p4_onset.keras` | best rolling model that is an LSTM | 17,089 | 16,912 | 0.990 (1 seed) | 8 bins |

How to feed them:
* Input is the v6 patch **time-reversed** (physical high-frequency-first order), time-max-pooled ×4
  (64 steps × 256 channels).
* Output shape is (64, 1).
* For true streaming, build the stateful twin with `scripts/roll_train.py: build(cfg, 1, 256, stateful=True)`.

Full stats (FLOPs, latency, training data): `docs/rolling_gru_stats.md`. Results: `results/summary_roll.md`.

## Historic models
`historic/models_2026-07/` (cnn_v1, lstm_v1–v3) and `historic/ablation_results/models/` hold the original
weights behind the 96–97 % numbers. They reproduce exactly (`results/summary_v6a.md`, Stage 0).
