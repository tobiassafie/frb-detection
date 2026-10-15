# FRB detection on CHIME data — models, results and history

Small CNN / LSTM / GRU classifiers that tell FRB from noise, aimed at FPGA deployment (hls4ml).
This branch is a curated, history-free snapshot of the working repo. Data files are not included.

## Repo map
| folder | what's in it |
|---|---|
| **`models/best/`** | the models to use: best CNNs, best LSTM/GRU, best rolling models. Model card: `models/README.md` |
| **`docs/rolling_gru_stats.md`** | params, FLOPs, latency, training-data size for the best rolling model |
| `results/` | summaries, notes, per-run CSV/JSON, figures, all trained weights by stage. Index: `results/README.md` |
| `scripts/` | everything that produced the results (dataset build, training, evaluation, hls4ml export) |
| `augmented_dataset/` | dataset **metadata only** + `README.md` (v6 construction, known issues, v7 rebuild) |
| `notebooks/` | `augmentation/` (v5, v6, v6_with_dm, DM extraction), `training/` (original v6 CNN/LSTM notebooks, ablation), `historic/` |
| `historic/` | earlier models (cnn_v1, lstm_v1–v3), ablation results, fpga_handoff configs, meeting reports |
| `hls_projects*/` | hls4ml synthesis READMEs + batch scripts. Project folders are regenerable with `scripts/v6a_export.py` |

## Headline models
| | model | params | cost | accuracy | file |
|---|---|---|---|---|---|
| best CNN (v6) | 256² CNN, Conv[4,8,16,16,16] + flatten | 22,561 | 14.8M MAC | 0.984 = v6 ceiling (3/3 seeds) | `models/best/cnn/cnn_v6_best_*` |
| smallest CNN (v6) | ½-width CNN, 128² | 2,601 | 1.07M MAC | 0.977 | `models/best/cnn/cnn_v6_smallest_*` |
| best CNN (leak-free v7) | Conv[4,8,16,16,16] + avg-pool + GAP | 6,433 | 3.69M MAC | 0.944 (held-out sources, S/N ≥ 7) | `models/best/cnn/cnn_v7_leakfree_*` |
| best LSTM (v6) | max-pool ×8 → Conv1D 32 → LSTM 32 | 18,721 | 526k MAC | 0.975 ± 0.002 | `models/best/lstm/lstm_v6_best_*` |
| cheapest GRU (v6) | max-pool ×16 → Conv1D 16 → GRU 32 | 10,001 | 140k MAC | 0.965 ± 0.002 | `models/best/lstm/gru_v6_cheapest_*` |
| **best rolling (v6)** | causal Conv1D 16 → GRU 32 → per-step P(FRB) | 9,457 | 9.2k MAC / 2.62 ms step | 0.990 ± 0.005 @1 % FA, latency ≈ 0 bins | `models/best/rolling/rolling_gru32_best_*` |

**Read before quoting v6 numbers.** `lstm_dataset_v6.npz` has only 5 unique noise images, and every
test noise image is also in train. 5 test FRB-labelled samples are pixel-identical to noise, so
0.9838 is the maximum. A no-learning lookup scores 0.984. v6 results are "leaky split, comparison
only"; the v7 rebuild (split by source burst) is the honest generalisation estimate. Details:
`augmented_dataset/README.md`.

**Hardware.** Nothing has been synthesised from this repo; every latency/resource number is `est_`.
Synthesis needs the Vitis HLS 2025.1 machine (see `hls_projects_v6c/README.md`). The one measured
FPGA result in the project history is the pooled CNN at 61.5 µs (`notebooks/historic/context/`).

## Running things
* Python 3.11, TensorFlow 2.20 / Keras 3.13, tf_keras + QKeras (QAT; `TF_USE_LEGACY_KERAS=1`),
  hls4ml 1.3.0, scikit-learn, optuna, pandas, h5py, matplotlib.
* Copy `augmented_dataset/lstm_dataset_v6.npz` (472 MB) into place; the v7 arrays can be rebuilt from
  the raw CHIME files.
* Rolling model: `python scripts/roll_prep.py` (onsets) → `python scripts/roll_train.py <cfgs.json>` →
  `python scripts/roll_midburst.py`.
