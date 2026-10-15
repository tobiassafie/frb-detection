# FRB detection — code, results and small models (snapshot 2026-10-14)

A curated, history-free snapshot of the working repo (`C:\Users\tobys\Desktop\frb`): all scripts, notebooks,
results (CSVs, summaries, figures, per-run JSONs) and the small trained weights.

**Not included** (too large for GitHub or regenerable):
* raw CHIME data (`realtime_data/`, ~34 GB);
* datasets: `augmented_dataset/*.npz` incl. **`lstm_dataset_v6.npz` (472 MB)**; v7 `X*.npy`; FETCH `classifier/fetch/*.hdf5`;
* generated hls4ml project folders (`hls_projects*/<project>/`, ~1.6 GB, mostly duplicated headers);
* `.venv`, `fpga_handoff/X_test.npy`.

Copy `augmented_dataset/lstm_dataset_v6.npz` separately (USB / cloud drive) to run anything on v6.

## Where to start
| file | what |
|---|---|
| `results/summary.md` | v7 (rebuilt, leak-free, file-grouped) CNN/LSTM results + hls4ml status |
| `results/summary_v6.md` | v6 benchmark, cross-dataset, streaming simulation |
| `results/summary_v6a.md` | best models on the ORIGINAL v6 split (CNN reaches the 0.9838 ceiling) |
| `results/summary_v6c.md` | smaller LSTMs + LSTM synthesis projects (est_ only) |
| `results/summary_v6d.md` | small-RNN sweep + Pareto plot (GRU32 @ 140k MACs = 0.965) |
| `results/NOTES*.md` | reasoning, assumptions, incidents for each stage |

## Key facts (details in the summaries)
* **v6 (`lstm_dataset_v6.npz`)**: only 5 unique noise images (oversampled); 5 test / 10 val / 36 train FRB-labelled
  samples are pixel-identical to noise images → max achievable test accuracy **0.9838** (val 0.9675).
  A no-learning lookup scores 0.984. All v6 numbers are "leaky split, comparison only".
* **v7** (`scripts/v7_*.py`): rebuilt from the raw CHIME files, split by source burst; best small CNN
  `s1mB_t0041_rfi` 0.944 test (S/N ≥ 7).
* Best on original v6: CNN `s1_H3_res256*` 0.984 (ceiling); RNN `c32_p8_lstm32_d64` 0.975 ± 0.002 (526k MACs);
  cheapest RNN within 1 pt of lstm_v3: `c16_p16_gru32_d32` 0.965 ± 0.002 (140k MACs).
* **Hardware**: nothing has been synthesised — every latency/resource number is `est_`. Synthesis needs the
  EC2 box with Vitis HLS 2025.1; see `hls_projects*/README.md` (`run_all_synthesis.sh`, 20-min timeout,
  `parse_synth_reports.py`, `check_accuracy.py`). Regenerate project folders with `scripts/v6a_export.py` /
  `scripts/hls_export.py`.

## Environment
Python 3.11, TensorFlow 2.20 / Keras 3.13, tf_keras + QKeras (QAT; run with `TF_USE_LEGACY_KERAS=1`),
hls4ml 1.3.0, scikit-learn, optuna, pandas, h5py, matplotlib.
