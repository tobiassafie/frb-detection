# Phase 1: Simple Numbers

Generated: 2026-09-23. Machine-readable version: [`01_basics.json`](01_basics.json).
Scripts: [`scripts/phase1_dataset_meta.py`](scripts/phase1_dataset_meta.py) (dataset shapes/keys, metadata only — no full-array loads for the HDF5/npz files; the small `.npz` label/DM vectors were loaded fully since they're a few KB).

Models covered, per your Phase 1 scope: the synthesized pooled model, `fpga_handoff/group1` (baseline, both precisions), `fpga_handoff/group2` (front-loaded, both precisions), and `models/cnn_v1` as a labeled reference row.

---

## 1. Model parameters, memory footprint, input/bottleneck size

| Model | Params | Input elements | Bottleneck (flatten) | Weight precision (source) | Memory @ fixed-point | Memory @ float32 |
|---|---|---|---|---|---|---|
| **Pooled (synthesized)** — `classifier/frb_classifier_pooled.h5` | 844 | 4,096 (64×64×1) | 392 | `ap_fixed<16,6>`, uniform — read from `frb_pooled_hls/frb_pooled_hls/firmware/defines.h` | **1,688 B** (1.65 KB) | 3,376 B (3.30 KB) |
| **group1 baseline** — `fpga_handoff/group1_w16_6` | 8,797 | 65,536 (256×256×1) | 256 | `ap_fixed<16,6>`, uniform — read from `fpga_handoff/group1_w16_6/firmware/defines.h` | **17,594 B** (17.18 KB) | 35,188 B (34.36 KB) |
| **group1 baseline** — `fpga_handoff/group1_w8_3` | 8,797 | 65,536 | 256 | `ap_fixed<8,3>`, uniform — same source, `_w8_3` variant | **8,797 B** (8.59 KB) | 35,188 B |
| **group2 front-loaded** — `fpga_handoff/group2_w16_6` | 8,885 | 65,536 | 256 | `ap_fixed<16,6>`, uniform | **17,770 B** (17.35 KB) | 35,540 B (34.71 KB) |
| **group2 front-loaded** — `fpga_handoff/group2_w8_3` | 8,885 | 65,536 | 256 | `ap_fixed<8,3>`, uniform | **8,885 B** (8.68 KB) | 35,540 B |
| **cnn_v1 (July 2026 baseline instance, reference only)** — `models/cnn_v1/model.keras` | 8,797 | 65,536 | 256 | none (plain Keras float32, never synthesized) | — | 35,188 B (34.36 KB) |

Weight and bias tensors use the **same** bit width in every layer of every synthesized model here (`weight_t` and `bias_t` typedefs are identical per layer, e.g. `conv_0_weight_t` = `conv_0_bias_t` = `ap_fixed<16,6>`) — I checked this directly rather than assuming it, since your task description flagged that per-layer precision might differ. It doesn't, for weights/biases. **Accumulator precision does grow per layer** (e.g. group1's `conv_0_accum_t` is `ap_fixed<37,17>`, its `dense_0_accum_t` is `ap_fixed<41,21>`) — that's an on-chip datapath-width question for Phase 2, not a stored-parameter-memory question, so it isn't in the table above.

Per-layer parameter breakdown for all four models is in `01_basics.json` under `models.*.layers`. Quick summary:
- **Pooled**: conv2d (20) → conv2d_1 (38) → dense (786). The final Dense layer alone is 93% of the model's parameters.
- **group1 baseline**: 5 conv layers (20+76+148+148+148 = 540) + dense_0 (8,224) + output (33) = 8,797. **dense_0 alone is 93.5% of all parameters** — the flatten→dense bottleneck dominates, not the convolutions.
- **group2 front-loaded**: 3 conv layers (40+296+292 = 628) + dense_0 (8,224) + output (33) = 8,885. Same story: dense_0 is 92.6% of parameters despite having 3 fewer conv layers than group1.

## 2. Datasets

### `augmented_dataset/lstm_dataset_v6.npz` (feeds `train_cnn.py` and the entire ablation study → all `fpga_handoff/group*` models)

| Key | Shape | Dtype |
|---|---|---|
| `X_train` / `y_train` | `(1436, 256, 256)` / `(1436,)` | float32 |
| `X_val` / `y_val` | `(308, 256, 256)` / `(308,)` | float32 |
| `X_test` / `y_test` | `(308, 256, 256)` / `(308,)` | float32 |
| `dm_train` / `dm_val` / `dm_test` | `(1436,)` / `(308,)` / `(308,)` | float32 |

- **Total: 2,052 samples** (1,436 train / 308 val / 308 test). File size on disk: 495,517,228 bytes (472.6 MB).
- **Class balance: exactly 50/50 in every split** (718/718 train, 154/154 val, 154/154 test) — confirms the confusion-matrix totals seen in the ablation JSON (154+154=308 test samples).
- **The `dm_*` arrays are populated only for FRB-labeled rows** (dispersion measure, pc/cm³, range ~100–2,500) and are `NaN` for every noise-labeled row — confirmed by direct inspection, not assumed.
- **Real vs. augmented — not determinable from this file alone.** There are 1,026 total FRB-labeled samples across the three splits (718+154+154), each with a distinct, near-unique DM value. Only **31 real CHIME FRB detections** exist anywhere in this repo (`realtime_data/FRBs/`). That gap is a strong indicator that most FRB-labeled samples here are synthetically redispersed/augmented from a much smaller set of real events, not independent real detections — but I can't quantify the exact real:synthetic ratio without opening `frb_augmentation_v5.ipynb` / `frb_augmentation_v6.ipynb`, which I did not do in this phase (say the word if you want that next).
- **Matches which model(s):** input shape `(256,256)` → `models/cnn_v1`, all `fpga_handoff/group1_*` and `group2_*` (after adding the trailing channel dim).

### `fpga_handoff/X_test.npy` + `y_test.npy`

- `X_test.npy`: shape `(308, 256, 256, 1)`, float32, 80,740,480 bytes.
- `y_test.npy`: shape `(308,)`, float32, 1,360 bytes.
- **This is the exact same 308-sample test split as `lstm_dataset_v6.npz`'s `X_test`/`y_test`**, confirmed by matching sample count and shape — re-saved standalone (mtime 2026-09-18, same as the rest of `fpga_handoff`) so each group's `check_accuracy.py` can run independently without needing the full augmented-dataset file. This is the test set Phase 4 will use.

### The pooled model's dataset — identified, not assumed

`classifier/frb_classifier_pooled.ipynb` loads `fetch/train_data.hdf5` and `fetch/test_data.hdf5` directly (confirmed from the notebook's own source, not inferred from file proximity):

| File | Size | Keys | Shapes |
|---|---|---|---|
| `classifier/fetch/train_data.hdf5` | 17,653,183,539 bytes (16.4 GB) | `data_dm_time`, `data_freq_time`, `data_labels` | `(40000, 256, 256, 1)` float32 ×2 arrays, `(40000,)` bool |
| `classifier/fetch/test_data.hdf5` | 6,036,831,364 bytes (5.6 GB) | same keys | `(13983, 256, 256, 1)` float32 ×2 arrays, `(13983,)` bool |

**Important, non-obvious finding:** the notebook uses **`data_dm_time`** (a dedispersion-transform image), not `data_freq_time` (the frequency-time spectrogram) — the `data_freq_time` load line is present but commented out. So the pooled model classifies DM-transform images, not raw spectrograms. It then resizes each `(256,256,1)` image down to `(64,64,1)` with `tf.image.resize()` (bilinear interpolation, not physical re-binning) entirely in-memory — there's no separate saved file for this step.

**Training log found — this answers your "accuracy never confirmed" flag from the task description:** the same notebook trains and evaluates a model with this exact architecture, dataset, and preprocessing, and reports:
- **Test accuracy: 95.20%** on the full 13,983-sample test set (7,319 RFI / 6,664 FRB — a real, reasonably balanced test set, unlike the excluded `model_outputs/performance_report.txt`'s 5-positive sample).
- Confusion matrix `[[7314, 5], [666, 5998]]` → FRB recall 90.0%, FRB precision 99.9%.
- **Caveat:** the notebook has no `model.save()` cell anywhere in its 12 cells. I'm treating this as high-confidence-but-not-cryptographically-proven evidence for `classifier/frb_classifier_pooled.h5`'s actual accuracy — same architecture, same data pipeline, same hyperparameters, just no direct proof the saved `.h5` weights are byte-identical to this exact run's weights (no fixed seed is set either, so re-running the notebook would not reproduce this exact number). I'd treat 95.20% as the number to cite, with this caveat attached, rather than treating it as unconfirmed.

## 3. File-size chain

There are **two separate pipelines** feeding two different model families — I did not collapse them into one chain since that would misrepresent which numbers apply to which model.

**Pipeline A — `fpga_handoff` / ablation study models (group1, group2, cnn_v1):**

| Stage | Size | Notes |
|---|---|---|
| Raw CHIME baseband file (1 real event) | ~1.09 GB avg (range 150.8 MB – 2.1 GB across 31 files) | `tiedbeam_power` alone is `(903 freq, 2 pol, 75,913 time)` float32 at 2.56 µs/sample native resolution; file also carries the complex64 baseband array at the same shape |
| Intermediate augmented-dataset generations | v3: 3.4 GB, v4: 4.2 GB, v5: 242.2 MB, v6: 472.6 MB (final) | Sizes are **not monotonic** across versions — v4 is the largest, v5/v6 much smaller — consistent with changing sample counts and/or preprocessing between iterations, not a simple linear reduction. Not investigated further here. |
| One sample in the final dataset (`lstm_dataset_v6.npz`) | 256×256 float32 = 262,144 bytes (256 KB) | 256×256 pixels at 0.655 ms/bin (256× downsample from the 2.56 µs native resolution — confirmed exactly: `2.56e-6 s × 256 = 6.5536e-4 s ≈ 0.655 ms`, and this matches `ablation_results`'s own `effective_time_res_ms: 0.655` for the r=1 baseline) |

**Reduction for one real event, raw file → one final training sample: ~1.09 GB → 256 KB ≈ 4,270×.** (This compares one full raw observation file to one cropped/downsampled spectrogram window used for training — not a like-for-like "the whole file became this sample," since the raw file covers a longer time window than just the burst.)

**Pipeline B — pooled model:**

| Stage | Size | Notes |
|---|---|---|
| Raw CHIME baseband file | ~1.09 GB avg | same as above |
| Intermediate: `classifier/fetch/{train,test}_data.hdf5` | 16.4 GB + 5.6 GB = 22.0 GB combined, 53,983 samples total | Each sample stored as **two** `256×256×1` float32 representations (freq-time AND dm-time) — only one (`data_dm_time`) is actually used |
| Final, in-memory only (no file): one sample after `tf.image.resize()` to 64×64 | 64×64 float32 = 16,384 bytes (16 KB) | Never written to disk as a separate dataset file — computed here from the model's actual input shape |

**Reduction for one real event, raw file → one final training sample: ~1.09 GB → 16 KB ≈ 68,300×** — about 16× more aggressive than Pipeline A, because of the extra 64×64 resize step.

## Open items for Phase 2

- Per-layer accumulator/activation precision (needed for the MACs/memory-per-layer breakdown) is already extracted from `defines.h` for all four models and ready to use — see the raw typedefs pulled during this phase.
- No synthesis reports exist for `fpga_handoff/group1` or `group2` (confirmed in Phase 0) — Phase 2 will give analytic MACs/FLOPs/activation-memory for these two, and will pull actual latency/DSP/LUT/etc. only for the pooled model's `csynth.rpt`, per your instruction.
