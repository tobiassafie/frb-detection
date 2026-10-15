# Phase 0: Inventory & Model Mapping

Generated: 2026-09-23. Root: `C:\Users\tobys\Desktop\frb`. Read-only scan (nothing modified).
Raw machine-readable dump: [`phase0_raw.json`](phase0_raw.json) (produced by
[`scripts/phase0_inventory.py`](scripts/phase0_inventory.py), rerunnable).

All dates below are filesystem mtimes, not necessarily creation/training dates.

---

## 1. Model files found

| Path | Size | Modified | Format |
|---|---|---|---|
| `models/cnn_v1/model.h5` | 166.1KB | 2026-07-21 21:14 | Keras H5 |
| `models/cnn_v1/model.keras` | 164.0KB | 2026-07-21 21:14 | Keras v3 |
| `models/cnn_v1/best_model.keras` | 164.0KB | 2026-07-21 21:14 | Keras v3 (checkpoint) |
| `fpga_handoff/group1_w16_6/keras_model.keras` | 168.0KB | 2026-09-18 11:53 | Keras v3 |
| `fpga_handoff/group1_w8_3/keras_model.keras` | 168.0KB | 2026-09-18 11:53 | Keras v3 |
| `fpga_handoff/group3_w16_6/keras_model.keras` | 168.0KB | 2026-09-18 11:53 | Keras v3 |
| `fpga_handoff/group3_w8_3/keras_model.keras` | 168.0KB | 2026-09-18 11:53 | Keras v3 |
| `fpga_handoff/group2_w16_6/keras_model.keras` | 153.8KB | 2026-09-18 11:53 | Keras v3 |
| `fpga_handoff/group2_w8_3/keras_model.keras` | 153.8KB | 2026-09-18 11:53 | Keras v3 |
| `classifier/frb_classifier_pooled.h5` | 45.6KB | 2026-02-03 23:58 | Keras H5 |
| `context/frb_classifier_pooled(1).h5` | 45.6KB | 2026-07-20 16:09 | Keras H5 (copy) |
| `frb_pooled_hls/frb_pooled_hls/keras_model.keras` | 27.1KB | 2026-02-04 00:11 | Keras v3 (won't load in current Keras — see §3) |
| `classifier/model_outputs/frb_classifier.h5` | 44.8KB | 2026-02-05 23:24 | Keras H5 |
| `classifier/model_outputs/frb_classifier.keras` | 41.6KB | 2026-02-05 23:24 | Keras v3 |
| `models/lstm_v1/model.h5`, `.keras`, `best_model.keras` | ~various | 2026 | Keras (LSTM, out of scope — see §5) |
| `models/lstm_v2/*`, `models/lstm_v3/*` | ~various | 2026 | Keras (LSTM, out of scope) |
| `ablation_results*/models/*.keras` (~40 files) | small | various | Keras (ablation sweep checkpoints, see §5) |
| `FRB_RFI_Classifier.ipynb` (defines a model inline, not a saved file) | — | — | Notebook-only, "unrelated early work" per your note — confirmed a separate CNN pipeline over FRB h5 + RFI pickle data, not the candidates below |

Full file-by-file listing (154 model files total, including ablation sweeps) is in `phase0_raw.json` under `inventory.models`.

**Note on a script bug I hit and corrected for:** my scanner's file-type filter checks `.h5`/`.hdf5` before checking filename keywords, so the 31 raw CHIME `realtime_data/FRBs/*_beamformed.h5` files got bucketed as "model" candidates and the script dutifully tried to `keras.load_model()` each one. They correctly failed to load (`ValueError: No model config found`), and the script's HDF5 fallback then read their raw structure instead — which turned out to be useful (see §4). They are **not** model files; see §2.

## 2. Datasets

| Path | Size | Modified | Notes |
|---|---|---|---|
| `classifier/fetch/train_data.hdf5` | 16.4GB | 2025-10-27 15:37 | Likely an intermediate/raw-derived dataset, pre-augmentation |
| `classifier/fetch/test_data.hdf5` | 5.6GB | 2025-10-27 19:59 | Same generation as above |
| `augmented_dataset/dataset.npz` | 472.5MB | 2026-03-13 00:05 | |
| `augmented_dataset/lstm_dataset_v3.npz` | 3.4GB | 2026-03-12 23:06 | |
| `augmented_dataset/lstm_dataset_v4.npz` | 4.2GB | 2026-03-12 23:18 | |
| `augmented_dataset/lstm_dataset_v5.npz` | 242.2MB | 2026-03-12 23:39 | |
| `augmented_dataset/lstm_dataset_v6.npz` | 472.6MB | **2026-07-21 00:01** | Used by `train_cnn.py` (current) — matches `models/cnn_v1` mtime same day |
| `augmented_dataset/lstm_dataset_v6 - Copy.npz` | 472.5MB | 2026-03-27 00:23 | Despite the "v6" name, this copy predates the real v6 by ~4 months — likely a stray copy of an older file, not an actual v6 backup. Flagged, not resolved. |
| `realtime_data/FRBs/*_beamformed.h5` (31 files) | 33.8GB total (150MB–2.1GB each) | 2024-03-20/22 | **Raw CHIME baseband data** — see §4 |
| `realtime_data/RFIs/LOFAR_Full_RFI_dataset.pkl` | — | — | RFI dataset used by `FRB_RFI_Classifier.ipynb` only |
| `fpga_handoff/X_test.npy` / `y_test.npy` | 77.0MB / 1.3KB | **2026-09-18 11:53** | Newest dataset artifact in repo; paired with `check_accuracy.py` in each `fpga_handoff/group*` dir — this is the test set for Phase 4 |
| `classifier/model_outputs/normalization_params.npy` | 415B | 2026-02-05 23:24 | Normalization stats for the pooled model |

Dataset shapes/keys/sample counts are deferred to Phase 1 per your instructions (metadata only, no full loads).

## 3. hls4ml projects & Vitis synthesis reports

Seven `hls4ml_config.yml` files found:

| Project dir | Precision | Strategy | ReuseFactor | IOType | Has completed Vitis `csynth` reports? |
|---|---|---|---|---|---|
| `frb_pooled_hls/frb_pooled_hls/` | `ap_fixed<16,6>` (single global) | **Latency** | 1 | io_stream | **YES** — 23 report pairs under `frb_classifier_prj/solution1/syn/report/` |
| `fpga_handoff/group1_w16_6/` | `ap_fixed<16,6>` default | Resource | 1 | io_stream | No — firmware/weights only, no `syn/report/` dir |
| `fpga_handoff/group1_w8_3/` | `ap_fixed<8,3>` default | Resource | 1 | io_stream | No |
| `fpga_handoff/group2_w16_6/` | `ap_fixed<16,6>` default | Resource | 1 | io_stream | No |
| `fpga_handoff/group2_w8_3/` | `ap_fixed<8,3>` default | Resource | 1 | io_stream | No |
| `fpga_handoff/group3_w16_6/` | `ap_fixed<16,6>` default | Resource | 1 | io_stream | No |
| `fpga_handoff/group3_w8_3/` | `ap_fixed<8,3>` default | Resource | 1 | io_stream | No |

**This is the single most important finding of Phase 0: only `frb_pooled_hls` has ever actually been through Vitis synthesis.** The six `fpga_handoff/group*` directories contain hls4ml-generated C++ firmware, weight headers, and TCL build scripts, but no `syn/report/*csynth*` files exist anywhere under `fpga_handoff/` — meaning either synthesis was never run for them, or it was run somewhere else and the reports weren't copied back into this repo. I did not find any Vivado utilization/timing reports outside the `frb_pooled_hls` tree either.

### Model ↔ hls4ml project mapping (with confidence)

**`frb_pooled_hls` → the 64×64×1, 2-output "pooled" model, Dropout variant (`classifier/frb_classifier_pooled.h5` / `context/frb_classifier_pooled(1).h5`). Confidence: high.**
Evidence:
- `frb_pooled_hls/frb_pooled_hls/hls4ml_config.yml` points `KerasModel` at `frb_pooled_hls/keras_model.keras`.
- That `keras_model.keras` fails to load under the current Keras 3 install (`could not be deserialized properly` — an old-format Sequential config), but the error message itself dumps the full architecture JSON: Conv2D(2,3×3)→MaxPool(2×2)→Dropout(0.5)→Conv2D(2,3×3)→MaxPool(2×2)→Flatten→Dense(2, softmax), input `(64,64,1)`. This is layer-for-layer identical (including layer names `conv2d`, `max_pooling2d`, `dropout`, `conv2d_1`, `max_pooling2d_1`, `dense`) to `classifier/frb_classifier_pooled.h5` and `context/frb_classifier_pooled(1).h5`.
- `context/frb_pooled_synthesis(1)(2).ipynb` (the notebook that produced this project) explicitly says: *"Solution: Create the CNN in notebook and load the weights via the `frb_classifer_poooled.h5` file"* [sic, their typo], builds `Conv2D(2,(3,3))` on `input_shape=(64,64,1)`, and sets `Precision: ap_fixed<16,6>`, `ReuseFactor: 1`, `Strategy: Latency`, `IOType: io_stream` — matching the config file exactly.
- The `csynth.rpt` numbers (12,299 cycles / 61.495 µs, 580 DSP, 274.05 MHz derived from 3.649 ns estimated clock) match your remembered numbers exactly. See Phase 2 for the full breakdown.

**`fpga_handoff/group1_*` and `fpga_handoff/group3_*` → the ablation study's 5-block "baseline" architecture (filters [2,4,4,4,4], 256×256×1 input, flatten→dense32→sigmoid, 8,797 params). Confidence: high, verified by exact weight comparison — correction from my first pass below.**

**Correction from Phase 0 v1:** I originally called this an unresolved conflict. Per your steer, I traced it to `ablation_results/` (the Sept 18 ablation study) and confirmed by direct weight comparison:

- `ablation_results/all_results.json` (mtime 2026-09-18 11:35) records a "group1" filter-count sweep over the 5-block architecture with `filters=[2,4,4,4,f]` for `f ∈ {1,2,4,8,16,32,64}`. `f4` (filters `[2,4,4,4,4]`, 8,797 params) is the sweep winner: highest test accuracy (0.9643) among a clearly-inflecting curve (f1=0.961, f2=0.958, **f4=0.964**, f8=0.912 — accuracy drops sharply above f4, then partially recovers at f16/f32 without beating f4 by enough to justify the extra params).
- `ablation_results/seed_check_results.json` (mtime 2026-09-18 11:53, **same minute** `fpga_handoff` was populated) reruns that winning "baseline" config at two more seeds (142, 242) to check stability: `seedcheck_baseline_s142` → test_acc 0.9675, `seedcheck_baseline_s242` → test_acc 0.9675. Both reruns score *higher* than the original `group1_f4` run (0.9643).
- **Direct weight comparison** (`np.array_equal` on every weight tensor): `fpga_handoff/group1_w16_6/keras_model.keras` is **byte-identical** to `ablation_results/models/seedcheck_baseline_s142.keras`. It does **not** match `ablation_results/models/group1_f4.keras` (same architecture, different trained weights — different seed). So the model actually exported to `fpga_handoff/group1` is the **seed-142 rerun of the baseline**, not the original sweep-winner checkpoint. **Test accuracy of the exact deployed weights: 96.75%**, on a balanced 308-sample test set (154/154 per class, confusion matrix `[[154,0],[10,144]]`) — this matches the 96.75% figure in your task description exactly.
- `group3` (time-resolution/downsampling sweep, `all_results.json`) tests `r ∈ {1,2,4,8,16,32}` (`r`=how much the time axis is coarsened before the model sees it; `r=1` is unmodified). Accuracy strictly degrades as `r` increases (r1: reuses `group1_f4.keras` directly, 0.964 → r2: 0.945 → r4: 0.948 → r8: 0.958 → r16: 0.955 → r32: 0.909), so the sweep winner is **r=1, i.e. the unchanged baseline** — there is no separate "group3 model," it's the same baseline architecture. That is exactly why `fpga_handoff/group3` is byte-identical to `fpga_handoff/group1`: both point at the same seed-142 baseline rerun. Confirmed by weight comparison (`group1_w8_3` == `group3_w8_3`, all tensors equal).
- Also confirmed: within each group, `_w16_6` and `_w8_3` share identical Keras weights (as expected — they're the same trained model exported at two different hls4ml quantization configs, not two different training runs).
- `context/cnn_synthesis(1).ipynb`'s "Test accuracy: 96.4%, ROC AUC 0.97" corresponds to the original `group1_f4` run (0.9643), not the seed-142 rerun actually shipped to `fpga_handoff` (0.9675) — a minor but real distinction if you need to cite a specific accuracy for the exact deployed weights vs. the architecture in general.

**`fpga_handoff/group2_*` → the ablation study's pooling-schedule sweep winner, `"b_front_loaded"` (filters `[4,8,4]`, pool schedule `[4×4, 4×4, 2×2]`, 8,885 params). Confidence: high, verified by exact weight comparison.**
- `ablation_results/all_results.json` "group2" sweep compares five pooling-schedule variants (`a_uniform`, `b_front_loaded`, `c_back_loaded`, `d_freq_preserving`, `e_time_preserving`) at matched ~8.7–8.9k param budgets. `b_front_loaded` wins on test accuracy (0.9643, tied with `a_uniform`/`d_freq_preserving` but with better val accuracy: 0.9448 vs 0.9318/0.9448).
- **Direct weight comparison:** `fpga_handoff/group2_w16_6/keras_model.keras` is byte-identical to `ablation_results/models/group2_b_front_loaded.keras` (the original seed-42 sweep run — not a seedcheck rerun this time; no seedcheck reruns of this model beat it, so the original checkpoint was kept). Test accuracy of the exact deployed weights: **96.43%**.
- This is **not** a draft of `train_cnn.py` — retracting that guess from my first pass. `scripts/ablation_arch.py` (the actual model-builder used by the ablation study) confirms this directly in its own source comment: *`"baseline" is the production architecture: uniform 5x(2,2) pooling, filters [2,4,4,4,4] ... It is Group 1's f4, Group 2's a_uniform, and Group 3's r1 -- literally the same architecture, trained once.`* Its `build_model()` uses the exact `conv_{i}/pool_{i}/dense_0/drop_0/output` naming seen in all `fpga_handoff/group*` models, and its `ARCHITECTURES["pool_front_loaded"]` entry is `filters=[4,8,4], pool_schedule=[(4,4),(4,4),(2,2)]` — an exact match to `group2`. `train_cnn.py`'s superficially similar naming convention and `[4,8,8]` filter choice is coincidental style resemblance from the same author, not a shared code path — it's a separate, standalone script that does not import `ablation_arch.py` and doesn't match any registry entry exactly.

**PyTorch model (`FRB_RFI_Classifier.ipynb`) → confirmed unrelated. Confidence: high.**
- This notebook builds its own from-scratch CNN pipeline (loading raw FRB `.h5` files directly + the LOFAR RFI pickle), independent of the Keras/hls4ml pipeline above. I did not check its exact layer sizes since you've already identified it as out of scope, but confirmed it doesn't share weights, data files, or hls4ml artifacts with any of the candidates above.

## 4. Raw CHIME file structure (bonus finding, useful for Phase 3)

One raw file inspected: `realtime_data/FRBs/FRB20181209A_22154974_beamformed.h5` (150.8MB, one of 31 similar files).

Top-level datasets:
| Dataset | Shape | Dtype |
|---|---|---|
| `tiedbeam_power` | `(903, 2, 75913)` | float32 |
| `tiedbeam_baseband` | `(903, 2, 75913)` | complex64 |
| `index_map/freq` | `(903,)` | struct `{centre, id}` |
| `index_map/beam` | `(2,)` | int64 |
| `index_map/time` | `(75913,)` | struct `{offset_fpga, offset_s}` |
| `index_map/good_channels` | `(760,)` | uint16 |

Key attrs: `delta_time = 2.56e-06` (seconds — native time resolution, 2.56 µs/sample), `archive_version = NT_3.1.0`, `__memh5_subclass = baseband_analysis.core.bbdata.BBData`.

**Axis order of `tiedbeam_power` is `(freq, beam/pol, time)`** — 903 frequency channels × 2 beams/pols × 75,913 time samples — confirmed directly from the array shape against `index_map/freq` (903) and `index_map/time` (75,913) lengths, not inferred. This resolves the disagreement your past notes had; whichever notes said `(time, beam, freq)` were wrong for this file. I'll re-verify this holds across a few more of the 31 files in Phase 3 rather than assuming uniformity.

## 4b. Ablation study results (source of the `fpga_handoff` models)

From `ablation_results/all_results.json`, `ablation_results/seed_check_results.json`, and `scripts/ablation_arch.py`. All models trained on `augmented_dataset/lstm_dataset_v6.npz` (`data_path` in the job dicts — confirmed same dataset feeding `train_cnn.py`), balanced test set of 308 samples (154/154). Full per-seed rows are in the JSON files; winners only, below:

| Sweep (fpga_handoff group) | Winning config | Filters | Pool schedule | Params | Original run test_acc (seed 42) | Deployed-weights test_acc (seed used in `fpga_handoff`) |
|---|---|---|---|---|---|---|
| group1 — filter-count | `baseline` (alias `f4`) | `[2,4,4,4,4]` | 5×(2,2) | 8,797 | 0.9643 | **0.9675** (seed 142 — confirmed by exact weight match) |
| group2 — pooling-schedule | `pool_front_loaded` (alias `b_front_loaded`) | `[4,8,4]` | `[(4,4),(4,4),(2,2)]` | 8,885 | 0.9643 | **0.9643** (seed 42, original run — confirmed by exact weight match) |
| group3 — time-resolution | `baseline` (alias `r1`, i.e. unmodified) | `[2,4,4,4,4]` | 5×(2,2) | 8,797 | 0.9643 | **0.9675** (same weights as group1 — r=1 has no `time_coarsen` layer, so it IS the baseline) |

Runner-up context (why each winner was picked):
- **group1 (filter count):** f1=0.961, f2=0.958, **f4=0.964**, f8=0.912, f16=0.974, f32=0.971, f64=0.974. f4 isn't the single highest test_acc in the whole sweep (f16/f32/f64 edge it out slightly), but it's the best value under ~9k params before the big f8 dip; the larger models weren't carried forward for synthesis. Worth flagging to your advisor if "best accuracy regardless of size" vs. "best accuracy near the current ~9k-param budget" is the framing they care about — the ablation study picked the latter.
- **group2 (pooling schedule):** a_uniform=0.964, **b_front_loaded=0.964**, c_back_loaded=0.955, d_freq_preserving=0.964, e_time_preserving=0.909. Three variants tie at 0.9643; `b_front_loaded` was preferred, val_acc (0.9448) was the tiebreaker over `a_uniform` (0.9318) and `d_freq_preserving` (0.9448, tied on val too — chosen ahead of it isn't explained in the JSON, possibly training-time or a manual call not recorded in the artifacts).
- **group3 (time resolution):** r1=0.964 (=baseline), r2=0.945, r4=0.948, r8=0.958, r16=0.955, r32=0.909 — accuracy degrades once time is coarsened at all, so the baseline (no coarsening) wins outright.

A second-generation, 5-seed-per-config rerun exists in `ablation_results_v2/raw_results.jsonl` (60 rows, mtime 2026-09-18 19:03 — **after** `fpga_handoff` was populated at 11:53). Since it postdates the handoff export, it did not feed the deployed weights; I did not cross-check it further unless you want a stability read using the larger seed count.

## 5. Out of scope but present (noted for completeness)

- `models/lstm_v1`, `lstm_v2`, `lstm_v3`: LSTM/Conv1D+LSTM hybrid models (38k/30k/30k params) on `(256,256)` flattened-ish input, trained via `lstm_training.ipynb` / `lstm_training_2.ipynb`. Not part of the CNN→FPGA candidate set; not investigated further unless you want them.
- `ablation_results/` and `ablation_results_v2/`: two generations of an ablation study (filter-count sweep, pooling-schedule sweep, time-resolution sweep, seed-check reruns) with ~55 small `.keras` checkpoints total, driven by `scripts/run_ablation_grid.py` and `frb_ablation_study.ipynb`. These look like architecture-search experiments feeding into the "5-block" and "3-block" design decisions, but are not themselves synthesis candidates.
- `frb_pooled_hls.zip`: a zip archive, appears to be a packaged copy of the `frb_pooled_hls/` directory. Not opened; noted by size/date only if needed later.
- `cnn_training.ipynb` / `cnn_training2.ipynb` / `cnn_training_v3.ipynb` / `cnn_training_v4.ipynb`: multiple generations of CNN training notebooks in the repo root — not yet cross-checked against which produced which model file. Flagging as a place more confusion could be hiding if you want me to dig further in a later phase.

## Open issues carried into later phases

1. **`train_cnn.py` doesn't build the model that's saved in `models/cnn_v1/`** (3-block script vs. 5-block artifact) — **per your decision, noted as a provenance gap and not investigated further.** `models/cnn_v1` is treated in Phase 1 onward only as a labeled reference point ("July 2026 baseline instance"), not as a synthesis candidate.
2. ~~`fpga_handoff/group1` and `group3` byte-identical duplicate weights~~ — **resolved in §4b**: both are the same ablation-study "baseline" architecture (group3's winning `r=1` config has no time-coarsening layer, so it IS the baseline; group1's winning `f4` config also IS the baseline). Not a bug.
3. **No synthesis reports exist for any of the six `fpga_handoff/group*` projects** — only `frb_pooled_hls` was actually carried through Vitis. **Per your decision:** Phase 2 will report analytic MACs/memory for the handoff models (group1, group2) but no resource/timing numbers, since none exist in this repo. You're checking the EC2 instance separately for those.
4. **`augmented_dataset/lstm_dataset_v6 - Copy.npz` predates the real `v6.npz` by ~4 months** despite the name — likely mislabeled, not investigated further.
5. Two distinct 64×64 "pooled" model files exist with the same param count (844) but different architectures: `classifier/frb_classifier_pooled.h5` (has Dropout, matches the synthesized project) vs. `classifier/model_outputs/frb_classifier.h5` (no Dropout, has a companion `performance_report.txt` claiming 100% test accuracy on only 1,125 test samples with just 5 positive-class examples). **Per your decision: `classifier/model_outputs/frb_classifier.h5` and its performance report are excluded from all reported numbers going forward.**
6. *(new)* `ablation_results/all_results.json` doesn't record why `b_front_loaded` was preferred over the tied/near-tied `a_uniform` and `d_freq_preserving` group2 variants (see §4b) — a val-accuracy tiebreaker explains it vs. `a_uniform` but not vs. `d_freq_preserving`, which tied on both metrics. Not investigated further unless you want it.
7. *(new)* `group1_f4`'s originally-reported 96.4% (cited in `context/cnn_synthesis(1).ipynb`) and the actual deployed `fpga_handoff/group1` weights' 96.75% (seed-142 rerun, see §4b) are two different numbers for "the 5-block baseline's accuracy" depending on which trained instance you mean. Phase 1/2 will cite 96.75% for the deployed weights specifically.

---

**Proceeding to Phase 1** with the models/datasets you specified: the synthesized pooled model (`frb_pooled_hls` / `classifier/frb_classifier_pooled.h5`), `fpga_handoff/group1` (baseline, both precisions), `fpga_handoff/group2` (front-loaded, both precisions), and `models/cnn_v1` as a labeled reference row. Datasets: `lstm_dataset_v6.npz`, `fpga_handoff/X_test.npy`+`y_test.npy`, and the pooled model's dataset (to be identified in Phase 1 — likely `classifier/fetch/*.hdf5` given the 7,495-sample count in the excluded `model_outputs/performance_report.txt`, but I'll confirm independently rather than assume).
