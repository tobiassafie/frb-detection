# Phase 2 (final): cnn_v1 evaluation + augmentation provenance

Scope per your last message: `fpga_handoff` and ablation folders dropped. Read-only; inference only, no training.

## 1. `models/cnn_v1/model.keras` evaluated on `lstm_dataset_v6.npz` test split

**Preprocessing used:** add a trailing channel dim (`X_test[..., np.newaxis]`) — **no normalization of any kind.** Confirmed this is what `train_cnn.py` does, and it's also what all four `cnn_training*.ipynb` variants do (identical loading cells in `cnn_training.ipynb`, `cnn_training2.ipynb`, `cnn_training_v3.ipynb`, `cnn_training_v4.ipynb` — none normalize either). `cnn_training_v4.ipynb`'s own `CONFIG` (`experiment_name: "cnn_v1"`, filters `[2,4,4,4,4]`, dense `[32]`, dropout `0.3`, sigmoid output) is an exact architecture match to the saved model, but its mtime (2026-08-18) is *after* `models/cnn_v1/model.keras` (2026-07-21), so it can't be the exact run that produced it — `cnn_training.ipynb` (2026-07-20, one day before) is the closer chronological match. Not investigated further, per your standing instruction on this provenance gap; it doesn't affect the normalization answer, which is consistent across every candidate script.

**Results** (308 test samples, 154 FRB / 154 noise, threshold 0.5 on the sigmoid output):

| Metric | Value |
|---|---|
| Accuracy | **97.40%** |
| Confusion matrix | TN=154, FP=0, FN=8, TP=146 |
| FRB precision | **1.000** |
| FRB recall | **0.9481** |
| ROC AUC | **0.9817** |

All 8 errors are false negatives (missed FRBs); zero false positives.

## 2. `augmented_dataset/lstm_dataset_v6.npz` provenance (from `frb_augmentation_v6_with_dm.ipynb`)

`frb_augmentation_v6.ipynb` (2026-04-01) does **not** depend on v5's code or outputs — v6 is a full rewrite. Its own header cell says v5's flaw was that the model learned a texture shortcut (FRB patches looked emptier than noise patches); v6's fix is injecting redispersed burst signal onto real noise backgrounds so both classes have matching pixel statistics. `frb_augmentation_v6_with_dm.ipynb` (2026-08-18) is a deliberate, documented byte-identical rerun of v6 (same seed, same RNG call order — its own comment states this explicitly) that additionally saves a per-sample DM value; its cached cell outputs match the actual `lstm_dataset_v6.npz` exactly (2052 samples, 1436/308/308, 495.5 MB), so its logic is what I'm reporting from.

**Pipeline:** of the 31 raw CHIME files in `realtime_data/FRBs/`, only **30** made it into the DM catalog (`dm_catalog.json`) that feeds this notebook — one was dropped earlier (`dm_extraction(1).ipynb`, not re-checked here). Of those 30:
- **29** passed the burst-quality filter (SNR ≥ 3.0); `FRB20190627D` (SNR 2.6) was dropped entirely.
- Of those 29, **27** actually produced valid FRB patches — `FRB20181231C` (102 live channels) and `FRB20190502B` (138 live channels) produced zero valid patches (too narrow-band for the injection/redispersion step to pass its validity checks).

**FRB samples — none are literal unmodified real detections.** Every FRB-class sample is a synthetic composite: the real burst's signal is isolated (background-subtracted), redispersed at a target DM, scaled to a randomly drawn SNR (uniform 0.5–5.0×), and added onto an unrelated real noise background. The only thing that varies between "real" and "augmented" is which DM value is used:
- **216 samples** ("original DM," 27 files × 8 temporal positions each) — redispersed at the burst's own real catalog DM.
- **810 samples** ("DM-augmented," 27 files × 30 draws each) — redispersed at a synthetic DM drawn uniform(100, 2500) pc/cm³.
- **Total: 1,026 FRB samples, all derived from only 27 distinct real events** (~38 synthetic variants per real event on average, uneven only in that 2 of the 29 candidate files contributed zero).

**Noise samples come from the same 29 real CHIME files** (sliding time-windows outside each burst's exclusion zone), but this extraction was extremely unproductive: **only 5 raw noise patches passed validity checks across all 29 files combined.** Since noise (5) was the minority class relative to FRB (1,026), the pipeline oversampled noise **with replacement** up to 1,026 — meaning **the final dataset's 1,026 noise-class samples are built from just 5 unique real noise images, each repeated ~205 times on average.**

**Split is by individual sample, not by source event.** `train_test_split` (sklearn, stratified only on the class label) runs on the fully shuffled, flattened array of 2,052 individual patches — there is no grouping by source file or by the underlying noise patch. Two direct consequences:
- Different DM-augmentations of the *same* real burst can land in both train and test.
- Because only 5 unique noise images exist, it is essentially certain that **identical noise-class images appear in train, val, and test simultaneously** — a real train/test leakage risk for the noise class specifically, on top of the near-certain leakage across FRB augmentations of the same source event.

This likely inflates `cnn_v1`'s measured accuracy above (97.40%) relative to what it would see on genuinely held-out real observations, given how the noise class in particular is constructed.
