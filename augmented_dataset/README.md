# Datasets (data files are NOT in git — only metadata)

## v6 — `lstm_dataset_v6.npz` (472 MB, copy separately)
Built by `notebooks/augmentation/frb_augmentation_v6.ipynb`. `frb_augmentation_v6_with_dm.ipynb` is a
byte-identical rerun that also saves per-sample DM; it produced the on-disk file. Its input is the DM
catalogue `dm_catalog.json`, made by `notebooks/augmentation/dm_extraction(1).ipynb`.

**Construction**
1. 30 raw CHIME baseband files are time-downsampled ×256 (0.655 ms bins) and cropped to live channels.
2. The "burst" is the region around the argmax of the band-averaged time series, background-subtracted.
3. The burst is redispersed at its catalogue DM (8 temporal positions per file) or at 30 random DMs in
   100–2500, then scaled to a random peak S/N of 0.5–5 and **added onto a noise patch**.
4. Each patch is normalised (per-patch z-score) and resampled to 256 time × 256 freq.
5. Classes are balanced and the data split by `train_test_split` (per sample): 1436 / 308 / 308.

**Known issues** (details: `results/NOTES.md` §0.1, `results/NOTES_v6a.md`)
* **Only 5 unique noise images** passed the validity check. They were oversampled with replacement to
  1,026, so every test noise image also appears in training. A no-learning exact-match lookup scores
  **0.984** on the test set.
* **5 test / 10 val / 36 train FRB-labelled samples are pixel-identical to noise images**: the burst
  landed outside the patch. That caps achievable accuracy at **0.9838** test / 0.9675 val.
* For most files the "burst" argmax hit an edge spike in the last time bin. The injected signal
  therefore carries edge/RFI content; on average 61 % of its energy precedes the visible sweep.
* The visible sweep is stored **low-frequency first** (CHIME baseband alignment). The rolling models
  reverse time to get physical order.
* The split is per sample, not per source burst, so augmentations of one burst appear in train and
  test.
* Per-sample metadata recovered later: burst onset/end, no-burst flags and sweep-track score are in
  `v6bench/roll_meta.npz` (`scripts/roll_prep.py`).

## v7 — rebuilt leak-free dataset (`scripts/v7_*.py`)
* Rebuilt from the raw CHIME files with per-sample provenance (`v7/meta.csv`, `v7/files.json`).
* Burst templates are fitted from the real sweep (`v7_cache/burst_fits.json`). Noise comes from real,
  in-painted backgrounds of the same file.
* The split is **by source file** (`results/splits/`). Mixed-file / RFI training pools are in `v7/mix_*/`.
* Rebuild: run `v7_cache_raw.py` → `v7_fit_bursts.py` → `v7_build_dataset.py` → `v7_make_splits.py`
  (requires `realtime_data/FRBs/*.h5`).
