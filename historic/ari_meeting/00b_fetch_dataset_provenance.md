# Provenance check: `classifier/fetch/train_data.hdf5` and `test_data.hdf5`

Requested before Phase 2. Read-only investigation — nothing modified. Scripts used are inline
Python one-liners run via Bash, not saved as reusable scripts since this was a one-shot forensic
check (say the word if you want them saved).

## Conclusion

**FETCH-format public/external data, not CHIME-derived.** Confidence: high, based on converging
structural, visual, and chronological evidence (below). Not 100% certain — I did not have network
access to diff this file byte-for-byte against a downloaded copy of the actual public FETCH
training set, so I can't rule out an unusual internal pipeline that reproduced FETCH's exact
schema from CHIME data. But nothing in this repo shows or claims to do that, and everything about
the file's structure and content matches the public FETCH single-pulse candidate format rather
than anything CHIME-specific.

## Evidence

### 1. No code anywhere in the repo writes these files

Searched every `.ipynb`, `.py`, and `.md` in the repo (excluding `.venv`) for:
- The literal filenames `train_data.hdf5` / `test_data.hdf5` — **10 hits, all reads.** The four
  notebooks that touch these files (`classifier/frb_classifier_pooled.ipynb`,
  `classifier/cnn_dropout.ipynb`, `classifier/lucky.ipynb`,
  `classifier/simple_model_accuracy.ipynb`) are near-identical variants that all open the file
  with `h5py.File(train_file, "r")` — read mode only. None contain a `model.save`-equivalent
  write call for these HDF5 files, no `h5py.File(..., "w")`, no `create_dataset(...)`.
- `h5py.File(..., 'w')` or `.create_dataset(` anywhere in the repo — **zero hits.**
- Download/fetch code (`requests`, `urllib`, `wget`, `curl`, a URL) anywhere near these files —
  **zero hits.** The `classifier/frb_classifier_pooled.ipynb` notebook (12 cells total) has no
  download cell; it starts directly from `train_file = "fetch/train_data.hdf5"` already present
  on disk.
- I also checked `frb_realtime_classifier.ipynb` (which *does* write files, but a different set —
  `classifier/model_outputs/frb_classifier.h5` and friends, from `realtime_data/FRBs/` +
  `realtime_data/RFIs/LOFAR_Full_RFI_dataset.pkl`, already excluded from your numbers). It's
  unrelated to `classifier/fetch/`.
- `dm_extraction(1).ipynb` — a DM-cataloging notebook — only reads `realtime_data/FRBs/*.h5`
  (the raw CHIME baseband files) to build a DM catalog for the *separate* augmentation pipeline
  (`frb_augmentation_v5/v6.ipynb` → `augmented_dataset/lstm_dataset_v6.npz`). It never touches
  `classifier/fetch/`.

No shell history files exist anywhere in the repo (checked with a glob for `*history*` and
`.bash_history` specifically — the only hits were Python/Keras library internals and this
project's own `*_history.json`/`.csv` training-curve logs, unrelated).

**There is no code path in this repository that could have produced `classifier/fetch/train_data.hdf5` or `test_data.hdf5`.** They exist only as pre-made inputs that four different notebooks all independently read from the same relative path.

### 2. No origin comments, URLs, or download cells in the pooled notebook or its siblings

Read all 12 cells of `classifier/frb_classifier_pooled.ipynb` in full, plus the three sibling
notebooks (`cnn_dropout.ipynb`, `lucky.ipynb`, `simple_model_accuracy.ipynb`, which are
near-duplicates of it). None mention where `fetch/` came from. No comments referencing CHIME,
FETCH, a paper, a survey name, or a URL. A repo-wide search for `agarwal`, `zenodo`, `HTRU`,
`PALFA`, `GBNCC`, `Parkes`, `Effelsberg`, `Arecibo`, and `FETCH ... (paper|survey|public|dataset)`
returned no genuine hits — the handful of regex matches were all inside embedded base64 PNG blobs
in notebook outputs (coincidental byte-sequence matches, not text).

### 3. File structure and pixel content

Inspected `classifier/fetch/train_data.hdf5` directly (read-only, small slices):

- **Keys:** `data_dm_time` `(40000, 256, 256, 1)` float32, `data_freq_time`
  `(40000, 256, 256, 1)` float32, `data_labels` `(40000,)` bool. **This exact three-key naming
  convention — `data_freq_time` / `data_dm_time` / `data_labels` — is the standard schema used by
  FETCH's candidate-maker output** (a frequency-time image, a DM-time image, and a boolean
  real/RFI label per candidate). It is not a naming convention used anywhere else in this repo's
  own pipeline (the CHIME-native pipeline's own files use `tiedbeam_power`, `X_train`/`y_train`,
  `data_labels` doesn't otherwise appear).
- **Zero metadata.** `f.attrs` is empty at the file level and at every dataset level — no
  frequency range, no telescope name, no beam/pointing info, no DM search range. This is a sharp
  contrast to the raw CHIME files in `realtime_data/FRBs/`, which (per Phase 0) carry rich
  telescope-specific attributes (`archive_version`, `__memh5_subclass:
  baseband_analysis.core.bbdata.BBData`, `delta_time`, `event_id`, etc.) and an `index_map` group
  with real frequency-channel and beam metadata. If this dataset had been derived in-house from
  the CHIME files, I'd expect at least some of that provenance to survive into the derived file;
  none did.
- **Per-sample z-score normalization**: every sample I checked has `std ≈ 1.0000` and `mean ≈ 0`
  computed independently per-image — the generic per-candidate normalization used by ML-ready
  candidate datasets (FETCH does exactly this), not raw physical units.
- **Chunking/compression** (`chunks=(1250,8,8,1)`, `gzip`) is generic HDF5, not diagnostic on its
  own, but consistent with a bulk-exported dataset rather than an incrementally-built one.

### 4. Visual inspection of the dispersion structure

Rendered 5 positive-labeled (`label=True`) samples' `data_freq_time` and `data_dm_time` images
(indices 6250–6254, chosen because they're chunk-aligned and known positive from the label array).
Saved image reviewed directly (not included in this repo per your read-only instruction, but the
description is fully reproducible from the inline commands run this session):

- **`data_freq_time` images show strong, fixed-position vertical banding** — bright and dark
  vertical stripes recurring at the *same x-coordinate* across all 5 different candidates. That
  pattern is diagnostic of either persistent RFI-flagged frequency channels or a static bandpass
  shape — both classic real-telescope artifacts. It confirms the array's horizontal axis is
  frequency (matching the "freq_time" name) and that this axis carries genuine
  instrumental/RFI structure, not a synthetic-clean spectrogram.
- **`data_dm_time` images show a textbook dedispersion "bowtie"/"nose-cone" pattern**: a sharp,
  bright, compact peak at one time/DM location, with a diagonal smear of partially-dedispersed
  signal fanning out above and below it. This is the exact, recognizable shape produced by a
  single-pulse DM-trial search (as in PRESTO's `single_pulse_search.py`, and reproduced
  extensively in FETCH's own training-candidate figures) — a real dispersed pulse imperfectly
  dedispersed at neighboring trial DMs. This specific visual signature is a strong positive
  identifier for FETCH-style candidate data; it is not something the repo's own from-scratch CNN
  training pipeline (which works directly on `(256,256)` spectrograms with no DM-trial search
  step) produces or uses anywhere else.
- I did attempt a numeric dispersion-curve fit (least-squares against a CHIME 400–800 MHz band
  model) as a cross-check; it did not converge cleanly on the individual sample I first tried
  (low single-sample SNR made an argmax-per-channel fit too noisy) — I'm not relying on that
  numeric fit for the conclusion, only on the direct visual identification of the bowtie/RFI-band
  pattern above, which is unambiguous.

### 5. File mtimes relative to CHIME raw files and the augmentation notebooks

| File | mtime |
|---|---|
| `realtime_data/FRBs/*.h5` (raw CHIME, sample) | 2024-03-20 |
| `classifier/fetch/train_data.hdf5` | **2025-10-27 15:37** |
| `classifier/fetch/test_data.hdf5` | **2025-10-27 19:59** |
| `classifier/frb_classifier_pooled.ipynb`, `cnn_dropout.ipynb`, `lucky.ipynb`, `simple_model_accuracy.ipynb` | 2025-11-16 (~3 weeks after `fetch/`) |
| `dm_extraction(1).ipynb` | 2026-04-01 |
| `frb_augmentation_v5.ipynb`, `frb_augmentation_v6.ipynb` | 2026-04-01 |
| `frb_augmentation_v6_with_dm.ipynb` | 2026-08-03 |

The chronology is consistent with (not proof of, but consistent with) the following narrative:
`classifier/fetch/*.hdf5` appeared as a ready-made dataset in late October 2025; the four
`classifier/*.ipynb` notebooks were written ~3 weeks later purely to consume it (no generation
step in between); and the CHIME-native pipeline (`dm_extraction`, `frb_augmentation_v5/v6`,
feeding `lstm_dataset_v6.npz`) was built from scratch **five months later**, using entirely
different code, a different file format, different key names, and no DM-time transform step. If
the team already had a working CHIME-derived dataset in October 2025, building an entirely
separate CHIME pipeline from raw baseband files five months later would be redundant — it's more
consistent with the October 2025 dataset being an external drop-in (FETCH) that got used for one
line of experiments while a genuine CHIME-native pipeline was developed in parallel/later for the
"real" model.

## What this changes

- The pooled model (`classifier/frb_classifier_pooled.h5`, synthesized as `frb_pooled_hls`) was
  most likely trained on **FETCH's public candidate data**, not CHIME data — despite living in a
  CHIME-focused repo and despite `classifier/frb_classifier_pooled.ipynb` printing "FRB vs RFI"
  class names. Its 95.20% test accuracy (found in Phase 1) should be read as accuracy on FETCH
  data, not as a validated number for real CHIME-detected FRBs.
- This directly affects the Phase 3 "why is the data so large" narrative if you want it to cover
  the pooled model specifically: its 256×256 (then resized to 64×64) inputs are single-pulse
  search-pipeline candidate images from another telescope/survey's pipeline, not CHIME baseband
  spectrograms — a different data-size story than the `lstm_dataset_v6.npz` / `fpga_handoff`
  pipeline, which genuinely does originate from the 31 raw CHIME files in this repo.
- I have not renamed or moved anything; `01_basics.md`'s description of the pooled dataset from
  Phase 1 is still accurate as far as file contents and shapes go — this note adds the origin
  determination on top of it. Let me know if you'd like me to add a correction line directly into
  `01_basics.md`/`00_inventory.md` or keep it as this standalone note.
