# NOTES — sweep of 2026-09-30 (assumptions, findings, caveats)

Everything here was done autonomously overnight; each assumption I had to make is listed so it
can be challenged in the meeting. Numbers are only quoted with the model / dataset / split they
belong to.

## Stage 0 — data audit (the most important part)

### 0.1 The old v6 split is leaky — earlier 96–99 % numbers are NOT valid generalisation numbers

Source: `frb_augmentation_v6_with_dm.ipynb` (byte-identical logic to v6, produces
`augmented_dataset/lstm_dataset_v6.npz`).

* **The noise class is 5 unique images.** Only 5 burst-free 256-bin windows passed the validity
  check across all 29 files; they were oversampled *with replacement* to 1,026. Verified by
  hashing `lstm_dataset_v6.npz`: exactly 5 unique noise images across train/val/test; **154/154
  test noise images (and 154/154 val) are byte-identical copies of training images.**
* **A lookup table scores 98.4 % on the v6 test split** — "predict noise iff the image is
  byte-identical to a training noise image, else FRB", no learning at all. So 96.4 % (group1_f4),
  96.75 % (fpga_handoff group1/3), 96.43 % (group2), 97.40 % (models/cnn_v1) and any 97–99 %
  number measured on that split cannot be read as generalisation accuracy.
* Every FRB sample is also *one of those same 5 backgrounds* + an injected signal, and 5 FRB test
  images are byte-identical to training images.
* The split was per-sample (`train_test_split` on shuffled patches), so ~38 augmentations of the
  same burst are spread across train/val/test.
* **The v6 "burst" was mostly an edge artefact.** v6 locates the burst with `argmax` of the
  band-averaged time series. Most raw files have a spike in the last sample, so `peak_idx` = last
  bin for most files (e.g. 295/296, 439/440, 492/494 — visible in the notebook's own printed
  output). The "burst signal" that was redispersed and injected was therefore a ±margin slice at
  the end of the file, not the FRB, for most sources.
* The raw CHIME baseband arrays are **not dedispersed**: each burst is a residual sweep spanning
  ~50–250 ms (fitted residual DM −1.4 … −9.7 pc cm⁻³ in the stored channel alignment). v6
  additionally redispersed at DM 100–2500, whose delays (seconds) are >> the 168 ms patch, so
  only a few top channels ever carry signal.

### 0.2 Rebuilt dataset: v7 (grouped by source, provenance per sample)

Scripts: `scripts/v7_cache_raw.py` → `v7_fit_bursts.py` → `v7_build_dataset.py` →
`v7_make_splits.py`. Output `augmented_dataset/v7/{X.npy, meta.csv, files.json}` (X not in git,
1.5 GB; rebuild is deterministic, seed 20260930, ~1 min).

* 31 raw files → 28 used. Excluded: `FRB20190410B` (truncated HDF5), `FRB20190627D`,
  `FRB20190628C` (no visible burst; the fit was spurious — keeping them would risk a real burst
  inside a "noise" sample). Note v6 used 27 different ones (it lost 190502B / 181231C; I keep them).
* Common frequency grid: 1024 CHIME channels 800→400 MHz binned ×4 to 256; time 0.655 ms
  (same as v6), 256-bin patches (168 ms). Files with partial band coverage keep their real
  missing channels as 0 (not stretched to fill 256 rows as v6 did). Per-pixel no-data masks are
  honoured (CHIME baseband channels have staggered time windows).
* Per file: residual DM fitted by brute-force incoherent dedispersion (S/N-maximising, both pols);
  burst template = rank-1 SVD (spectrum × profile) of the dedispersed burst window; background =
  file with the burst track in-painted by N(0,1) in the per-channel-normalised units.
* **Both classes are made from the same file's backgrounds** (random window, random pol, random
  time-reversal); FRB = background + template redispersed at residual DM ~ U(−12, −0.5), random
  arrival time, scaled to an *optimal (matched-filter) S/N* ~ log-uniform(3, 50). 200 noise +
  200 FRB per file → 11,200 samples. Class-wise pixel statistics are identical (std 0.673 vs
  0.673), i.e. no global shortcut.
* Every sample records file, group, pol, window start, reversal, S/N, peak-pixel amplitude, DM.

### 0.3 Splits (saved: `results/splits/v7_splits.json`, `v7_primary_{train,val,test}_idx.npy`)

* **primary**: by file, seed 42 → train 17 files / val 5 / test 6, June-2019 cluster spread across
  splits. Test files: 181226E, 181231C, 190210B, 190609C, 190609D, 190630C.
* **heldout_family**: train on non-June-2019 files, test on all 8 June-2019 files (441–799 MHz
  band configuration) — held-out source *family* test.
* **cv4**: 4 grouped folds over all 28 files.

### 0.4 Evaluation protocol (fixed before the sweep)

* Primary test set = all noise samples of the test files + FRB samples with optimal S/N ≥ 7
  (below ~7 no detector can be reliable; the full-range number is reported too as
  `test_allsnr_acc`, plus recall per S/N bin).
* Model selection uses **val only**; test numbers are reported, never optimised.
* Threshold 0.5 on the sigmoid; recall@1 %FPR from the ROC.

## Assumptions / deviations from the plan

* "Augmentation strength" in the sweep = minimum injected S/N kept for training
  (`train_snr_min` ∈ {3,5,7}) + optional time-reversal (`time_flip`).
* Search space additionally has head ∈ {flatten, global-avg-pool} and pool type ∈ {max, avg}.
  Both are hls4ml-supported layers; avg-pool is the S/N-optimal matched filter for faint tracks.
  Track B stays Conv+ReLU+Pool+Dense, no BatchNorm.
* Input resolution 256/128/64 = block-mean of the 256×256 patch by 1/2/4 in both axes (a cheap
  upstream binning, not part of the FPGA model).

## Stage 1 — CNN sweep: what changed along the way (chronological)

1. **Pilot at 200 samples/file** (archived in `results/archive_stage1_n200/`): best val ≈ 0.88,
   test ≈ 0.83–0.91. Regenerated v7 at **600 samples/class/file (33,600 samples)** — cheap, more
   S/N / DM / position / background diversity. Everything below uses the 600/file version.
2. **Same-file training overfits per file.** Every run's best epoch was 2–3; with patience 30
   the train accuracy keeps rising (0.66 → 0.89) while val loss triples (0.67 → 1.95). Per-file
   breakdown: FRB recall ≈ 0 even on *training* narrowband files (190627A, 181231C, 190502B,
   400–480 MHz, 29–51 live channels) and noise accuracy ≈ 0.4–0.5 on a few files (190609D,
   190519H) — i.e. the model memorises each file's single burst template + backgrounds.
   Those same-file runs stay in `sweep_results.csv` with `stage=stage1_optuna` (no `train_pool`).
3. **Fix: cross-file mixed training pool** (`scripts/v7_build_mixed.py`,
   `augmented_dataset/v7/mix_primary/`, 24,000 samples): background from one TRAIN file, burst
   template from another TRAIN file, plus random sub-band masking (p = 0.35) on both classes.
   Val/test are unchanged (same-file samples of held-out files), so no leakage is introduced.
   Same architecture: val 0.734 → 0.835, test 0.815 → 0.845. The sweep was restarted with it
   (`stage=stage1_optuna_mix`, run ids `s1m*`).
4. Input clipping to [−3, 5] was tried (inconclusive, ±5 pts noise between runs) and not used.
5. 256×256 input was dropped from the TPE space (4× slower, not better in the pilot); top
   configs are re-run at 256 explicitly.
6. **Sweep budget.** Trials early-stop by epoch 4–11 on the mixed pool, so the sweep runs at 30
   epochs / patience 8. Track B got ~45 Optuna trials and track A ~17 (its trials were 3–5× slower
   and never better). The search converged on 128×128 input, avg-pool, global-avg-pool head,
   non-decreasing filters, train S/N ≥ 3.
7. **False positives were real RFI.** At that point all test false positives came from one file
   (190609D). Its in-painted background is clean of the burst but has short, bright narrowband
   RFI dashes. Fix: a second mixed pool with synthetic RFI injected into BOTH classes
   (`MIX_P_RFI=0.5`: narrowband dashes, persistent lines, broadband zero-DM spikes;
   `mix_primary_rfi`). On the top 8 configs val went up 1–3 pts. For `s1mB_t0041`: val
   0.923 → 0.940, test 0.924 → 0.944, specificity 0.90 → 0.996. All final CNNs use it. RNNs got
   worse with it (val 0.84 → 0.75), so the RNNs use the plain mixed pool.
8. **Local search** around `s1mB_t0041_rfi` (lr, batch, kernel, width, dense, pooling, res 64,
   time-flip, train S/N ≥ 5, +BN, ×2–×4 width): nothing beat it on val by more than seed noise.
9. **Selection noise.** With 5 val files, configs within ~2 pts of val are indistinguishable;
   3-seed reruns differ by up to 4 pts on test. Hence the 3-seed / 4-fold CV / held-out-family
   rows in the summary.
10. **Classical reference** (`scripts/classical_baseline.py`): incoherent dedispersion over residual
    DM −13…−0.4 (step 0.2), 32 sub-bands, boxcar 1–16, sigma-clipped noise, threshold tuned on val.
    It scores 84.5 % on the primary test set and 74.4 % on the held-out family.

## Stage 1 QAT
* QKeras (tf_keras; run with `TF_USE_LEGACY_KERAS=1`). Per-layer integer bits come from the float
  model. QAT starts from the float weights (lr/3, ≤30 epochs, early stop on val loss).
  The model **input** is not quantised by QKeras; it is ap_fixed<16,6> in hls4ml.
* Track-B top 3: `s1mB_t0041_rfi` (8b 0.9425, 6b 0.9336; float 0.9443), `loc_t0041_w4`
  (8b 0.9339; float 0.9365), `s1mB_t0048_rfi` (8b 0.9245; float 0.9212).

## Stage 2/3 — hls4ml (NOT synthesised)
* This machine has no Vitis HLS, no g++/cl (so not even `hls_model.compile()`), no WSL distro and
  no EC2 credentials. The plan's `run_vitis_synthesis.sh` did not exist in the repo. The existing
  wrapper was `fpga_handoff/*/run_synthesis.sh`, so each project gets a `run_vitis_synthesis.sh`
  modelled on it (same Vitis 2025.1 settings, part, clock). It runs csynth only, with the same
  options/command as hls4ml's `VitisBackend.build()`, under `timeout 1200`. A timeout is recorded
  as "timeout", and failures keep the last ERROR lines of the log.
* Conversion issues hit and fixed along the way:
  * hls4ml reserves the layer name `input` (models are renamed on export).
  * QKeras models need hls4ml's Keras-2 frontend: hls4ml chooses its frontend from the installed
    `keras.__version__`, so it is temporarily reported as 2.15 for QKeras models.
* Every PTQ project sets weights/biases/results explicitly to ap_fixed<W,I,AP_RND_CONV,AP_SAT>
  (accumulators auto, input ap_fixed<16,6>) so that the local emulation and the hardware agree.
* RNN: io_parallel, Resource, `static=True` (verified `use_static = true` in parameters.h),
  RF 1/4/16, ap_fixed<12,6>/<14,6>/<16,6>. The GRU fallback and a tiny 4-unit LSTM were exported
  too, in case the main LSTM fails in Vitis.
* No synthesis failures can be reported yet, because none were run.

## Things I did NOT do / caveats
* No synthesis, no C-sim, no cosim (see above). All latency/resource numbers are `est_*`.
* Stage-1 searches excluded 256×256 input from TPE (time). Pilot runs at 256 were not better.
* Only 28 real source bursts exist. Every FRB sample is a synthetic injection of one of them
  (real morphology, real background, controlled DM/S/N). No unmodified real detection is in the
  test set, and noise comes from the same 28 observations.
* Commits: the first commit also picked up your pre-existing uncommitted `.gitignore` edit, and
  a later `git add scripts/*.py` included your untracked ablation scripts (unchanged).
