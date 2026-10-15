# NOTES_v6 — v6 benchmark + streaming simulation (2026-10-01)

## Stage 0 — audit (done before any training)

Sources read: `frb_streaming_pipeline.ipynb` (cell/line numbers refer to the notebook's code
cells; line numbers count from 1 inside each cell), `scripts/frb_train.py`,
`scripts/v7_build_dataset.py`, `scripts/v7_build_mixed.py`, `scripts/v7_common.py`,
`scripts/eval_headline.py` and `scripts/classical_baseline.py`.

### a) Independent fixed-size patches, or a time-ordered stream?

**Independent fixed-size patches everywhere. Nothing is fed as a causal, chunk-by-chunk stream.**

* **v7 training/eval:**
  * Each sample is one 256×256 patch cut at a random start from a file
    (`v7_build_dataset.py:95-102`, `window()`, `s = rng.integers(0, T - PATCH_T + 1)` at line 101;
    the same function is used by `v7_build_mixed.py`).
  * Training batches are drawn from a random permutation of sample indices
    (`frb_train.py:200-227`, `BatchSeq`, `self.rng.permutation(idx)` at lines 213/227).
  * Val/test are the same kind of independent patches, predicted in one batch
    (`frb_train.py:188` `predict`; `eval_headline.py:38-40`).
* **Streaming notebook:**
  * Patches are cut with a sliding window over each file (cell 13, lines 158-171:
    `for start_t in range(0, n_time - patch_time + 1, stride)`, stride 128, window 256).
  * Those windows are then pooled, shuffled and split at random (cell 31, lines 5-7 and 17-19:
    `np.random.permutation`, `train_test_split(..., stratify=y)`). So the model never sees a
    time-ordered sequence; overlapping windows from the same file land in both train and val.
  * Windows are **native 2.56 µs samples**: 256 samples = 0.66 ms (cell 4, lines 25-27;
    no time downsampling in `load_frb_file`, cell 13 lines 14-17). The 168 ms v6/v7 patches are
    not the same thing.
* **The notebook's training/streaming parts never actually ran on this data:**
  * The saved outputs show extraction failing for 29/31 files (`IndexError: boolean index did
    not match`, a channel mask built from the 903-channel first file is applied to files with
    108-895 channels; cell 15 lines 12-14 and 21-22, cell 13 line 31).
  * Only 592 patches survived, **2 of them positive** (cell 15 output).
  * Cells 20 onward (RFI loading, training, streaming) have **no saved outputs**, so there is
    no record that they ever executed.

### b) High-frequency-first arrival (partial burst), or the whole dispersed burst?

* **v7: the whole (residual-)dispersed burst, placed inside the patch; arrival order is never
  simulated.**
  * The template is dispersed over the full patch in one step (`v7_build_dataset.py:105-121`,
    `inject`; `canvas = shift_channels(canvas, d + frac)` at line 121), centred at
    `t_centre ~ U(0.15, 0.85)·256` (line 178; `v7_build_mixed.py:114`).
  * Measured over all 16,800 v7 FRB samples: on average **97.3 %** of the live channels'
    arrivals fall inside the patch; 77.3 % of samples contain the complete sweep, and none
    contain less than half of it.
  * Partial sweeps only happen by accident at the patch edge, not as "only the
    high-frequency part has arrived yet".
  * Note: in these CHIME baseband arrays the stored residual sweep runs *low-frequency-first*
    (fitted residual DM < 0, `DM_RANGE = (-12, -0.5)` at line 40). Physical
    high-frequency-first order only appears if the data are re-delayed with the true DM.
* **Notebook:**
  * `streaming_inference` (cell 40, lines 8-40) takes a *fixed* patch and progressively
    zeroes / un-zeroes whole frequency channels according to a dispersion-delay schedule
    (lines 30-35: `partial[:, arrived] = spec2d[:, arrived]`). The patch content never moves
    in time.
  * The DM is drawn at random, unrelated to the patch (cell 42 line 10:
    `dm = np.random.uniform(300, 1000)`; cell 44 line 7: `uniform(200, 1500)`). Those DMs
    give 0.5-29 s of delay across the band, against a 0.66 ms window. So it is a channel-mask
    schedule, not a physical arrival simulation.

### c) Any measurement of detection latency or false alarms per unit time?

* **v7: none.** The metrics are per-sample accuracy / precision / recall / F1 / AUC / recall@1 %
  FPR (`frb_train.py` `metrics`; `eval_headline.py`). There is no clock and no stream duration,
  so nothing per unit time.
* **Notebook:**
  * It defines a "trigger %" (the % of the delay schedule at which P ≥ 0.5; cell 44,
    lines 1-12) and counts "false triggers" **per sample** (cell 44 lines 20-27). Neither is a
    time-to-detect in seconds nor false alarms per hour.
  * As noted above, these cells have no recorded outputs.
* The only latency figures in the repo are hardware latencies: the csynth report of
  frb_pooled_hls, and my `est_latency_us`. Neither is detection latency.

**Conclusion → Stage 2 is needed:** there is no time-ordered, causal streaming evaluation
anywhere in the repo. I build one (`scripts/stream_sim.py`, see Stage 2 below).

## Stage 1 — v6 benchmark: decisions & incidents

* **Which v6 file is "original".** Git HEAD's `lstm_dataset_v6.npz` and the on-disk file have
  the same samples, split and labels, but pixel values differ by ≤ 4e-6 (float rounding from the
  `_with_dm` rerun). The on-disk file is byte-identical to `fpga_handoff/X_test.npy`, i.e. the
  data behind every 96–99 % number, so it is used as-is. Original train/val/test split
  (1436/308/308), no re-splitting.
* **Inputs.** v6 patches are (256 time × 256 freq) per-patch z-scored, the same orientation as
  v7 (index 0 = highest live channel). Each model gets the patch block-mean-pooled by its v7
  `(t_pool, f_pool)`:
  * CNNs: 128×128 (2×2 mean);
  * LSTM/GRU: 64 timesteps × 8 channels (4×32 mean).

  No other preprocessing. In particular there is **no per-channel renormalisation**: the v6
  models see v6's own normalisation. Stored as float16, as for v7.
* **Training data on v6.** v7's cross-file mixed pool and RFI augmentation cannot be built
  from v6 (no file/source ids, no backgrounds), so v6 models train on v6 train as-is.
* **Epoch budget.** First attempt: literally the v7 winners' 30 epochs / patience 8. On v6
  (1436 samples → ~22 optimizer steps/epoch vs ~320 on v7) every CNN stayed at loss 0.693
  (chance) and stopped by epoch 9–21, and the LSTM was still improving at epoch 30.
  Re-run with the same architecture / lr / batch / class weights and a **step-matched budget:
  ≤ 400 epochs, patience 60** (`stage=v6_bench`). These are the numbers reported.
* The short-budget results are kept here because their run JSONs were lost: my relabel script
  deleted them without writing the renamed copies, and the relaunch overwrote the log; models
  survive as `results/models/v6e_*.keras`. Values copied from that run's output (v6 val acc /
  v6 test acc / v6 test AUC):

| model (v7 source) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| s1mB_t0041_rfi | 0.500 / 0.500 / 0.619 | 0.494 / 0.539 / 0.661 | 0.494 / 0.539 / 0.618 |
| loc_t0041_b32 | 0.500 / 0.500 / 0.523 | 0.500 / 0.500 / 0.635 | 0.494 / 0.539 / 0.624 |
| loc_t0041_A_w8k5bn (68k BN) | 0.536 / 0.510 / 0.619 | 0.500 / 0.500 / 0.672 | 0.546 / 0.584 / 0.685 |
| LSTM s3r_lstm_u16_T64_F8… | 0.708 / 0.724 / 0.780 | 0.789 / 0.831 / 0.891 | 0.539 / 0.588 / 0.748 |
| GRU s3r_gru_u16_T64_F8… | 0.692 / 0.695 / 0.821 | 0.692 / 0.701 / 0.790 | 0.588 / 0.617 / 0.691 |
* **Dead seeds.** With the step-matched budget most runs learn, but some seeds never leave
  loss = ln 2: the network outputs a constant (`v6_loc_t0041_b32_s0`,
  `v6_loc_t0041_A_w8k5bn_s2`, sweep `v6_sw_r128_avg_gap_d16`). With the same recipe they all
  train on v7. Avg-pool + GAP on v6's per-patch z-scored input seems prone to this, because
  v6's burst is a thin sliver in the top channels and GAP dilutes it. They are reported as-is
  (test 0.50), not re-seeded.
* **Dedup row caveat.** All 154 v6 test noise images (and 5 FRBs) appear verbatim in train, so
  the deduplicated test set is **149 FRBs and 0 noise**. Only recall is defined on it, and a
  constant "FRB" predictor (e.g. the dead seeds) scores 1.00. Read it together with the full-test
  specificity.
* **Cross-dataset normalisation.** Each model gets the other dataset's patches renormalised to
  its *own* training convention (v6 models: per-patch z-score; v7 models: per-channel robust).
  Results are ~chance in both directions (v6→v7 0.41–0.59, v7→v6 0.47–0.53): the two datasets
  describe different signals (v6 = DM 100–2500 slivers in the top channels of a
  stretched-to-256-channel band; v7 = real residual sweeps on a fixed 400–800 MHz grid).
* **Sweep budget (≤ 2 h).** Grid: input res {64,128,256} × pool {avg,max} × head {GAP,flatten}
  at dense 16, plus 2 filter-width variants; then dense {8,32} on the top config; top-3 configs
  get 3 seeds. Running in parallel with the benchmark, the sweep had passed its 2 h budget by
  03:45. I stopped it there: **the four 256×256 configs and the two filter-width variants did
  not finish and are not reported.** Top 3 by v6 val (tie-break val loss):
  r128/avg/flatten, r128/max/flatten, r64/avg/flatten.
* **QAT on v6.** QKeras 8-bit QAT fine-tuned from each v6-trained `s1mB_t0041_rfi` seed: test
  0.942 / 0.938 / 0.955. Naive PTQ inside the QKeras graph loses much more on v6 (0.60–0.87)
  than on v7 (0.94). v6 inputs are per-patch z-scored with a long bright tail, so the
  activation ranges are wider.

## Stage 2 — streaming simulation (built: `scripts/stream_sim.py`, `scripts/stream_eval.py`)

**Stream construction (v7-style)**
* Each held-out file becomes a 10-min "session" (val files: 6 min) of continuous data.
  It is made of random ≥ 64-bin segments of that file's real, burst-in-painted background (both
  pols, random time reversal), 0.655 ms × 256 channels.
* Bursts use the same file's template, injected every U(1, 3) s, with |residual DM| ~ U(0.5, 12)
  and optimal S/N ~ logU(3, 50).
* The test stream is 1.0 h (1,716 bursts) per orientation; the val stream is 0.5 h.
* Seeds are fixed, so every model, including the QAT ones scored in a separate legacy-Keras
  process, sees identical data.

**Orientation**
* *stored* = the sweep direction present in every v6/v7 training sample (low frequency first,
  as stored in the CHIME baseband arrays).
* *physical* = high frequency first, the order a real-time stream delivers.

**Windows and normalisation**
* Causal sliding window of 256 bins, stride 64 (= 1/4 window). The window ending at t only
  uses data before t.
* Each window is normalised on its own data: per-channel robust for v7 models, per-window
  z-score for v6 models. It is then pooled to the model input, as in training.

**Metrics**
* Per burst: detected or not, at the first window with p ≥ threshold that overlaps
  [first arrival, last arrival + 1 window].
* Time-to-detect = end of that window − first channel arrival.
* Fraction of the live band that has arrived at that moment.
* False alarms = clusters of consecutive above-threshold windows outside every burst interval,
  per hour of stream.

**Thresholds**
* Lowest threshold with ≤ 10 FA/h on the matching **val-file** stream; also fixed 0.5.
* **Caveat found:** FA/h measured on the val files does not transfer to the test files. The
  test files have more/different RFI, so the same threshold gives 80–500 FA/h on test. Treat
  test FA/h as the honest number and the threshold as the val-chosen operating point.

**Stream-trained CNN** (`--train_pool stream_primary`, `scripts/v7_build_stream_pool.py`)
* Causal windows cut from synthetic TRAIN-file streams, both orientations 50/50.
* The burst's first arrival is uniform over [−span, 256) relative to the window, so windows
  hold the leading part only, the whole sweep, or the trailing part.
* Labelling:
  * positive if the in-window S/N ≥ 5;
  * negative if there is no burst energy in the window;
  * dropped if ambiguous (6,296 dropped).
* 13 % of positives contain < 50 % of the band.
* Same mixed templates, RFI and sub-band augmentation as `mix_primary_rfi`.
* Same architecture/hyper-parameters as `s1mB_t0041_rfi`; 3 seeds (`strm_t0041_s0..2`). The
  stream evaluation uses the best-val seed (s1).

**v6 stream (comparison only)**
* v6 has no continuous data. Its stream tiles the v6 test patches end-to-end: each FRB patch is
  followed by 3–10 noise patches drawn from the 5 unique v6 noise images. That gives 0.052 h,
  154 bursts; the val stream is built the same way from v6 val.
* Its false-alarm numbers are dominated by the artificial patch-to-patch discontinuities, and
  its S/N is undefined, so only the all-burst detection rate is reported. **Not a
  meaningful FA estimate.**

**Model selection for streaming**
* For v6-trained architectures with dead seeds, the best-val seed is used: `v6_s1mB_t0041_rfi_s2`,
  `v6_loc_t0041_b32_s1`, the LSTM s1 and the GRU s1.
* v7 models are the original single runs.
