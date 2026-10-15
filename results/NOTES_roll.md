# NOTES_roll — causal rolling LSTM f(t) → P(FRB | data up to t) on the original v6 split (2026-10-15)

**v6 benchmark - leaky split, comparison only.**
* Data: `augmented_dataset/lstm_dataset_v6.npz` as on disk, original split.
* Lookup baseline 0.984; patch ceiling 0.9838.

## Stage 0 — audit (`scripts/roll_prep.py` → `augmented_dataset/v6bench/roll_meta.npz`, `results/fig_roll_onsets.png`)

**What v6 stores per sample.** Only `X`, `y` and `dm` (FRB rows; NaN for noise). Nothing about
burst position, width or S/N. Those are drawn inside `frb_augmentation_v6_with_dm.ipynb` (cell 17:
`pos`, `target_snr`) but never saved. The stored DM (100–2500) is the redispersion DM, whose delays
(seconds) far exceed the 168 ms patch. **It does not describe the sweep that is actually visible**,
so onset/end cannot come from the dispersion law with `dm`.

**Orientation.**
* Frequency index 0 = highest channel: `load_spectrogram` keeps the CHIME file order, 800 → 400 MHz.
* The visible sweep in v6 patches runs **low frequency first** along increasing time (see the top
  row of `fig_roll_onsets.png`; fitted track slopes 0.35–0.8 bins/channel). It is the residual sweep
  of the stored CHIME baseband alignment, not physical arrival.
* **Conversion to physical high-frequency-first order: reverse the time axis**, `x_phys = x[::-1, :]`.
  This is applied to every split for training **and** evaluation, noise samples included, so both
  classes see the same transform.

**Recovering per-sample burst onset/end.**
1. Every v6 FRB sample is `normalize(bg_k + burst)` with `bg_k` one of the **5 unique noise images**.
   A least-squares fit `x ≈ a·N_k + c` (trimmed refits) against each stored noise image gives the
   residual `burst/s`.
2. **Bad samples:** a residual of ≈ 0 everywhere (max energy per bin < 1e-6) means the sample is
   pixel-identical to a noise image. Found **36 train / 10 val / 5 test**, exactly the known counts.
   They are **excluded** from per-timestep training and from all rolling metrics.
3. **Sweep track:** a straight track `t(ch) = t0 + k·(255 − ch)` (stored frame) is searched on the
   per-channel robust-normalised sample (steady RFI rows cancel), maximising the summed z along the
   track (k ∈ [0, 2]). `track_score` is that z-sum/√n.
4. **Onset / end (physical frame):** the earliest / latest physical arrival among channels that
   actually carry the sweep (along-track z, smoothed over 9 channels, > 1.5).
   * Test: median onset bin 80, median duration 48 bins.
   * Weak tracks (score < 8): 141 train / 19 val / 32 test FRB samples. They are kept, with a
     less certain onset; latency is also reported for score ≥ 8 only.
5. Verified visually on 5 test FRB samples (`results/fig_roll_onsets.png`): the fitted track lies on
   the visible sweep, and onset/end bracket it in the reversed frame.

## Per-timestep label definitions
* **onset (default):** y_t = 0 for every step of a noise sample; for an FRB sample y_t = 0 for
  t < onset and y_t = 1 for t ≥ onset ("a burst has started by now").
* **window (variant):** y_t = 1 only for onset ≤ t ≤ end, else 0 ("a burst is arriving now").
* With time pooling ×p, onset/end are mapped to `onset // p`, `end // p`.

## Model / training
* Physical-order patch → causal time max-pool ×p (non-overlapping blocks, emitted when complete) →
  Conv1D(c, k1, relu), a per-step dense layer → LSTM/GRU(u, return_sequences=True, unidirectional) →
  Conv1D(16, k1, relu) → Conv1D(1, k1, sigmoid) = P(FRB | data up to t). There is no lookahead
  anywhere: every layer after the pooling is per-step or causal-recurrent.
* Loss: per-step BCE with per-step class weights (positive steps weight 0.5/frac_pos, negative
  0.5/frac_neg, computed on train).
* Recipe: lstm_v3 (Adam 1e-3, batch 128, ReduceLROnPlateau(0.5, patience 5), 150 epochs,
  **no early stopping**, the best-val-loss weights restored at the end by a custom callback).
* Retry: a run whose val per-step prediction std is < 1e-3 or whose val per-step AUC is < 0.6 is
  rerun with seed + 1000·k. Attempts are logged; nothing is dropped.
* Streaming: a stateful twin (batch 1, one time bin per call, carried state, `reset_state` per
  sample) gets the same weights. Its outputs are asserted equal to the batch outputs to 1e-5.

## Stage 1 — outcomes (11 runs, ~5 min wall clock; full table in results/summary_roll.md)
* **GRU32 / front 16 / pool ×4** is best: patch acc @1 % FA threshold 0.990 ± 0.005, per-step AUC
  0.986, latency ≈ 0 bins (3 seeds). It is followed by GRU32 / front 32 (0.993 @thr, step AUC 0.992,
  1 seed). LSTM64 3 seeds: 0.925 ± 0.074 @thr.
* Top-2 for the 3-seed rule were chosen by **val** patch accuracy @0.5: GRU32 (0.997) and LSTM64
  (0.876).
* Patch accuracy at a fixed 0.5 swings with seed (GRU32: 0.983 / 0.993 / 0.901) because the patch
  score is a max over 64 steps. The val-chosen threshold removes most of that swing.
* Detection "before onset" (30–54 % for GRU32) reflects the uncertainty in the recovered onset (it
  is the first channel above a smoothed 1.5 σ along a straight-track fit) plus the ×4 pooling (±4
  bins). It is not lookahead: the streaming check proves causality.
* Not done here (later stages of the plan): mid-burst truncation tests, v7 cross-check, streaming
  false alarms per hour.
