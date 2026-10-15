# Rolling (causal) RNN — P(FRB | data up to t) on the original v6 split (2026-10-15)

**v6 benchmark - leaky split, comparison only.** Definitions, orientation and the onset recovery are in `results/NOTES_roll.md`. Per-run rows:
`results/v7r_runs.csv`; run JSONs: `results/runs_roll/`; weights: `results/models_roll/`.

* **Input:** physical high-frequency-first order (v6 time axis reversed); causal time max-pool ×p;
  one time bin (256 channels) per step.
* **Model:** Conv1D(k1) front end → unidirectional LSTM/GRU, return_sequences → per-step head → sigmoid.
* **Labels (default "onset"):** noise 0 at every step; FRB 0 before the recovered onset, 1 from onset on.
* **Excluded:** the 36/10/5 FRB-labelled samples that are pixel-identical to noise (no burst).
  So test = 303 patches and the patch ceiling is 1.0 here, not 0.9838.
* **Patch score = max over t of P_t.**
* **Threshold:** "@thr" uses the lowest threshold whose VAL patch false-alarm rate is ≤ 1 %;
  "@0.5" uses a fixed 0.5.
* **Latency:** time bins (0.655 ms) from the recovered onset to the first step with P ≥ thr;
  negative = before the recovered onset. The onset itself is uncertain by a few bins.
* **Streaming:** every model was checked one bin at a time with carried state against the batch
  output (max |diff| ≤ 1.2e-7, asserted < 1e-5).
* **Runs:** 11 (6 variants + 2 extra seeds each for the top 2 + GRU with front end 32), all seed
  attempts = 1, no collapses.

## Result
* **Best: GRU32, Conv1D(16) front end, time pool ×4.**
  * 9,457 params, 9,232 MACs per step, 64 steps per 168 ms patch.
  * Patch accuracy at the val-chosen 1 % FA threshold **0.990 ± 0.005** (3 seeds).
  * Per-step AUC 0.986 ± 0.000.
  * Median detection latency ≈ 0 bins after the recovered onset (−4…+4, i.e. within one pooled
    step).
* **GRU beats LSTM** at equal or smaller size. LSTM64 is less stable across seeds.
* **Pooling ×4 beats no pooling:** 256 steps learns much worse (per-step AUC 0.90).
* **Window labels** give the best per-step AUC (0.990) but poorly calibrated patch scores at 0.5.
* **Caveats, as for every v6 number:**
  * The memorisation probe (noise accuracy with invisible noise added) is 0.61–0.86: partial
    memorisation of the 5 noise images.
  * v6 noise is 5 images, so FA rates are optimistic.

v6 benchmark - leaky split, comparison only

| model | run_ids | params | MACs/step | steps | per-step AUC | patch acc @thr | patch recall @thr | patch acc @0.5 | precision @0.5 | recall @0.5 | F1 @0.5 | patch AUC | recall@1%FPR | latency median (bins) | latency p90 (bins) | probe noise acc @thr | lookup | max stream diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU32, front 16, pool ×4 (best) | `r_gru32_f16_p4_onset`, `r_gru32_f16_p4_onset_s142`, `r_gru32_f16_p4_onset_s242` | 9,457 | 9,232 | 64 | 0.986 ± 0.000 | 0.990 ± 0.005 | 0.980 ± 0.009 | 0.959 ± 0.041 | 0.947 ± 0.075 | 0.980 ± 0.009 | 0.961 ± 0.038 | 0.991 ± 0.005 | 0.980 ± 0.009 | 0.0 ± 3.3 | 16.0 ± 8.6 | 0.677 ± 0.056 | 0.984 | 1.2e-07 |
| LSTM64, front 16, pool ×4 | `r_lstm64_f16_p4_onset`, `r_lstm64_f16_p4_onset_s142`, `r_lstm64_f16_p4_onset_s242` | 25,905 | 25,616 | 64 | 0.951 ± 0.027 | 0.925 ± 0.076 | 0.848 ± 0.155 | 0.803 ± 0.080 | 0.727 ± 0.087 | 0.993 ± 0.005 | 0.836 ± 0.057 | 0.952 ± 0.052 | 0.848 ± 0.155 | 4.0 ± 0.0 | 56.4 ± 14.6 | 0.725 ± 0.065 | 0.984 | 1.2e-07 |
| GRU32, front 32, pool ×4 | `r_gru32_f32_p4_onset` | 15,105 | 14,864 | 64 | 0.992 | 0.993 | 0.987 | 0.983 | 1.000 | 0.966 | 0.983 | 0.993 | 0.987 | -4.0 | 4.0 | 0.649 | 0.984 | 1.2e-07 |
| LSTM32, front 32, pool ×4 | `r_lstm32_f32_p4_onset` | 17,089 | 16,912 | 64 | 0.968 | 0.990 | 0.980 | 0.904 | 0.841 | 0.993 | 0.911 | 0.995 | 0.980 | 8.0 | 164.0 | 0.753 | 0.984 | 8.9e-08 |
| LSTM32, front 16, pool ×4 (base) | `r_lstm32_f16_p4_onset` | 10,929 | 10,768 | 64 | 0.946 | 0.964 | 0.926 | 0.802 | 0.713 | 1.000 | 0.832 | 0.985 | 0.926 | 32.0 | 192.0 | 0.857 | 0.984 | 1.2e-07 |
| LSTM32, front 16, pool ×4, WINDOW label | `r_lstm32_f16_p4_window` | 10,929 | 10,768 | 64 | 0.990 | 0.974 | 0.946 | 0.594 | 0.548 | 1.000 | 0.708 | 0.980 | 0.946 | 0.0 | 8.0 | 0.812 | 0.984 | 1.2e-07 |
| LSTM32, front 16, no pooling | `r_lstm32_f16_p1_onset` | 10,929 | 10,768 | 256 | 0.904 | 0.822 | 0.638 | 0.591 | 0.546 | 0.993 | 0.705 | 0.862 | 0.638 | 51.0 | 87.6 | 0.818 | 0.984 | 1.2e-07 |
