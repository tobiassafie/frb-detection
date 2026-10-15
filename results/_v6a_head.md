# v6a — winning models on the ORIGINAL v6 split (2026-10-07)

All numbers are on `augmented_dataset/lstm_dataset_v6.npz` as on disk (byte-identical to
`fpga_handoff/X_test.npy`), original train/val/test split, **v6 benchmark - leaky split, comparison
only**. Every run is in `results/v6a_runs.csv` (`results/runs_v6a/*.json`); reasoning, pre-registered
hypotheses and incidents are in `results/NOTES_v6a.md`. 26 new training runs, ~1.5 h wall clock.

## TL;DR

1. **Every saved historical model reproduces exactly.** Retraining with the original recipes also
   reproduces, within run-to-run noise:
   * `cnn_v1` 0.9740, `seedcheck_baseline_s142` 0.9675, `group1_f64` 0.9740 and `lstm_v3` 0.9643
     all match bit-for-bit from their saved weights;
   * the [2,4,4,4,4] retrain scores 0.977 vs 0.9675;
   * the lstm_v3 retrain scores 0.961 vs 0.964, but only when trained for all 150 epochs as the
     original was. Its early stop would otherwise fire at epoch 49.

   No ~99 % v6 model exists anywhere in the repo or its history, and `train_cnn(2).py` does not
   exist.
2. **The v6 test set has a hard ceiling of 303/308 = 0.9838.** 5 test samples labelled FRB are
   *byte-identical* to noise images (the injected burst landed outside the patch). Identical
   inputs must get identical outputs, so no model can exceed 0.9838 on test (0.9675 on val; 36
   such samples are in train). **99 % is mathematically impossible on this split.**
3. **Best CNN: 0.984 ± 0.000 over 3 seeds = the ceiling.** It is `s1_H3_res256*`: 256 × 256 input,
   Conv[4,8,16,16,16] + MaxPool + Flatten + Dense16, 22,561 params, val 0.9675 = also the val
   ceiling. Every real FRB in the deduplicated test is detected (149/149), with precision 1.000.
4. **Smallest CNN within 0.5 pt:** `s1_H4_half`, 2,601 params, 0.977. Quartering the model (667
   params) drops it to 0.951.
5. **Best many-to-one RNN: lstm_v3 recipe**, 0.963 ± 0.003 over 3 seeds. It reads 256 time steps
   × 256 channels, applies a 1×1 Conv1D(64), then LSTM(32) → Dense32, for 29,953 params.
   * Feeding fewer time steps with 64–128 channels per step (H6) or a Conv1D front end (H8) did
     not beat it.
   * LSTM64 on 64 × 128 reaches 0.953.
   * GRU ≈ LSTM.
6. **The caveat beside every number.** The models that reach the ceiling do so partly by
   recognising the 5 memorised noise images:
   * adding invisible N(0, 0.05) noise to the test noise drops the best CNN's noise accuracy
     to 0.42;
   * the old-recipe CNN (0.966 ± 0.008) stays at 1.00, but gives up ~2 pt of v6 test;
   * on the v7 test set and on time-reversed v6 every v6 model is near chance.

   v6 accuracy shows how well a model fits *this* dataset, not how well it detects FRBs in
   new data.
7. **Synthesis.** No Vitis and no C++ compiler on this machine, so C-sim was not run.
   * hls4ml projects are written for the best CNN, the smallest CNN, the robust CNN and both
     RNNs (`hls_projects_v6a/`).
   * Fixed-point emulation at ap_fixed<16,6> matches float for the CNNs; it costs 0.3 pt (lstm_v3)
     and 1 pt (LSTM64).
   * All hardware numbers are `est_`.
   * The RNNs carry real synthesis risk: their io_parallel inputs are 65,536 and 8,192 values,
     ~7–13k multipliers at RF 4.

