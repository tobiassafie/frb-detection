# FRB-vs-noise: CNN / LSTM sweep — summary for the 2026-10-01 meeting

All numbers below are on the **rebuilt v7 dataset, split by source burst (file)**. Each number is
printed next to the `run_id` it belongs to (rows of `results/sweep_results.csv`; weights in
`results/models/<run_id>.keras|.h5`). Assumptions and the full audit trail are in `results/NOTES.md`.

## TL;DR

1. **The old 96–99 % accuracies are not valid generalisation numbers.** The v6 noise class is
   5 unique images repeated ~205×. All 154 test noise images also appear verbatim in train, and a
   lookup table with no learning scores **98.4 %** on the v6 test split. On top of that, v6's
   "burst" was mostly a file-edge artefact. See NOTES §0.1.
2. **Rebuilt dataset (v7):** 28 real CHIME bursts → 33,600 samples. The burst template is fitted
   from the real sweep and the backgrounds are real, with the burst inpainted out. Train, val and
   test use **disjoint source files**. FRB S/N is controlled (optimal S/N 3–50, logged per
   sample).
3. **Best synthesis-friendly CNN (`s1mB_t0041_rfi`, 6,433 params):**
   * 94.4 % test accuracy at S/N ≥ 7 (seed 0); **92.1 ± 1.7 %** over 3 seeds; **88.5 ± 2.6 %** in
     4-fold file-grouped CV; 81.0 % on a held-out source family.
   * **≥ 98 % only for bursts with S/N ≥ 15** (98.3 %; 98.6 % at S/N ≥ 20). Specificity is 99.6 %.
   * A classical dedispersion + boxcar search scores 84.5 % on the same test set.
   * 8-bit QAT: 94.3 % (−0.2 pts). 6-bit QAT: 93.4 %.
   * Fixed-point emulation shows no PTQ loss down to ap_fixed<8,3>.
4. **The ≥ 98 % target on the S/N ≥ 7 test set was not reached.** No CNN (up to 68k params) beat the
   6.4k one. The errors sit at S/N 5–12 (see `fig_recall_vs_snr.png`), so the remaining gap is
   detection physics, not architecture.
5. **Best LSTM (`s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0`, 1,617 params):** 83.5 % (82.6 ± 0.6 % over
   3 seeds; 73 % CV). It is clearly weaker than the CNNs, and GRU was weaker still.
6. **Synthesis status: NOT RUN.** This machine has no Vitis HLS, no C++ compiler and no EC2 access.
   * 25 hls4ml projects are written and ready, with a 20-min-timeout batch script. See
     `hls_projects/README.md`.
   * Every hardware number below is an **analytic estimate**, flagged `est_`. DSP/LUT/FF/BRAM,
     Fmax, real latency and C-sim accuracy appear once `run_all_synthesis.sh` has been run on EC2.
   * **Whether the LSTM synthesises is therefore still unknown.**

## Test protocol

* Test = all noise samples of the 6 held-out test files + their FRB samples with optimal S/N ≥ 7.
  This was fixed before the sweep; other S/N cuts are reported as extra rows.
* Model selection used val files only. Test files were never used for any choice.
* Val and test hold only 5–6 source files each, so ±2–3 pts between seeds/configs is noise.
  The 3-seed, CV and held-out-family rows are the honest error bars.

## Headline models

