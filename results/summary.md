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

| | **Best CNN - accuracy track (any size; best 3-seed val)** | **Best CNN - synthesis track (<=10k, exported to HLS)** | **Best LSTM/GRU** | **Largest CNN tried (68k, BN)** | **8-bit QAT of the synthesis CNN** | classical dedispersion search |
|---|---|---|---|---|---|---|
| run_id | `loc_t0041_b32` | `s1mB_t0041_rfi` | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | `loc_t0041_A_w8k5bn` | `qat_s1mB_t0041_rfi_8b` | `scripts/classical_baseline.py` |
| architecture | 128² in; Conv[4, 8, 16, 16, 16] k3, avg-pool [2, 2, 2, 2, 2], gap, Dense16 | 128² in; Conv[4, 8, 16, 16, 16] k3, avg-pool [2, 2, 2, 2, 2], gap, Dense16 | LSTM(16) on 64 t × 8 f | 128² in; Conv[8, 16, 32, 32, 32] k5, avg-pool [2, 2, 2, 2, 2], gap, Dense16, BN | 128² in; Conv[4, 8, 16, 16, 16] k3, avg-pool [2, 2, 2, 2, 2], gap, Dense16 | DM −13…−0.4, boxcar 1–16 |
| params | 6,433 | 6,433 | 1,617 | 68,425 | 6,433 | — |
| MACs / inference | 3,686,672 | 3,686,672 | 98,320 | 37,683,728 | 3,686,672 | — |
| **test acc** (seed 0; noise + FRB S/N ≥ 7) | **0.9155** | **0.9443** | **0.8354** | **0.9236** | **0.9425** | 0.8453 |
| test acc, 3 training seeds | 0.915 ± 0.002 (n=3) | 0.921 ± 0.017 (n=3) | 0.826 ± 0.006 (n=3) | 0.924 | 0.943 | — |
| test acc, 4-fold file-grouped CV | — | 0.885 ± 0.026 (n=4) | 0.732 ± 0.040 (n=4) | — | — | — |
| test acc, held-out June-2019 family | — | 0.810 | 0.560 | — | — | 0.744 |
| precision / recall | 0.893 / 0.905 | 0.994 / 0.871 | 0.840 / 0.744 | 0.974 / 0.838 | 0.963 / 0.895 | 0.874 / 0.732 |
| F1 | 0.8986 | 0.9283 | 0.7891 | 0.9008 | 0.9280 | 0.7966 |
| ROC-AUC | 0.9622 | 0.9716 | 0.8734 | 0.9683 | 0.9710 | 0.8932 |
| recall @ 1 % FPR | 0.853 | 0.888 | 0.593 | 0.831 | 0.870 | 0.000 |
| specificity (noise samples correct) | 0.9231 | 0.9961 | 0.9000 | 0.9842 | 0.9758 | — |
| test acc, FRB S/N ≥ 10 | 0.9316 | 0.9712 | 0.8670 | 0.9552 | 0.9637 | — |
| test acc, FRB S/N ≥ 15 | 0.9360 | 0.9830 | 0.8963 | 0.9694 | 0.9710 | — |
| test acc, FRB S/N ≥ 20 | 0.9347 | 0.9861 | 0.9073 | 0.9749 | 0.9718 | — |
| test acc, v6-brightness-equivalent FRBs (peak 0.5–5 σ) | 0.8150 | 0.8291 | 0.7443 | 0.8070 | 0.8365 | — |
| test acc, all S/N 3–50 | 0.8190 | 0.8268 | 0.7460 | 0.8036 | 0.8350 | 0.7336 |

Notes on the table:

* **Accuracy track vs synthesis track are the same architecture.** The accuracy-track winner,
  picked by best 3-seed *val* mean, is `loc_t0041_b32`: the synthesis CNN trained with batch 32
  instead of 64. It has a lower seed-0 test number. That difference is model-selection noise from
  having only 5 val files (3-seed test means: 0.915 vs 0.921). Bigger models did not help (68k-param
  BN CNN: 0.924), so **the synthesis-friendly CNN is also the best CNN we have.**
* **"v6-brightness-equivalent"** keeps only FRBs whose injected peak pixel is 0.5–5 σ, i.e. v6's own
  `SNR_INJECT_RANGE`. Many of those are faint (low S/N) in the matched-filter sense, hence ~83 %.
* The classical search uses incoherent dedispersion over residual DM −13…−0.4 with boxcar widths
  1–16. Its threshold was tuned on val.

## Hardware (hls4ml projects for the synthesis CNN, its QAT versions, and the RNNs)

**Every `est_*` column is an analytic estimate, NOT a synthesis result:**
* `est_mult_total` = Σ multipliers instantiated (exact from hls4ml's per-layer configs; an upper
  bound on DSPs, since narrow multipliers go to LUTs).
* `est_latency_us` = io_stream dataflow bound calibrated on the one design ever synthesised here
  (frb_pooled_hls: 3 cycles per input pixel, 5 ns clock). RNN latency is uncalibrated.

`emu_test_acc` = Python fixed-point emulation of the same ap_fixed types; for QAT models it is the
QKeras forward pass. It is **not C-sim**; each project's `check_accuracy.py` gives the real C-sim
accuracy on EC2. The accuracy-track CNN has the same architecture as `s1mB_t0041_rfi`, so its
hardware is identical.

| project | precision | strategy | reuse_factor | float_test_acc | emu_test_acc | est_mult_total | est_weight_kbits | est_latency_us | status |
|---|---|---|---|---|---|---|---|---|---|
| s1mB_t0041_rfi__Res_rf1_w16_6 | ap_fixed<16,6> | Resource | 1 | 0.9443 | 0.9446 | 6356 | 102.9000 | 278.8500 | written |
| s1mB_t0041_rfi__Res_rf4_w16_6 | ap_fixed<16,6> | Resource | 4 | 0.9443 | 0.9446 | 1592 | 102.9000 | 360.4400 | written |
| s1mB_t0041_rfi__Res_rf16_w16_6 | ap_fixed<16,6> | Resource | 16 | 0.9443 | 0.9446 | 387 | 102.9000 | 1673.1000 | written |
| s1mB_t0041_rfi__Res_rf64_w16_6 | ap_fixed<16,6> | Resource | 64 | 0.9443 | 0.9446 | 90 | 102.9000 | 5767.1600 | written |
| s1mB_t0041_rfi__Lat_rf1_w16_6 | ap_fixed<16,6> | Latency | 1 | 0.9443 | 0.9446 | 6356 | 102.9000 | 278.8500 | written |
| s1mB_t0041_rfi__Res_rf4_w8_3 | ap_fixed<8,3> | Resource | 4 | 0.9443 | 0.9445 | 1592 | 51.5000 | 360.4400 | written |
| s1mB_t0041_rfi__Res_rf4_w10_4 | ap_fixed<10,4> | Resource | 4 | 0.9443 | 0.9445 | 1592 | 64.3000 | 360.4400 | written |
| s1mB_t0041_rfi__Res_rf4_w12_5 | ap_fixed<12,5> | Resource | 4 | 0.9443 | 0.9446 | 1592 | 77.2000 | 360.4400 | written |
| qat_s1mB_t0041_rfi_8b__Res_rf1_qkeras | qkeras | Resource | 1 | 0.9443 | 0.9425 | 6356 | 51.5000 | 278.8500 | written |
| qat_s1mB_t0041_rfi_8b__Res_rf4_qkeras | qkeras | Resource | 4 | 0.9443 | 0.9425 | 1592 | 51.5000 | 360.4400 | written |
| qat_s1mB_t0041_rfi_8b__Res_rf16_qkeras | qkeras | Resource | 16 | 0.9443 | 0.9425 | 387 | 51.5000 | 1673.1000 | written |
| qat_s1mB_t0041_rfi_8b__Res_rf64_qkeras | qkeras | Resource | 64 | 0.9443 | 0.9425 | 90 | 51.5000 | 5767.1600 | written |
| qat_s1mB_t0041_rfi_8b__Lat_rf1_qkeras | qkeras | Latency | 1 | 0.9443 | 0.9425 | 6356 | 51.5000 | 278.8500 | written |
| qat_s1mB_t0041_rfi_6b__Res_rf1_qkeras | qkeras | Resource | 1 | 0.9443 | 0.9336 | 6356 | 38.6000 | 278.8500 | written |
| qat_s1mB_t0041_rfi_6b__Res_rf4_qkeras | qkeras | Resource | 4 | 0.9443 | 0.9336 | 1592 | 38.6000 | 360.4400 | written |
| qat_s1mB_t0041_rfi_6b__Res_rf16_qkeras | qkeras | Resource | 16 | 0.9443 | 0.9336 | 387 | 38.6000 | 1673.1000 | written |
| s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0__Res_rf1_w16_6 | ap_fixed<16,6> | Resource | 1 | 0.8354 | 0.8357 | 1552 | 25.9000 | 4.1600 | written |
| s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0__Res_rf4_w16_6 | ap_fixed<16,6> | Resource | 4 | 0.8354 | 0.8357 | 388 | 25.9000 | 5.1200 | written |
| s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0__Res_rf16_w16_6 | ap_fixed<16,6> | Resource | 16 | 0.8354 | 0.8357 | 97 | 25.9000 | 8.9600 | written |
| s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0__Res_rf4_w12_6 | ap_fixed<12,6> | Resource | 4 | 0.8354 | 0.8385 | 388 | 19.4000 | 5.1200 | written |
| s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0__Res_rf4_w14_6 | ap_fixed<14,6> | Resource | 4 | 0.8354 | 0.8325 | 388 | 22.6000 | 5.1200 | written |
| s3r_gru_u16_T64_F8_lr0.003_do0.0_d16__Res_rf4_w16_6 | ap_fixed<16,6> | Resource | 4 | 0.7810 | 0.7807 | 356 | 24.6000 | 5.1200 | written |
| s3r_gru_u16_T64_F8_lr0.003_do0.0_d16__Res_rf16_w16_6 | ap_fixed<16,6> | Resource | 16 | 0.7810 | 0.7807 | 89 | 24.6000 | 8.9600 | written |
| s3_lstm_u4_T32_F16__Res_rf1_w16_6 | ap_fixed<16,6> | Resource | 1 | 0.7991 | 0.7988 | 324 | 5.5000 | 2.0800 | written |
| s3_lstm_u4_T32_F16__Res_rf4_w16_6 | ap_fixed<16,6> | Resource | 4 | 0.7991 | 0.7988 | 81 | 5.5000 | 2.5600 | written |

Reading the estimates: at RF = 4 the CNN needs ≈1.6k multipliers and ≈0.36 ms per 168 ms patch.
The LSTM needs ≈0.4k multipliers and ≈5 µs. Both are far inside real time; the open questions
are resources/Fmax (CNN) and whether Vitis accepts the static LSTM (RNN).

## Figures

* `fig_accuracy_vs_params.png`: test accuracy vs parameters for every sweep run (CNN A/B, QAT,
  LSTM/GRU), with the classical-search line.
* `fig_recall_vs_snr.png`: FRB recall per S/N bin for the three headline models + classical
  search. This is where the remaining errors are.
* `fig_accuracy_vs_latency.png`: quantised accuracy vs **estimated** latency for the exported
  projects. It will be re-plotted from csynth once EC2 results exist.

## Next steps

1. EC2: `bash hls_projects/run_all_synthesis.sh 4`, then `check_accuracy.py` on the RF=4
   projects. Then fill the DSP/LUT/BRAM/Fmax columns (`python scripts/make_summary.py ...`).
2. If ≥ 98 % at S/N ≥ 7 is a hard requirement, the levers are in the data, not the architecture:
   * more real source bursts (only 17 training files);
   * a pre-dedispersion front end (a few DM trials before the CNN);
   * or an agreed minimum S/N for the spec.
