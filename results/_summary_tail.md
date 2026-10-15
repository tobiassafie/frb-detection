

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

