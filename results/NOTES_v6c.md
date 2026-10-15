# NOTES_v6c — smaller LSTMs on the original v6 split (2026-10-07)

**v6 benchmark - leaky split, comparison only.**

* Data: `augmented_dataset/lstm_dataset_v6.npz` as on disk, original split.
* Runs: `results/runs_v6c/*.json` → `results/v6c_lstm_runs.csv`. Trainer: `scripts/v6a_train.py`
  with `V6A_TAG=v6c`.
* Known v6 ceiling: test 0.9838 / val 0.9675. 5 test / 10 val FRB-labelled samples are
  byte-identical to noise images (see NOTES_v6a).

## Recipe (exact lstm_v3)
* Raw v6 (256 time × 256 freq, no extra normalisation).
* Adam 1e-3, batch 128, class weights, ReduceLROnPlateau(0.5, patience 5, min 1e-6),
  150 epochs.
* **No early stopping** (patience 150); the best-val-loss epoch is restored at the end, as the
  original did.
* Seed 42.
* Retry rule: restart with seed + 1000·k if best val < 0.65 **or** the val-prediction std
  < 1e-3 (constant output). Every attempt is logged in `attempts`; nothing is dropped.

Reference: lstm_v3 (`s0_lstm_v3_recipe_full150*`): Conv1D(64,k1) → LSTM(32) → Dense32 → drop 0.3,
29,953 params, 7.34M MACs, test **0.963 ± 0.003** (3 seeds), val 0.947 ± 0.004.

## Stage 1 — hypotheses (written before the runs)

| # | run_id | change vs lstm_v3 | hypothesis |
|---|---|---|---|
| L1a | `c_L1_u16` | LSTM 32 → 16 | stays within 1 pt of 0.963 |
| L1b | `c_L1_u8` | LSTM 32 → 8 | drops > 1 pt; U* = smallest within 1 pt |
| L2a | `c_L2_T64F256` | time pool ×4 (64 steps × 256 ch; 1×1 Conv1D(64) kept), LSTM U* | ~same acc, ~4× fewer MACs |
| L2b | `c_L2_T128F128` | pool 2 × 2 (128 steps × 128 ch), LSTM U* | ~same acc; same input size as L2a, different layout |
| L3 | `c_L3_gru` | GRU(U*) at the better L2 layout (full res if L2 loses > 1 pt) | ≈ LSTM accuracy with fewer gate multipliers |
| L4 | `c_L4_dense8` | Dense 32 → 8 on the best so far | holds within 1 pt |

The top 2 by val get 3 seeds. lstm_v3 already has 3.

## Stage 1 — outcomes (8 training runs, ~12 min wall clock; no seed collapsed, no constant output)

**v6 benchmark - leaky split, comparison only.**
* Lookup baseline 0.984; ceiling 0.9838.
* probe = test-noise accuracy with invisible N(0, 0.05) added; lstm_v3 reference probe ≈ 0.89.
* All runs have precision 1.000 (all test noise images are memorised training images).

| # | run_id | params | MACs | val | test | recall | AUC | rec@1%FPR | probe | v7 acc | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | lstm_v3 (`s0_lstm_v3_recipe_full150*`, 3 seeds) | 29,953 | 7.34M | 0.947 ± 0.004 | **0.963 ± 0.003** | 0.926 | 0.979 | 0.955 | 0.89 | 0.43–0.55 | — |
| L1a | `c_L1_u16` | 22,209 | 5.51M | 0.909 | 0.916 | 0.831 | 0.971 | 0.929 | 0.97 | 0.414 | **rejected** (−4.7 pt) |
| L1b | `c_L1_u8` | 19,105 | 4.78M | 0.886 | 0.912 | 0.825 | 0.962 | 0.935 | 0.97 | 0.423 | **confirmed drop**; no size below 32 stays within 1 pt → U* = 32 |
| L2a | `c_L2_T64F256` | 29,953 | **1.84M** | 0.925 | 0.945 | 0.890 | 0.977 | 0.935 | 0.84 | 0.476 | **rejected** (−1.8 pt) — but 4× fewer MACs |
| L2b | `c_L2_T128F128` | 21,761 | 2.62M | 0.828 | 0.831 | 0.662 | 0.889 | 0.805 | 0.97 | 0.419 | **rejected** (−13 pt) |
| L3 | `c_L3_gru` + 2 seeds | 26,945 | 6.55M | 0.936 ± 0.017 | 0.944 ± 0.016 | 0.887 | 0.974 | 0.942 | 0.88 | 0.453 | **rejected on 3 seeds** (seed 42 alone: 0.958) |
| L4 | `c_L4_dense8` | 29,137 | 7.34M | 0.682 | 0.740 | 0.890 | 0.886 | 0.610 | 0.59 | 0.416 | **rejected** (−22 pt) |

**Reading**
* Halving or quartering the LSTM, or shrinking the head, costs 5–22 pt.
* Losing frequency resolution (128 ch/step) costs 13 pt. Losing time resolution (64 steps)
  costs only 1.8 pt for 4× fewer MACs.
* The MACs are dominated by the per-step 1×1 Conv1D (256 → 64 channels × 256 steps = 4.2M),
  not by the recurrent cell. Shrinking the LSTM barely changes cost.
* **No new configuration stays within ~1 pt of 0.963. lstm_v3 remains the smallest RNN that
  does.** The best smaller trade-offs are LSTM 64 × 256 (`c_L2_T64F256`, 1.84M MACs, 0.945)
  and GRU32 (0.944 ± 0.016).
* Top 2 by val for the 3-seed rule: `c_L3_gru` (0.958) and lstm_v3 (already 3 seeds) → 2 extra
  seed runs; total 8.

## Stage 2 — synthesis (projects only)

`which vitis_hls g++` → `which: no vitis_hls …`, `which: no g++ …` (exit 2; full text in
`results/logs/v6c_which.txt`). So, per the instructions, projects are written and every hardware
number is `est_`. Exactly what is needed to run them elsewhere: `hls_projects_v6c/README.md`
(Vitis HLS 2025.1 Build 6135595 = the version stamped in the 61.5 µs `frb_pooled_hls` report,
xcvu9p-flga2104-2L-e, 5 ns, g++ + hls4ml 1.3.0 for C-sim).

**Which model to synthesise.**
* No model smaller than lstm_v3 stays within 1 pt.
* The "best small" pick is therefore `c_L2_T64F256`: same layers as lstm_v3 at 64 time steps
  (4× fewer MACs, 4× smaller parallel input: 16,384 values, ~4× lower est. latency). It scores
  0.945, −1.8 pt, single seed.
* lstm_v3 (`s0_lstm_v3_recipe_full150_seed1`) is exported with the same sweep as the
  within-1-pt reference.
* GRU32 is exported at RF16 <16,6> as a fallback.

**Fixed-point emulation** (`hls_export.emulate_rnn`; not C-sim; float sigmoid/tanh instead of
hls4ml tables).
* Validated: ap_fixed<32,10> reproduces float (mean |Δp| ≈ 0.0005).
* Accuracy depends only on the number of fractional bits; 6 integer bits are enough (pre-RNN
  activations ≤ 6.7).

| fractional bits | precisions | c_L2_T64F256 (float 0.945) | lstm_v3 seed1 (float 0.961) |
|---|---|---|---|
| 12 | <18,6> | 0.945 | 0.961 |
| 10 | <16,6> | 0.964 (mean \|Δp\| 0.037 — a favourable perturbation, not a real gain) | 0.958 |
| 8 | <14,6> | 0.636 | 0.669 |
| 6 | <12,6> | 0.692 | 0.500 |

→ **<16,6> is the narrowest usable width and <18,6> the safe one.** <12,6>/<14,6> are written
as requested but are expected to fail C-sim accuracy.

**est_ hardware** (`results/v6c_hls_index.csv`, analytic; not measured):
* est. multipliers: RF4 7,432 → RF16 1,858 → RF64 465 (same for both LSTMs);
* est. latency: 64-step model 5.1 / 9.0 / 24.3 µs; 256-step lstm_v3 20.5 / 35.8 / 97.3 µs at
  RF 4 / 16 / 64.

The RNN latency formula (timesteps × (RF + 12) cycles) is uncalibrated. No DSP/LUT/FF/BRAM/Fmax
numbers exist until csynth runs.
