# v6c — smaller LSTMs on the original v6 split + synthesis prep (2026-10-07)

**v6 benchmark - leaky split, comparison only.**
* Data: `lstm_dataset_v6.npz` as on disk, original split.
* Recipe: exact lstm_v3 (Adam 1e-3, batch 128, class weights, RLROP 0.5/5, 150 epochs, no early stop).
* 8 training runs (~12 min).
* Lookup baseline **0.984**; max achievable on this test set **0.9838**.
* probe = test-noise accuracy after adding invisible N(0, 0.05).
* Details: `results/NOTES_v6c.md`; per-run rows: `results/v6c_lstm_runs.csv`.

## TL;DR
* **No LSTM/GRU smaller than lstm_v3 stays within 1 pt of its 0.963.**
  * Halving the units, shrinking the head, pooling frequency and switching to GRU all cost 1.9–22 pt.
  * The cheapest acceptable trade-off is `c_L2_T64F256`: the lstm_v3 layers on 64 time steps
    (× 256 channels). It scores **0.945** (−1.8 pt, 1 seed) with **4× fewer MACs** (1.84M vs 7.34M)
    and a 4× smaller parallel input.
* **Synthesis was not run.** `which vitis_hls g++` finds neither.
  * Projects for `c_L2_T64F256` and lstm_v3 (RF {4,16,64} × ap_fixed<12/14/16,6> + <18,6>) and a GRU
    fallback are in `hls_projects_v6c/`, with exact EC2 instructions in its README.
  * Every hardware number below is `est_`.
* **Precision finding (fixed-point emulation, not C-sim).**
  * The LSTM state needs ≥ 10 fractional bits: <16,6> is ≈ float and <18,6> = float.
  * <12,6> and <14,6> collapse to 0.50–0.69.
* As before, none of these numbers transfers off v6: the v7 test accuracy is 0.41–0.55.

## Hypotheses — v6 benchmark - leaky split, comparison only

| hypothesis | run_ids | params | MACs | val_acc | acc | precision | recall | f1 | auc | recall_at_1pct_fpr | noise_acc_perturbed_0p05 | v7acc | lookup | prediction | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_v3 reference | `s0_lstm_v3_recipe_full150`, `s0_lstm_v3_recipe_full150_seed1`, `s0_lstm_v3_recipe_full150_seed2` | 29,953 | 7,341,088 | 0.947 ± 0.004 | 0.963 ± 0.003 | 1.000 ± 0.000 | 0.926 ± 0.006 | 0.962 ± 0.003 | 0.979 ± 0.002 | 0.955 ± 0.005 | 0.885 ± 0.031 | 0.489 ± 0.052 | 0.984 | — | reference (0.963 ± 0.003) |
| L1a LSTM16 | `c_L1_u16` | 22,209 | 5,505,568 | 0.909 | 0.916 | 1.000 | 0.831 | 0.908 | 0.971 | 0.929 | 0.968 | 0.414 | 0.984 | stays within 1 pt | rejected (−4.7 pt) |
| L1b LSTM8 | `c_L1_u8` | 19,105 | 4,784,416 | 0.886 | 0.912 | 1.000 | 0.825 | 0.904 | 0.962 | 0.935 | 0.968 | 0.423 | 0.984 | drops > 1 pt | confirmed (−5.0 pt) → U* = 32 |
| L2a 64 t × 256 f | `c_L2_T64F256` | 29,953 | 1,836,064 | 0.925 | 0.945 | 1.000 | 0.890 | 0.942 | 0.977 | 0.935 | 0.838 | 0.476 | 0.984 | ~same acc, 4× fewer MACs | rejected (−1.8 pt), but 4× fewer MACs |
| L2b 128 t × 128 f | `c_L2_T128F128` | 21,761 | 2,622,496 | 0.828 | 0.831 | 1.000 | 0.662 | 0.797 | 0.889 | 0.805 | 0.968 | 0.419 | 0.984 | ~same acc | rejected (−13 pt) |
| L3 GRU32 | `c_L3_gru`, `c_L3_gru_seed1`, `c_L3_gru_seed2` | 26,945 | 6,554,656 | 0.936 ± 0.017 | 0.944 ± 0.016 | 1.000 ± 0.000 | 0.887 ± 0.031 | 0.940 ± 0.018 | 0.974 ± 0.005 | 0.942 ± 0.014 | 0.881 ± 0.029 | 0.453 ± 0.015 | 0.984 | ≈ LSTM | rejected on 3 seeds (−1.9 pt) |
| L4 Dense8 head | `c_L4_dense8` | 29,137 | 7,340,296 | 0.682 | 0.740 | 0.685 | 0.890 | 0.774 | 0.885 | 0.610 | 0.591 | 0.416 | 0.984 | holds | rejected (−22 pt) |

## Synthesis candidates — v6 benchmark - leaky split, comparison only (accuracy on v6 test; hardware = est_, NOT measured)

| project | run_id | params | MACs | precision | RF | float acc | fixed-point emu acc (not C-sim) | est_mults | est_weight_kbits | est_latency_us | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `c_L2_T64F256__Res_rf4_w12_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<12,6> | 4 | 0.945 | 0.692 | 7432 | 359.4 | 5.12 | written |
| `c_L2_T64F256__Res_rf4_w14_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<14,6> | 4 | 0.945 | 0.636 | 7432 | 419.3 | 5.12 | written |
| `c_L2_T64F256__Res_rf4_w16_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<16,6> | 4 | 0.945 | 0.964 | 7432 | 479.2 | 5.12 | written |
| `c_L2_T64F256__Res_rf16_w12_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<12,6> | 16 | 0.945 | 0.692 | 1858 | 359.4 | 8.96 | written |
| `c_L2_T64F256__Res_rf16_w14_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<14,6> | 16 | 0.945 | 0.636 | 1858 | 419.3 | 8.96 | written |
| `c_L2_T64F256__Res_rf16_w16_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<16,6> | 16 | 0.945 | 0.964 | 1858 | 479.2 | 8.96 | written |
| `c_L2_T64F256__Res_rf16_w18_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<18,6> | 16 | 0.945 | 0.945 | 1858 | 539.2 | 8.96 | written |
| `c_L2_T64F256__Res_rf64_w12_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<12,6> | 64 | 0.945 | 0.692 | 465 | 359.4 | 24.32 | written |
| `c_L2_T64F256__Res_rf64_w14_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<14,6> | 64 | 0.945 | 0.636 | 465 | 419.3 | 24.32 | written |
| `c_L2_T64F256__Res_rf64_w16_6` | `c_L2_T64F256` | 29,953 | 1,836,064 | ap_fixed<16,6> | 64 | 0.945 | 0.964 | 465 | 479.2 | 24.32 | written |
| `c_L3_gru__Res_rf16_w16_6` | `c_L3_gru` | 26,945 | 6,554,656 | ap_fixed<16,6> | 16 | 0.958 | 0.961 | 1666 | 431.1 | 35.84 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf4_w12_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<12,6> | 4 | 0.961 | 0.500 | 7432 | 359.4 | 20.48 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf4_w14_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<14,6> | 4 | 0.961 | 0.669 | 7432 | 419.3 | 20.48 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf4_w16_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<16,6> | 4 | 0.961 | 0.958 | 7432 | 479.2 | 20.48 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf16_w12_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<12,6> | 16 | 0.961 | 0.500 | 1858 | 359.4 | 35.84 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf16_w14_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<14,6> | 16 | 0.961 | 0.669 | 1858 | 419.3 | 35.84 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf16_w16_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<16,6> | 16 | 0.961 | 0.958 | 1858 | 479.2 | 35.84 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf16_w18_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<18,6> | 16 | 0.961 | 0.961 | 1858 | 539.2 | 35.84 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf64_w12_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<12,6> | 64 | 0.961 | 0.500 | 465 | 359.4 | 97.28 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf64_w14_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<14,6> | 64 | 0.961 | 0.669 | 465 | 419.3 | 97.28 | written |
| `s0_lstm_v3_recipe_full150_seed1__Res_rf64_w16_6` | `s0_lstm_v3_recipe_full150_seed1` | 29,953 | 7,341,088 | ap_fixed<16,6> | 64 | 0.961 | 0.958 | 465 | 479.2 | 97.28 | written |

**Not available until synthesis runs:** DSP/LUT/FF/BRAM, Fmax, measured latency and C-sim accuracy.

## What to run on EC2

On the box with Vitis HLS 2025.1 (Build 6135595; the 61.5 µs `frb_pooled_hls` machine), from `hls_projects_v6c/`:

* `bash run_all_synthesis.sh 4` — csynth, 20-min timeout per project
* `python3 parse_synth_reports.py`
* `python3 check_accuracy.py <project>` — C-sim vs float

Start with `c_L2_T64F256__Res_rf16_w16_6` and `__w18_6`.
