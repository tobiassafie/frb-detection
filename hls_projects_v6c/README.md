# hls_projects_v6c — LSTM/GRU synthesis handoff (NOT synthesised)

`which vitis_hls g++` on the machine that wrote these projects returned nothing (see
`results/logs/v6c_which.txt`). No Vitis HLS and no C++ compiler, so neither csynth nor hls4ml
C-sim was run. Every hardware number in `results/v6c_hls_index.csv` is an `est_` estimate.

## What is needed to run them (= the EC2 box that produced the 61.5 µs `frb_pooled_hls` result)
* Vitis HLS **2025.1** (Build 6135595), as stamped in
  `frb_pooled_hls/frb_pooled_hls/frb_classifier_prj/solution1/syn/report/frb_classifier_csynth.rpt`,
  with `vitis-run` on PATH (settings sourced from `/opt/Xilinx/Vitis/2025.1/settings64.sh` or
  `/opt/Xilinx/2025.1/Vitis/settings64.sh`, as in `fpga_handoff/*/run_synthesis.sh`).
* Part `xcvu9p-flga2104-2L-e`, 5 ns clock (written into every project).
* For C-sim: g++ plus Python with hls4ml 1.3.0, TensorFlow 2.20 / Keras 3, numpy, pyyaml.

## Commands (from this directory)
```bash
bash run_all_synthesis.sh 4            # TCL wrapper per project: vitis-run --tcl build_prj.tcl --mode hls,
                                       # csynth only, `timeout 1200` (20 min) -> status ok / timeout / error_N
python3 parse_synth_reports.py         # -> synth_results.csv: latency cycles/us, II, est. clock -> Fmax,
                                       #    DSP/LUT/FF/BRAM/URAM, synth seconds, last ERROR lines on failure
python3 check_accuracy.py <project>    # bit-accurate C-sim accuracy on the v6 test set vs float (csim_accuracy.csv)
```
Copy `synth_results.csv` and `csim_accuracy.csv` back to `results/`.

## Projects
* hls4ml 1.3.0, Vitis backend, io_parallel, Resource, `static=True` (`use_static = true` checked
  in parameters.h).
* Weights/bias/results are ap_fixed<W,6,AP_RND_CONV,AP_SAT>; the input is ap_fixed<16,6>.

| model | run_id | input | sweep |
|---|---|---|---|
| best small LSTM | `c_L2_T64F256` (runs_v6c) | 64 t × 256 f | RF {4,16,64} × ap_fixed<12,6>/<14,6>/<16,6> + RF16 <18,6> |
| reference lstm_v3 | `s0_lstm_v3_recipe_full150_seed1` (runs_v6a) | 256 t × 256 f | same |
| GRU fallback | `c_L3_gru` (runs_v6c) | 256 t × 256 f | RF16 <16,6> |

**Expect <12,6> and <14,6> to lose accuracy.** Python fixed-point emulation shows the LSTM state
needs ≥ 10 fractional bits: <12,6> / <14,6> give 0.50–0.69 test accuracy, <16,6> is near float
and <18,6> equals float. Check C-sim before trusting a narrow variant.
