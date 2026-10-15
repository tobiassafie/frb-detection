

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
