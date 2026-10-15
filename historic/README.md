# Historic material (kept for provenance)
* `models_2026-07/`: cnn_v1 and lstm_v1–v3 saved weights (formerly `models/`). Reported vs reproduced
  accuracy: `results/summary_v6a.md`, Stage 0 (all reproduce exactly).
* `ablation_results/`: the 2026-09-18 ablation sweep (results JSON + small checkpoints).
* `fpga_handoff/`: the six hls4ml handoff configs, group1–3 × <16,6>/<8,3>. Only the configs and
  checker scripts are kept; `X_test.npy` is not.
* `ari_meeting/`: the 2026-09-23 inventory and provenance reports, and the synth-ablation scripts.
* `misc_scripts/`: small plotting / dataset-check helpers.

Note: `scripts/v6a_eval_saved.py` still points at the old paths (`models/...`, `ablation_results/...`).
Prefix them with `historic/` (`historic/models_2026-07/...`) to rerun it from this layout.
