"""results/summary_v6c.md from results/runs_v6c/*.json, results/runs_v6a (lstm_v3 reference),
results/v6c_hls_index.csv. Every table is labelled as the leaky v6 benchmark."""
import json

import numpy as np
import pandas as pd

R = "results"
LABEL = "v6 benchmark - leaky split, comparison only"


def load(rid):
    for d in ("runs_v6c", "runs_v6a"):
        try:
            return json.load(open(f"{R}/{d}/{rid}.json"))
        except FileNotFoundError:
            pass
    raise FileNotFoundError(rid)


def agg(ids, k):
    v = [load(i)[k] for i in ids]
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


ROWS = [  # label, ids (seeds), hypothesis, verdict
    ("lstm_v3 reference", ["s0_lstm_v3_recipe_full150", "s0_lstm_v3_recipe_full150_seed1", "s0_lstm_v3_recipe_full150_seed2"],
     "—", "reference (0.963 ± 0.003)"),
    ("L1a LSTM16", ["c_L1_u16"], "stays within 1 pt", "rejected (−4.7 pt)"),
    ("L1b LSTM8", ["c_L1_u8"], "drops > 1 pt", "confirmed (−5.0 pt) → U* = 32"),
    ("L2a 64 t × 256 f", ["c_L2_T64F256"], "~same acc, 4× fewer MACs", "rejected (−1.8 pt), but 4× fewer MACs"),
    ("L2b 128 t × 128 f", ["c_L2_T128F128"], "~same acc", "rejected (−13 pt)"),
    ("L3 GRU32", ["c_L3_gru", "c_L3_gru_seed1", "c_L3_gru_seed2"], "≈ LSTM", "rejected on 3 seeds (−1.9 pt)"),
    ("L4 Dense8 head", ["c_L4_dense8"], "holds", "rejected (−22 pt)"),
]
KEYS = ["val_acc", "test_acc", "test_precision", "test_recall", "test_f1", "test_auc", "test_recall_at_1pct_fpr",
        "noise_acc_perturbed_0p05", "v7test_acc"]


def main():
    L = [f"# v6c — smaller LSTMs on the original v6 split + synthesis prep (2026-10-07)", "",
         f"**{LABEL}.**",
         "* Data: `lstm_dataset_v6.npz` as on disk, original split.",
         "* Recipe: exact lstm_v3 (Adam 1e-3, batch 128, class weights, RLROP 0.5/5, 150 epochs, no early stop).",
         "* 8 training runs (~12 min).",
         "* Lookup baseline **0.984**; max achievable on this test set **0.9838**.",
         "* probe = test-noise accuracy after adding invisible N(0, 0.05).",
         "* Details: `results/NOTES_v6c.md`; per-run rows: `results/v6c_lstm_runs.csv`.", "",
         "## TL;DR",
         "* **No LSTM/GRU smaller than lstm_v3 stays within 1 pt of its 0.963.**",
         "  * Halving the units, shrinking the head, pooling frequency and switching to GRU all cost 1.9–22 pt.",
         "  * The cheapest acceptable trade-off is `c_L2_T64F256`: the lstm_v3 layers on 64 time steps",
         "    (× 256 channels). It scores **0.945** (−1.8 pt, 1 seed) with **4× fewer MACs** (1.84M vs 7.34M)",
         "    and a 4× smaller parallel input.",
         "* **Synthesis was not run.** `which vitis_hls g++` finds neither.",
         "  * Projects for `c_L2_T64F256` and lstm_v3 (RF {4,16,64} × ap_fixed<12/14/16,6> + <18,6>) and a GRU",
         "    fallback are in `hls_projects_v6c/`, with exact EC2 instructions in its README.",
         "  * Every hardware number below is `est_`.",
         "* **Precision finding (fixed-point emulation, not C-sim).**",
         "  * The LSTM state needs ≥ 10 fractional bits: <16,6> is ≈ float and <18,6> = float.",
         "  * <12,6> and <14,6> collapse to 0.50–0.69.",
         "* As before, none of these numbers transfers off v6: the v7 test accuracy is 0.41–0.55.", "",
         f"## Hypotheses — {LABEL}", "",
         "| hypothesis | run_ids | params | MACs | " + " | ".join(k.replace('test_', '') for k in KEYS) + " | lookup | prediction | verdict |",
         "|---|---|---|---|" + "---|" * len(KEYS) + "---|---|---|"]
    for lab, ids, hyp, ver in ROWS:
        r0 = load(ids[0])
        L.append(f"| {lab} | {', '.join('`' + i + '`' for i in ids)} | {r0['params']:,} | {r0['macs']:,} | "
                 + " | ".join(agg(ids, k) for k in KEYS) + f" | 0.984 | {hyp} | {ver} |")
    hw = pd.read_csv(f"{R}/v6c_hls_index.csv")
    L += ["", f"## Synthesis candidates — {LABEL} (accuracy on v6 test; hardware = est_, NOT measured)", "",
          "| project | run_id | params | MACs | precision | RF | float acc | fixed-point emu acc (not C-sim) | "
          "est_mults | est_weight_kbits | est_latency_us | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, h in hw.sort_values(["run_id", "reuse_factor", "precision"]).iterrows():
        L.append(f"| `{h.project}` | `{h.run_id}` | {int(h.params):,} | {int(h.macs):,} | {h.precision} | {h.reuse_factor} | "
                 f"{h.float_test_acc_v6:.3f} | {h.emu_test_acc_v6:.3f} | {int(h.est_mult_total)} | {h.est_weight_kbits} | "
                 f"{h.est_latency_us} | {h.status} |")
    L += ["", "**Not available until synthesis runs:** DSP/LUT/FF/BRAM, Fmax, measured latency and C-sim accuracy.",
          "", "## What to run on EC2", "",
          "On the box with Vitis HLS 2025.1 (Build 6135595; the 61.5 µs `frb_pooled_hls` machine), from `hls_projects_v6c/`:",
          "",
          "* `bash run_all_synthesis.sh 4` — csynth, 20-min timeout per project",
          "* `python3 parse_synth_reports.py`",
          "* `python3 check_accuracy.py <project>` — C-sim vs float",
          "",
          "Start with `c_L2_T64F256__Res_rf16_w16_6` and `__w18_6`."]
    open(f"{R}/summary_v6c.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
