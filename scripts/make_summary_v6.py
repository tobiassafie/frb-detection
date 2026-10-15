"""results/summary_v6.md tables from results/v6_benchmark.csv, results/sweep_results.csv,
results/stream_results.csv, results/v6_baselines.json. Writes results/_v6_*.md fragments."""
import json
import os
import re

import numpy as np
import pandas as pd

R = "results"
M = ["acc", "precision", "recall", "f1", "auc", "recall_at_1pct_fpr"]


def f(x, p=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def ms(s, p=3):
    s = s.dropna()
    if not len(s):
        return "—"
    return f"{s.mean():.{p}f} ± {s.std(ddof=0):.{p}f}" if len(s) > 1 else f"{s.iloc[0]:.{p}f}"


def v6_table():
    b = pd.read_csv(f"{R}/v6_benchmark.csv")
    L = ["**v6 benchmark - leaky split, comparison only** (original `lstm_dataset_v6.npz` split, "
         "test = 154 FRB + 154 noise; mean ± std over the 3 seeds; *dead* = seed that never left loss ln 2)", "",
         "| model (architecture source) | run_ids | params | acc | precision | recall | F1 | ROC-AUC | recall@1%FPR | "
         "dedup-test recall (149 FRB, 0 noise) | same model on **v7** test (acc) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in b[b.row_type == "baseline"].iterrows():
        L.append(f"| **no-learning baseline: {r.run_id}** | — | 0 | " + " | ".join(f(r[k]) for k in M) +
                 f" | {f(r.dedup_test_recall)} | — |")
    m = b[b.row_type == "model"].copy()

    def fam(rid):
        rid = re.sub(r"^(qat_)?v6_", "", rid)
        rid = re.sub(r"_(s\d|seed\d)(_8b)?$", "", rid)
        return rid + (" (8-bit QAT)" if "qat_" in rid else "")
    m["fam"] = m.run_id.apply(lambda r: ("QAT8 " if r.startswith("qat_") else "") + fam(r))
    for fm, g in m.groupby("fam", sort=False):
        dead = g[(g.acc <= 0.53) & (g.auc <= 0.70)]
        ok = g.drop(dead.index)
        ids = ", ".join(f"`{x}`" for x in g.run_id)
        note = f" ({len(dead)} dead seed{'s' if len(dead) > 1 else ''} excluded from mean)" if len(dead) and len(ok) else (
            " (all seeds dead)" if len(dead) else "")
        use = ok if len(ok) else g
        L.append(f"| {fm}{note} | {ids} | {int(g.params.iloc[0]):,} | " + " | ".join(ms(use[k]) for k in M) +
                 f" | {ms(use.dedup_test_recall)} | {ms(use.v7test_acc)} |")
    L += ["", "**v7-trained models evaluated on the v6 test set** (cross-dataset; input renormalised per channel "
          "as in v7 training)", "", "| v7 model | params | v6 acc | precision | recall | F1 | ROC-AUC | recall@1%FPR | "
          "dedup recall | its own v7 test acc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in b[b.row_type == "cross_v7_to_v6"].iterrows():
        L.append(f"| `{r.run_id}` | {int(r.params):,} | " + " | ".join(f(r[k]) for k in M) +
                 f" | {f(r.dedup_test_recall)} | {f(r.v7test_acc)} |")
    return "\n".join(L)


def stream_table():
    d = pd.read_csv(f"{R}/stream_results.csv")
    L = []
    for mode in ("val_fa<=10/h", "fixed_0.5"):
        L += [f"**Threshold: {mode}** "
              + ("(lowest threshold with ≤ 10 false alarms/h on the matching VAL-files stream)" if mode != "fixed_0.5" else ""),
              "", "| model | run_id | stream | thr | test FA/h | det S/N≥7 | det S/N≥15 | det all S/N | median time-to-detect "
              "(ms, S/N≥7) | p90 TTD | median frac. of band arrived at detection | detected before half the band |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        x = d[d.threshold_mode == mode]
        for _, r in x.iterrows():
            st = {"v7_test_stored": "v7 stored (as trained)", "v7_test_physical": "v7 physical (high-f first)",
                  "v6_test": "v6 tiled (comparison only)"}[r.stream]
            L.append(f"| {r.model} | `{r.run_id}` | {st} | {f(r.threshold)} | {f(r.fa_per_hour, 0)} | "
                     f"{f(r.det_rate_snr_7_50)} | {f(r.det_rate_snr_15_50)} | {f(r.det_rate_all)} | "
                     f"{f(r.ttd_ms_median_snr_ge7, 0)} | {f(r.ttd_ms_p90_snr_ge7, 0)} | "
                     f"{f(r.frac_band_at_detect_median_snr_ge7, 2)} | {f(r.frac_detected_before_half_band_snr_ge7, 2)} |")
        L.append("")
    return "\n".join(L)


def v7_table():
    s = pd.read_csv(f"{R}/sweep_results.csv").set_index("run_id")
    b = pd.read_csv(f"{R}/v6_benchmark.csv")
    rows = ["s1mB_t0041_rfi", "loc_t0041_b32", "loc_t0041_A_w8k5bn", "qat_s1mB_t0041_rfi_8b",
            "s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0", "s3r_gru_u16_T64_F8_lr0.003_do0.0_d16", "strm_t0041_s0",
            "strm_t0041_s1", "strm_t0041_s2"]
    L = ["**v7 dataset (file-grouped, leak-free; test = 6 held-out source files, noise + FRB S/N ≥ 7)**", "",
         "| model | run_id | trained on | params | acc | precision | recall | F1 | ROC-AUC | recall@1%FPR |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for rid in rows:
        if rid not in s.index:
            continue
        r = s.loc[rid]
        tr = "v7 stream-sampled pool" if rid.startswith("strm") else "v7 train"
        name = {"s1mB_t0041_rfi": "v7 synthesis CNN", "loc_t0041_b32": "v7 accuracy-track CNN",
                "loc_t0041_A_w8k5bn": "v7 68k BN CNN", "qat_s1mB_t0041_rfi_8b": "v7 8-bit QAT CNN",
                "s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0": "v7 LSTM", "s3r_gru_u16_T64_F8_lr0.003_do0.0_d16": "v7 GRU"
                }.get(rid, "stream-trained CNN (t0041 cfg)")
        L.append(f"| {name} | `{rid}` | {tr} | {int(r.params):,} | " +
                 " | ".join(f(r[f'test_{k}']) for k in M) + " |")
    x = b[b.row_type == "model"].dropna(subset=["v7test_acc"])
    for _, r in x.iterrows():
        L.append(f"| v6-trained (cross) | `{r.run_id}` | v6 train | {int(r.params):,} | " +
                 " | ".join(f(r[f'v7test_{k}']) for k in M) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    open(f"{R}/_v6_table.md", "w", encoding="utf-8").write(v6_table())
    open(f"{R}/_v7_table_for_v6.md", "w", encoding="utf-8").write(v7_table())
    open(f"{R}/_stream_table.md", "w", encoding="utf-8").write(stream_table())
    print("ok")
