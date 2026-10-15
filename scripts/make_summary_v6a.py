"""results/_v6a_*.md tables for results/summary_v6a.md (from results/runs_v6a/*.json,
results/v6a_hls_index.csv, results/stream_results_v6a.csv)."""
import glob
import json
import os

import numpy as np
import pandas as pd

R = "results"
LABEL = "**v6 benchmark - leaky split, comparison only**"
CEIL = 303 / 308


def runs():
    return {json.load(open(f))["run_id"]: json.load(open(f)) for f in glob.glob(f"{R}/runs_v6a/*.json")}


def f(x, p=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def ctx(r):
    return (f"{f(r.get('dedup_recall'))} | {f(r.get('v7test_acc'))} | {f(r.get('v6test_timereversed_acc'))} | "
            f"{f(r.get('noise_acc_perturbed_0p05'))}")


CTXH = "dedup FRB recall (149) | v7 test acc | v6 test time-reversed acc | noise acc, +0.05σ probe"


def repro(R_):
    L = [LABEL + " — Stage 0: reproduced vs originally reported (v6 test, 308 samples; lookup baseline 0.984; "
         "max achievable 0.984)", "",
         f"| model | how | run_id | params | reported | reproduced | {CTXH} |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(R_.items()):
        if r.get("stage") == "stage0_saved":
            L.append(f"| {k.replace('saved__', '')} | saved weights | `{k}` | {r['params']:,} | {f(r['reported_test_acc'], 4)} | "
                     f"{f(r['test_acc'], 4)} | {ctx(r)} |")
    rep = {"s0_baseline_2_4_4_4_4_s142": ("seedcheck_baseline_s142", 0.9675), "s0_train_cnn_py_4_8_8": ("train_cnn.py", None),
           "s0_lstm_v3_recipe": ("lstm_v3 (ES patience 10)", 0.9643), "s0_lstm_v3_recipe_full150": ("lstm_v3 (all 150 epochs)", 0.9643)}
    for k, (n, v) in rep.items():
        r = R_[k]
        L.append(f"| {n} | retrained, original recipe | `{k}` | {r['params']:,} | {f(v, 4)} | {f(r['test_acc'], 4)} | {ctx(r)} |")
    return "\n".join(L)


def seeds(R_, ids):
    rs = [R_[i] for i in ids if i in R_]
    m = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))  # noqa: E731
    return rs, m


def pm(t):
    return f"{t[0]:.3f} ± {t[1]:.3f}"


def headline(R_, hw):
    groups = [
        ("Best CNN (by 3-seed val)", ["s1_H3_res256", "s1_H3_res256_seed1", "s1_H3_res256_seed2"], "s1_H3_res256_seed2"),
        ("Best smallest CNN within 0.5 pt (½ filters/dense)", ["s1_H4_half"], "s1_H4_half"),
        ("Most robust CNN (old v6 recipe, 3 seeds)", ["s0_baseline_2_4_4_4_4_s142", "s0_baseline_2_4_4_4_4_seed1",
                                                     "s0_baseline_2_4_4_4_4_seed2"], "s0_baseline_2_4_4_4_4_s142"),
        ("Best RNN (lstm_v3 recipe, 3 seeds)", ["s0_lstm_v3_recipe_full150", "s0_lstm_v3_recipe_full150_seed1",
                                                "s0_lstm_v3_recipe_full150_seed2"], "s0_lstm_v3_recipe_full150_seed1"),
        ("Smaller-input RNN (LSTM64, 64×128, 3 seeds)", ["s2_H7_lstm64", "s2_H7_lstm64_seed1", "s2_H7_lstm64_seed2"],
         "s2_H7_lstm64"),
    ]
    L = [LABEL + " — headline models (test metrics = mean ± std over the listed seeds; context columns for the "
         "exported run)", "",
         "| model | run_ids | exported run | params | MACs | acc | precision | recall | F1 | ROC-AUC | recall@1%FPR | "
         "lookup baseline | max achievable | " + CTXH + " | fixed-point emu acc (16,6) | est. mults (RF4) | est. latency µs |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, ids, exp in groups:
        rs, m = seeds(R_, ids)
        e = R_[exp]
        h = hw[hw.run_id == exp]
        hh = h.iloc[0] if len(h) else {}
        L.append(f"| {name} | {', '.join('`' + i + '`' for i in ids)} | `{exp}` | {e['params']:,} | {e['macs']:,} | "
                 + " | ".join(pm(m(f'test_{k}')) if len(rs) > 1 else f(rs[0][f'test_{k}'])
                              for k in ("acc", "precision", "recall", "f1", "auc", "recall_at_1pct_fpr"))
                 + f" | 0.984 | {CEIL:.4f} | {ctx(e)} | {f(hh.get('emu_test_acc_v6')) if len(h) else '—'} | "
                   f"{int(hh['est_mult_total']) if len(h) else '—'} | {f(hh.get('est_latency_us'), 1) if len(h) else '—'} |")
    return "\n".join(L)


def hyp(R_):
    rows = [("H1", "flatten > GAP", "s1_H1_gap", "confirmed"), ("H2", "max > avg pool", "s1_H2_avgpool", "inconclusive"),
            ("H3", "256 vs 128", "s1_H3_res256", "weakly confirmed (3 seeds: 0.984 vs 0.979)"),
            ("H3", "64 vs 128", "s1_H3_res64", "inconclusive"),
            ("H4", "½ filters/dense", "s1_H4_half", "confirmed: within 0.5 pt (smallest acceptable)"),
            ("H4", "¼ filters/dense", "s1_H4_quarter", "confirmed drop (−2.8 pt)"),
            ("H5a", "L2 1e-4 + dropout 0.3", "s1_H5a_l2_drop", "rejected (no gain)"),
            ("H5b", "time-shift ±16 + noise 0.1 aug", "s1_H5b_aug", "rejected for v6 acc (−4 pt); only CNN on this recipe that stops memorising"),
            ("H6", "64 t × 64 f", "s2_H6_T64_F64", "rejected (−1.5 pt vs 256×256+1×1)"),
            ("H6", "32 t × 128 f", "s2_H6_T32_F128", "rejected"), ("H6", "64 t × 128 f", "s2_H6_T64_F128", "rejected (best H6)"),
            ("H7", "LSTM64 vs 32 (64×128)", "s2_H7_lstm64", "weakly confirmed (+1.6 pt seed 42; 3-seed 0.953)"),
            ("H7", "GRU32 vs LSTM32", "s2_H7_gru32", "inconclusive (≈)"),
            ("H8", "Conv1D front end", "s2_H8_conv1d_front", "rejected")]
    L = [LABEL + " — hypotheses (one change off the base; base = 128² max/flatten CNN, 3 seeds test 0.977–0.984; "
         "RNN base = lstm_v3 recipe, test 0.963 ± 0.003)", "",
         f"| id | change | run_id | params | val | test | {CTXH} | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, ch, rid, v in rows:
        r = R_[rid]
        L.append(f"| {i} | {ch} | `{rid}` | {r['params']:,} | {f(r['val_acc'])} | {f(r['test_acc'])} | {ctx(r)} | {v} |")
    return "\n".join(L)


def stream():
    p = f"{R}/stream_results_v6a.csv"
    if not os.path.exists(p):
        return "_(streaming results pending)_"
    d = pd.read_csv(p)
    L = [LABEL + " — streaming context (v6-trained models; causal 256-bin windows, stride 64; 1 h v7-style test "
         "stream, 1,716 bursts per orientation; threshold = lowest with ≤ 10 FA/h on the val stream; 1.000 = "
         "target not reachable)", "",
         "| model | run_id | stream | thr | test FA/h | det S/N≥7 | det S/N≥15 | det all | median TTD ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in d[d.threshold_mode == "val_fa<=10/h"].iterrows():
        st = {"v7_test_stored": "v7 stored", "v7_test_physical": "v7 physical (high-f first)", "v6_test": "v6 tiled"}[r.stream]
        L.append(f"| {r.model} | `{r.run_id}` | {st} | {f(r.threshold)} | {f(r.fa_per_hour, 0)} | {f(r.det_rate_snr_7_50)} | "
                 f"{f(r.det_rate_snr_15_50)} | {f(r.det_rate_all)} | {f(r.ttd_ms_median_snr_ge7, 0)} |")
    return "\n".join(L)


if __name__ == "__main__":
    R_ = runs()
    hw = pd.read_csv(f"{R}/v6a_hls_index.csv")
    for n, t in (("repro", repro(R_)), ("headline", headline(R_, hw)), ("hyp", hyp(R_)), ("stream", stream())):
        open(f"{R}/_v6a_{n}.md", "w", encoding="utf-8").write(t)
    print("ok")
