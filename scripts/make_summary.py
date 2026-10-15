"""Stage 4: results/summary.md + plots from results/sweep_results.csv,
results/hls_projects_index.csv, results/synth_results.csv (EC2, if present),
results/classical_baseline.json.

Every number in summary.md is printed next to the run_id / project it came from.
"""
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = "results"
# reference palette slots 1-4 (validated default instance), + marker shape as secondary encoding
FAM = {"CNN track A (any size)": ("#2a78d6", "o"), "CNN track B (<=10k, HLS-friendly)": ("#eb6834", "s"),
       "CNN track B + QAT (QKeras)": ("#1baf7a", "D"), "LSTM/GRU": ("#eda100", "^")}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"


def fam(r):
    c = json.loads(r.cfg_json)
    if r.kind == "rnn":
        return "LSTM/GRU"
    if str(c.get("quant", "")).startswith("qkeras"):
        return "CNN track B + QAT (QKeras)"
    return "CNN track A (any size)" if c.get("track") == "A" else "CNN track B (<=10k, HLS-friendly)"


def load():
    d = pd.read_csv(f"{R}/sweep_results.csv")
    d = d[d.status == "ok"].copy()
    d["cfgd"] = d.cfg_json.apply(json.loads)
    d["family"] = d.apply(fam, axis=1)
    return d


def robustness(d, rid):
    """seed mean/std (primary), held-out family, 4-fold CV for a parent run id."""
    out = {}
    fam_rows = d[(d.run_id == rid) | d.run_id.str.startswith(rid + "_seed")]
    out["seed_n"] = len(fam_rows)
    out["seed_test_acc"] = (fam_rows.test_acc.mean(), fam_rows.test_acc.std(ddof=0))
    ho = d[d.run_id == rid + "_heldout"]
    out["heldout_test_acc"] = ho.test_acc.iloc[0] if len(ho) else None
    cv = d[d.run_id.str.match(re.escape(rid) + r"_cv\d$")]
    out["cv_n"] = len(cv)
    out["cv_test_acc"] = (cv.test_acc.mean(), cv.test_acc.std(ddof=0)) if len(cv) else None
    return out


def hw_table():
    idx = pd.read_csv(f"{R}/hls_projects_index.csv") if os.path.exists(f"{R}/hls_projects_index.csv") else None
    syn = pd.read_csv(f"{R}/synth_results.csv") if os.path.exists(f"{R}/synth_results.csv") else None
    if idx is not None and syn is not None:
        idx = idx.merge(syn, on="project", how="left", suffixes=("", "_synth"))
    return idx


def fmt(x, p=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def plots(d, hw):
    main = d[d.stage.isin(["stage1_optuna_mix", "stage1_refine_rfi", "stage1_local", "stage3_grid", "stage3_refine", "stage1_qat"])]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, (col, mk) in FAM.items():
        s = main[main.family == name]
        if len(s):
            ax.scatter(s.params, s.test_acc, s=36, c=col, marker=mk, edgecolors="white", linewidths=1.5,
                       label=f"{name} (n={len(s)})", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)", color=INK2)
    ax.set_ylabel("test accuracy (held-out files, S/N >= 7)", color=INK2)
    ax.set_title("Accuracy vs model size - v7 file-grouped test split", color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2)
    cb = json.load(open(f"{R}/classical_baseline.json")) if os.path.exists(f"{R}/classical_baseline.json") else None
    if cb:
        ax.axhline(cb["primary"]["acc"], color=INK2, lw=1, ls="--", zorder=1)
        ax.text(ax.get_xlim()[0] * 1.1, cb["primary"]["acc"] + 0.003, "classical dedispersion + boxcar search",
                color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_accuracy_vs_params.png")
    plt.close(fig)

    if hw is not None and len(hw):
        h = hw.copy()
        have_syn = "latency_us" in h and h["latency_us"].notna().any()
        h["lat"] = h["latency_us"] if have_syn else h["est_latency_us"]
        h["acc"] = h["emu_test_acc"].fillna(h["float_test_acc"])
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for name, (col, mk) in FAM.items():
            s = h[h.family == name] if "family" in h else h.iloc[0:0]
            if len(s):
                ax.scatter(s.lat, s.acc, s=40, c=col, marker=mk, edgecolors="white", linewidths=1.5, label=name, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(("csynth latency (us)" if have_syn else "ESTIMATED latency (us) - not synthesised") +
                      ", log scale", color=INK2)
        ax.set_ylabel("quantised test accuracy (fixed-point emulation)", color=INK2)
        ax.set_title("Accuracy vs latency per 168 ms patch - hls4ml projects" + ("" if have_syn else " (latency = estimate)"), color=INK,
                     loc="left", fontsize=11)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=INK2)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
        fig.tight_layout()
        fig.savefig(f"{R}/fig_accuracy_vs_latency.png")
        plt.close(fig)

    # recall vs S/N for the three headline models
    return main


def snr_plot(d, picks):
    bins = ["3_5", "5_7", "7_10", "10_15", "15_25", "25_50"]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for (label, rid), (name, (col, mk)) in zip(picks, [(k, v) for k, v in FAM.items() if k != "CNN track B + QAT (QKeras)"]):
        r = d[d.run_id == rid]
        if not len(r):
            continue
        y = [r[f"test_recall_snr_{b}"].iloc[0] for b in bins]
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=8, mec="white", mew=1.5, label=f"{label}: {rid}")
    cb = json.load(open(f"{R}/classical_baseline.json")) if os.path.exists(f"{R}/classical_baseline.json") else None
    if cb:
        ax.plot(x, [cb["primary"][f"recall_snr_{b}"] for b in bins], color=INK2, lw=1.5, ls="--",
                label="classical dedispersion search")
    ax.set_xticks(x, [b.replace("_", "-") for b in bins])
    ax.set_xlabel("injected optimal S/N", color=INK2)
    ax.set_ylabel("FRB recall (test files)", color=INK2)
    ax.set_title("Where the errors are: recall vs S/N", color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{R}/fig_recall_vs_snr.png")
    plt.close(fig)


def seedstats(d, rid, col="test_acc"):
    rows = d[(d.run_id == rid) | d.run_id.isin([f"{rid}_seed1", f"{rid}_seed2"])]
    return rows[col].mean(), rows[col].std(ddof=0), len(rows)


def cvstats(d, rid):
    rows = d[d.run_id.isin([f"{rid}_cv{k}" for k in range(4)])]
    return (rows.test_acc.mean(), rows.test_acc.std(ddof=0), len(rows)) if len(rows) else (np.nan, np.nan, 0)


def pm(m, s, n):
    if n == 0 or np.isnan(m):
        return "—"
    return f"{m:.3f} ± {s:.3f} (n={n})" if n > 1 else f"{m:.3f}"


def arch(c):
    if c["kind"] == "rnn":
        return (f"{c['cell'].upper()}({c['units']}) on {256 // c['t_pool']} t × {256 // c['f_pool']} f"
                + (f" + Dense{c['dense']}" if c.get("dense") else ""))
    return (f"{256 // c['t_pool']}² in; Conv{c['filters']} k{c['kernel']}, {c.get('pool_type', 'max')}-pool "
            f"{c['pools']}, {c.get('head', 'flatten')}, Dense{c['dense']}" + (", BN" if c.get("batchnorm") else ""))


def model_table(d, picks):
    """Accuracy table for the headline models (columns) + classical reference."""
    he = json.load(open(f"{R}/headline_eval.json"))
    cb = json.load(open(f"{R}/classical_baseline.json"))
    g = lambda rid: d[d.run_id == rid].iloc[0]  # noqa: E731
    L = ["| | " + " | ".join(f"**{lab}**" for lab, _ in picks) + " | classical dedispersion search |",
         "|---|" + "---|" * (len(picks) + 1)]

    def row(name, f, cbv="—"):
        L.append(f"| {name} | " + " | ".join(f(rid) for _, rid in picks) + f" | {cbv} |")

    def ho(r):
        x = d[d.run_id == r + "_heldout"]
        return f"{x.test_acc.iloc[0]:.3f}" if len(x) else "—"

    hev = lambda r, k, p=4: f"{he[r][k]:.{p}f}" if r in he else "—"  # noqa: E731
    row("run_id", lambda r: f"`{r}`", "`scripts/classical_baseline.py`")
    row("architecture", lambda r: arch(g(r).cfgd), "DM −13…−0.4, boxcar 1–16")
    row("params", lambda r: f"{int(g(r).params):,}")
    row("MACs / inference", lambda r: f"{int(g(r).macs):,}")
    row("**test acc** (seed 0; noise + FRB S/N ≥ 7)", lambda r: f"**{g(r).test_acc:.4f}**", f"{cb['primary']['acc']:.4f}")
    row("test acc, 3 training seeds", lambda r: pm(*seedstats(d, r)))
    row("test acc, 4-fold file-grouped CV", lambda r: pm(*cvstats(d, r)))
    row("test acc, held-out June-2019 family", ho, f"{cb['heldout_family']['acc']:.3f}")
    row("precision / recall", lambda r: f"{g(r).test_precision:.3f} / {g(r).test_recall:.3f}",
        f"{cb['primary']['precision']:.3f} / {cb['primary']['recall']:.3f}")
    row("F1", lambda r: f"{g(r).test_f1:.4f}", f"{cb['primary']['f1']:.4f}")
    row("ROC-AUC", lambda r: f"{g(r).test_auc:.4f}", f"{cb['primary']['auc']:.4f}")
    row("recall @ 1 % FPR", lambda r: f"{g(r).test_recall_at_1pct_fpr:.3f}", f"{cb['primary']['recall_at_1pct_fpr']:.3f}")
    row("specificity (noise samples correct)", lambda r: hev(r, "noise_acc(specificity)"))
    for t in (10, 15, 20):
        row(f"test acc, FRB S/N ≥ {t}", lambda r, t=t: hev(r, f"acc_snr_ge_{t}"))
    row("test acc, v6-brightness-equivalent FRBs (peak 0.5–5 σ)", lambda r: hev(r, "acc_v6_brightness_equiv"))
    row("test acc, all S/N 3–50", lambda r: f"{g(r).test_allsnr_acc:.4f}", f"{cb['primary']['allsnr_acc']:.4f}")
    return "\n".join(L)


def hw_md(hw):
    cols = ["project", "precision", "strategy", "reuse_factor", "float_test_acc", "emu_test_acc", "est_mult_total",
            "est_weight_kbits", "est_latency_us"]
    syn_cols = [c for c in ("status", "latency_worst", "latency_us", "fmax_mhz_est", "DSP", "LUT", "FF", "BRAM_18K",
                            "synth_seconds") if c in hw.columns]
    L = ["| " + " | ".join(cols + syn_cols) + " |", "|" + "---|" * len(cols + syn_cols)]
    for _, r in hw.iterrows():
        L.append("| " + " | ".join(fmt(r.get(c), 4) if isinstance(r.get(c), float) else str(r.get(c))
                                   for c in cols + syn_cols) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    picks = json.loads(sys.argv[1])
    d = load()
    hw = hw_table()
    if hw is not None:
        fam_of = {r.run_id: r.family for r in d.itertuples()}
        hw["family"] = hw.run_id.map(fam_of)
    plots(d, hw)
    snr_plot(d, picks[:3])
    open(f"{R}/_model_table.md", "w", encoding="utf-8").write(model_table(d, picks))
    if hw is not None:
        open(f"{R}/_hw_table.md", "w", encoding="utf-8").write(hw_md(hw))
    print("figures + tables written")
