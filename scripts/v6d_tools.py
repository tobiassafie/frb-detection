"""v6d helpers.
  python scripts/v6d_tools.py seeds   -> results/logs/v6d_C.json: seeds 142, 242 for every grid config
                                         whose seed-42 test >= 0.915 (prune rule: > 3 pt below 0.945
                                         after seed 1), ordered by seed-42 val (best first)
  python scripts/v6d_tools.py summary -> results/v6d_configs.csv (per-config mean ± std, n seeds),
                                         results/fig_v6d_pareto.png, results/_v6d_tables.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

R = "results"
PRUNE_TEST = 0.945 - 0.03
REF_ACC, REF_MACS = 0.963, 7341088
LABEL = "v6 benchmark - leaky split, comparison only"
M = ["val_acc", "test_acc", "test_precision", "test_recall", "test_f1", "test_auc", "test_recall_at_1pct_fpr",
     "noise_acc_perturbed_0p05", "v7test_acc"]


def runs():
    out = []
    for f in glob.glob(f"{R}/runs_v6d/*.json"):
        r = json.load(open(f))
        if r["cfg"].get("stage") in ("v6d_B", "v6d_C"):
            out.append(r)
    return out


def seeds():
    rs = [r for r in runs() if r["cfg"]["seed"] == 42]
    keep = [r for r in rs if r["test_acc"] >= PRUNE_TEST and not r.get("collapsed_all")]
    keep.sort(key=lambda r: -r["val_acc"])
    cfgs = []
    for r in keep:
        for s in (142, 242):
            c = dict(r["cfg"], seed=s, stage="v6d_C", run_id=r["run_id"].replace("_s42", f"_s{s}"), hypothesis="grid seeds")
            cfgs.append(c)
    json.dump(cfgs, open(f"{R}/logs/v6d_C.json", "w"), indent=1)
    print(f"{len(rs)} seed-42 runs, {len(rs) - len(keep)} pruned (test < {PRUNE_TEST:.3f}), {len(cfgs)} seed runs queued")


def summary():
    rs = runs()
    df = pd.DataFrame([dict({k: r[k] for k in M}, run_id=r["run_id"], key=r["cfg"]["cfg_key"], seed=r["cfg"]["seed"],
                            params=r["params"], macs=r["macs"], c=r["cfg"]["freq_reduction"], p=r["cfg"]["t_pool"],
                            cell=r["cfg"]["cell"], units=r["cfg"]["units"], dense=r["cfg"]["dense"],
                            attempts=r["n_attempts"]) for r in rs])
    g = df.groupby("key")
    agg = g[M].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    first = g[["params", "macs", "c", "p", "cell", "units", "dense"]].first()
    agg = first.join(agg)
    agg["n_seeds"] = g.size()
    agg["run_ids"] = g.run_id.apply(lambda s: ";".join(sorted(s)))
    agg["pruned_after_seed42"] = (agg.n_seeds == 1) & (agg.test_acc_mean < PRUNE_TEST)
    agg = agg.sort_values("macs")
    agg.insert(0, "table", LABEL)
    agg["lookup_baseline_acc"] = 0.984
    agg.to_csv(f"{R}/v6d_configs.csv")
    full = agg[agg.n_seeds >= 3]
    # Pareto front on (macs down, test mean up) among 3-seed configs
    front, best = [], -1
    for k, r in full.sort_values("macs").iterrows():
        if r.test_acc_mean > best:
            front.append(k)
            best = r.test_acc_mean
    within = full[full.test_acc_mean >= REF_ACC - 0.01].sort_values(["macs", "test_acc_mean"], ascending=[True, False])
    under1m = full[full.macs < 1e6].sort_values("test_acc_mean", ascending=False)
    pick_within = within.index[0] if len(within) else None
    pick_1m = under1m.index[0] if len(under1m) else None
    plot(agg, full, front, pick_within, pick_1m)
    tables(agg, full, front, pick_within, pick_1m, df)
    print("within 1 pt (lowest MACs):", pick_within, "| best < 1M MACs:", pick_1m)


def plot(agg, full, front, pw, p1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"
    COL = {"lstm": "#2a78d6", "gru": "#eb6834"}            # palette slots 1-2
    MK = {16: "o", 32: "s", 64: "D"}                      # Conv1D width as secondary encoding
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    for (cell, c), s in full.groupby(["cell", "c"]):
        ax.errorbar(s.macs, s.test_acc_mean, yerr=s.test_acc_std, fmt=MK[c], color=COL[cell], ms=6, mec="white", mew=1,
                    elinewidth=0.8, capsize=0, alpha=0.9, label=f"{cell.upper()}, Conv1D {c}", zorder=3)
    p1s = agg[agg.n_seeds < 3]
    ax.scatter(p1s.macs, p1s.test_acc_mean, s=14, facecolors="none", edgecolors=INK2, lw=0.8,
               label="seed 42 only (pruned / not run)", zorder=2)
    fr = full.loc[front].sort_values("macs")
    ax.plot(fr.macs, fr.test_acc_mean, color=INK, lw=1.2, drawstyle="steps-post", zorder=1, label="Pareto front (3 seeds)")
    ax.axhline(REF_ACC, color=INK2, ls="--", lw=1)
    ax.text(2.6e6, 0.905, "dashed: lstm_v3 0.963 (7.34M MACs)\nshaded: within 1 pt", color=INK2, fontsize=8)
    ax.axhspan(REF_ACC - 0.01, REF_ACC, color=GRID, alpha=0.5, zorder=0)
    ax.axvline(1e6, color=INK2, ls=":", lw=1)
    for (k, lab), off in zip(((pw, "cheapest within 1 pt"), (p1, "best < 1M MACs")), ((20, -150), (30, -190))):
        if k is not None:
            r = full.loc[k]
            ax.annotate(f"{lab}: {k}", (r.macs, r.test_acc_mean), xytext=off, textcoords="offset points",
                        fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("MACs per inference (log scale)", color=INK2)
    ax.set_ylabel("v6 test accuracy (mean ± std, 3 seeds)", color=INK2)
    ax.set_title("Small RNNs: accuracy vs cost — v6 benchmark, leaky split, comparison only", color=INK, loc="left", fontsize=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2)
    lo = max(0.80, min(agg.test_acc_mean.min(), 0.90) - 0.01)
    ax.set_ylim(lo, 0.99)
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_v6d_pareto.png")
    plt.close(fig)


def fmt(r, k):
    m, s = r[f"{k}_mean"], r[f"{k}_std"]
    return f"{m:.3f} ± {s:.3f}" if r.n_seeds > 1 and not np.isnan(s) else f"{m:.3f}"


def tables(agg, full, front, pw, p1, df):
    cols = ["test_acc", "test_precision", "test_recall", "test_f1", "test_auc", "test_recall_at_1pct_fpr",
            "noise_acc_perturbed_0p05", "v7test_acc"]
    head = ("| config | n seeds | run_ids | params | MACs | val | " + " | ".join(c.replace("test_", "") for c in cols)
            + " | lookup |")
    sep = "|---|---|---|---|---|---|" + "---|" * len(cols) + "---|"

    def row(k, r):
        return (f"| `{k}` | {r.n_seeds} | {', '.join('`' + x + '`' for x in r.run_ids.split(';'))} | {int(r.params):,} | "
                f"{int(r.macs):,} | {fmt(r, 'val_acc')} | " + " | ".join(fmt(r, c) for c in cols) + " | 0.984 |")
    L = [f"**{LABEL}** — named picks (3 seeds)", "", head, sep]
    for k in [x for x in (pw, p1) if x is not None]:
        L.append(row(k, full.loc[k]))
    L += ["", f"**{LABEL}** — Pareto front (3-seed configs; accuracy rises with MACs along the front)", "", head, sep]
    for k in front:
        L.append(row(k, full.loc[k]))
    L += ["", f"**{LABEL}** — all configs (sorted by MACs; n seeds = 1 → pruned after seed 42 or not reached in time)",
          "", head, sep]
    for k, r in agg.iterrows():
        L.append(row(k, r))
    open(f"{R}/_v6d_tables.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    {"seeds": seeds, "summary": summary}[sys.argv[1]]()
