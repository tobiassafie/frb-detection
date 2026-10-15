"""
Parallel ablation grid: 16 distinct architectures (cross-group deduplicated),
5 seeds each initially, escalating specific borderline comparisons to 10 seeds,
with paired significance testing against the production baseline. Must be run
as a script (not a notebook cell) -- ProcessPoolExecutor on Windows uses the
'spawn' start method, which needs the worker function importable from a real
module (ablation_arch.py), not defined in a notebook's __main__.

Usage: .venv/Scripts/python.exe scripts/run_ablation_grid.py
"""
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from ablation_arch import ARCHITECTURES, EPOCHS_CAP, PATIENCE, train_one_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (REPO_ROOT / "augmented_dataset" / "lstm_dataset_v6.npz").as_posix()
RESULTS_DIR = REPO_ROOT / "ablation_results_v2"
RAW_JSONL = RESULTS_DIR / "raw_results.jsonl"
N_WORKERS = 8
SEEDS_INITIAL = [0, 1, 2, 3, 4]
SEEDS_ESCALATION = [5, 6, 7, 8, 9]
BASELINE_NAME = "baseline"
FPGA_HANDOFF_DIR = REPO_ROOT / "fpga_handoff"
FLATTEN_BUDGET = 500
PARAMS_BUDGET = 50000


def load_existing():
    """key: (config_name, seed) -> result dict."""
    existing = {}
    if RAW_JSONL.exists():
        with open(RAW_JSONL, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing[(r["config_name"], r["seed"])] = r
    return existing


def append_result(result):
    with open(RAW_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")


def build_jobs(seeds, existing):
    jobs = []
    for config_name, spec in ARCHITECTURES.items():
        for seed in seeds:
            if (config_name, seed) in existing:
                continue
            jobs.append({
                "config_name": config_name,
                "config_hash": spec["config_hash"],
                "group": ",".join(spec["groups"]),
                "build_args": spec["build_args"],
                "seed": seed,
                "epochs": EPOCHS_CAP,
                "patience": PATIENCE,
                "data_path": DATA_PATH,
                "results_dir": RESULTS_DIR.as_posix(),
            })
    return jobs


def run_jobs(jobs, label):
    if not jobs:
        print(f"[{label}] nothing to do -- all requested (config, seed) pairs already exist")
        return
    print(f"[{label}] submitting {len(jobs)} jobs across {N_WORKERS} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(train_one_seed, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  [FAILED] {job['config_name']} seed={job['seed']}: {e}")
                continue
            append_result(result)
            done += 1
            retry_note = f" ({result['collapsed_retries']} collapse-retries)" if result["collapsed_retries"] else ""
            print(f"  [{done}/{len(jobs)}] {result['config_name']:>22} seed={result['seed']}  "
                  f"test_acc={result['test_acc']:.4f}  val_acc={result['val_acc']:.4f}  "
                  f"epochs={result['n_epochs_ran']:>3}  {result['train_time_sec']:5.0f}s{retry_note}")


def aggregate(existing, config_name, metric="test_acc"):
    vals = [r[metric] for (cn, seed), r in existing.items() if cn == config_name]
    n = len(vals)
    if n == 0:
        return None
    arr = np.array(vals)
    return {"n": n, "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "sem": float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")}


def paired_diff_test(existing, config_name, baseline_name, seeds, metric="test_acc"):
    """Paired t-test on metric, matched by seed. Returns None if either side is
    missing any of the requested seeds."""
    cand = {}
    base = {}
    for seed in seeds:
        c = existing.get((config_name, seed))
        b = existing.get((baseline_name, seed))
        if c is None or b is None:
            return None
        cand[seed] = c[metric]
        base[seed] = b[metric]
    cand_arr = np.array([cand[s] for s in seeds])
    base_arr = np.array([base[s] for s in seeds])
    diff = cand_arr - base_arr
    n = len(diff)
    mean_diff = float(diff.mean())
    if n > 1 and diff.std(ddof=1) > 0:
        t_stat, p_value = stats.ttest_rel(cand_arr, base_arr)
        sem_diff = diff.std(ddof=1) / math.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_low = float(mean_diff - t_crit * sem_diff)
        ci_high = float(mean_diff + t_crit * sem_diff)
        p_value = float(p_value)
    else:
        # zero variance in the paired difference (degenerate / n=1) -- can't test
        p_value = float("nan")
        ci_low, ci_high = float("nan"), float("nan")
    significant = bool((not math.isnan(ci_low)) and (ci_low > 0 or ci_high < 0))
    return {"n": n, "mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high,
            "p_value": p_value, "significant": significant}


def main():
    os.makedirs(RESULTS_DIR / "models", exist_ok=True)

    print(f"Architectures: {len(ARCHITECTURES)} distinct (cross-group deduplicated)")
    for name, spec in ARCHITECTURES.items():
        print(f"  {name:>22}  groups={spec['groups']}  hash={spec['config_hash']}")

    # ---- Pass 1: initial 5 seeds per config ----
    existing = load_existing()
    jobs = build_jobs(SEEDS_INITIAL, existing)
    run_jobs(jobs, "initial 5-seed pass")
    existing = load_existing()

    # ---- Aggregate + paired significance vs baseline (5-seed) ----
    print("\n=== 5-seed aggregation ===")
    summary = {}
    for name in ARCHITECTURES:
        agg = aggregate(existing, name)
        summary[name] = agg
        print(f"  {name:>22}  n={agg['n']}  test_acc={agg['mean']:.4f} +/- {agg['std']:.4f}  "
              f"(sem={agg['sem']:.4f})")

    print("\n=== paired vs baseline (5-seed) ===")
    escalate = set()
    sig_results = {}
    for name, spec in ARCHITECTURES.items():
        if name == BASELINE_NAME:
            continue
        pt = paired_diff_test(existing, name, BASELINE_NAME, SEEDS_INITIAL)
        sig_results[name] = pt
        baseline_std = summary[BASELINE_NAME]["std"]
        borderline = (not pt["significant"]) and (abs(pt["mean_diff"]) < baseline_std)
        tag = "SIGNIFICANT" if pt["significant"] else ("borderline -> escalating" if borderline else "not significant")
        print(f"  {name:>22}  mean_diff={pt['mean_diff']:+.4f}  95% CI=[{pt['ci_low']:+.4f}, {pt['ci_high']:+.4f}]  "
              f"p={pt['p_value']:.3f}  {tag}")
        if borderline:
            escalate.add(name)

    # ---- Pass 2: escalate borderline comparisons to 10 seeds (baseline included) ----
    if escalate:
        print(f"\n=== escalating to 10 seeds: {sorted(escalate)} (+ baseline) ===")
        escalate_configs = escalate | {BASELINE_NAME}
        escalation_jobs = []
        for config_name in escalate_configs:
            spec = ARCHITECTURES[config_name]
            for seed in SEEDS_ESCALATION:
                if (config_name, seed) in existing:
                    continue
                escalation_jobs.append({
                    "config_name": config_name, "config_hash": spec["config_hash"],
                    "group": ",".join(spec["groups"]), "build_args": spec["build_args"],
                    "seed": seed, "epochs": EPOCHS_CAP, "patience": PATIENCE,
                    "data_path": DATA_PATH, "results_dir": RESULTS_DIR.as_posix(),
                })
        run_jobs(escalation_jobs, "escalation 10-seed pass")
        existing = load_existing()

        print("\n=== paired vs baseline (10-seed, escalated configs only) ===")
        for name in sorted(escalate):
            pt = paired_diff_test(existing, name, BASELINE_NAME, SEEDS_INITIAL + SEEDS_ESCALATION)
            sig_results[name] = pt
            tag = "SIGNIFICANT" if pt["significant"] else "not distinguishable from baseline at n=10"
            print(f"  {name:>22}  mean_diff={pt['mean_diff']:+.4f}  95% CI=[{pt['ci_low']:+.4f}, {pt['ci_high']:+.4f}]  "
                  f"p={pt['p_value']:.3f}  {tag}")
    else:
        print("\nNo borderline comparisons -- no escalation needed.")

    # ---- Final summary table ----
    print("\n=== FINAL SUMMARY ===")
    final_summary = {}
    for name, spec in ARCHITECTURES.items():
        agg = aggregate(existing, name)
        flatten_dim = next(r["flatten_dim"] for (cn, s), r in existing.items() if cn == name)
        n_params = next(r["n_params"] for (cn, s), r in existing.items() if cn == name)
        in_budget = flatten_dim < FLATTEN_BUDGET and n_params < PARAMS_BUDGET
        pt = sig_results.get(name)
        final_summary[name] = {
            "groups": spec["groups"], "n_seeds": agg["n"],
            "test_acc_mean": agg["mean"], "test_acc_std": agg["std"], "test_acc_sem": agg["sem"],
            "flatten_dim": flatten_dim, "n_params": n_params, "in_budget": in_budget,
            "vs_baseline": pt,
            "config_hash": spec["config_hash"],
        }
        sig_str = "n/a (is baseline)" if name == BASELINE_NAME else (
            "SIGNIFICANT" if pt["significant"] else "not significant")
        print(f"  {name:>22}  n_seeds={agg['n']:>2}  test_acc={agg['mean']:.4f}+/-{agg['std']:.4f}  "
              f"flatten={flatten_dim:>5}  params={n_params:>7}  in_budget={in_budget}  vs_baseline={sig_str}")

    with open(RESULTS_DIR / "final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(final_summary, fh, indent=2)

    package_fpga(final_summary, existing)


def package_fpga(final_summary, existing):
    """Winner per group = baseline unless a candidate SIGNIFICANTLY beats it.
    Dedup by config_hash -- never package the same architecture twice. If fewer
    than 3 distinct architectures result, fill remaining slots (up to 3) with
    the highest-mean in-budget config not already included."""
    print("\n=== FPGA handoff selection ===")
    groups = ["pipe_size", "staged_pooling", "time_resolution"]
    chosen = {}  # config_hash -> (config_name, reason)
    for group in groups:
        candidates = [name for name, s in final_summary.items() if group in s["groups"]]
        baseline_mean = final_summary[BASELINE_NAME]["test_acc_mean"]
        best_name, best_mean = BASELINE_NAME, baseline_mean
        for name in candidates:
            if name == BASELINE_NAME:
                continue
            s = final_summary[name]
            pt = s["vs_baseline"]
            if pt["significant"] and pt["mean_diff"] > 0 and s["test_acc_mean"] > best_mean:
                best_name, best_mean = name, s["test_acc_mean"]
        h = final_summary[best_name]["config_hash"]
        reason = "baseline (no significant improvement found)" if best_name == BASELINE_NAME else "significant winner"
        if h in chosen:
            print(f"  {group:>16}: winner is {best_name} -- hash-identical to already-chosen "
                  f"{chosen[h][0]} ({chosen[h][1]}), not packaging a duplicate slot")
        else:
            chosen[h] = (best_name, reason)
            print(f"  {group:>16}: winner = {best_name}  ({reason})")

    # baseline is always included as a reference point
    baseline_hash = final_summary[BASELINE_NAME]["config_hash"]
    if baseline_hash not in chosen:
        chosen[baseline_hash] = (BASELINE_NAME, "baseline (always included as reference)")
        print(f"  {'(reference)':>16}: winner = {BASELINE_NAME}  (baseline, always included)")

    # fill free slots (cap 3 distinct configs) with the next-best in-budget,
    # not-already-chosen config, ranked by mean test_acc
    MAX_SLOTS = 3
    if len(chosen) < MAX_SLOTS:
        remaining = [
            (name, s["test_acc_mean"]) for name, s in final_summary.items()
            if s["config_hash"] not in chosen and s["in_budget"]
        ]
        remaining.sort(key=lambda t: -t[1])
        for name, mean_acc in remaining:
            if len(chosen) >= MAX_SLOTS:
                break
            h = final_summary[name]["config_hash"]
            chosen[h] = (name, f"budget-filler (highest remaining in-budget mean test_acc={mean_acc:.4f})")
            print(f"  {'(filler)':>16}: added = {name}  (budget-filler, mean test_acc={mean_acc:.4f})")

    print(f"\n{len(chosen)} distinct architecture(s) selected for FPGA packaging:")
    for h, (name, reason) in chosen.items():
        s = final_summary[name]
        in_budget_str = "IN BUDGET" if s["in_budget"] else "OVER BUDGET (flatten>=500 or params>=50k)"
        print(f"  {name:>22}  flatten={s['flatten_dim']:>5}  params={s['n_params']:>7}  "
              f"{in_budget_str}  -- {reason}")
        if not s["in_budget"]:
            print(f"    ^^ WARNING: {name} does not satisfy the FPGA budget and should not be packaged as-is")

    with open(RESULTS_DIR / "fpga_selection.json", "w", encoding="utf-8") as fh:
        json.dump({h: {"config_name": n, "reason": r} for h, (n, r) in chosen.items()}, fh, indent=2)

    return chosen


if __name__ == "__main__":
    main()
