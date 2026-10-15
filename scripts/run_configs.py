"""Run a list of configs (JSON file: list of cfg dicts, each with a run_id) across
N worker processes. Usage: python scripts/run_configs.py cfgs.json [workers] [threads]"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))


def _one(args):
    cfg, threads = args
    import frb_train as ft
    try:
        r = ft.train_eval(cfg, run_id=cfg["run_id"], threads=threads)
    except Exception as e:  # log failures as results, never hide them
        r = dict(run_id=cfg["run_id"], status=f"error: {type(e).__name__}: {e}", cfg=cfg)
        os.makedirs(ft.RUNS, exist_ok=True)
        json.dump(r, open(os.path.join(ft.RUNS, cfg["run_id"] + ".json"), "w"), indent=1, default=str)
    return r


if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1]))
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    done = set(f[:-5] for f in os.listdir("results/runs")) if os.path.isdir("results/runs") else set()
    todo = [c for c in cfgs if c["run_id"] not in done]
    print(f"{len(todo)} to run ({len(cfgs) - len(todo)} already done)", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_one, (c, threads)) for c in todo]
        for f in as_completed(futs):
            r = f.result()
            print(r["run_id"], r["status"], "params", r.get("params"), "val_acc", round(r.get("val_acc", -1), 4),
                  "test_acc", round(r.get("test_acc", -1), 4), "auc", round(r.get("test_auc", -1), 4), flush=True)
