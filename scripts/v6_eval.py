"""v6 benchmark table -> results/v6_benchmark.csv  ("v6 benchmark - leaky split, comparison only").

Rows:
  * no-learning baselines on v6 test (results/v6_baselines.json)
  * every v6-trained run (stage v6_bench / v6_sweep): v6 test metrics, dedup-test recall,
    and the same model on the v7 primary test set (cross-dataset)
  * v7-trained headline models evaluated on the v6 test set (cross-dataset)
Cross-dataset inputs are renormalised to the convention the model was TRAINED with:
  v6-trained -> per-patch z-score (v6 convention) of the v7 patch;
  v7-trained -> per-channel robust normalisation (v7 convention) of the v6 patch.
Then block-mean pooled to the model's input shape, as in training.
Usage: [TF_USE_LEGACY_KERAS=1] python scripts/v6_eval.py [--qat]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import frb_train as ft  # noqa: E402
from v7_common import channel_normalise  # noqa: E402

OUT = os.path.join(ft.ROOT, "results", "v6_benchmark.csv")
V7_HEADLINE = ["s1mB_t0041_rfi", "loc_t0041_b32", "loc_t0041_A_w8k5bn", "s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0",
               "s3r_gru_u16_T64_F8_lr0.003_do0.0_d16"]
METRIC_KEYS = ["acc", "precision", "recall", "f1", "auc", "recall_at_1pct_fpr"]


def load_model(run):
    if run["model_path"].endswith(".h5"):
        import tensorflow as tf
        from qkeras.utils import _add_supported_quantized_objects
        co = {}
        _add_supported_quantized_objects(co)
        return tf.keras.models.load_model(os.path.join(ft.ROOT, run["model_path"]), custom_objects=co)
    import keras
    return keras.models.load_model(os.path.join(ft.ROOT, run["model_path"]))


def pooled(Xfull, tp, fp):
    n, T, F = Xfull.shape
    return Xfull.reshape(n, T // tp, tp, F // fp, fp).mean((2, 4))


def zscore(X):
    m = X.mean((1, 2), keepdims=True); s = X.std((1, 2), keepdims=True)
    return (X - m) / np.where(s > 0, s, 1)


_full = {}


def full_res(dataset, which):
    key = (dataset, which)
    if key not in _full:
        X, meta = ft.load_data(1, 1, dataset=dataset)
        idx = ft.split_indices(meta, "v6orig" if dataset == "v6" else "primary")[which]
        _full[key] = (np.asarray(X[idx], np.float32), meta.label.values[idx].astype(float))
    return _full[key]


def eval_on(model, cfg, dataset, which, renorm):
    Xf, y = full_res(dataset, which)
    if renorm == "v6":
        Xf = zscore(Xf)
    elif renorm == "v7":
        live = np.ones(256, bool)
        Xf = np.stack([channel_normalise(x, live) for x in Xf]).clip(-20, 60)
    x = pooled(Xf, cfg["t_pool"], cfg["f_pool"])
    if cfg.get("clip"):
        x = np.clip(x, *cfg["clip"])
    x = x[..., None] if cfg["kind"] == "cnn" else x
    p = ft.predict(model, x)
    return ft.metrics(y, p), p


def main(qat=False):
    rows = []
    old = pd.read_csv(OUT) if os.path.exists(OUT) else None
    if not qat:
        b = json.load(open(os.path.join(ft.ROOT, "results", "v6_baselines.json")))
        for name in ("lookup_exact_noise", "nearest_neighbour_1nn"):
            r = dict(row_type="baseline", run_id=name, trained_on="(no learning)", evaluated_on="v6 test (original split)")
            r.update({k: b[name][k] for k in METRIC_KEYS})
            r["dedup_test_recall"] = b[f"{name}_on_dedup"]["recall"]
            rows.append(r)
    runs = sorted(glob.glob(os.path.join(ft.RUNS, "v6_*.json"))) + sorted(glob.glob(os.path.join(ft.RUNS, "qat_v6_*.json")))
    for f in runs:
        run = json.load(open(f))
        if run.get("status") != "ok":
            continue
        is_q = run["model_path"].endswith(".h5")
        if is_q != qat:
            continue
        cfg = run["cfg"]
        m = load_model(run)
        r = dict(row_type="model", run_id=run["run_id"], trained_on="v6 train (original split)",
                 evaluated_on="v6 test (original split)", stage=cfg.get("stage"), source_v7_run=cfg.get("source_v7_run",
                 cfg.get("parent")), seed=cfg.get("seed"), params=run["params"], kind=cfg["kind"],
                 quant=cfg.get("quant", "float32"))
        met, p = eval_on(m, cfg, "v6", "test", None)
        r.update({k: met[k] for k in METRIC_KEYS})
        X, meta = ft.load_data(1, 1, dataset="v6")
        te = ft.split_indices(meta, "v6orig")
        dedup_mask = ~meta.dup_in_train.values[te["test"]]
        r["dedup_test_recall"] = float((p[dedup_mask] >= 0.5).mean())
        r["dedup_test_n"] = int(dedup_mask.sum())
        met7, _ = eval_on(m, cfg, "v7", "test", "v6")
        r.update({f"v7test_{k}": met7[k] for k in METRIC_KEYS})
        rows.append(r)
        print(r["run_id"], round(r["acc"], 4), "dedup", round(r["dedup_test_recall"], 4), "v7", round(r["v7test_acc"], 4))
    # v7-trained headline models on the v6 test set
    heads = ["qat_s1mB_t0041_rfi_8b"] if qat else V7_HEADLINE
    for rid in heads:
        run = json.load(open(os.path.join(ft.RUNS, f"{rid}.json")))
        cfg = run["cfg"]
        m = load_model(run)
        met, p = eval_on(m, cfg, "v6", "test", "v7")
        X, meta = ft.load_data(1, 1, dataset="v6")
        dedup_mask = ~meta.dup_in_train.values[ft.split_indices(meta, "v6orig")["test"]]
        r = dict(row_type="cross_v7_to_v6", run_id=rid, trained_on="v7 train (file-grouped)",
                 evaluated_on="v6 test (original split)", params=run["params"], kind=cfg["kind"],
                 quant=cfg.get("quant", "float32"), dedup_test_recall=float((p[dedup_mask] >= 0.5).mean()),
                 dedup_test_n=int(dedup_mask.sum()))
        r.update({k: met[k] for k in METRIC_KEYS})
        r.update({f"v7test_{k}": run[f"test_{k}"] for k in METRIC_KEYS})
        rows.append(r)
        print(rid, "on v6 test", round(r["acc"], 4))
    df = pd.DataFrame(rows)
    if old is not None:
        old = old.drop(columns=["table"], errors="ignore")
        df = pd.concat([old[~old.run_id.isin(df.run_id)], df], ignore_index=True)
    df.insert(0, "table", "v6 benchmark - leaky split, comparison only")
    df.to_csv(OUT, index=False)
    print(len(df), "rows ->", OUT)


if __name__ == "__main__":
    main(qat="--qat" in sys.argv)
