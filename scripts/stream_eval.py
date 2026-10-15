"""Stage 2 driver: score every model on identical (seeded) causal streams; write
results/stream_results.csv and raw scores to results/stream/<stream>.npz.

Streams (deterministic seeds, so a second process -- QAT models need TF_USE_LEGACY_KERAS=1 --
regenerates exactly the same data):
  v7_val_<orient>  : the 5 primary VAL files, 6 min each   -> thresholds
  v7_test_<orient> : the 6 primary TEST files, 10 min each -> reported metrics
  v6_val / v6_test : tiled v6 patches (comparison only)
Thresholds: 'fa10' = lowest threshold with <= 10 false alarms/hour on the matching VAL
stream; 'p0.5' = fixed 0.5.
Usage: python scripts/stream_eval.py spec.json [--qat]
  spec.json: list of {label, run_id, preproc: v7|v6, trained_on}
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import stream_sim as ss  # noqa: E402

ROOT = ss.ROOT
OUT = os.environ.get("STREAM_OUT", os.path.join(ROOT, "results", "stream_results.csv"))
SD = os.path.join(ROOT, "results", "stream")
FA_TARGET = 10.0


def load(run):
    if run["model_path"].endswith(".h5"):
        import tensorflow as tf
        from qkeras.utils import _add_supported_quantized_objects
        co = {}
        _add_supported_quantized_objects(co)
        return tf.keras.models.load_model(os.path.join(ROOT, run["model_path"]), custom_objects=co)
    import keras
    return keras.models.load_model(os.path.join(ROOT, run["model_path"]))


def main(spec_path):
    spec = json.load(open(spec_path))
    specs, meta = [], {}
    for s in spec:
        rp = os.path.join(ROOT, "results", "runs", f"{s['run_id']}.json")
        if not os.path.exists(rp):  # v6a runs live in results/runs_v6a
            rp = os.path.join(ROOT, "results", "runs_v6a", f"{s['run_id']}.json")
        run = json.load(open(rp))
        specs.append(ss.ModelSpec(s["label"], run, load(run), s["preproc"]))
        meta[s["label"]] = dict(s, params=run["params"], kind=run["cfg"]["kind"])
    sp = json.load(open(os.path.join(ROOT, "results", "splits", "v7_splits.json")))
    os.makedirs(SD, exist_ok=True)
    rows = []
    streams = {}
    for orient in ("stored", "physical"):
        t = time.time()
        streams[f"v7_val_{orient}"] = ss.run_v7_stream(specs, sorted(sp["primary"]["val"]), 6.0, orient, seed=101)
        streams[f"v7_test_{orient}"] = ss.run_v7_stream(specs, sorted(sp["primary"]["test"]), 10.0, orient, seed=202)
        print(orient, "streams done", round(time.time() - t), "s", flush=True)
    streams["v6_val"] = ss.run_v6_stream(specs, "val", seed=303)
    streams["v6_test"] = ss.run_v6_stream(specs, "test", seed=404)
    for name, (ends, sc, bursts, hours) in streams.items():
        np.savez_compressed(os.path.join(SD, f"{name}__{os.path.basename(spec_path).replace('.json', '')}.npz"),
                            ends=ends, hours=hours, **{f"p_{k}": v for k, v in sc.items()},
                            burst_t_first=[b["t_first"] for b in bursts], burst_snr=[b["snr"] for b in bursts])
    pairs = [("v7_val_stored", "v7_test_stored", "v7-style stream, stored orientation (low freq first, as trained)"),
             ("v7_val_physical", "v7_test_physical", "v7-style stream, physical orientation (high freq first)"),
             ("v6_val", "v6_test", "v6 tiled-patch stream (comparison only)")]
    for s in specs:
        for vname, tname, desc in pairs:
            ve, vs, vb, vh = streams[vname]
            te, ts, tb, th = streams[tname]
            thr_fa = ss.pick_threshold(ve, vs[s.label], vb, vh, FA_TARGET)
            for mode, thr in (("val_fa<=10/h", thr_fa), ("fixed_0.5", 0.5)):
                r = dict(model=s.label, **{k: v for k, v in meta[s.label].items() if k != "label"},
                         stream=tname, stream_desc=desc, threshold_mode=mode)
                r.update(ss.stream_metrics(te, ts[s.label], tb, th, thr))
                r["val_fa_per_hour_at_thr"] = ss.stream_metrics(ve, vs[s.label], vb, vh, thr)["fa_per_hour"]
                rows.append(r)
        print(s.label, "done", flush=True)
    df = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.model.isin(df.model)], df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(len(df), "rows ->", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
