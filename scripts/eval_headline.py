"""Extra evaluation of headline models on the SAME primary test files:
accuracy when the FRB class is restricted to optimal S/N >= {5,7,10,15,20},
on a v6-brightness-equivalent subset (FRB peak pixel 0.5-5 sigma, v6's
SNR_INJECT_RANGE convention), and per test file. Noise samples are always all
noise samples of the test files. Appends to results/headline_eval.json.

Usage: [TF_USE_LEGACY_KERAS=1] python scripts/eval_headline.py run_id [run_id ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

import frb_train as ft  # noqa: E402


def load_model(run):
    if run["model_path"].endswith(".h5"):
        import tensorflow as tf
        from qkeras.utils import _add_supported_quantized_objects
        co = {}
        _add_supported_quantized_objects(co)
        return tf.keras.models.load_model(os.path.join(ft.ROOT, run["model_path"]), custom_objects=co)
    import keras
    return keras.models.load_model(os.path.join(ft.ROOT, run["model_path"]))


def main(rids):
    out_path = os.path.join(ft.ROOT, "results", "headline_eval.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for rid in rids:
        run = json.load(open(os.path.join(ft.RUNS, f"{rid}.json")))
        cfg = run["cfg"]
        X, meta = ft.load_data(cfg["t_pool"], cfg["f_pool"])
        idx = ft.split_indices(meta, cfg.get("split", "primary"), cfg.get("fold"))
        ii = idx["test_all"]
        m = meta.iloc[ii]
        p = ft.predict(load_model(run), ft.prep_input(X[ii], cfg["kind"], cfg.get("clip")))
        y = m.label.values
        ok = (p >= 0.5) == y
        noise = y == 0
        res = {"n_noise": int(noise.sum()), "noise_acc(specificity)": float(ok[noise].mean())}
        for t in (3, 5, 7, 10, 15, 20):
            sel = noise | (m.snr.values >= t)
            res[f"acc_snr_ge_{t}"] = float(ok[sel].mean())
            res[f"recall_snr_ge_{t}"] = float(ok[(~noise) & (m.snr.values >= t)].mean())
        sel = noise | m.peak_amp.between(0.5, 5).values
        res["acc_v6_brightness_equiv"] = float(ok[sel].mean())
        res["n_frb_v6_brightness_equiv"] = int(((~noise) & m.peak_amp.between(0.5, 5).values).sum())
        per = {}
        for g, sub in m.assign(ok=ok).groupby("group_id"):
            ev = (sub.label == 0) | (sub.snr >= 7)
            per[sub.file.iloc[0][:14]] = dict(acc_snr_ge_7=float(sub.ok[ev].mean()),
                                               noise_acc=float(sub.ok[sub.label == 0].mean()),
                                               frb_recall_snr_ge_7=float(sub.ok[(sub.label == 1) & (sub.snr >= 7)].mean()))
        res["per_test_file"] = per
        out[rid] = res
        print(rid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "per_test_file"})
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
