"""Causal rolling LSTM/GRU on the original v6 split: P(FRB | data up to t) at every time step.
See results/NOTES_roll.md for the label / orientation / onset definitions.
Usage: python scripts/roll_train.py cfgs.json [workers] [threads]
Each run -> results/runs_roll/<run_id>.json (+ model in results/models_roll/), aggregated into
results/v7r_runs.csv (table label: "v6 benchmark - leaky split, comparison only").
"""
import glob
import json
import os
import sys
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs_roll")
MODELS = os.path.join(ROOT, "results", "models_roll")
CSV = os.path.join(ROOT, "results", "v7r_runs.csv")
LOOKUP = 0.9837662337662337
LABEL = "v6 benchmark - leaky split, comparison only"
DEF = dict(cell="lstm", units=32, front=16, head=16, t_pool=1, label="onset", lr=1e-3, batch=128, epochs=150,
           lr_patience=5, seed=42)
_D = {}


def data(t_pool):
    """Physical-order (time-reversed) v6, causal time max-pool, bad samples removed."""
    if t_pool in _D:
        return _D[t_pool]
    d = np.load(os.path.join(ROOT, "augmented_dataset", "lstm_dataset_v6.npz"))
    m = np.load(os.path.join(ROOT, "augmented_dataset", "v6bench", "roll_meta.npz"))
    out = {}
    for s in ("train", "val", "test"):
        X = d[f"X_{s}"][:, ::-1, :]                         # physical high-frequency-first order
        y = d[f"y_{s}"].astype(int)
        bad = m[f"{s}_bad"]
        keep = ~bad
        X, y = X[keep], y[keep]
        n, T, F = X.shape
        Xp = X.reshape(n, T // t_pool, t_pool, F).max(2) if t_pool > 1 else X
        out[s] = dict(X=np.ascontiguousarray(Xp, np.float32), y=y, onset=m[f"{s}_onset"][keep],
                      end=m[f"{s}_end"][keep], score=m[f"{s}_track_score"][keep], n_bad=int(bad.sum()),
                      orig_idx=np.where(keep)[0])
    _D[t_pool] = out
    return out


def step_labels(D, t_pool, kind):
    n, T, _ = D["X"].shape
    Y = np.zeros((n, T), np.float32)
    for i in np.where(D["y"] == 1)[0]:
        on, en = D["onset"][i] // t_pool, D["end"][i] // t_pool
        if kind == "window":
            Y[i, on:en + 1] = 1
        else:
            Y[i, on:] = 1
    return Y


def build(cfg, T, F, stateful=False):
    from tensorflow import keras
    from tensorflow.keras import layers
    if stateful:
        inp = keras.Input(batch_shape=(1, 1, F), name="input_data")
    else:
        inp = keras.Input(shape=(T, F), name="input_data")
    x = layers.Conv1D(cfg["front"], 1, activation="relu", name="front")(inp)
    cell = layers.LSTM if cfg["cell"] == "lstm" else layers.GRU
    x = cell(cfg["units"], return_sequences=True, stateful=stateful, name="rnn")(x)
    x = layers.Conv1D(cfg["head"], 1, activation="relu", name="head")(x)
    out = layers.Conv1D(1, 1, activation="sigmoid", name="p_frb")(x)
    return keras.Model(inp, out)


def macs_per_step(cfg, F):
    g = 4 if cfg["cell"] == "lstm" else 3
    u, c, h = cfg["units"], cfg["front"], cfg["head"]
    return F * c + g * u * (c + u) + u * h + h


class BestVal:
    def __new__(cls):
        from tensorflow import keras

        class B(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best, self.w, self.ep = np.inf, None, -1

            def on_epoch_end(self, epoch, logs=None):
                if logs["val_loss"] < self.best:
                    self.best, self.w, self.ep = logs["val_loss"], self.model.get_weights(), epoch + 1

            def on_train_end(self, logs=None):
                if self.w is not None:
                    self.model.set_weights(self.w)
        return B()


def patch_metrics(y, s, thr):
    from sklearn.metrics import roc_auc_score, roc_curve
    yh = (s >= thr).astype(int)
    tp, fp = int(((yh == 1) & (y == 1)).sum()), int(((yh == 1) & (y == 0)).sum())
    fn, tn = int(((yh == 0) & (y == 1)).sum()), int(((yh == 0) & (y == 0)).sum())
    fpr, tpr, _ = roc_curve(y, s)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(y), precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                auc=float(roc_auc_score(y, s)), recall_at_1pct_fpr=float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0)


def evaluate(model, cfg, D, thr_val=None):
    """Per-step AUC, patch-level metrics with patch score = MAX_t P_t, detection latency."""
    from sklearn.metrics import roc_auc_score
    tp = cfg["t_pool"]
    P = model.predict(D["X"], batch_size=256, verbose=0)[..., 0]
    Y = step_labels(D, tp, cfg["label"])
    res = dict(step_auc=float(roc_auc_score(Y.ravel(), P.ravel())))
    s = P.max(1)
    res.update({f"patch_{k}": v for k, v in patch_metrics(D["y"], s, 0.5).items()})
    if thr_val is not None:
        res.update({f"patch_{k}_at_thr": v for k, v in patch_metrics(D["y"], s, thr_val).items()})
        fr = np.where(D["y"] == 1)[0]
        lat, early = [], 0
        for i in fr:
            hit = np.where(P[i] >= thr_val)[0]
            if len(hit):
                dt = (hit[0] - D["onset"][i] // tp) * tp       # native 0.655 ms bins after onset
                lat.append(dt)
                early += dt < 0
        lat = np.array(lat)
        strong = np.array([D["score"][i] >= 8 for i in fr])
        res.update(latency_bins_median=float(np.median(lat)) if len(lat) else np.nan,
                   latency_bins_p90=float(np.percentile(lat, 90)) if len(lat) else np.nan,
                   frac_detected_before_onset=float(early / max(len(lat), 1)),
                   detected_frac_at_thr=float(len(lat) / len(fr)))
        lat_s = []
        for i in fr[strong]:
            hit = np.where(P[i] >= thr_val)[0]
            if len(hit):
                lat_s.append((hit[0] - D["onset"][i] // tp) * tp)
        res["latency_bins_median_strongtrack"] = float(np.median(lat_s)) if lat_s else np.nan
    return res, P


def pick_threshold(model, Dv, target_fpr=0.01):
    """Lowest threshold whose VAL patch-level false-alarm rate (max_t P over noise patches) <= target."""
    s = model.predict(Dv["X"], batch_size=256, verbose=0)[..., 0].max(1)
    noise = np.sort(s[Dv["y"] == 0])
    k = int(np.floor(target_fpr * len(noise)))
    thr = float(noise[len(noise) - 1 - k]) + 1e-6 if len(noise) else 0.5
    return min(thr, 0.999999)


def stream_check(model, cfg, D, n=4):
    """Feed one time bin at a time through a stateful twin with carried state; compare to batch output."""
    F = D["X"].shape[2]
    sm = build(cfg, 1, F, stateful=True)
    sm.set_weights(model.get_weights())
    rnn = sm.get_layer("rnn")
    Pb = model.predict(D["X"][:n], verbose=0)[..., 0]
    md = 0.0
    for i in range(n):
        rnn.reset_state()
        ps = np.array([float(sm(D["X"][i:i + 1, t:t + 1], training=False).numpy()[0, 0, 0]) for t in range(D["X"].shape[1])])
        md = max(md, float(np.abs(ps - Pb[i]).max()))
    assert md < 1e-5, f"streaming != batch: max |diff| = {md}"
    return md


def run(cfg, threads=4):
    import tensorflow as tf
    from tensorflow import keras
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    cfg = dict(DEF, **cfg)
    D = data(cfg["t_pool"])
    tr, va, te = D["train"], D["val"], D["test"]
    T, F = tr["X"].shape[1:]
    Ytr, Yva = step_labels(tr, cfg["t_pool"], cfg["label"]), step_labels(va, cfg["t_pool"], cfg["label"])
    fp = Ytr.mean()
    wpos, wneg = 0.5 / fp, 0.5 / (1 - fp)
    Wtr = np.where(Ytr == 1, wpos, wneg).astype(np.float32)
    Wva = np.where(Yva == 1, wpos, wneg).astype(np.float32)
    attempts, t0 = [], time.time()
    from sklearn.metrics import roc_auc_score
    for a in range(3):
        seed = cfg["seed"] + 1000 * a
        keras.utils.set_random_seed(seed)
        model = build(cfg, T, F)
        model.compile(keras.optimizers.Adam(cfg["lr"]), "binary_crossentropy")
        bv = BestVal()
        model.fit(tr["X"], Ytr[..., None], sample_weight=Wtr, batch_size=cfg["batch"], epochs=cfg["epochs"],
                  validation_data=(va["X"], Yva[..., None], Wva), verbose=0, shuffle=True,
                  callbacks=[bv, keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5,
                                                                   patience=cfg["lr_patience"], min_lr=1e-6)])
        Pv = model.predict(va["X"], batch_size=256, verbose=0)[..., 0]
        auc_v = float(roc_auc_score(Yva.ravel(), Pv.ravel()))
        attempts.append(dict(seed=seed, best_epoch=bv.ep, val_loss=float(bv.best), val_step_auc=auc_v,
                             val_pred_std=float(Pv.std())))
        if Pv.std() >= 1e-3 and auc_v >= 0.6:
            break
    thr = pick_threshold(model, va)
    rv, _ = evaluate(model, cfg, va, thr)
    rt, Pt = evaluate(model, cfg, te, thr)
    # memorisation probe: v6 test noise + N(0, 0.05) per pixel, patch-level noise accuracy
    Xn = te["X"][te["y"] == 0] + np.random.default_rng(0).normal(0, 0.05, te["X"][te["y"] == 0].shape).astype(np.float32)
    sn = model.predict(Xn, batch_size=256, verbose=0)[..., 0].max(1)
    md = stream_check(model, cfg, te)
    os.makedirs(MODELS, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
    mp = os.path.join(MODELS, cfg["run_id"] + ".keras")
    model.save(mp)
    r = dict(run_id=cfg["run_id"], cfg=cfg, attempts=attempts, n_attempts=len(attempts),
             collapsed_all=attempts[-1]["val_pred_std"] < 1e-3 or attempts[-1]["val_step_auc"] < 0.6,
             params=int(model.count_params()), macs_per_step=macs_per_step(cfg, F), steps=int(T),
             macs_per_patch=macs_per_step(cfg, F) * int(T), threshold_val_fa1pct=thr,
             stream_max_abs_diff=md, train_seconds=round(time.time() - t0, 1),
             n_bad_excluded=dict(train=tr["n_bad"], val=va["n_bad"], test=te["n_bad"]),
             n_test=int(len(te["y"])), lookup_baseline_acc=LOOKUP, patch_score="max over t of P_t",
             probe_noise_acc_0p05_at_0p5=float((sn < 0.5).mean()), probe_noise_acc_0p05_at_thr=float((sn < thr).mean()),
             model_path=os.path.relpath(mp, ROOT))
    r.update({f"val_{k}": v for k, v in rv.items()})
    r.update({f"test_{k}": v for k, v in rt.items()})
    np.save(os.path.join(MODELS, cfg["run_id"] + "_test_P.npy"), Pt.astype(np.float32))
    json.dump(r, open(os.path.join(RUNS, cfg["run_id"] + ".json"), "w"), indent=1, default=float)
    return r


def aggregate():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(f))
        rows.append({k: v for k, v in r.items() if k not in ("cfg", "attempts", "n_bad_excluded")}
                    | {k: r["cfg"].get(k) for k in ("cell", "units", "front", "head", "t_pool", "label", "seed")})
    df = pd.DataFrame(rows)
    df.insert(0, "table", LABEL)
    df.to_csv(CSV, index=False)
    return df


def _one(args):
    cfg, th = args
    try:
        r = run(cfg, th)
        return (f"{r['run_id']} att={r['n_attempts']} params={r['params']} macs/step={r['macs_per_step']} "
                f"val_patch_acc={r['val_patch_acc']:.4f} test_patch_acc={r['test_patch_acc']:.4f} "
                f"step_auc={r['test_step_auc']:.4f} lat_med={r['test_latency_bins_median']} stream_diff={r['stream_max_abs_diff']:.1e}")
    except Exception as e:
        import traceback
        return f"{cfg['run_id']} ERROR {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cfgs = json.load(open(sys.argv[1]))
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    th = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    done = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(RUNS, "*.json"))}
    todo = [c for c in cfgs if c["run_id"] not in done]
    print(f"{len(todo)} to run", flush=True)
    with ProcessPoolExecutor(w) as ex:
        for f in as_completed([ex.submit(_one, (c, th)) for c in todo]):
            print(f.result(), flush=True)
    aggregate()
