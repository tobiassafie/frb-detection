"""v6a: targeted experiments on the ORIGINAL v6 split (lstm_dataset_v6.npz as on disk; float32,
no re-splitting). One run = one JSON in results/runs_v6a/ + one row in results/v6a_runs.csv.

Every run also records the mandatory context columns:
  lookup_baseline_acc  : no-learning exact-match noise lookup on the same test set (0.984)
  dedup_recall         : recall on the 149 test FRBs whose pixels do not appear in train
                         (dedup removes all 154 noise samples -> recall only)
  v7test_acc           : the same trained model on the v7 primary test set (v7 patch z-scored
                         per patch = v6 convention, then pooled like the model input)
  v6test_timereversed_acc : v6 test with the time axis reversed (opposite sweep direction)
Recipe defaults = the old v6 recipe (scripts/ablation_arch.py / cnn_training*.ipynb): Adam 1e-3,
batch 128, class weights, EarlyStopping(val_loss, patience 30, restore best),
ReduceLROnPlateau(0.5, patience 10, min 1e-6), up to 150 epochs, and a COLLAPSE RETRY: if best
val acc < 0.65 the run is restarted with seed + 1000*attempt (up to 3 attempts); every attempt
is logged (n_attempts), never dropped silently.
Usage: python scripts/v6a_train.py cfgs.json [workers] [threads]
"""
import hashlib
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
sys.path.insert(0, os.path.dirname(__file__))
TAG = os.environ.get("V6A_TAG", "v6a")  # e.g. V6A_TAG=v6c -> results/runs_v6c, models_v6c, v6c_lstm_runs.csv
RUNS = os.path.join(ROOT, "results", f"runs_{TAG}")
MODELS = os.path.join(ROOT, "results", f"models_{TAG}")
CSV = os.path.join(ROOT, "results", {"v6a": "v6a_runs.csv", "v6c": "v6c_lstm_runs.csv"}.get(TAG, f"{TAG}_runs.csv"))
LOOKUP_ACC = 0.9837662337662337
DEFAULTS = dict(lr=1e-3, batch=128, epochs=150, patience=30, lr_patience=10, class_weight=True, seed=42,
                t_pool=1, f_pool=1, aug_shift=0, aug_noise=0.0, l2=0.0, oversample=1)
_D = {}


def data():
    if "v6" not in _D:
        d = np.load(os.path.join(ROOT, "augmented_dataset", "lstm_dataset_v6.npz"))
        _D["v6"] = {k: d[k] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}
        h = lambda x: hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()  # noqa: E731
        tr = {h(x) for x in _D["v6"]["X_train"]}
        _D["v6"]["dedup"] = np.array([h(x) not in tr for x in _D["v6"]["X_test"]])
    return _D["v6"]


def pool(X, tp, fp, mode="avg"):
    """Upstream block pooling: mean over (tp x fp) blocks; mode='max' takes the max over the tp
    time bins (frequency still block-averaged)."""
    if tp == 1 and fp == 1:
        return X
    n, T, F = X.shape
    Y = X.reshape(n, T // tp, tp, F // fp, fp).mean(4)
    return Y.max(2) if mode == "max" else Y.mean(2)


def v7_test(tp, fp, mode="avg"):
    """v7 primary test (noise + FRB S/N>=7), per-patch z-scored at full res, then pooled."""
    key = ("v7", tp, fp, mode)
    if key not in _D:
        import frb_train as ft
        X, meta = ft.load_data(1, 1)
        idx = ft.split_indices(meta, "primary")["test"]
        out = []
        for s in range(0, len(idx), 400):
            x = np.asarray(X[idx[s:s + 400]], np.float32)
            m, sd = x.mean((1, 2), keepdims=True), x.std((1, 2), keepdims=True)
            out.append(pool((x - m) / np.where(sd > 0, sd, 1), tp, fp, mode))
        _D[key] = (np.concatenate(out), meta.label.values[idx].astype(float))
    return _D[key]


# ------------------------------------------------------------------ models
def build(cfg, shape):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    reg = regularizers.l2(cfg["l2"]) if cfg.get("l2") else None
    if cfg["kind"] == "cnn":
        inp = keras.Input(shape=shape + (1,), name="input_data")
        x = inp
        for i, f in enumerate(cfg["filters"]):
            x = layers.Conv2D(f, cfg.get("kernel", 3), padding="same", activation="relu", kernel_regularizer=reg,
                              name=f"conv_{i}")(x)
            P = layers.AveragePooling2D if cfg.get("pool_type", "max") == "avg" else layers.MaxPooling2D
            x = P(tuple(cfg["pools"][i]) if isinstance(cfg["pools"][i], list) else (cfg["pools"][i],) * 2,
                  name=f"pool_{i}")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x) if cfg.get("head", "flatten") == "gap" else \
            layers.Flatten(name="flatten")(x)
        if cfg.get("dense"):
            x = layers.Dense(cfg["dense"], activation="relu", kernel_regularizer=reg, name="dense_0")(x)
            if cfg.get("dropout", 0) > 0:
                x = layers.Dropout(cfg["dropout"], name="drop_0")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        return keras.Model(inp, out)
    # many-to-one RNN: input (timesteps, freq features)
    inp = keras.Input(shape=shape, name="input_data")
    x = inp
    for j, c in enumerate(cfg.get("conv1d", [])):  # Conv1D over TIME with freq as channels (H8 front end)
        x = layers.Conv1D(c["filters"], c.get("kernel", 3), padding="same", activation="relu", name=f"c1d_{j}")(x)
        if c.get("pool", 1) > 1:
            x = layers.MaxPooling1D(c["pool"], name=f"c1p_{j}")(x)
    if cfg.get("freq_reduction"):  # per-timestep dense reduction (lstm_training_2 style), 1x1 conv
        x = layers.Conv1D(cfg["freq_reduction"], 1, activation="relu", name="freq_reduce")(x)
    cell = layers.LSTM if cfg["cell"] == "lstm" else layers.GRU
    rnn = cell(cfg["units"], name="rnn_0")
    x = layers.Bidirectional(rnn, name="birnn_0")(x) if cfg.get("bidirectional") else rnn(x)
    if cfg.get("dense"):
        x = layers.Dense(cfg["dense"], activation="relu", name="dense_0")(x)
        if cfg.get("dropout", 0) > 0:
            x = layers.Dropout(cfg["dropout"], name="drop_0")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out)


def macs(model):
    import frb_train as ft
    m = ft.count_macs(model)
    for l in model.layers:  # Conv1D not covered by ft.count_macs
        if l.__class__.__name__ == "Conv1D":
            T, c = l.output.shape[1], l.output.shape[2]
            m += T * c * l.kernel_size[0] * l.input.shape[-1]
        if l.__class__.__name__ == "Bidirectional":
            T, d = l.input.shape[1], l.input.shape[2]
            u = l.forward_layer.units
            g = 4 if "LSTM" in l.forward_layer.__class__.__name__ else 3
            m += 2 * T * g * u * (d + u)
    return int(m)


class Seq:
    """Batches with optional oversampling (more steps per epoch, same LR schedule) and light
    augmentation: random circular time shift (|s| <= aug_shift bins) + Gaussian noise."""

    def __new__(cls, X, y, cfg, seed):
        from tensorflow import keras

        class S(keras.utils.PyDataset):
            def __init__(self):
                super().__init__()
                self.rng = np.random.default_rng(seed)
                self.idx = np.tile(np.arange(len(X)), cfg["oversample"])
                self.order = self.rng.permutation(self.idx)

            def __len__(self):
                return int(np.ceil(len(self.idx) / cfg["batch"]))

            def __getitem__(self, i):
                b = self.order[i * cfg["batch"]:(i + 1) * cfg["batch"]]
                xb = X[b].copy()
                if cfg["aug_shift"]:
                    for k, s in enumerate(self.rng.integers(-cfg["aug_shift"], cfg["aug_shift"] + 1, len(b))):
                        xb[k] = np.roll(xb[k], s, axis=0)
                if cfg["aug_noise"]:
                    xb = xb + self.rng.normal(0, cfg["aug_noise"], xb.shape).astype(np.float32)
                return (xb[..., None] if cfg["kind"] == "cnn" else xb), y[b]

            def on_epoch_end(self):
                self.order = self.rng.permutation(self.idx)

        return S()


def run(cfg, threads=4):
    import tensorflow as tf
    from tensorflow import keras
    sys.path.insert(0, os.path.dirname(__file__))
    import frb_train as ft
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    cfg = dict(DEFAULTS, **cfg)
    D = data()
    tp, fp = cfg["t_pool"], cfg["f_pool"]
    pm = cfg.get("t_pool_type", "avg")
    Xtr, Xva, Xte = (pool(D[k], tp, fp, pm) for k in ("X_train", "X_val", "X_test"))
    ytr, yva, yte = D["y_train"], D["y_val"], D["y_test"]
    prep = (lambda a: a[..., None]) if cfg["kind"] == "cnn" else (lambda a: a)
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())} if cfg["class_weight"] else None
    attempts, t0 = [], time.time()
    for a in range(3):
        seed = cfg["seed"] + 1000 * a
        keras.utils.set_random_seed(seed)
        model = build(cfg, Xtr.shape[1:])
        model.compile(keras.optimizers.Adam(cfg["lr"]), "binary_crossentropy", metrics=["accuracy"])
        h = model.fit(Seq(Xtr, ytr, cfg, seed), validation_data=(prep(Xva), yva), epochs=cfg["epochs"],
                      class_weight=cw, verbose=0, callbacks=[
                          keras.callbacks.EarlyStopping("val_loss", patience=cfg["patience"], restore_best_weights=True),
                          keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=cfg["lr_patience"],
                                                            min_lr=1e-6)])
        bv = int(np.argmin(h.history["val_loss"]))
        pstd = float(np.std(ft.predict(model, prep(Xva))))  # constant output = never trained
        attempts.append(dict(seed=seed, best_val_acc=float(h.history["val_accuracy"][bv]), epochs=len(h.history["loss"]),
                             val_pred_std=pstd))
        if h.history["val_accuracy"][bv] >= 0.65 and pstd >= 1e-3:
            break
    p = ft.predict(model, prep(Xte))
    r = dict(run_id=cfg["run_id"], stage=cfg.get("stage"), hypothesis=cfg.get("hypothesis", ""), kind=cfg["kind"],
             params=int(model.count_params()), macs=macs(model), input_shape=list(Xtr.shape[1:]),
             n_attempts=len(attempts), attempts=attempts,
             collapsed_all=attempts[-1]["best_val_acc"] < 0.65 or attempts[-1]["val_pred_std"] < 1e-3,
             epochs_run=attempts[-1]["epochs"], steps_per_epoch=int(np.ceil(len(Xtr) * cfg["oversample"] / cfg["batch"])),
             train_seconds=round(time.time() - t0, 1), cfg=cfg)
    r.update({f"val_{k}": v for k, v in ft.metrics(yva, ft.predict(model, prep(Xva))).items()})
    r.update({f"test_{k}": v for k, v in ft.metrics(yte, p).items()})
    r["lookup_baseline_acc"] = LOOKUP_ACC
    r["dedup_recall"] = float((p[D["dedup"]] >= 0.5).mean())
    r["dedup_n"] = int(D["dedup"].sum())
    # memorisation probe: v6 test noise images + N(0, 0.05) per pixel (v6 pixels are z-scored, so
    # 0.05 sigma is visually invisible). A detector should not care; a lookup-like model flips.
    pn = ft.predict(model, prep(Xte + np.random.default_rng(0).normal(0, 0.05, Xte.shape).astype(np.float32)))
    r["noise_acc_perturbed_0p05"] = float((pn[yte == 0] < 0.5).mean())
    r["frb_recall_perturbed_0p05"] = float((pn[yte == 1] >= 0.5).mean())
    pr = ft.predict(model, prep(Xte[:, ::-1]))
    r["v6test_timereversed_acc"] = float(((pr >= 0.5) == yte).mean())
    X7, y7 = v7_test(tp, fp, pm)
    p7 = ft.predict(model, prep(X7))
    r["v7test_acc"] = float(((p7 >= 0.5) == y7).mean())
    from sklearn.metrics import roc_auc_score
    r["v7test_auc"] = float(roc_auc_score(y7, p7))
    os.makedirs(MODELS, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
    mp = os.path.join(MODELS, cfg["run_id"] + ".keras")
    model.save(mp)
    r["model_path"] = os.path.relpath(mp, ROOT)
    json.dump(r, open(os.path.join(RUNS, cfg["run_id"] + ".json"), "w"), indent=1, default=float)
    return r


def aggregate():
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(f))
        rows.append({k: v for k, v in r.items() if k not in ("cfg", "attempts")} | {"cfg_json": json.dumps(r["cfg"])})
    df = pd.DataFrame(rows)
    df.insert(0, "table", "v6 benchmark - leaky split, comparison only")
    df.to_csv(CSV, index=False)
    return df


def _one(args):
    cfg, threads = args
    try:
        r = run(cfg, threads)
        return f"{r['run_id']} attempts={r['n_attempts']} params={r['params']} val={r['val_acc']:.4f} test={r['test_acc']:.4f} " \
               f"auc={r['test_auc']:.4f} dedup={r['dedup_recall']:.3f} rev={r['v6test_timereversed_acc']:.3f} v7={r['v7test_acc']:.3f}"
    except Exception as e:
        import traceback
        return f"{cfg['run_id']} ERROR {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"


def reeval(run_id):
    """Recompute the context/probe columns for an existing run from its saved model."""
    import keras
    import frb_train as ft
    f = os.path.join(RUNS, run_id + ".json")
    r = json.load(open(f))
    cfg = r["cfg"]
    m = keras.models.load_model(os.path.join(ROOT, r["model_path"]), compile=False)
    D = data()
    Xte = pool(D["X_test"], cfg.get("t_pool", 1), cfg.get("f_pool", 1), cfg.get("t_pool_type", "avg"))
    prep = (lambda a: a[..., None]) if cfg["kind"] == "cnn" else (lambda a: a)
    yte = D["y_test"]
    pn = ft.predict(m, prep(Xte + np.random.default_rng(0).normal(0, 0.05, Xte.shape).astype(np.float32)))
    r["noise_acc_perturbed_0p05"] = float((pn[yte == 0] < 0.5).mean())
    r["frb_recall_perturbed_0p05"] = float((pn[yte == 1] >= 0.5).mean())
    json.dump(r, open(f, "w"), indent=1, default=float)
    print(run_id, r["noise_acc_perturbed_0p05"], r["frb_recall_perturbed_0p05"])


if __name__ == "__main__" and sys.argv[1] == "--reeval":
    for rid in sys.argv[2:]:
        reeval(rid)
    aggregate()
    sys.exit(0)

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cfgs = json.load(open(sys.argv[1]))
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    th = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    done = {f[:-5] for f in os.listdir(RUNS)} if os.path.isdir(RUNS) else set()
    todo = [c for c in cfgs if c["run_id"] not in done]
    print(f"{len(todo)} to run", flush=True)
    with ProcessPoolExecutor(w) as ex:
        for f in as_completed([ex.submit(_one, (c, th)) for c in todo]):
            print(f.result(), flush=True)
    aggregate()
