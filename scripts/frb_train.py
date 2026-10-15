"""Shared training / evaluation harness for the v7 sweeps (CNN + RNN).

Every run writes one JSON to results/runs/<run_id>.json (process-safe);
scripts/aggregate_results.py turns them into results/sweep_results.csv.

Evaluation protocol (fixed before any sweep, see results/NOTES.md):
  * split by source file (results/splits/v7_splits.json)
  * primary metric set = all noise samples + FRB samples with optimal S/N >= 7
  * also reported: accuracy on the full S/N range (3-50) and FRB recall per S/N bin
"""
import hashlib
import json
import os
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "augmented_dataset", "v7")
RUNS = os.path.join(ROOT, "results", "runs")
MODELS = os.path.join(ROOT, "results", "models")
EVAL_SNR_MIN = 7.0
SNR_BINS = [3, 5, 7, 10, 15, 25, 50]

_cache = {}


def set_threads(n):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(2)


class F32View:
    """Read-only view of a float16 memmap that returns float32 on indexing, so
    parallel workers share one OS-cached copy instead of each holding float32."""

    def __init__(self, mm):
        self.mm, self.shape = mm, mm.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        return np.asarray(self.mm[i], np.float32)


DATASETS = {"v7": DATA, "v6": os.path.join(ROOT, "augmented_dataset", "v6bench")}


def load_data(t_pool=1, f_pool=1, pool=None, dataset="v7"):
    """X pooled by (t_pool, f_pool) block-means, cached on disk as float16 and
    memory-mapped; indexing returns float32 (N, T, F). pool=None -> the same-file
    v7 set; pool='mix_primary' etc. -> a cross-file training pool (v7_build_mixed.py)."""
    key = (t_pool, f_pool, pool, dataset)
    if key in _cache:
        return _cache[key]
    base = DATASETS[dataset] if pool is None else os.path.join(DATASETS[dataset], pool)
    meta = pd.read_csv(os.path.join(base, "meta.csv"))
    path = os.path.join(base, f"X_t{t_pool}_f{f_pool}.npy")
    src = os.path.join(base, "X.npy")
    if (t_pool, f_pool) == (1, 1):
        path = src
    elif not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(src):
        X = np.load(src, mmap_mode="r")
        N, T, F = X.shape
        tmp = path + f".tmp{os.getpid()}.npy"
        out = np.lib.format.open_memmap(tmp, "w+", np.float16, (N, T // t_pool, F // f_pool))
        for s in range(0, N, 1000):
            b = np.asarray(X[s:s + 1000], np.float32)
            out[s:s + 1000] = b.reshape(len(b), T // t_pool, t_pool, F // f_pool, f_pool).mean((2, 4))
        out.flush(); del out
        try:
            os.replace(tmp, path)
        except OSError:  # another worker won the race
            os.remove(tmp)
    out = F32View(np.load(path, mmap_mode="r"))
    _cache[key] = (out, meta)
    return out, meta


def split_indices(meta, split="primary", fold=None, train_snr_min=3.0):
    if split == "v6orig":  # original v6 per-sample split, used as-is (no re-splitting)
        w = lambda s: np.where(meta.split == s)[0]  # noqa: E731
        return dict(train=w("train"), val=w("val"), test=w("test"), test_all=w("test"), val_all=w("val"),
                    test_dedup=np.where((meta.split == "test") & ~meta.dup_in_train)[0])
    sp = json.load(open(os.path.join(ROOT, "results", "splits", "v7_splits.json")))
    s = sp[split] if fold is None else sp[split][fold]
    ev = (meta.label == 0) | (meta.snr >= EVAL_SNR_MIN)
    tr = meta.group_id.isin(s["train"]) & ((meta.label == 0) | (meta.snr >= train_snr_min))
    va = meta.group_id.isin(s["val"]) & ev
    te_all = meta.group_id.isin(s["test"])
    return dict(train=np.where(tr)[0], val=np.where(va)[0], test=np.where(te_all & ev)[0],
                test_all=np.where(te_all)[0], val_all=np.where(meta.group_id.isin(s["val"]))[0])


# ---------------------------------------------------------------- models

def build_cnn(cfg, input_shape):
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=input_shape + (1,), name="input_data")
    x = inp
    H, W = input_shape
    for i, f in enumerate(cfg["filters"]):
        x = layers.Conv2D(f, cfg["kernel"], padding="same", use_bias=not cfg.get("batchnorm"), name=f"conv_{i}")(x)
        if cfg.get("batchnorm"):
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        p = cfg["pools"][i]
        if p > 1:
            P = layers.AveragePooling2D if cfg.get("pool_type") == "avg" else layers.MaxPooling2D
            x = P((p, p), name=f"pool_{i}")(x)
    if cfg.get("head", "flatten") == "gap":
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Flatten(name="flatten")(x)
    if cfg.get("dense", 0):
        x = layers.Dense(cfg["dense"], name="dense_0")(x)
        x = layers.Activation("relu", name="relu_d")(x)
        if cfg.get("dropout", 0) > 0:
            x = layers.Dropout(cfg["dropout"], name="drop_0")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name="frb_cnn")


def build_rnn(cfg, input_shape):
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=input_shape, name="input_data")
    cell = layers.LSTM if cfg["cell"] == "lstm" else layers.GRU
    kw = dict(name="rnn_0", return_sequences=False)
    if cfg["cell"] == "gru":
        kw["reset_after"] = cfg.get("reset_after", True)  # hls4ml supports both; Keras default
    x = cell(cfg["units"], **kw)(inp)
    if cfg.get("dropout", 0) > 0:
        x = layers.Dropout(cfg["dropout"], name="drop_0")(x)
    if cfg.get("dense", 0):
        x = layers.Dense(cfg["dense"], activation="relu", name="dense_0")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name="frb_rnn")


def count_macs(model):
    """Multiply-accumulates per inference (conv, dense, LSTM/GRU)."""
    macs = 0
    for l in model.layers:
        cls = l.__class__.__name__
        if cls.startswith("QConv2D") or cls == "Conv2D":
            _, h, w, c = l.output.shape
            kh, kw = l.kernel_size
            macs += h * w * c * kh * kw * l.input.shape[-1]
        elif cls in ("Dense", "QDense"):
            macs += l.input.shape[-1] * l.units
        elif cls in ("LSTM", "GRU", "QLSTM", "QGRU"):
            T, d = l.input.shape[1], l.input.shape[2]
            u = l.units
            g = 4 if "LSTM" in cls else 3
            macs += T * g * u * (d + u)
    return int(macs)


# ---------------------------------------------------------------- metrics

def metrics(y, p, thr=0.5):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
    yhat = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    rec_at_1 = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tp = int(((yhat == 1) & (y == 1)).sum())
    return dict(acc=float((yhat == y).mean()),
                bal_acc=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
                precision=float(precision_score(y, yhat, zero_division=0)),
                recall=float(recall_score(y, yhat, zero_division=0)),
                f1=float(f1_score(y, yhat, zero_division=0)),
                auc=float(roc_auc_score(y, p)), recall_at_1pct_fpr=rec_at_1,
                tn=tn, fp=fp, fn=fn, tp=tp)


def recall_by_snr(meta_sub, p):
    out = {}
    frb = meta_sub.label.values == 1
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
        m = frb & (meta_sub.snr.values >= lo) & (meta_sub.snr.values < hi)
        out[f"recall_snr_{lo}_{hi}"] = float((p[m] >= 0.5).mean()) if m.any() else None
    return out


def predict(model, X, bs=512):
    return model.predict(X, batch_size=bs, verbose=0).ravel()


# ---------------------------------------------------------------- training

def prep_input(X, kind, clip=None):
    if clip:  # RFI clipping in units of per-channel noise sigma (done upstream of the FPGA model)
        X = np.clip(X, clip[0], clip[1])
    return X[..., None] if kind == "cnn" else X


class BatchSeq:
    """Index-based batch feeder (no copy of the train subset); optional random
    per-sample time reversal (augmentation) instead of doubling the data."""

    def __new__(cls, X, y, idx, batch, kind, flip, seed, clip=None):
        from tensorflow import keras

        Base = getattr(keras.utils, "PyDataset", None) or keras.utils.Sequence

        class _Seq(Base):
            def __init__(self):
                super().__init__()
                self.rng = np.random.default_rng(seed)
                self.order = self.rng.permutation(idx)

            def __len__(self):
                return int(np.ceil(len(idx) / batch))

            def __getitem__(self, i):
                b = np.sort(self.order[i * batch:(i + 1) * batch])
                xb = X[b]
                if flip:
                    m = self.rng.random(len(b)) < 0.5
                    xb[m] = xb[m][:, ::-1]
                return prep_input(xb, kind, clip), y[b]

            def on_epoch_end(self):
                self.order = self.rng.permutation(idx)

        return _Seq()


def train_eval(cfg, run_id=None, save_model=True, threads=4, extra_eval=True):
    """cfg keys: kind ('cnn'|'rnn'), t_pool, f_pool, arch keys, lr, batch, epochs, patience,
    dropout, class_weight, train_snr_min, time_flip, seed, split, fold, track, stage."""
    import tensorflow as tf
    from tensorflow import keras
    set_threads(threads)
    seed = cfg.get("seed", 0)
    keras.utils.set_random_seed(seed)
    X, meta = load_data(cfg["t_pool"], cfg["f_pool"], dataset=cfg.get("dataset", "v7"))
    idx = split_indices(meta, cfg.get("split", "primary"), cfg.get("fold"), cfg.get("train_snr_min", 3.0))
    y = meta.label.values.astype(np.float32)
    shape = X.shape[1:]
    model = build_cnn(cfg, shape) if cfg["kind"] == "cnn" else build_rnn(cfg, shape)
    params = int(model.count_params())
    macs = count_macs(model)
    if cfg.get("max_params") and params > cfg["max_params"]:
        return dict(run_id=run_id, status="skipped_params", params=params)

    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="binary_crossentropy", metrics=["accuracy"])
    tr = idx["train"]
    Xtr_src, ytr_src = X, y
    if cfg.get("train_pool"):  # cross-file mixed pool built from the TRAIN files of this split only
        Xtr_src, mtr = load_data(cfg["t_pool"], cfg["f_pool"], cfg["train_pool"])
        ytr_src = mtr.label.values.astype(np.float32)
        tr = np.where((mtr.label == 0) | (mtr.snr >= cfg.get("train_snr_min", 3.0)))[0]
    cw = None
    if cfg.get("class_weight"):
        n1 = ytr_src[tr].sum(); n0 = len(tr) - n1
        cw = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
    clip = cfg.get("clip")
    train_seq = BatchSeq(Xtr_src, ytr_src, tr, cfg["batch"], cfg["kind"], cfg.get("time_flip", False), seed, clip)
    Xva, yva = prep_input(X[idx["val"]], cfg["kind"], clip), y[idx["val"]]
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.get("patience", 8),
                                        restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, cfg.get("patience", 8) // 2))]
    t0 = time.time()
    h = model.fit(train_seq, validation_data=(Xva, yva), epochs=cfg.get("epochs", 40),
                  class_weight=cw, callbacks=cb, verbose=0)
    train_s = time.time() - t0
    run_id = run_id or hashlib.sha1((json.dumps(cfg, sort_keys=True) + str(time.time())).encode()).hexdigest()[:10]

    res = dict(run_id=run_id, status="ok", params=params, macs=macs, flops=2 * macs, epochs_run=len(h.history["loss"]),
               best_epoch=int(np.argmin(h.history["val_loss"])) + 1, train_seconds=round(train_s, 1),
               val_loss=float(np.min(h.history["val_loss"])), cfg=cfg, input_shape=list(shape),
               history={k: [round(float(v), 5) for v in vs] for k, vs in h.history.items() if k != "learning_rate"})
    pv = predict(model, Xva)
    res.update({f"val_{k}": v for k, v in metrics(yva, pv).items()})
    Xte = prep_input(X[idx["test"]], cfg["kind"], clip)
    pt = predict(model, Xte)
    res.update({f"test_{k}": v for k, v in metrics(y[idx["test"]], pt).items()})
    if extra_eval:
        pa = predict(model, prep_input(X[idx["test_all"]], cfg["kind"], clip))
        res["test_allsnr_acc"] = float(((pa >= 0.5) == y[idx["test_all"]]).mean())
        res.update({f"test_{k}": v for k, v in recall_by_snr(meta.iloc[idx["test_all"]], pa).items()})
    if save_model:
        os.makedirs(MODELS, exist_ok=True)
        path = os.path.join(MODELS, f"{run_id}.keras")
        model.save(path)
        res["model_path"] = os.path.relpath(path, ROOT)
    os.makedirs(RUNS, exist_ok=True)
    json.dump(res, open(os.path.join(RUNS, f"{run_id}.json"), "w"), indent=1, default=float)
    return res
