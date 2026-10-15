"""
Architecture registry + the picklable worker function for the parallel ablation
grid. Must be an importable module (not defined inline in a notebook/script's
__main__) because ProcessPoolExecutor on Windows uses the 'spawn' start method,
which re-imports this module in each worker process.
"""
import hashlib
import json
import os
import time
from pathlib import Path


def build_pool_schedule(t_exp, f_exp=5, n_stages=5):
    """Per-stage (time_pool, freq_pool) tuples: stage i pools time by 2 while
    i < t_exp, and pools freq by 2 while i < f_exp; 1 (no-op) otherwise."""
    return [(2 if i < t_exp else 1, 2 if i < f_exp else 1) for i in range(n_stages)]


def config_hash(build_args, input_shape=(256, 256, 1)):
    """Stable hash of the architecture + preprocessing, used for cross-group
    dedup (e.g. Group 1's flatten=256 baseline IS Group 3's r=1)."""
    payload = {
        "filters": list(build_args["filters"]),
        "pool_schedule": [list(p) for p in build_args["pool_schedule"]],
        "time_avgpool": build_args.get("time_avgpool"),
        "input_shape": list(input_shape),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


BASELINE_ARGS = {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None}

# canonical_name -> (group, aliases-from-the-old-per-group-naming, build_args)
# "baseline" is the production architecture: uniform 5x(2,2) pooling, filters
# [2,4,4,4,4], flatten=256, no time coarsening. It is Group 1's f4, Group 2's
# a_uniform, and Group 3's r1 -- literally the same architecture, trained once.
ARCHITECTURES = {
    "baseline": {
        "groups": ["pipe_size", "staged_pooling", "time_resolution"],
        "aliases": {"pipe_size": "f4", "staged_pooling": "a_uniform", "time_resolution": "r1"},
        "build_args": BASELINE_ARGS,
    },
    "pipe_f1": {
        "groups": ["pipe_size"], "aliases": {"pipe_size": "f1"},
        "build_args": {"filters": [2, 4, 4, 4, 1], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    },
    "pipe_f2": {
        "groups": ["pipe_size"], "aliases": {"pipe_size": "f2"},
        "build_args": {"filters": [2, 4, 4, 4, 2], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    },
    "pipe_f8": {
        "groups": ["pipe_size"], "aliases": {"pipe_size": "f8"},
        "build_args": {"filters": [2, 4, 4, 4, 8], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    },
    "pipe_f16": {
        "groups": ["pipe_size"], "aliases": {"pipe_size": "f16"},
        "build_args": {"filters": [2, 4, 4, 4, 16], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    },
    "pipe_f32": {
        "groups": ["pipe_size"], "aliases": {"pipe_size": "f32"},
        "build_args": {"filters": [2, 4, 4, 4, 32], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    },
    "pipe_f64": {
        "groups": ["pipe_size"], "aliases": {"pipe_size": "f64"},
        "build_args": {"filters": [2, 4, 4, 4, 64], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    },
    "pool_front_loaded": {
        "groups": ["staged_pooling"], "aliases": {"staged_pooling": "b_front_loaded"},
        "build_args": {"filters": [4, 8, 4], "pool_schedule": [(4, 4), (4, 4), (2, 2)], "time_avgpool": None},
    },
    "pool_back_loaded": {
        "groups": ["staged_pooling"], "aliases": {"staged_pooling": "c_back_loaded"},
        "build_args": {"filters": [2, 4, 8, 4], "pool_schedule": [(2, 2), (2, 2), (4, 4), (2, 2)], "time_avgpool": None},
    },
    "pool_freq_preserving": {
        "groups": ["staged_pooling"], "aliases": {"staged_pooling": "d_freq_preserving"},
        "build_args": {"filters": [4, 8, 1], "pool_schedule": [(4, 2), (4, 2), (2, 2)], "time_avgpool": None},
    },
    "pool_time_preserving": {
        "groups": ["staged_pooling"], "aliases": {"staged_pooling": "e_time_preserving"},
        "build_args": {"filters": [4, 8, 1], "pool_schedule": [(2, 4), (2, 4), (2, 2)], "time_avgpool": None},
    },
    "timeres_r2": {
        "groups": ["time_resolution"], "aliases": {"time_resolution": "r2"},
        "build_args": {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(4, 5), "time_avgpool": 2},
    },
    "timeres_r4": {
        "groups": ["time_resolution"], "aliases": {"time_resolution": "r4"},
        "build_args": {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(3, 5), "time_avgpool": 4},
    },
    "timeres_r8": {
        "groups": ["time_resolution"], "aliases": {"time_resolution": "r8"},
        "build_args": {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(2, 5), "time_avgpool": 8},
    },
    "timeres_r16": {
        "groups": ["time_resolution"], "aliases": {"time_resolution": "r16"},
        "build_args": {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(1, 5), "time_avgpool": 16},
    },
    "timeres_r32": {
        "groups": ["time_resolution"], "aliases": {"time_resolution": "r32"},
        "build_args": {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(0, 5), "time_avgpool": 32},
    },
}

for _name, _spec in ARCHITECTURES.items():
    _spec["config_hash"] = config_hash(_spec["build_args"])

EPOCHS_CAP = 300
PATIENCE = 30
LR_PATIENCE = 10
COLLAPSE_VAL_ACC = 0.65
COLLAPSE_MAX_ATTEMPTS = 3


def train_one_seed(job):
    """Picklable top-level worker function. job is a plain dict (must be
    picklable for ProcessPoolExecutor). Returns a plain dict matching the
    Requirement-2 JSONL schema. Runs entirely inside a fresh worker process --
    imports TF locally so nothing is shared across processes."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import Model, callbacks, layers, optimizers
    from sklearn.metrics import auc, confusion_matrix, roc_curve

    config_name = job["config_name"]
    seed = job["seed"]
    args = job["build_args"]

    data = np.load(job["data_path"])
    X_train, y_train = data["X_train"][..., np.newaxis], data["y_train"]
    X_val, y_val = data["X_val"][..., np.newaxis], data["y_val"]
    X_test, y_test = data["X_test"][..., np.newaxis], data["y_test"]
    n_pos, n_neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    def build_model():
        inp = layers.Input(shape=(256, 256, 1), name="input_data")
        x = inp
        if args.get("time_avgpool") and args["time_avgpool"] > 1:
            x = layers.AveragePooling2D(pool_size=(args["time_avgpool"], 1), name="time_coarsen")(x)
        for i, (f, (pt, pf)) in enumerate(zip(args["filters"], args["pool_schedule"])):
            x = layers.Conv2D(f, (3, 3), activation="relu", padding="same", name=f"conv_{i}")(x)
            x = layers.MaxPooling2D((pt, pf), name=f"pool_{i}")(x)
        x = layers.Flatten(name="flatten")(x)
        flatten_dim = int(np.prod(x.shape[1:]))
        x = layers.Dense(32, activation="relu", name="dense_0")(x)
        x = layers.Dropout(0.3, name="drop_0")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        return Model(inp, out), flatten_dim

    model_path = (Path(job["results_dir"]) / "models" / f"{config_name}_s{seed}.keras").as_posix()
    collapsed_retries = 0
    attempt = 1
    while True:
        effective_seed = seed if attempt == 1 else seed + 1000 * (attempt - 1)
        tf.random.set_seed(effective_seed)
        model, flatten_dim = build_model()
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss="binary_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=job["epochs"], batch_size=128, class_weight=class_weight,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=job["patience"],
                                         restore_best_weights=True, verbose=0),
                callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE,
                                             min_lr=1e-6, verbose=0),
            ],
            verbose=0,
        )
        train_time = time.time() - t0
        best_epoch = int(np.argmin(history.history["val_loss"]))
        val_acc = float(history.history["val_accuracy"][best_epoch])
        val_loss = float(history.history["val_loss"][best_epoch])
        if val_acc >= COLLAPSE_VAL_ACC or attempt >= COLLAPSE_MAX_ATTEMPTS:
            break
        collapsed_retries += 1
        attempt += 1

    y_pred_prob = model.predict(X_test, batch_size=128, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    test_auc = float(auc(fpr, tpr))

    return {
        "config_name": config_name,
        "config_hash": job["config_hash"],
        "group": job["group"],
        "seed": seed,
        "flatten_dim": flatten_dim,
        "n_params": int(model.count_params()),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "test_acc": float(test_acc),
        "test_auc": test_auc,
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "best_epoch": best_epoch + 1,
        "n_epochs_ran": len(history.history["loss"]),
        "train_time_sec": train_time,
        "model_path": model_path,
        "collapsed_retries": collapsed_retries,
    }
