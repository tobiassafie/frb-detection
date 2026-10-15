"""
Generates frb_ablation_study.ipynb (unexecuted) from the cell sources below.
Run, then execute the notebook separately with nbclient so real results get baked in.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""# FRB CNN Ablation Study — Groups 1-3

Three architecture ablations run against the production baseline (5-stage `(2,2)`
pooling, filters `[2,4,4,4,4]`, flatten dim 256) trained on `augmented_dataset/lstm_dataset_v6.npz`.

- **Group 1 — pipe-size sweep**: hold depth fixed at 5 stages of `(2,2)` pooling, sweep
  only the final stage's filter count over `{1,2,4,8,16,32,64}` -> flatten dim
  `{64,128,256,512,1024,2048,4096}`.
- **Group 2 — staged pooling ablation**: hold the bottleneck fixed at flatten=256,
  vary *how* you get there (uniform / front-loaded / back-loaded / asymmetric
  freq-preserving / asymmetric time-preserving).
- **Group 3 — time-resolution coarsening**: average-pool the 256-bin
  (0.655 ms/bin) time axis by factor `r in {1,2,4,8,16,32}`, with the CNN's own
  pooling schedule adjusted per `r` so the bottleneck stays fixed at flatten=256
  regardless of `r` (bottleneck width and time resolution must not be confounded).

**Axis convention**: input images are `(256, 256, 1)` = `(time, freq, channel)`. A
pool tuple `(pt, pf)` means `MaxPooling2D(pool_size=(pt, pf))`, i.e. `pt` pools the
time axis and `pf` pools the frequency axis.

The best config (by validation accuracy at the early-stopping checkpoint) from each
group is then converted to hls4ml/Vitis HLS projects at two post-training
quantization precisions (`ap_fixed<16,6>` and `ap_fixed<8,3>`) for FPGA synthesis on
EC2 (no Vitis toolchain is invoked here -- only project files are generated; the
bit-accurate quantized-accuracy check also needs a C/C++ compiler this machine
doesn't have, so it's deferred to EC2 too -- see the FPGA handoff section).
""")

code(r"""import os, json, time, math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, Model
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
""")

code(r"""DATA_PATH = "./augmented_dataset/lstm_dataset_v6.npz"
data = np.load(DATA_PATH)
X_train, y_train = data["X_train"][..., np.newaxis], data["y_train"]
X_val, y_val = data["X_val"][..., np.newaxis], data["y_val"]
X_test, y_test = data["X_test"][..., np.newaxis], data["y_test"]

print(f"Train: {X_train.shape} ({np.mean(y_train):.1%} FRB)")
print(f"Val:   {X_val.shape} ({np.mean(y_val):.1%} FRB)")
print(f"Test:  {X_test.shape} ({np.mean(y_test):.1%} FRB)")

n_pos, n_neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
CLASS_WEIGHT = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
print("Class weights:", CLASS_WEIGHT)
""")

md("""## Shared helpers

`build_pool_schedule` turns a required number of time/frequency halvings into a
per-stage `(pt, pf)` list. `build_cnn` stacks Conv2D/MaxPooling2D stages from a
`filters` list and matching `pool_schedule`, with an optional `AveragePooling2D`
time-coarsening layer at the input (Group 3). `train_and_eval` runs the same
training protocol as `train_cnn.py` (Adam 1e-3, batch 128, class-weighted,
early-stopping on val_loss) and reports test-set metrics.""")

code(r'''RESULTS_DIR = "ablation_results"
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)


def build_pool_schedule(t_exp, f_exp=5, n_stages=5):
    """Per-stage (time_pool, freq_pool) tuples: stage i pools time by 2 while
    i < t_exp, and pools freq by 2 while i < f_exp; 1 (no-op on that axis)
    otherwise."""
    return [(2 if i < t_exp else 1, 2 if i < f_exp else 1) for i in range(n_stages)]


def build_cnn(filters, pool_schedule, time_avgpool=None, dense_units=32, dropout=0.3,
              input_shape=(256, 256, 1), name="cnn"):
    assert len(filters) == len(pool_schedule)
    inp = layers.Input(shape=input_shape, name="input_data")
    x = inp
    if time_avgpool and time_avgpool > 1:
        x = layers.AveragePooling2D(pool_size=(time_avgpool, 1), name="time_coarsen")(x)
    for i, (f, (pt, pf)) in enumerate(zip(filters, pool_schedule)):
        x = layers.Conv2D(f, (3, 3), activation="relu", padding="same", name=f"conv_{i}")(x)
        x = layers.MaxPooling2D((pt, pf), name=f"pool_{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    flatten_dim = int(np.prod(x.shape[1:]))
    x = layers.Dense(dense_units, activation="relu", name="dense_0")(x)
    x = layers.Dropout(dropout, name="drop_0")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out, name=name)
    return model, flatten_dim


def train_and_eval(name, build_fn, epochs=150, batch_size=128, lr=1e-3,
                    early_stop_patience=30, lr_patience=10,
                    max_attempts=3, collapse_val_acc=0.65, base_seed=None):
    """build_fn() must return a *fresh* (model, flatten_dim) on every call.

    A handful of configs in this sweep hit a dead-ReLU-style collapse (val_acc
    stuck at chance, early-stopping after ~30 epochs) purely from unlucky weight
    initialization, not from the architecture itself -- confirmed by rerunning
    the exact same architecture with a different init seed and getting normal
    (0.92-0.96) accuracy. Detect that (val_acc well below chance+margin after
    training) and retry with a bumped seed rather than let an init fluke stand
    in as the result, since some of these configs get reused as baselines for
    other groups.

    base_seed (default: the module-level SEED) is also used for the seed
    robustness check later in this notebook, where the same architecture is
    deliberately retrained under a few different base seeds to see whether a
    close-margin "winner" survives resampling.
    """
    if base_seed is None:
        base_seed = SEED
    attempt = 1
    while True:
        tf.random.set_seed(base_seed + attempt - 1)
        model, flatten_dim = build_fn()
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                      loss="binary_crossentropy", metrics=["accuracy"])
        ckpt_path = os.path.join(RESULTS_DIR, "models", f"{name}.keras")
        t0 = time.time()
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, class_weight=CLASS_WEIGHT,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=early_stop_patience,
                                         restore_best_weights=True, verbose=0),
                callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience,
                                             min_lr=1e-6, verbose=0),
            ],
            verbose=0,
        )
        train_time = time.time() - t0
        best_epoch = int(np.argmin(history.history["val_loss"]))
        best_val_acc = float(history.history["val_accuracy"][best_epoch])
        best_val_loss = float(history.history["val_loss"][best_epoch])

        if best_val_acc >= collapse_val_acc or attempt >= max_attempts:
            if attempt > 1:
                tag = "recovered" if best_val_acc >= collapse_val_acc else "still collapsed, giving up"
                print(f"[{name:>20}] attempt {attempt}: {tag} (seed={SEED + attempt - 1})")
            break
        print(f"[{name:>20}] attempt {attempt} collapsed (val_acc={best_val_acc:.3f}, near chance) "
              f"-- retrying with a different init seed")
        attempt += 1

    y_pred_prob = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec_vals, prec_vals)
    f1 = f1_score(y_test, y_pred)

    result = {
        "name": name,
        "base_seed": base_seed,
        "seed_used": base_seed + attempt - 1,
        "params": int(model.count_params()),
        "flatten_dim": flatten_dim,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "train_time_sec": train_time,
        "model_path": ckpt_path,
        "train_attempts": attempt,
    }
    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as fh:
        json.dump(history.history, fh)
    attempts_note = f", {attempt} attempts" if attempt > 1 else ""
    print(f"[{name:>20}] params={result['params']:>7}  best_val_acc={best_val_acc:.3f}  "
          f"test_acc={test_acc:.3f}  roc_auc={roc_auc:.3f}  f1={f1:.3f}  "
          f"({result['epochs_run']:>3} epochs, {train_time:5.0f}s{attempts_note})")
    return result
''')

md("""## Group 1 — pipe-size sweep

Filters `[2, 4, 4, 4, F]`, uniform `(2,2)` pooling x5, `F` swept over
`{1,2,4,8,16,32,64}` -> flatten `{64,...,4096}`.""")

code(r'''GROUP1_RESULTS = []
for final_filters in [1, 2, 4, 8, 16, 32, 64]:
    filters = [2, 4, 4, 4, final_filters]
    pool_schedule = build_pool_schedule(t_exp=5, f_exp=5)
    name = f"group1_f{final_filters}"
    build_fn = lambda filters=filters, pool_schedule=pool_schedule, name=name: \
        build_cnn(filters, pool_schedule, name=name)
    result = train_and_eval(name, build_fn)
    assert result["flatten_dim"] == 64 * final_filters, (result["flatten_dim"], final_filters)
    result["final_filters"] = final_filters
    result["build_args"] = {"filters": filters, "pool_schedule": pool_schedule, "time_avgpool": None}
    GROUP1_RESULTS.append(result)
''')

code(r'''fig, ax = plt.subplots(figsize=(6, 4))
flat_dims = [r["flatten_dim"] for r in GROUP1_RESULTS]
test_accs = [r["test_acc"] for r in GROUP1_RESULTS]
ax.plot(flat_dims, test_accs, "o-")
ax.set_xscale("log", base=2)
ax.set_xlabel("Flatten (bottleneck) dimension")
ax.set_ylabel("Test accuracy")
ax.set_title("Group 1: accuracy vs. bottleneck width")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "group1_accuracy_vs_flatten.png"), dpi=120)
plt.show()

print(f"{'final_filters':>14} {'flatten':>8} {'params':>8} {'val_acc':>8} {'test_acc':>9} {'f1':>6}")
for r in GROUP1_RESULTS:
    print(f"{r['final_filters']:>14} {r['flatten_dim']:>8} {r['params']:>8} "
          f"{r['best_val_acc']:>8.3f} {r['test_acc']:>9.3f} {r['f1_score']:>6.3f}")
''')

md("""## Group 2 — staged pooling ablation

Bottleneck fixed at flatten=256, five different pooling *shapes* to get there:

| label | filters | pool_schedule |
|---|---|---|
| a_uniform (production baseline) | `[2,4,4,4,4]` | `[(2,2)]*5` |
| b_front_loaded | `[4,8,4]` | `[(4,4),(4,4),(2,2)]` |
| c_back_loaded | `[2,4,8,4]` | `[(2,2),(2,2),(4,4),(2,2)]` |
| d_freq_preserving (compress time faster) | `[4,8,1]` | `[(4,2),(4,2),(2,2)]` |
| e_time_preserving (compress freq faster) | `[4,8,1]` | `[(2,4),(2,4),(2,2)]` |

`a_uniform` is architecturally identical to the Group 1 `final_filters=4` config, so
its result is reused rather than retrained.""")

code(r'''GROUP2_CONFIGS = {
    "a_uniform":         dict(filters=[2, 4, 4, 4, 4], pool_schedule=[(2, 2)] * 5),
    "b_front_loaded":    dict(filters=[4, 8, 4],        pool_schedule=[(4, 4), (4, 4), (2, 2)]),
    "c_back_loaded":     dict(filters=[2, 4, 8, 4],     pool_schedule=[(2, 2), (2, 2), (4, 4), (2, 2)]),
    "d_freq_preserving": dict(filters=[4, 8, 1],        pool_schedule=[(4, 2), (4, 2), (2, 2)]),
    "e_time_preserving": dict(filters=[4, 8, 1],        pool_schedule=[(2, 4), (2, 4), (2, 2)]),
}

GROUP1_F4 = next(r for r in GROUP1_RESULTS if r["final_filters"] == 4)

GROUP2_RESULTS = []
for label, cfg in GROUP2_CONFIGS.items():
    name = f"group2_{label}"
    if label == "a_uniform":
        result = dict(GROUP1_F4)
        result["name"] = name
        print(f"[{name:>20}] reusing group1_f4 result (identical architecture)")
    else:
        build_fn = lambda cfg=cfg, name=name: build_cnn(cfg["filters"], cfg["pool_schedule"], name=name)
        result = train_and_eval(name, build_fn)
        assert result["flatten_dim"] == 256, (label, result["flatten_dim"])
        result["build_args"] = {"filters": cfg["filters"], "pool_schedule": cfg["pool_schedule"], "time_avgpool": None}
    result["label"] = label
    GROUP2_RESULTS.append(result)
''')

code(r'''print(f"{'label':>18} {'params':>8} {'val_acc':>8} {'test_acc':>9} {'f1':>6}")
for r in GROUP2_RESULTS:
    print(f"{r['label']:>18} {r['params']:>8} {r['best_val_acc']:>8.3f} "
          f"{r['test_acc']:>9.3f} {r['f1_score']:>6.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
labels = [r["label"] for r in GROUP2_RESULTS]
accs = [r["test_acc"] for r in GROUP2_RESULTS]
ax.bar(labels, accs)
ax.set_ylabel("Test accuracy")
ax.set_title("Group 2: accuracy by pooling schedule shape")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "group2_accuracy_by_schedule.png"), dpi=120)
plt.show()
''')

md("""## Group 3 — time-resolution coarsening

`AveragePooling2D(pool_size=(r,1))` coarsens the time axis by `r` before the conv
stack. The remaining CNN pooling schedule is chosen so the bottleneck stays at
flatten=256 regardless of `r`: time needs `32/r` total reduction (`t_exp = 5 -
log2(r)` halvings across 5 stages), frequency always needs the full `32x`
(`f_exp=5`). `r=1` is identical to the Group 1 `final_filters=4` / Group 2
`a_uniform` config and is reused.

Effective time resolution after coarsening: `r * 0.655 ms/bin`.""")

code(r'''GROUP3_RESULTS = []
for r_factor in [1, 2, 4, 8, 16, 32]:
    name = f"group3_r{r_factor}"
    if r_factor == 1:
        result = dict(GROUP1_F4)
        result["name"] = name
        print(f"[{name:>20}] reusing group1_f4 result (r=1 = no coarsening)")
    else:
        t_exp = 5 - int(math.log2(r_factor))
        pool_schedule = build_pool_schedule(t_exp=t_exp, f_exp=5)
        filters = [2, 4, 4, 4, 4]
        build_fn = lambda filters=filters, pool_schedule=pool_schedule, r_factor=r_factor, name=name: \
            build_cnn(filters, pool_schedule, time_avgpool=r_factor, name=name)
        result = train_and_eval(name, build_fn)
        assert result["flatten_dim"] == 256, (r_factor, result["flatten_dim"])
        result["build_args"] = {"filters": filters, "pool_schedule": pool_schedule, "time_avgpool": r_factor}
    result["r"] = r_factor
    result["effective_time_res_ms"] = r_factor * 0.655
    GROUP3_RESULTS.append(result)
''')

code(r'''print(f"{'r':>4} {'eff_res_ms':>11} {'params':>8} {'val_acc':>8} {'test_acc':>9} {'f1':>6}")
for r in GROUP3_RESULTS:
    print(f"{r['r']:>4} {r['effective_time_res_ms']:>11.3f} {r['params']:>8} "
          f"{r['best_val_acc']:>8.3f} {r['test_acc']:>9.3f} {r['f1_score']:>6.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
eff_res = [r["effective_time_res_ms"] for r in GROUP3_RESULTS]
accs = [r["test_acc"] for r in GROUP3_RESULTS]
ax.plot(eff_res, accs, "o-")
ax.set_xscale("log", base=2)
ax.set_xlabel("Effective time resolution (ms/bin)")
ax.set_ylabel("Test accuracy")
ax.set_title("Group 3: accuracy vs. effective time resolution")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "group3_accuracy_vs_time_resolution.png"), dpi=120)
plt.show()
''')

md("""## Winners

Selected by validation accuracy at the early-stopping checkpoint (`best_val_acc`),
to avoid picking based on the test set itself. Test-set metrics are reported for
the winner only.""")

code(r'''def pick_best(results, key="best_val_acc"):
    return max(results, key=lambda r: r[key])


WINNERS = {
    "group1": pick_best(GROUP1_RESULTS),
    "group2": pick_best(GROUP2_RESULTS),
    "group3": pick_best(GROUP3_RESULTS),
}

ALL_RESULTS = {"group1": GROUP1_RESULTS, "group2": GROUP2_RESULTS, "group3": GROUP3_RESULTS}
with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as fh:
    json.dump(ALL_RESULTS, fh, indent=2)

for gname, res in WINNERS.items():
    print(f"{gname}: {res['name']}  params={res['params']}  flatten={res.get('flatten_dim', 'n/a')}  "
          f"val_acc={res['best_val_acc']:.3f}  test_acc={res['test_acc']:.3f}  f1={res['f1_score']:.3f}")
''')

md(r"""## Seed robustness check

The single-seed picks above are close calls in two places that matter: Group 2's
top three (`d_freq_preserving` 0.951, `b_front_loaded`/`c_back_loaded` 0.938) are
within about one standard error of each other at n=308 validation samples, and
Group 3's top four (`r1`/`r2` 0.935, `r8` 0.942, `r16` 0.929) are effectively a
4-way tie. Group 1's raw accuracy-winner (`f32`, flatten=2048, 67k params) also
blows past this project's FPGA sizing budget (flatten<500, params<50k), so the
config that actually ships to the FPGA package needs to be chosen from the
in-budget subset (`f1`, `f2`, `f4`) instead -- and `f1` (val_acc=0.731) is
genuinely worse, not noise, leaving `f2` vs `f4` as the real in-budget call.

Rather than lock in single-seed results for decisions that close, retrain each
architecture actually in contention under 2 additional seeds (reusing each
architecture's existing seed-42 run as the third point) and pick winners by
mean validation accuracy across all 3 seeds. Architectures that are the same
across groups (the production baseline `[2,4,4,4,4]`/uniform pooling/no time
coarsening -- `group1_f4` = `group2_a_uniform` = `group3_r1`) are only retrained
once and reused.""")

code(r'''SEED_TRIALS = [42, 142, 242]  # 42 reuses the already-trained result above

BASELINE_ARGS = {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None}

CANDIDATES = {
    ("group1", "f2"): {"filters": [2, 4, 4, 4, 2], "pool_schedule": build_pool_schedule(5, 5), "time_avgpool": None},
    ("group1", "f4"): BASELINE_ARGS,
    ("group2", "a_uniform"): BASELINE_ARGS,
    ("group2", "b_front_loaded"): {"filters": [4, 8, 4], "pool_schedule": [(4, 4), (4, 4), (2, 2)], "time_avgpool": None},
    ("group2", "c_back_loaded"): {"filters": [2, 4, 8, 4], "pool_schedule": [(2, 2), (2, 2), (4, 4), (2, 2)], "time_avgpool": None},
    ("group2", "d_freq_preserving"): {"filters": [4, 8, 1], "pool_schedule": [(4, 2), (4, 2), (2, 2)], "time_avgpool": None},
    ("group3", "r1"): BASELINE_ARGS,
    ("group3", "r2"): {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(4, 5), "time_avgpool": 2},
    ("group3", "r8"): {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(2, 5), "time_avgpool": 8},
    ("group3", "r16"): {"filters": [2, 4, 4, 4, 4], "pool_schedule": build_pool_schedule(1, 5), "time_avgpool": 16},
}

# The seed=42 point for each candidate was already trained above -- reuse it
# instead of retraining, and only run the 2 new seeds.
EXISTING_SEED42 = {
    ("group1", "f2"): next(r for r in GROUP1_RESULTS if r["final_filters"] == 2),
    ("group1", "f4"): GROUP1_F4,
    ("group2", "a_uniform"): GROUP1_F4,
    ("group2", "b_front_loaded"): next(r for r in GROUP2_RESULTS if r["label"] == "b_front_loaded"),
    ("group2", "c_back_loaded"): next(r for r in GROUP2_RESULTS if r["label"] == "c_back_loaded"),
    ("group2", "d_freq_preserving"): next(r for r in GROUP2_RESULTS if r["label"] == "d_freq_preserving"),
    ("group3", "r1"): GROUP1_F4,
    ("group3", "r2"): next(r for r in GROUP3_RESULTS if r["r"] == 2),
    ("group3", "r8"): next(r for r in GROUP3_RESULTS if r["r"] == 8),
    ("group3", "r16"): next(r for r in GROUP3_RESULTS if r["r"] == 16),
}

SEED_CHECK_TRIALS = {}  # architecture tag (dedup'd) -> list of per-seed result dicts
ARCH_TAG = {  # dedup shared architectures so each is only ever trained once per seed
    ("group1", "f4"): "baseline", ("group2", "a_uniform"): "baseline", ("group3", "r1"): "baseline",
}
for key in CANDIDATES:
    ARCH_TAG.setdefault(key, f"{key[0]}_{key[1]}")

for key, args in CANDIDATES.items():
    tag = ARCH_TAG[key]
    if tag in SEED_CHECK_TRIALS:
        continue  # already trained under this tag via an earlier (group, name) key
    trials = [EXISTING_SEED42[key]]
    for trial_seed in [s for s in SEED_TRIALS if s != 42]:
        build_fn = lambda args=args, tag=tag: build_cnn(
            args["filters"], args["pool_schedule"], time_avgpool=args["time_avgpool"], name=f"seedcheck_{tag}"
        )
        r = train_and_eval(f"seedcheck_{tag}_s{trial_seed}", build_fn, base_seed=trial_seed)
        trials.append(r)
    SEED_CHECK_TRIALS[tag] = trials

print(f"\n{'architecture':>20} {'seeds':>16} {'val_acc (mean +/- std)':>24} {'test_acc (mean)':>17}")
for tag, trials in SEED_CHECK_TRIALS.items():
    val_accs = np.array([t["best_val_acc"] for t in trials])
    test_accs = np.array([t["test_acc"] for t in trials])
    seeds = [t.get("seed_used", t.get("base_seed", "?")) for t in trials]
    print(f"{tag:>20} {str(seeds):>16} {val_accs.mean():>10.4f} +/- {val_accs.std():<9.4f} {test_accs.mean():>17.4f}")
''')

code(r'''def mean_val_acc(key):
    return np.mean([t["best_val_acc"] for t in SEED_CHECK_TRIALS[ARCH_TAG[key]]])


def best_trial(key):
    """The single trial (of the 3 seeds) with the highest val_acc for this
    candidate -- this is the concrete model that gets exported to hls4ml."""
    return max(SEED_CHECK_TRIALS[ARCH_TAG[key]], key=lambda t: t["best_val_acc"])


# Group 1's FPGA package must satisfy the project's sizing budget (flatten<500,
# params<50k); f1 is excluded as a genuinely worse config (0.73 vs 0.93+), not a
# close call, leaving f2 vs f4 as the real in-budget decision.
group1_in_budget_keys = [("group1", "f2"), ("group1", "f4")]
group1_final_key = max(group1_in_budget_keys, key=mean_val_acc)

group2_keys = [("group2", "a_uniform"), ("group2", "b_front_loaded"),
               ("group2", "c_back_loaded"), ("group2", "d_freq_preserving")]
group2_final_key = max(group2_keys, key=mean_val_acc)

group3_keys = [("group3", "r1"), ("group3", "r2"), ("group3", "r8"), ("group3", "r16")]
group3_final_key = max(group3_keys, key=mean_val_acc)

FINAL_WINNER_KEYS = {"group1": group1_final_key, "group2": group2_final_key, "group3": group3_final_key}
FINAL_WINNERS = {}
for gname, key in FINAL_WINNER_KEYS.items():
    trial = best_trial(key)
    result = dict(trial)
    result["build_args"] = CANDIDATES[key]
    result["mean_val_acc_3seed"] = float(mean_val_acc(key))
    result["std_val_acc_3seed"] = float(np.std([t["best_val_acc"] for t in SEED_CHECK_TRIALS[ARCH_TAG[key]]]))
    FINAL_WINNERS[gname] = result
    print(f"{gname}: {key[1]}  mean_val_acc(3-seed)={result['mean_val_acc_3seed']:.4f} "
          f"+/- {result['std_val_acc_3seed']:.4f}  best-trial test_acc={result['test_acc']:.4f}  "
          f"(best trial seed={result.get('seed_used')})")

with open(os.path.join(RESULTS_DIR, "seed_check_results.json"), "w") as fh:
    json.dump({tag: trials for tag, trials in SEED_CHECK_TRIALS.items()}, fh, indent=2)
''')

md(r"""## FPGA handoff packaging

**Quantization approach note:** this originally used QKeras quantization-aware
training, but QKeras requires Keras 2 while plain `tensorflow` on this machine
defaults to Keras 3 (TF 2.20), and the `tf_keras` legacy shim only works if
*every* import -- layers, callbacks, utils, not just the model -- consistently
comes from `tf_keras`. Mixing them breaks at the callback level. This has failed
twice before for the same root cause, so instead of QKeras this uses hls4ml's own
**post-training quantization**: the plain float Keras models trained above are
converted straight to fixed-point HLS projects via a per-layer `ap_fixed<W,I>`
precision config, no QKeras and no Keras-2 shim involved.

For each of the three 3-seed-verified winning float architectures (`FINAL_WINNERS`
above -- not the naive single-seed picks, and Group 1's is additionally restricted
to the in-budget subset), two candidate precisions are generated: `ap_fixed<16,6>`
(matches the existing `frb_pooled_hls/` project) and `ap_fixed<8,3>` (a narrower
candidate) -- 6 hls4ml/Vitis HLS projects total.
`hls4ml.converters.convert_from_keras_model(...).write()` only writes C++/TCL
project files to disk; it never invokes the Vitis toolchain, so it's safe to run
here.

**Bit-accurate quantized accuracy is deferred to EC2.** hls4ml's own NumPy
emulation (`hls_model.compile()` + `hls_model.predict()`) needs to compile a small
C++ bridge locally via `g++`, and this machine has no C/C++ compiler at all (no
gcc/g++/cl.exe/MinGW found) -- confirmed by testing this exact path on a toy model
before writing this cell. So instead of a fabricated or skipped accuracy number,
`X_test`/`y_test` are saved alongside the generated projects so that on EC2 (which
should have a compiler available next to Vitis) the same `compile()`/`predict()`
check can run before or after synthesis. No LUT/DSP/BRAM/latency/Fmax numbers are
fabricated here either -- those only exist after real synthesis.

Target: `xcvu9p-flga2104-2L-e` (Virtex UltraScale+ VU9P) at 200 MHz (5 ns clock),
`io_stream` + `Resource` strategy, per this project's FPGA constraints. The
synthesis driver matches the `run_vitis_synthesis.sh`-style pattern already used in
`frb_pooled_hls/` (a hand-written `run_synthesis.tcl` invoked via
`vitis-run --mode hls`, working around a known hls4ml `.build()` / Vitis version
mismatch).""")

code(r'''import hls4ml

FPGA_HANDOFF_DIR = "fpga_handoff"
os.makedirs(FPGA_HANDOFF_DIR, exist_ok=True)

PART = "xcvu9p-flga2104-2L-e"
CLOCK_PERIOD_NS = 5  # 200 MHz
PRECISIONS = ["ap_fixed<16,6>", "ap_fixed<8,3>"]

GROUP_DESC = {
    "group1": "pipe-size sweep (bottleneck width)",
    "group2": "staged pooling ablation (pooling schedule shape)",
    "group3": "time-resolution coarsening",
}

# Shared test set for the deferred EC2 bit-accurate check (hls_model.compile() +
# .predict() needs a local C++ compiler, which this machine does not have).
np.save(os.path.join(FPGA_HANDOFF_DIR, "X_test.npy"), X_test)
np.save(os.path.join(FPGA_HANDOFF_DIR, "y_test.npy"), y_test)
''')

code(r'''RUN_SH = """#!/bin/bash
source /opt/Xilinx/Vitis/2025.1/settings64.sh 2>/dev/null || source /opt/Xilinx/2025.1/Vitis/settings64.sh 2>/dev/null || true
vitis-run --mode hls --tcl run_synthesis.tcl 2>&1 | tee synthesis.log
"""

RUN_TCL = """open_project {proj}_prj
set_top {proj}

add_files firmware/{proj}.cpp -cflags "-std=c++0x"
add_files -tb {proj}_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data

open_solution "solution1"
set_part {{{part}}}
create_clock -period {clock} -name default

config_compile -pipeline_loops 64

puts "Starting C synthesis..."
csynth_design

puts "Synthesis complete!"
exit
"""

CHECK_ACCURACY_PY = """# Run on EC2 (needs a C/C++ compiler next to Vitis -- not available on the
# machine this project was generated on). Bit-accurate NumPy emulation of the
# quantized model, no Vitis invoked. Rebuilds the hls4ml project in-place from
# keras_model.keras (rather than trusting paths recorded in hls4ml_config.yml,
# which are relative to the original repo layout and not portable) so this also
# self-heals against any hls4ml version difference between here and EC2.
import numpy as np
import tensorflow as tf
import hls4ml

PROJECT_NAME = "{proj}"
PRECISION = "{precision}"
PART = "{part}"
CLOCK_PERIOD_NS = {clock}

model = tf.keras.models.load_model("keras_model.keras")
hls_config = hls4ml.utils.config_from_keras_model(
    model, granularity="name", backend="Vitis", default_precision=PRECISION,
)
hls_config["Model"]["Strategy"] = "Resource"
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=hls_config, output_dir=".", project_name=PROJECT_NAME,
    part=PART, backend="Vitis", io_type="io_stream", clock_period=CLOCK_PERIOD_NS,
)
hls_model.compile()

X_test = np.load("../X_test.npy")
y_test = np.load("../y_test.npy")
y_pred = hls_model.predict(X_test).flatten()
acc = ((y_pred >= 0.5).astype(int) == y_test).mean()
print(f"quantized test accuracy ({{PROJECT_NAME}}): {{acc:.4f}}")
"""

README = """# FPGA Handoff -- {config_name}

Source: {group} -- {group_desc}, from `frb_ablation_study.ipynb`.

## Architecture
- filters: `{filters}`
- pool_schedule: `{pool_schedule}`  (axis convention: `(time_pool, freq_pool)`)
- time_avgpool (Group 3 only): `{time_avgpool}`
- flatten (bottleneck) dimension: {flatten_dim}
- parameter count: {params}
- 3-seed mean validation accuracy: {mean_val_acc_3seed:.4f} +/- {std_val_acc_3seed:.4f}
- Float test accuracy of the best of those 3 seeds (this repo, CPU, no quantization): {float_test_acc:.4f}

## Quantization
Post-training fixed-point quantization via hls4ml (no QKeras): every layer's
weights/activations/accumulators default to `{precision}`
(`hls4ml.utils.config_from_keras_model(model, granularity="name",
default_precision="{precision}")`). This was NOT applied on the float model ahead of
time (no quantization-aware fine-tuning) -- hls4ml quantizes at conversion time.

**Quantized test accuracy is not in this README.** It requires
`hls_model.compile()` + `hls_model.predict()`, which compiles a small C++ bridge
via `g++` -- unavailable on the machine that generated this project. Run
`check_accuracy.py` in this folder on a machine with a C/C++ compiler (e.g. the EC2
box, which should already have one alongside Vitis) to get it:

    cd fpga_handoff/{config_name}
    python check_accuracy.py

## FPGA target
- Part: `{part}` (Xilinx Virtex UltraScale+ VU9P)
- Clock: 200 MHz ({clock} ns period)
- IOType: `io_stream`, Strategy: `Resource`

## Static constraint check (not synthesis-verified -- these are just facts about
## the model itself, per this project's documented FPGA constraints)
- Flatten dim < 500: {flatten_ok}
- Params < 50k: {params_ok}
{constraint_note}
## Running synthesis (on the EC2 box with Vitis HLS installed -- NOT here)

    cd fpga_handoff/{config_name}
    chmod +x run_synthesis.sh
    ./run_synthesis.sh

This produces `{proj}_prj/solution1/syn/report/{proj}_csynth.rpt` with real LUT /
DSP / BRAM / latency / Fmax numbers -- those are not available in this repo and are
intentionally left out of this README; read them from that report (or
`synthesis.log`) after running.

## Files
- `firmware/` -- generated HLS C++, weights baked in as headers
- `{proj}_test.cpp` -- hls4ml-generated testbench
- `tb_data/` -- testbench I/O data
- `hls4ml_config.yml` -- exact hls4ml config used to generate this project
- `keras_model.keras` -- the float Keras model this was converted from
- `check_accuracy.py` -- run on EC2 for the deferred bit-accurate quantized accuracy
- `run_synthesis.sh` / `run_synthesis.tcl` -- Vitis HLS synthesis driver (same
  hand-written-tcl-via-`vitis-run` pattern as `frb_pooled_hls/`)
"""

EXPORT_SUMMARY = []
for gname, res in FINAL_WINNERS.items():
    args = res["build_args"]
    float_model = tf.keras.models.load_model(res["model_path"])
    for precision in PRECISIONS:
        precision_tag = precision.replace("ap_fixed<", "w").replace(",", "_").replace(">", "")
        config_name = f"{gname}_{precision_tag}"
        proj = f"frb_{config_name}"

        hls_config = hls4ml.utils.config_from_keras_model(
            float_model, granularity="name", backend="Vitis", default_precision=precision,
        )
        hls_config["Model"]["Strategy"] = "Resource"

        out_dir = os.path.join(FPGA_HANDOFF_DIR, config_name)
        hls_model = hls4ml.converters.convert_from_keras_model(
            float_model, hls_config=hls_config, output_dir=out_dir,
            project_name=proj, part=PART, backend="Vitis",
            io_type="io_stream", clock_period=CLOCK_PERIOD_NS,
        )
        hls_model.write()

        with open(os.path.join(out_dir, "run_synthesis.sh"), "w", newline="\n") as fh:
            fh.write(RUN_SH)
        with open(os.path.join(out_dir, "run_synthesis.tcl"), "w", newline="\n") as fh:
            fh.write(RUN_TCL.format(proj=proj, part=PART, clock=CLOCK_PERIOD_NS))
        with open(os.path.join(out_dir, "check_accuracy.py"), "w", newline="\n") as fh:
            fh.write(CHECK_ACCURACY_PY.format(proj=proj, precision=precision, part=PART, clock=CLOCK_PERIOD_NS))

        flatten_dim = res.get("flatten_dim", 256)
        params = res["params"]
        flatten_ok = "yes" if flatten_dim < 500 else "NO"
        params_ok = "yes" if params < 50000 else "NO"
        constraint_note = ""
        if flatten_ok == "NO" or params_ok == "NO":
            constraint_note = (
                "\n**Caution:** this config exceeds the documented FPGA sizing constraint "
                "(flatten<500 and/or params<50k). It won a group on accuracy; whether it "
                "actually fits/synthesizes in reasonable time is unverified until EC2 "
                "synthesis runs.\n"
            )

        readme = README.format(
            config_name=config_name, group=gname, group_desc=GROUP_DESC[gname],
            filters=args["filters"], pool_schedule=args["pool_schedule"], time_avgpool=args["time_avgpool"],
            flatten_dim=flatten_dim, params=params, float_test_acc=res["test_acc"],
            mean_val_acc_3seed=res["mean_val_acc_3seed"], std_val_acc_3seed=res["std_val_acc_3seed"],
            precision=precision, part=PART, clock=CLOCK_PERIOD_NS,
            flatten_ok=flatten_ok, params_ok=params_ok, constraint_note=constraint_note, proj=proj,
        )
        with open(os.path.join(out_dir, "README.md"), "w") as fh:
            fh.write(readme)

        EXPORT_SUMMARY.append({
            "config_name": config_name, "group": gname, "precision": precision,
            "flatten_dim": flatten_dim, "params": params, "float_test_acc": res["test_acc"],
            "mean_val_acc_3seed": res["mean_val_acc_3seed"], "std_val_acc_3seed": res["std_val_acc_3seed"],
            "flatten_ok": flatten_ok, "params_ok": params_ok, "path": out_dir,
        })
        print(f"[{config_name}] wrote {out_dir}  (flatten={flatten_dim}, params={params}, "
              f"flatten_ok={flatten_ok}, params_ok={params_ok})")
''')

code(r'''top_readme_rows = "\n".join(
    f"| {e['config_name']} | {e['group']} | {e['precision']} | {e['flatten_dim']} | {e['params']} | "
    f"{e['mean_val_acc_3seed']:.4f} +/- {e['std_val_acc_3seed']:.4f} | {e['float_test_acc']:.4f} | "
    f"{e['flatten_ok']} | {e['params_ok']} | "
    f"run `check_accuracy.py` then synthesis on EC2, see `fpga_handoff/{e['config_name']}/` |"
    for e in EXPORT_SUMMARY
)

top_readme = f"""# FPGA Handoff

Generated by `frb_ablation_study.ipynb` on {datetime.now().isoformat(timespec='seconds')}.
Six hls4ml/Vitis HLS projects: the 3-seed-verified best config from each of Groups
1-3 (Group 1 additionally restricted to the in-budget subset -- flatten<500,
params<50k -- since its raw accuracy-winner blew past both), each at two
post-training-quantization precisions (ap_fixed<16,6>, ap_fixed<8,3>). Target:
xcvu9p-flga2104-2L-e, 200 MHz (5 ns), io_stream + Resource strategy.

`X_test.npy` / `y_test.npy` in this directory are the held-out test set, needed by
each config's `check_accuracy.py` (quantized-accuracy check deferred to EC2 -- this
machine has no C/C++ compiler to run hls4ml's local NumPy emulation bridge).

Copy this whole directory to the EC2 box with Vitis HLS (and a C/C++ compiler)
installed. Per config: run `check_accuracy.py` for the bit-accurate quantized
accuracy, then `run_synthesis.sh` for the real LUT/DSP/BRAM/latency/Fmax numbers.

| config | group | precision | flatten | params | val_acc (3-seed mean +/- std) | best-seed test_acc | flatten<500 | params<50k | next step |
|---|---|---|---|---|---|---|---|---|---|
{top_readme_rows}
"""
with open(os.path.join(FPGA_HANDOFF_DIR, "README.md"), "w") as fh:
    fh.write(top_readme)

print(top_readme)
''')

md("""## Summary

Single-seed picks (informational) vs. the 3-seed-verified configs that actually
shipped to the FPGA package.""")

code(r'''print("Naive single-seed picks:")
for gname, res in WINNERS.items():
    print(f"  {gname}: {res['name']}  flatten={res.get('flatten_dim','n/a')}  "
          f"params={res['params']}  test_acc={res['test_acc']:.4f}")

print("\n3-seed-verified winners (what was actually packaged):")
for gname, key in FINAL_WINNER_KEYS.items():
    res = FINAL_WINNERS[gname]
    print(f"  {gname}: {key[1]}  flatten={res.get('flatten_dim','n/a')}  params={res['params']}  "
          f"mean_val_acc(3-seed)={res['mean_val_acc_3seed']:.4f} +/- {res['std_val_acc_3seed']:.4f}  "
          f"best-trial test_acc={res['test_acc']:.4f}")

print("\nPackaged for FPGA handoff:")
for e in EXPORT_SUMMARY:
    print(f"  {e['config_name']} ({e['precision']})  -> {e['path']}")
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("frb_ablation_study.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote frb_ablation_study.ipynb with", len(cells), "cells")
