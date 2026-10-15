"""Stage 1 (end): quantization-aware training (QKeras) of track-B configs.

Runs under TF_USE_LEGACY_KERAS=1 (QKeras needs Keras 2 / tf_keras). For each
(float run_id, bits): rebuild the architecture with QKeras quantizers, copy the
float weights in, fine-tune with QAT (early stop on val loss), evaluate the
quantized model on the same file-grouped splits, and log a run JSON
(stage=stage1_qat, cfg.quant='qkeras_<bits>b', cfg.parent=<float run_id>).

Integer bits per layer are set from the float model: weights -> ceil(log2 max|w|),
activations -> ceil(log2 of the 99.99th percentile on 2k train samples).
The model INPUT is not quantized by QKeras; hls4ml gets ap_fixed<16,6> for it.

Usage: TF_USE_LEGACY_KERAS=1 python scripts/qat_train.py <float_run_id> <bits> [threads]
"""
import json
import math
import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

import frb_train as ft  # noqa: E402


def int_bits(v):
    return max(0, int(math.ceil(math.log2(max(float(v), 1e-6)))))


def build_qcnn(cfg, input_shape, bits, wint, aint):
    from qkeras import QActivation, QConv2D, QDense, quantized_bits, quantized_relu
    from tensorflow import keras
    L = keras.layers
    q = lambda n: quantized_bits(bits, wint[n], alpha=1)  # noqa: E731
    inp = keras.Input(shape=tuple(input_shape) + (1,), name="input_data")
    x = inp
    for i, f in enumerate(cfg["filters"]):
        x = QConv2D(f, cfg["kernel"], padding="same", kernel_quantizer=q(f"conv_{i}"), bias_quantizer=q(f"conv_{i}"),
                    name=f"conv_{i}")(x)
        x = QActivation(quantized_relu(bits, aint[f"relu_{i}"]), name=f"relu_{i}")(x)
        p = cfg["pools"][i]
        if p > 1:
            P = L.AveragePooling2D if cfg.get("pool_type") == "avg" else L.MaxPooling2D
            x = P((p, p), name=f"pool_{i}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x) if cfg.get("head", "flatten") == "gap" else L.Flatten(name="flatten")(x)
    if cfg.get("dense", 0):
        x = QDense(cfg["dense"], kernel_quantizer=q("dense_0"), bias_quantizer=q("dense_0"), name="dense_0")(x)
        x = QActivation(quantized_relu(bits, aint["relu_d"]), name="relu_d")(x)
        if cfg.get("dropout", 0) > 0:
            x = L.Dropout(cfg["dropout"], name="drop_0")(x)
    x = QDense(1, kernel_quantizer=q("output"), bias_quantizer=q("output"), name="output_dense")(x)
    out = L.Activation("sigmoid", name="output")(x)
    return keras.Model(inp, out, name="frb_qcnn")


def main(parent, bits, threads=4):
    from tensorflow import keras
    ft.set_threads(threads)
    pr = json.load(open(os.path.join(ft.RUNS, f"{parent}.json")))
    cfg = dict(pr["cfg"])
    keras.utils.set_random_seed(cfg.get("seed", 0))
    X, meta = ft.load_data(cfg["t_pool"], cfg["f_pool"], dataset=cfg.get("dataset", "v7"))
    idx = ft.split_indices(meta, cfg.get("split", "primary"), cfg.get("fold"), cfg.get("train_snr_min", 3.0))
    y = meta.label.values.astype(np.float32)

    # float reference model (Keras 3 file) -> weights + activation ranges, via a tf_keras rebuild
    fl = ft.build_cnn(cfg, tuple(pr["input_shape"]))
    import keras as k3
    f3 = k3.models.load_model(os.path.join(ft.ROOT, pr["model_path"]))
    w3 = {l.name: l.get_weights() for l in f3.layers if l.get_weights()}
    for l in fl.layers:
        if l.name in w3 and l.weights:
            l.set_weights(w3[l.name])
    wint = {n: int_bits(max(np.abs(w).max() for w in ws)) for n, ws in w3.items()}
    probe = ft.prep_input(X[np.sort(np.random.default_rng(0).choice(idx["train"], min(2000, len(idx["train"])), replace=False))], "cnn",
                          cfg.get("clip"))
    aint = {}
    for l in fl.layers:
        if l.name.startswith("relu"):
            a = keras.Model(fl.input, l.output).predict(probe, batch_size=256, verbose=0)
            aint[l.name] = int_bits(np.percentile(a, 99.99))
    qm = build_qcnn(cfg, pr["input_shape"], bits, wint, aint)
    for l in qm.layers:
        src = "output" if l.name == "output_dense" else l.name
        if src in w3 and l.weights:
            l.set_weights(w3[src])
    qm.compile(optimizer=keras.optimizers.Adam(cfg["lr"] / 3), loss="binary_crossentropy", metrics=["accuracy"])

    # float-weights-loaded-into-quantized-graph accuracy (= naive PTQ via QKeras, before QAT)
    clip = cfg.get("clip")
    Xte = ft.prep_input(X[idx["test"]], "cnn", clip)
    p_ptq = ft.predict(qm, Xte)
    Xtr, ytr, tr = X, y, idx["train"]
    if cfg.get("train_pool"):
        Xtr, mtr = ft.load_data(cfg["t_pool"], cfg["f_pool"], cfg["train_pool"])
        ytr = mtr.label.values.astype(np.float32)
        tr = np.where((mtr.label == 0) | (mtr.snr >= cfg.get("train_snr_min", 3.0)))[0]
    seq = ft.BatchSeq(Xtr, ytr, tr, cfg["batch"], "cnn", cfg.get("time_flip", False), cfg.get("seed", 0),
                      cfg.get("clip"))
    Xva, yva = ft.prep_input(X[idx["val"]], "cnn", clip), y[idx["val"]]
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    h = qm.fit(seq, validation_data=(Xva, yva), epochs=30, callbacks=cb, verbose=0)

    qcfg = dict(cfg, quant=f"qkeras_{bits}b", parent=parent, stage="stage1_qat", wint=wint, aint=aint)
    run_id = f"qat_{parent}_{bits}b"
    if cfg.get("dataset") == "v6":
        qcfg["stage"] = "v6_bench"
    res = dict(run_id=run_id, status="ok", params=pr["params"], macs=pr["macs"], flops=pr["flops"], cfg=qcfg,
               input_shape=pr["input_shape"], epochs_run=len(h.history["loss"]),
               val_loss=float(min(h.history["val_loss"])), float_test_acc=pr["test_acc"],
               ptq_qkeras_test_acc=float(((p_ptq >= 0.5) == y[idx["test"]]).mean()))
    res.update({f"val_{k}": v for k, v in ft.metrics(yva, ft.predict(qm, Xva)).items()})
    pt = ft.predict(qm, Xte)
    res.update({f"test_{k}": v for k, v in ft.metrics(y[idx["test"]], pt).items()})
    pa = ft.predict(qm, ft.prep_input(X[idx["test_all"]], "cnn", clip))
    res["test_allsnr_acc"] = float(((pa >= 0.5) == y[idx["test_all"]]).mean())
    res.update({f"test_{k}": v for k, v in ft.recall_by_snr(meta.iloc[idx["test_all"]], pa).items()})
    path = os.path.join(ft.MODELS, f"{run_id}.h5")
    qm.save(path)
    res["model_path"] = os.path.relpath(path, ft.ROOT)
    json.dump(res, open(os.path.join(ft.RUNS, f"{run_id}.json"), "w"), indent=1, default=float)
    print(run_id, "float", pr["test_acc"], "ptq(qkeras)", res["ptq_qkeras_test_acc"], "qat", res["test_acc"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4)
