"""Stage 2/3: write hls4ml Vitis HLS projects for selected models + a local
fixed-point emulation of the same precision settings.

This machine has no Vitis HLS and no C++ compiler, so: projects are written
(hls_model.write()) and synthesised by hls_projects/run_all_synthesis.sh on the
EC2 box (20-min timeout per project, csynth only). The emulation here is a
Python model of the same fixed-point types (weights/bias/results
ap_fixed<W,I,AP_RND_CONV,AP_SAT>, wide accumulators, input ap_fixed<16,6>) and is
labelled `emu_*` -- it is NOT C-simulation; each project's check_accuracy.py runs
the real bit-accurate C-sim on EC2.

Usage: python scripts/hls_export.py plan.json   (list of {run_id, variants:[{prec, rf, strategy, io}]})
"""
import json
import os
import re
import shutil
import sys
import textwrap

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

import frb_train as ft  # noqa: E402

PART = "xcvu9p-flga2104-2L-e"
CLOCK_NS = 5
OUT = os.path.join(ft.ROOT, "hls_projects")
N_TB = 64          # test vectors written to tb_data (C-sim smoke test during synthesis)
INPUT_PREC = "ap_fixed<16,6>"


def parse_prec(p):
    w, i = map(int, re.match(r"ap_fixed<(\d+),(\d+)", p).groups())
    return w, i


def full_prec(p):
    w, i = parse_prec(p)
    return f"ap_fixed<{w},{i},AP_RND_CONV,AP_SAT>"


def qfix(x, w, i):
    f = w - i
    lo, hi = -2.0 ** (i - 1), 2.0 ** (i - 1) - 2.0 ** -f
    return np.clip(np.round(x * 2.0 ** f) / 2.0 ** f, lo, hi)  # round-half-even == AP_RND_CONV


def load_float(run):
    """Load a Keras-3 float model; rename a legacy 'input' layer (reserved in hls4ml)."""
    import keras
    m0 = keras.models.load_model(os.path.join(ft.ROOT, run["model_path"]))
    cfg = json.loads(json.dumps(m0.get_config()).replace('"input"', '"input_data"'))
    m = keras.Model.from_config(cfg)
    m.set_weights(m0.get_weights())
    return m


def emulate(model, X, prec, batch=400):
    return np.concatenate([_emulate(model, X[i:i + batch], prec) for i in range(0, len(X), batch)])


def _emulate(model, X, prec):
    """Layer-by-layer forward with fixed-point quantisation of weights, biases and
    every layer's output (sigmoid output left in float ~ hls4ml lookup table)."""
    w, i = parse_prec(prec)
    iw, ii = parse_prec(INPUT_PREC)
    x = qfix(X, iw, ii)
    for l in model.layers[1:]:
        cls = l.__class__.__name__
        if cls == "Dropout":
            continue
        if l.get_weights():
            orig = l.get_weights()
            l.set_weights([qfix(a, w, i) for a in orig])
        act = getattr(l, "activation", None)
        is_sig = act is not None and getattr(act, "__name__", "") == "sigmoid"
        if is_sig:  # Dense(..., activation=sigmoid): quantise the logit, then float sigmoid
            W, b = l.get_weights()
            y = qfix(x.reshape(len(x), -1) @ W + b, w, i)
            y = 1 / (1 + np.exp(-y))
        else:
            y = np.asarray(l(x, training=False))
            if cls not in ("Flatten",):
                y = qfix(y, w, i)
        if l.get_weights():
            l.set_weights(orig)
        x = y
    return np.asarray(x).ravel()


def emulate_rnn(model, X, prec):
    """Fixed-point LSTM/GRU (Keras gate order, GRU reset_after) + dense head: weights, gate
    pre-activations, gates, cell and hidden states quantised to ap_fixed<W,I>; sigmoid/tanh in
    float (hls4ml uses lookup tables -- small extra error not modelled)."""
    w, i = parse_prec(prec)
    iw, ii = parse_prec(INPUT_PREC)
    q = lambda a: qfix(a, w, i)  # noqa: E731
    sig = lambda a: 1 / (1 + np.exp(-a))  # noqa: E731
    x = qfix(X, iw, ii)
    rnn = [l for l in model.layers if l.__class__.__name__ in ("LSTM", "GRU")][0]
    for l in model.layers[1:model.layers.index(rnn)]:  # pre-RNN layers (e.g. 1x1 Conv1D reduction)
        if l.get_weights():
            orig = l.get_weights()
            l.set_weights([q(a) for a in orig])
            x = q(np.asarray(l(x.astype(np.float32))))
            l.set_weights(orig)
        else:
            x = q(np.asarray(l(x.astype(np.float32))))
    Wk, Uk, bk = [q(a) for a in rnn.get_weights()]
    N, T, _ = x.shape
    u = rnn.units
    h = np.zeros((N, u)); c = np.zeros((N, u))
    for t in range(T):
        xt = x[:, t]
        if rnn.__class__.__name__ == "LSTM":
            z = q(q(xt @ Wk) + q(h @ Uk) + bk)
            ig, fg, cg, og = (q(sig(z[:, :u])), q(sig(z[:, u:2 * u])), q(np.tanh(z[:, 2 * u:3 * u])),
                              q(sig(z[:, 3 * u:])))
            c = q(fg * c + ig * cg)
            h = q(og * q(np.tanh(c)))
        else:
            bx, bh = (bk[0], bk[1]) if bk.ndim == 2 else (bk, np.zeros_like(bk))
            mx = q(xt @ Wk + bx); mh = q(h @ Uk + bh)
            zg = q(sig(mx[:, :u] + mh[:, :u])); rg = q(sig(mx[:, u:2 * u] + mh[:, u:2 * u]))
            hh = q(np.tanh(mx[:, 2 * u:] + q(rg * mh[:, 2 * u:])))
            h = q(zg * h + (1 - zg) * hh)
    y = h
    after = model.layers[model.layers.index(rnn) + 1:]
    for l in after:
        cls = l.__class__.__name__
        if cls == "Dropout":
            continue
        Wd, bd = [q(a) for a in l.get_weights()]
        z = q(y @ Wd + bd)
        act = getattr(getattr(l, "activation", None), "__name__", "linear")
        y = sig(z) if act == "sigmoid" else (q(np.maximum(z, 0)) if act == "relu" else z)
    return y.ravel()


def write_project(model, run, var, name, Xtb, ytb, is_qkeras=False, testset=None):
    import hls4ml
    import keras
    d = os.path.join(OUT, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    kw = dict(granularity="name", backend="Vitis")
    if not is_qkeras:
        kw["default_precision"] = full_prec(var["prec"])
    _v = keras.__version__
    if is_qkeras:
        keras.__version__ = "2.15.0"  # route tf_keras/QKeras models to hls4ml's Keras-2 frontend
    try:
        cfg = hls4ml.utils.config_from_keras_model(model, **kw)
        cfg["Model"]["Strategy"] = var["strategy"]
        cfg["Model"]["ReuseFactor"] = var["rf"]
        for lname, lc in cfg["LayerName"].items():
            lc["ReuseFactor"] = var["rf"]
            if "static" in var and any(k in lname for k in ("rnn", "lstm", "gru")):
                lc["static"] = var["static"]
            if lname == "input_data":
                lc.setdefault("Precision", {})["result"] = INPUT_PREC
            elif not is_qkeras and isinstance(lc.get("Precision"), dict):
                for k in ("weight", "bias", "result", "recurrent_weight"):
                    if k in lc["Precision"]:
                        lc["Precision"][k] = full_prec(var["prec"])
        hm = hls4ml.converters.convert_from_keras_model(
            model, hls_config=cfg, output_dir=d, project_name="frb", part=PART, backend="Vitis",
            io_type=var["io"], clock_period=CLOCK_NS, input_data_tb=None, output_data_tb=None)
        hm.write()
    finally:
        keras.__version__ = _v
    # test vectors for the csim smoke test inside synthesis
    tb = os.path.join(d, "tb_data")
    os.makedirs(tb, exist_ok=True)
    np.savetxt(os.path.join(tb, "tb_input_features.dat"), Xtb.reshape(len(Xtb), -1), fmt="%.6f")
    np.savetxt(os.path.join(tb, "tb_output_predictions.dat"), ytb.reshape(-1, 1), fmt="%.6f")
    # model + full test set for the bit-accurate C-sim accuracy check on EC2
    shutil.copy(os.path.join(ft.ROOT, run["model_path"]), os.path.join(d, "source_model" + os.path.splitext(run["model_path"])[1]))
    json.dump(dict(run_id=run["run_id"], variant=var, qkeras=is_qkeras, input_shape=run["input_shape"],
                   kind=run["cfg"]["kind"], float_test_acc=run.get("float_test_acc", run["test_acc"]), testset=testset,
                   input_clip=run["cfg"].get("clip")),
              open(os.path.join(d, "frb_export.json"), "w"), indent=1)
    # same options as hls4ml's VitisBackend.build(reset=True, csim=False, synth=True): csynth only
    with open(os.path.join(d, "build_opt.tcl"), "w", newline="\n") as f:
        f.write("array set opt {\n    reset      1\n    csim       0\n    synth      1\n    cosim      0\n"
                "    validation 0\n    export     0\n    vsynth     0\n    fifo_opt   0\n}\n")
    with open(os.path.join(d, "run_vitis_synthesis.sh"), "w", newline="\n") as f:
        f.write(textwrap.dedent(f"""\
            #!/bin/bash
            # csynth only (no cosim / no Vivado implementation), killed after 20 min -> "timeout".
            # Wrapper pattern copied from fpga_handoff/*/run_synthesis.sh (Vitis 2025.1, {PART}, {CLOCK_NS} ns);
            # command identical to hls4ml VitisBackend.build().
            cd "$(dirname "$0")"
            source /opt/Xilinx/Vitis/2025.1/settings64.sh 2>/dev/null || source /opt/Xilinx/2025.1/Vitis/settings64.sh 2>/dev/null || true
            start=$(date +%s)
            timeout 1200 vitis-run --tcl build_prj.tcl --mode hls > synthesis.log 2>&1
            status=$?
            end=$(date +%s)
            if [ $status -eq 124 ]; then s=timeout; elif [ $status -eq 0 ]; then s=ok; else s=error_$status; fi
            echo "{name},$s,$((end-start))" > synth_status.csv
            """))
    return d


def main(plan_path):
    plan = json.load(open(plan_path))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for item in plan:
        run = json.load(open(os.path.join(ft.RUNS, f"{item['run_id']}.json")))
        cfg = run["cfg"]
        is_q = cfg.get("quant", "float32").startswith("qkeras")
        X, meta = ft.load_data(cfg["t_pool"], cfg["f_pool"])
        idx = ft.split_indices(meta, cfg.get("split", "primary"), cfg.get("fold"))
        y = meta.label.values
        Xte = X[idx["test"]]
        Xte_in = Xte[..., None] if cfg["kind"] == "cnn" else Xte
        if is_q:
            import tensorflow as tf
            from qkeras.utils import _add_supported_quantized_objects
            co = {}
            _add_supported_quantized_objects(co)
            model = tf.keras.models.load_model(os.path.join(ft.ROOT, run["model_path"]), custom_objects=co)
        else:
            model = load_float(run)
        pf = ft.predict(model, Xte_in)
        # save full test set once per input shape for check_accuracy.py (C-sim on EC2)
        tsdir = os.path.join(OUT, "_testsets")
        os.makedirs(tsdir, exist_ok=True)
        tsname = f"test_t{cfg['t_pool']}_f{cfg['f_pool']}_{cfg['kind']}"
        if not os.path.exists(os.path.join(tsdir, tsname + "_X.npy")):
            np.save(os.path.join(tsdir, tsname + "_X.npy"), Xte_in.astype(np.float32))
            np.save(os.path.join(tsdir, tsname + "_y.npy"), y[idx["test"]].astype(np.float32))
        for var in item["variants"]:
            name = f"{item['run_id']}__{var['strategy'][:3]}_rf{var['rf']}_{var['prec'].replace('ap_fixed<', 'w').replace(',', '_').replace('>', '') if not is_q else 'qkeras'}"
            emu_acc = None
            if not is_q and cfg["kind"] == "cnn":
                pe = emulate(model, Xte_in, var["prec"])
                emu_acc = float(((pe >= 0.5) == y[idx["test"]]).mean())
            elif is_q:
                emu_acc = float(((pf >= 0.5) == y[idx["test"]]).mean())  # QKeras forward = quantised model
            elif cfg["kind"] == "rnn":
                pe = emulate_rnn(model, Xte_in, var["prec"])
                emu_acc = float(((pe >= 0.5) == y[idx["test"]]).mean())
            try:
                write_project(model, run, var, name, Xte_in[:N_TB], pf[:N_TB], is_qkeras=is_q, testset=tsname)
                st = "written"
            except Exception as e:
                st = f"convert_error: {type(e).__name__}: {str(e)[:300]}"
            rows.append(dict(project=name, run_id=item["run_id"], kind=cfg["kind"], quant=cfg.get("quant", "float32"),
                             precision="qkeras" if is_q else var["prec"], reuse_factor=var["rf"],
                             strategy=var["strategy"], io_type=var["io"], params=run["params"], macs=run["macs"],
                             float_test_acc=run.get("float_test_acc", run["test_acc"]), emu_test_acc=emu_acc,
                             testset=tsname, status=st))
            print(rows[-1], flush=True)
    import pandas as pd
    out = os.path.join(ft.ROOT, "results", "hls_projects_index.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat([old[~old.project.isin(df.project)], df])
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1])
