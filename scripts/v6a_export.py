"""Stage 4: hls4ml projects for the v6a winners (no Vitis, no C++ compiler on this machine ->
projects are written only; C-simulation not run; every hardware number is an `est_` estimate).

CNN: io_stream, Resource, RF 4.  RNN: io_parallel, static=True, Resource, RF 4.
Precision ap_fixed<16,6> (weights/bias/results AP_RND_CONV/AP_SAT; input ap_fixed<16,6>).
Writes hls_projects_v6a/<project>/ (+ _testsets for check_accuracy.py on EC2) and
results/v6a_hls_index.csv with float vs fixed-point-EMULATION accuracy on the v6 test set
(emulation = scripts/hls_export.py emulate/emulate_rnn; NOT C-sim).
Usage: python scripts/v6a_export.py run_id [run_id ...]
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import frb_train as ft  # noqa: E402
import hls_export as he  # noqa: E402
import v6a_train as va  # noqa: E402

he.OUT = os.path.join(va.ROOT, os.environ.get("EXPORT_DIR", "hls_projects_v6a"))
PREC = "ap_fixed<16,6>"
# EXPORT_SWEEP="4:ap_fixed<16,6>;16:ap_fixed<12,6>;..." -> one project per (rf, precision); default RF4 <16,6>
SWEEP = [(int(a), b) for a, b in (v.split(":") for v in os.environ.get("EXPORT_SWEEP", "4:ap_fixed<16,6>").split(";"))]
INDEX = os.environ.get("EXPORT_INDEX", "v6a_hls_index.csv")


def main(rids):
    import keras
    D = va.data()
    rows = []
    os.makedirs(os.path.join(he.OUT, "_testsets"), exist_ok=True)
    for rid in rids:
        rp = os.path.join(va.RUNS, f"{rid}.json")
        if not os.path.exists(rp):
            rp = os.path.join(ft.RUNS, f"{rid}.json")
        run = json.load(open(rp))
        cfg = run["cfg"]
        m0 = keras.models.load_model(os.path.join(va.ROOT, run["model_path"]), compile=False)
        model = keras.Model.from_config(json.loads(json.dumps(m0.get_config()).replace('"input"', '"input_data"')))
        model.set_weights(m0.get_weights())
        Xte = va.pool(D["X_test"], cfg.get("t_pool", 1), cfg.get("f_pool", 1), cfg.get("t_pool_type", "avg"))
        Xin = Xte[..., None] if cfg["kind"] == "cnn" else Xte
        y = D["y_test"]
        pf = ft.predict(model, Xin)
        ts = f"v6_test_t{cfg.get('t_pool', 1)}_f{cfg.get('f_pool', 1)}_{cfg['kind']}"
        np.save(os.path.join(he.OUT, "_testsets", ts + "_X.npy"), Xin.astype(np.float32))
        np.save(os.path.join(he.OUT, "_testsets", ts + "_y.npy"), y.astype(np.float32))
        emu = {}
        for rf, prec in SWEEP:
            var = dict(prec=prec, rf=rf, strategy="Resource", io="io_stream" if cfg["kind"] == "cnn" else "io_parallel")
            if cfg["kind"] != "cnn":
                var["static"] = True
            if prec not in emu:
                pe = he.emulate(model, Xin, prec) if cfg["kind"] == "cnn" else he.emulate_rnn(model, Xin, prec)
                emu[prec] = float(((pe >= 0.5) == y).mean())
            w, i = he.parse_prec(prec)
            name = f"{rid}__Res_rf{rf}_w{w}_{i}"
            run_for_writer = dict(run, cfg=dict(cfg), test_acc=run["test_acc"])
            st = "written"
            try:
                he.write_project(model, run_for_writer, var, name, Xin[:64], pf[:64], is_qkeras=False, testset=ts)
            except Exception as e:  # conversion failures are results too
                st = f"convert_error: {type(e).__name__}: {str(e)[:300]}"
            rows.append(dict(project=name, run_id=rid, kind=cfg["kind"], params=run["params"], macs=run.get("macs"),
                             precision=prec, reuse_factor=rf, strategy="Resource", io_type=var["io"],
                             static=var.get("static"), float_test_acc_v6=float(((pf >= 0.5) == y).mean()),
                             emu_test_acc_v6=emu[prec], status=st))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "table", "v6 benchmark - leaky split, comparison only")
    out = os.path.join(va.ROOT, "results", INDEX)
    if os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat([old[~old.project.isin(df.project)], df], ignore_index=True)
    df.to_csv(out, index=False)
    for f in ("run_all_synthesis.sh", "parse_synth_reports.py", "check_accuracy.py"):
        shutil.copy(os.path.join(va.ROOT, "hls_projects", f), os.path.join(he.OUT, f))


if __name__ == "__main__":
    main(sys.argv[1:])
