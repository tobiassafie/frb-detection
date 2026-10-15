"""Bit-accurate C-simulation accuracy of one exported project (needs g++, hls4ml 1.3, TF).
   python3 check_accuracy.py <project_dir> [n_samples]      -> appends to csim_accuracy.csv
Rebuilds the hls4ml model from the project's own hls4ml_config.yml (same precision / RF / strategy),
compiles the C++ emulation library and runs the file-grouped v7 TEST split through it."""
import csv
import json
import os
import sys

import numpy as np

proj = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else None
meta = json.load(open(os.path.join(proj, "frb_export.json")))
if meta["qkeras"]:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
import hls4ml  # noqa: E402
import yaml  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(proj, "hls4ml_config.yml")).read().replace("!keras_model", ""))
cfg["OutputDir"] = proj
src = [f for f in os.listdir(proj) if f.startswith("source_model")][0]
cfg["KerasModel"] = os.path.join(proj, src)
if meta["qkeras"]:
    import keras
    keras.__version__ = "2.15.0"
hm = hls4ml.converters.keras_to_hls(cfg) if not meta["qkeras"] else hls4ml.converters.keras_v2_to_hls(cfg)
hm.compile()
X = np.load(os.path.join("_testsets", meta["testset"] + "_X.npy"))  # recorded by scripts/hls_export.py
y = np.load(os.path.join("_testsets", meta["testset"] + "_y.npy"))
if n:
    X, y = X[:n], y[:n]
p = hm.predict(np.ascontiguousarray(X.astype(np.float32))).ravel()
acc = float(((p >= 0.5) == y).mean())
new = not os.path.exists("csim_accuracy.csv")
with open("csim_accuracy.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["project", "n", "csim_acc", "float_test_acc"])
    w.writerow([proj, len(y), acc, meta["float_test_acc"]])
print(proj, "C-sim acc", acc, "float", meta["float_test_acc"])
