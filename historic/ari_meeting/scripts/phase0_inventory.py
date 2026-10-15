"""
Phase 0 inventory script.
Walks the repo (excluding .venv, .git, node_modules) and records, for each
relevant file: path, size, mtime.
For model files (.h5, .keras), attempts to load architecture via Keras and
records input shape, layer stack, output shape, param count.
For hls4ml project dirs, records config + weight file listing so they can be
mapped to a model.

Read-only: does not modify any file.
"""
import os
import sys
import json
import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXCLUDE_DIRS = {".venv", ".git", "node_modules", "__pycache__"}

MODEL_EXTS = (".h5", ".hdf5", ".keras", ".pth", ".pkl", ".onnx")
DATASET_EXTS = (".npz", ".npy")
SCRIPT_EXTS_ARCH_HINT = ("train_cnn", "train_", "architecture")

def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def walk_files():
    results = {
        "models": [],
        "datasets": [],
        "notebooks": [],
        "train_scripts": [],
        "hls4ml_dirs": [],
        "vitis_reports": [],
        "tcl_scripts": [],
        "raw_chime": [],
    }
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel_dir = os.path.relpath(dirpath, ROOT)
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            try:
                st = os.stat(full)
            except OSError:
                continue
            size = st.st_size
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()
            entry = {"path": rel.replace("\\", "/"), "size": size, "size_h": human(size), "mtime": mtime}

            lower = fn.lower()
            if lower.endswith(MODEL_EXTS):
                results["models"].append(entry)
            elif lower.endswith(DATASET_EXTS):
                results["datasets"].append(entry)
            elif lower.endswith(".ipynb"):
                results["notebooks"].append(entry)
            elif lower == "hls4ml_config.yml":
                results["hls4ml_dirs"].append({"config": entry, "dir": rel_dir.replace("\\", "/")})
            elif "csynth" in lower and (lower.endswith(".rpt") or lower.endswith(".xml")):
                results["vitis_reports"].append(entry)
            elif lower.endswith(".tcl") and ("run_synthesis" in lower or "build_prj" in lower or "vivado_synth" in lower):
                results["tcl_scripts"].append(entry)
            elif fn.startswith("train_cnn") and lower.endswith(".py"):
                results["train_scripts"].append(entry)
            elif ("beamformed" in lower or "chime" in lower) and lower.endswith((".h5", ".hdf5")):
                results["raw_chime"].append(entry)
    return results


def try_load_keras(path):
    """Attempt to load a .h5/.keras model and extract architecture summary."""
    info = {"loadable": False, "error": None}
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path, compile=False)
        info["loadable"] = True
        info["input_shape"] = str(model.input_shape)
        info["output_shape"] = str(model.output_shape)
        info["total_params"] = int(model.count_params())
        trainable = int(sum(int(tf.keras.backend.count_params(w)) for w in model.trainable_weights))
        info["trainable_params"] = trainable
        info["non_trainable_params"] = info["total_params"] - trainable
        layers = []
        for layer in model.layers:
            try:
                out_shape = str(layer.output_shape)
            except Exception:
                out_shape = "?"
            layers.append({
                "name": layer.name,
                "class": layer.__class__.__name__,
                "output_shape": out_shape,
                "params": int(layer.count_params()),
                "config": {k: v for k, v in layer.get_config().items()
                           if k in ("filters", "kernel_size", "pool_size", "strides",
                                     "units", "activation", "padding")},
            })
        info["layers"] = layers
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def try_load_h5_raw(path):
    """Fallback: inspect an HDF5 file's raw structure (for weight-only or
    non-standard-Keras h5 files that tf.keras can't load)."""
    info = {"keys": [], "attrs": {}}
    try:
        import h5py
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info["keys"].append({"path": name, "shape": obj.shape, "dtype": str(obj.dtype)})
            f.visititems(visit)
            info["attrs"] = {k: str(v) for k, v in f.attrs.items()}
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def main():
    inv = walk_files()

    # Sort by path for stability
    for k in inv:
        inv[k].sort(key=lambda e: e.get("dir", e.get("path", "")))

    # Try loading each model file (skip huge duplicated hls4ml keras_model.keras copies individually,
    # but still record metadata; architecture-load them too since they're small).
    model_arch = {}
    for entry in inv["models"]:
        path = entry["path"]
        full = os.path.join(ROOT, path)
        if path.lower().endswith((".h5", ".keras")):
            print(f"Loading {path} ...", file=sys.stderr)
            arch = try_load_keras(full)
            if not arch["loadable"] and path.lower().endswith(".h5"):
                arch["raw_h5"] = try_load_h5_raw(full)
            model_arch[path] = arch

    out = {
        "root": ROOT,
        "generated": datetime.datetime.now().isoformat(),
        "inventory": inv,
        "model_architectures": model_arch,
    }

    out_path = os.path.join(ROOT, "results", "ari_meeting", "phase0_raw.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
