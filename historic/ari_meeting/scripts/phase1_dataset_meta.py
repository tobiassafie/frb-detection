"""
Phase 1 dataset metadata script. Reads only shapes/dtypes/keys from datasets,
never loads full arrays into memory. Read-only.
"""
import json
import os
import sys
import zipfile

import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def npz_meta(path):
    """Inspect an .npz's member arrays' shape/dtype without loading data,
    by reading each member's .npy header directly from the zip."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                key = name[:-4]
                out[key] = {"shape": shape, "dtype": str(dtype)}
    return out


def hdf5_meta(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}
        f.visititems(visit)
        attrs = {k: str(v) for k, v in f.attrs.items()}
    return out, attrs


def npy_meta(path):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return {"shape": shape, "dtype": str(dtype)}


def main():
    targets = {
        "lstm_dataset_v6.npz": "augmented_dataset/lstm_dataset_v6.npz",
        "fpga_handoff_X_test.npy": "fpga_handoff/X_test.npy",
        "fpga_handoff_y_test.npy": "fpga_handoff/y_test.npy",
        "classifier_train_data.hdf5": "classifier/fetch/train_data.hdf5",
        "classifier_test_data.hdf5": "classifier/fetch/test_data.hdf5",
    }
    result = {}
    for label, rel in targets.items():
        full = os.path.join(ROOT, rel)
        if not os.path.exists(full):
            result[label] = {"error": "file not found", "path": rel}
            continue
        size = os.path.getsize(full)
        try:
            if rel.endswith(".npz"):
                meta = npz_meta(full)
            elif rel.endswith(".npy"):
                meta = npy_meta(full)
            elif rel.endswith((".hdf5", ".h5")):
                meta, attrs = hdf5_meta(full)
                result[label] = {"path": rel, "size_bytes": size, "keys": meta, "attrs": attrs}
                continue
            result[label] = {"path": rel, "size_bytes": size, "keys": meta}
        except Exception as e:
            result[label] = {"path": rel, "size_bytes": size, "error": f"{type(e).__name__}: {e}"}

    out_path = os.path.join(ROOT, "results", "ari_meeting", "phase1_dataset_meta.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
