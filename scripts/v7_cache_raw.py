"""Stage 0a: cache every raw CHIME baseband file as a small downsampled
(time, freq) power array per beam, so the dataset builder never has to touch
the 34 GB of raw HDF5 again.

Output: augmented_dataset/v7_cache/<fname>.npz with
  power  : (2, n_time_ds, 903) float32, time-downsampled by CACHE_DS (nanmean)
  freq   : (903,) channel centre frequencies (MHz)
Downsample factor 64 (0.164 ms) so later stages can re-bin by 4 to reproduce
v6's 0.655 ms bins, or use other resolutions.
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

RAW_DIR = "realtime_data/FRBs"
OUT_DIR = "augmented_dataset/v7_cache"
CACHE_DS = 64


def cache_one(fpath):
    fname = os.path.basename(fpath)
    out = os.path.join(OUT_DIR, fname.replace(".h5", ".npz"))
    if os.path.exists(out):
        return fname, "cached"
    try:
        with h5py.File(fpath, "r") as f:
            ds = f["tiedbeam_power"]
            n_freq, n_beam, n_time = ds.shape
            n_trim = (n_time // CACHE_DS) * CACHE_DS
            beams = []
            for b in range(n_beam):
                p = ds[:, b, :n_trim].astype(np.float32)  # (freq, time)
                p[p == 0] = np.nan
                p = np.nanmean(p.reshape(n_freq, n_trim // CACHE_DS, CACHE_DS), axis=2)
                beams.append(p.T)  # (time, freq)
            freq = f["index_map/freq"]["centre"][:].astype(np.float32)
        np.savez_compressed(out, power=np.stack(beams), freq=freq)
        return fname, f"ok {np.stack(beams).shape}"
    except Exception as e:  # corrupted files are reported, not hidden
        return fname, f"FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.h5")))
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for fname, status in ex.map(cache_one, files):
            print(f"{fname:45s} {status}", flush=True)
