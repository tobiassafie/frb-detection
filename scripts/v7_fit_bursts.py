"""Stage 0b: for every cached file, put it on the common 256-channel grid,
fit the residual dispersion sweep (incoherent dedispersion search over a
residual DM that may be negative), and record burst time / width / S/N.

Writes augmented_dataset/v7_cache/burst_fits.json and a diagnostic figure.
"""
import glob
import json
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from v7_common import (EDGE_TRIM, FREQ_256, WORK_REBIN, channel_normalise,  # noqa: E402
                       delays_bins, live_mask, rebin_time, to_common_grid)

CACHE = "augmented_dataset/v7_cache"
DM_TRIALS = np.arange(-25, -0.5, 0.05)  # every visually-confirmed sweep has negative residual DM; excludes zero-DM RFI
WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]


def load_work(path, beam, return_valid=False):
    d = np.load(path)
    p = rebin_time(d["power"][beam], WORK_REBIN)
    p = to_common_grid(p, d["freq"])
    p = p[EDGE_TRIM:-EDGE_TRIM]
    live = live_mask(p)
    valid = np.isfinite(p) & (p != 0)
    if return_valid:
        return channel_normalise(p, live, valid), live, valid
    return channel_normalise(p, live, valid), live


def dedisp_ts(x, live, dm):
    """Integer-shift dedispersion (undo delays) and average live channels."""
    T = x.shape[0]
    sh = np.clip(np.round(delays_bins(dm)).astype(int), -T + 1, T - 1)
    acc = np.zeros(3 * T, np.float32)
    cnt = np.zeros(3 * T, np.float32)
    lc = np.where(live)[0]
    for c in lc:
        s = sh[c]
        acc[T - s:2 * T - s] += x[:, c]
        cnt[T - s:2 * T - s] += 1
    acc, cnt = acc[T:2 * T], cnt[T:2 * T]
    ok = cnt >= 0.8 * max(len(lc), 1)   # only bins where >=80% of channels overlap
    # renormalise so the noise std is ~1/sqrt(n) regardless of coverage
    return np.where(ok, acc / np.maximum(cnt, 1), np.nan)


def best_snr(ts):
    """Boxcar matched filter; robust noise from median/MAD of the series."""
    best = (-np.inf, 0, 1)
    valid = np.isfinite(ts)
    if valid.sum() < 64:
        return best
    idx = np.where(valid)[0]
    seg = ts[idx[0]:idx[-1] + 1]
    seg = np.where(np.isfinite(seg), seg, np.nanmedian(seg))
    for w in WIDTHS:
        if w >= len(seg) // 4:
            break
        sm = np.convolve(seg, np.ones(w) / np.sqrt(w), mode="valid")
        med = np.median(sm)
        sig = 1.4826 * np.median(np.abs(sm - med))
        if sig <= 0:
            continue
        k = int(np.argmax(sm))
        k_full = idx[0] + k + w // 2
        snr = (sm[k] - med) / sig
        if snr > best[0]:
            best = (snr, k_full, w)
    return best


def fit_file(path):
    fname = os.path.basename(path).replace(".npz", ".h5")
    x, live = load_work(path, 0)
    x1, _ = load_work(path, 1)
    x = x + x1  # both beams (pols) for the fit
    res = []
    for dm in DM_TRIALS:
        snr, k, w = best_snr(dedisp_ts(x, live, dm))
        res.append((snr, dm, k, w))
    snr, dm, k, w = max(res)
    # edge check: a fit whose peak sits within 3 bins of either end is suspect
    edge = k < 3 or k > x.shape[0] - 4
    return fname, dict(dm_res=float(dm), snr=float(snr), t_peak=int(k), width=int(w),
                       n_time=int(x.shape[0]), n_live=int(live.sum()),
                       f_hi=float(FREQ_256[live].max()) if live.any() else None,
                       f_lo=float(FREQ_256[live].min()) if live.any() else None,
                       edge_peak=bool(edge),
                       snr_curve=[float(r[0]) for r in res])


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
    out = {}
    with ProcessPoolExecutor(12) as ex:
        for fname, r in ex.map(fit_file, files):
            out[fname] = r
            print(f"{fname[:24]} dm_res={r['dm_res']:+6.1f} snr={r['snr']:7.1f} t={r['t_peak']:4d}/{r['n_time']} "
                  f"w={r['width']:3d} live={r['n_live']:3d} band={r['f_lo']:.0f}-{r['f_hi']:.0f} edge={r['edge_peak']}",
                  flush=True)
    json.dump(out, open(os.path.join(CACHE, "burst_fits.json"), "w"), indent=1)
