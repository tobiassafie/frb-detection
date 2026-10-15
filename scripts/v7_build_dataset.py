"""Stage 0c: build the v7 FRB-vs-noise dataset with full per-sample provenance,
so train/val/test can be split by SOURCE FILE (= one real CHIME burst).

Per usable source file (both beams/pols):
  * working array  : 0.655 ms x 256 channels on the common 800->400 MHz grid,
                     robust per-channel normalised (unit noise per channel).
  * burst template : dedispersed at the fitted residual DM, rank-1 SVD
                     (spectrum x profile) of the burst window -> denoised.
  * background     : working array with the burst track in-painted by N(0,1)
                     -> real burst-free noise with the file's RFI / flagging.
Samples (all from ONE file, so both classes share its backgrounds exactly):
  * noise : random 256-bin window of a background (random beam, random
            time-reversal).
  * FRB   : same kind of window + template redispersed at a residual DM drawn
            U(DM_RANGE), placed at a random time, scaled to an optimal
            (matched-filter) S/N drawn log-uniform in SNR_RANGE, then the
            patch is re-normalised per channel exactly like the noise class.
Output: augmented_dataset/v7/{X.npy (float16, N x 256 x 256 [time, freq]),
        meta.csv, files.json}
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from v7_common import (PATCH_T, channel_normalise, delays_bins,  # noqa: E402
                       shift_channels)
from v7_fit_bursts import load_work  # noqa: E402

CACHE = "augmented_dataset/v7_cache"
OUT = "augmented_dataset/v7"
SEED = 20260930
N_PER_CLASS = 600                 # per source file
SNR_RANGE = (3.0, 50.0)           # optimal S/N, log-uniform; primary test metric uses S/N >= 7
DM_RANGE = (-12.0, -0.5)          # residual DM (pc/cm^3); observed fits span -1.4 .. -9.7
MIN_FIT_SNR = 8.0
EXCLUDE = {  # reason strings go into files.json and NOTES.md
    "FRB20190627D_42637411_beamformed.h5": "no visible burst; fit S/N 10 at max width 32 (spurious)",
    "FRB20190628C_42842438_beamformed.h5": "no visible burst; fit S/N 6 (spurious)",
}


def burst_window(x, live, fit):
    """Dedispersed band-averaged profile -> contiguous window around the peak where
    the smoothed profile exceeds 2 sigma, padded by 3 bins."""
    dd = shift_channels(x, -delays_bins(fit["dm_res"]))
    ts = dd[:, live].mean(1)
    w = max(1, fit["width"])
    sm = np.convolve(ts, np.ones(w) / w, mode="same")
    med = np.median(sm)
    sig = 1.4826 * np.median(np.abs(sm - med))
    t0 = fit["t_peak"]
    lo = t0
    while lo > 0 and sm[lo - 1] > med + 2 * sig:
        lo -= 1
    hi = t0
    while hi < len(sm) - 1 and sm[hi + 1] > med + 2 * sig:
        hi += 1
    lo, hi = max(0, lo - w - 3), min(len(sm), hi + w + 4)
    return dd, lo, hi


def make_template(dd, live, lo, hi):
    """Rank-1 SVD of the dedispersed burst window (time x live-chan), clipped >= 0."""
    win = dd[lo:hi][:, live]
    u, s, vt = np.linalg.svd(win, full_matrices=False)
    r1 = s[0] * np.outer(u[:, 0], vt[0])
    if r1.sum() < 0:
        r1 = -r1
    r1 = np.clip(r1, 0, None)
    tpl = np.zeros((hi - lo, dd.shape[1]), np.float32)
    tpl[:, live] = r1
    return tpl


def inpaint_track(x, valid, live, fit, lo, hi, rng):
    """Replace every VALID pixel of the burst track (+/-3 bins beyond the burst window) by N(0,1)."""
    bg = x.copy()
    d = delays_bins(fit["dm_res"])
    T = x.shape[0]
    for c in np.where(live)[0]:
        a = int(np.floor(lo + d[c])) - 3
        b = int(np.ceil(hi + d[c])) + 3
        a, b = max(0, a), min(T, b)
        if b > a:
            bg[a:b, c] = np.where(valid[a:b, c], rng.standard_normal(b - a), 0.0)
    return bg


def window(bg, valid, rng):
    T = bg.shape[0]
    if T < PATCH_T:  # short file: reflect-pad (both classes get the same treatment)
        bg = np.concatenate([bg, bg[::-1]], 0)
        valid = np.concatenate([valid, valid[::-1]], 0)
        T = bg.shape[0]
    s = rng.integers(0, T - PATCH_T + 1)
    return bg[s:s + PATCH_T], valid[s:s + PATCH_T], int(s)


def inject(patch, tpl, live, dm, t_centre, snr, vmask):
    """Place template (dedispersed, peak-aligned at 800 MHz ref) so the band-centre
    arrival is at t_centre, redisperse at dm, scale to optimal S/N = snr."""
    L = tpl.shape[0]
    canvas = np.zeros_like(patch)
    lc = np.where(live)[0]
    d = delays_bins(dm)
    d_mid = d[lc].mean()
    start = t_centre - d_mid - L / 2.0
    # place at integer part, then fractional + dispersion shift via interpolation
    s0 = int(np.floor(start))
    frac = start - s0
    a, b = max(0, s0), min(PATCH_T, s0 + L)
    if b <= a:
        return None, 0.0, 0.0
    canvas[a:b] = tpl[a - s0:b - s0]
    canvas = shift_channels(canvas, d + frac)
    canvas[:, ~live] = 0.0
    canvas[~vmask] = 0.0  # no signal where the instrument recorded no data
    e = np.sqrt((canvas ** 2).sum())
    if e <= 0:
        return None, 0.0, 0.0
    return patch + canvas * (snr / e), float(snr), float(canvas.max() * snr / e)


def main():
    rng = np.random.default_rng(SEED)
    fits = json.load(open(os.path.join(CACHE, "burst_fits.json")))
    os.makedirs(OUT, exist_ok=True)

    files, ok_files = {}, []
    for fn, fit in fits.items():
        reason = EXCLUDE.get(fn)
        if reason is None and fit["snr"] < MIN_FIT_SNR:
            reason = f"fit S/N {fit['snr']:.1f} < {MIN_FIT_SNR}"
        files[fn] = dict(fit={k: v for k, v in fit.items() if k != "snr_curve"}, used=reason is None,
                         reason=reason)
        if reason is None:
            ok_files.append(fn)
    files["FRB20190410B_36154661_beamformed.h5"] = dict(used=False, reason="truncated HDF5 (unreadable)")

    X = np.zeros((len(ok_files) * 2 * N_PER_CLASS, PATCH_T, 256), np.float16)
    meta = []
    k = 0
    for fid, fn in enumerate(ok_files):
        fit = fits[fn]
        path = os.path.join(CACHE, fn.replace(".h5", ".npz"))
        beams, bgs, valids = [], [], []
        for beam in (0, 1):
            x, live, valid = load_work(path, beam, return_valid=True)
            beams.append((x, live))
            valids.append(valid)
        # template from Stokes-I-like sum of both pols (higher S/N), background per pol
        xs = beams[0][0] + beams[1][0]
        live = beams[0][1] & beams[1][1]
        dd, lo, hi = burst_window(xs, live, fit)
        tpl = make_template(dd, live, lo, hi)
        for beam in (0, 1):
            bgs.append(inpaint_track(beams[beam][0], valids[beam], live, fit, lo, hi, rng))
        files[fn].update(template_len=int(hi - lo), n_live=int(live.sum()), group_id=fid)

        for label in (0, 1):
            n_done = 0
            while n_done < N_PER_CLASS:
                beam = int(rng.integers(0, 2))
                patch, vmask, s = window(bgs[beam], valids[beam], rng)
                rev = bool(rng.integers(0, 2))
                if rev:
                    patch, vmask = patch[::-1], vmask[::-1]
                row = dict(idx=k, file=fn, group_id=fid, label=label, beam=beam, bg_start=s, reversed=rev,
                           snr=np.nan, dm_res=np.nan, t_centre=np.nan, peak_amp=np.nan)
                if label == 1:
                    dm = rng.uniform(*DM_RANGE)
                    tc = rng.uniform(0.15, 0.85) * PATCH_T
                    snr = float(np.exp(rng.uniform(np.log(SNR_RANGE[0]), np.log(SNR_RANGE[1]))))
                    inj, snr, peak = inject(patch.copy(), tpl, live & (vmask.sum(0) >= 16), dm, tc, snr, vmask)
                    if inj is None:
                        continue
                    patch = inj
                    row.update(snr=snr, dm_res=dm, t_centre=tc, peak_amp=peak)
                X[k] = channel_normalise(patch, live, vmask).clip(-20, 60)
                meta.append(row)
                k += 1
                n_done += 1
        print(f"[{fid:2d}] {fn[:24]} T={bgs[0].shape[0]} live={live.sum()} tpl_len={hi-lo} done", flush=True)

    X = X[:k]
    if "--meta-only" in sys.argv:  # same seed & draw order -> identical X; only refresh meta.csv
        old = np.load(os.path.join(OUT, "X.npy"), mmap_mode="r")
        assert np.array_equal(np.asarray(old[:50]), X[:50]) and np.array_equal(np.asarray(old[-50:]), X[-50:])
    else:
        np.save(os.path.join(OUT, "X.npy"), X)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "meta.csv"), index=False)
    json.dump(dict(seed=SEED, n_per_class=N_PER_CLASS, snr_range=SNR_RANGE, dm_range=DM_RANGE,
                   files=files), open(os.path.join(OUT, "files.json"), "w"), indent=1, default=str)
    print("saved", X.shape)


if __name__ == "__main__":
    main()
