"""Stage 0e: cross-file MIXED training pool for one split (training files only).

Why: with same-file samples the CNN memorises each training file's single burst
template + backgrounds (val loss bottoms out at epoch 2-3 and then explodes).
Here every sample takes its background from one TRAINING file and (for FRB)
its burst template from another TRAINING file of the same split, so no val/test
file is ever touched. Augmentations applied identically to both classes:
  * random sub-band mask (p=0.35): keep a contiguous 15-60 % slice of the live
    band, flag the rest (teaches band-limited bursts / narrowband files)
  * pol, window, time-reversal as in v7_build_dataset.py
Val/test stay the same-file samples of held-out files (protocol unchanged).

Usage: python scripts/v7_build_mixed.py <split> [fold] [n_per_class]
Output: augmented_dataset/v7/mix_<split>[_f<fold>]/{X.npy, meta.csv}
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from v7_build_dataset import (CACHE, DM_RANGE, OUT, SNR_RANGE, burst_window, inject,  # noqa: E402
                              inpaint_track, make_template, window)
from v7_common import PATCH_T, channel_normalise  # noqa: E402
from v7_fit_bursts import load_work  # noqa: E402

P_SUBBAND = 0.35
P_RFI = float(os.environ.get("MIX_P_RFI", "0"))  # synthetic RFI on BOTH classes (hard negatives)


def add_rfi(patch, vmask, live, rng):
    """Narrowband dashes (1-3 ch, 3-80 bins, 2-12 sigma), persistent narrowband lines
    and broadband zero-DM spikes (1-3 bins, 0.5-3 sigma per channel). Added before
    the final per-channel normalisation, identically distributed for both classes."""
    T, C = patch.shape
    lc = np.where(live)[0]
    if len(lc) == 0:
        return patch, []
    kinds = []
    for _ in range(int(rng.integers(1, 5))):
        r = rng.random()
        if r < 0.55:
            c0 = int(rng.choice(lc)); w = int(rng.integers(1, 4)); L = int(rng.integers(3, 81))
            t0 = int(rng.integers(0, T - L)); a = rng.uniform(2, 12)
            patch[t0:t0 + L, c0:c0 + w] += a; kinds.append("dash")
        elif r < 0.8:
            c0 = int(rng.choice(lc)); patch[:, c0] += rng.uniform(0.5, 3) * (1 + 0.3 * rng.standard_normal(T))
            kinds.append("line")
        else:
            t0 = int(rng.integers(0, T - 3)); L = int(rng.integers(1, 4))
            patch[t0:t0 + L, live] += rng.uniform(0.5, 3); kinds.append("spike")
    return patch, kinds


def prepare(fn, fit, rng):
    path = os.path.join(CACHE, fn.replace(".h5", ".npz"))
    xs, vs, lives = [], [], []
    for beam in (0, 1):
        x, live, valid = load_work(path, beam, return_valid=True)
        xs.append(x); vs.append(valid); lives.append(live)
    live = lives[0] & lives[1]
    dd, lo, hi = burst_window(xs[0] + xs[1], live, fit)
    tpl = make_template(dd, live, lo, hi)
    bgs = [inpaint_track(xs[b], vs[b], live, fit, lo, hi, rng) for b in (0, 1)]
    return dict(bgs=bgs, valids=vs, live=live, tpl=tpl)


def main(split, fold=None, n_per_class=12000, seed=777):
    rng = np.random.default_rng(seed + (0 if fold is None else 100 + int(fold)))
    sp = json.load(open("results/splits/v7_splits.json"))
    s = sp[split] if fold is None else sp[split][int(fold)]
    g2f = {int(k): v for k, v in sp["group_to_file"].items()}
    fits = json.load(open(os.path.join(CACHE, "burst_fits.json")))
    train_groups = sorted(s["train"])
    prep = {g: prepare(g2f[g], fits[g2f[g]], rng) for g in train_groups}
    name = f"mix_{split}" + ("" if fold is None else f"_f{fold}") + ("_rfi" if P_RFI > 0 else "")
    od = os.path.join(OUT, name)
    os.makedirs(od, exist_ok=True)
    N = 2 * n_per_class
    X = np.lib.format.open_memmap(os.path.join(od, "X.npy.tmp"), "w+", np.float16, (N, PATCH_T, 256))
    meta, k = [], 0
    for label in (0, 1):
        n_done = 0
        while n_done < n_per_class:
            gb = int(rng.choice(train_groups))
            B = prep[gb]
            beam = int(rng.integers(0, 2))
            patch, vmask, st = window(B["bgs"][beam], B["valids"][beam], rng)
            rev = bool(rng.integers(0, 2))
            if rev:
                patch, vmask = patch[::-1], vmask[::-1]
            live = B["live"].copy()
            sub = rng.random() < P_SUBBAND
            if sub:
                lc = np.where(live)[0]
                w = max(8, int(len(lc) * rng.uniform(0.15, 0.6)))
                a = int(rng.integers(0, max(1, len(lc) - w + 1)))
                keep = np.zeros_like(live)
                keep[lc[a:a + w]] = True
                vmask = vmask & keep[None, :]
                live = live & keep
            rfi = []
            if P_RFI > 0 and rng.random() < P_RFI:
                patch, rfi = add_rfi(patch.copy(), vmask, live, rng)
            row = dict(idx=k, label=label, bg_group=gb, tpl_group=-1, beam=beam, bg_start=st, reversed=rev,
                       subband=sub, rfi="+".join(rfi), snr=np.nan, dm_res=np.nan, t_centre=np.nan, peak_amp=np.nan)
            if label == 1:
                gt = int(rng.choice(train_groups))
                dm = rng.uniform(*DM_RANGE)
                tc = rng.uniform(0.15, 0.85) * PATCH_T
                snr = float(np.exp(rng.uniform(np.log(SNR_RANGE[0]), np.log(SNR_RANGE[1]))))
                inj, snr, peak = inject(patch.copy(), prep[gt]["tpl"], live & (vmask.sum(0) >= 16), dm, tc, snr, vmask)
                if inj is None:
                    continue
                patch = inj
                row.update(tpl_group=gt, snr=snr, dm_res=dm, t_centre=tc, peak_amp=peak)
            X[k] = channel_normalise(patch, live, vmask).clip(-20, 60)
            meta.append(row)
            k += 1
            n_done += 1
        print(f"label {label} done", flush=True)
    X.flush(); del X
    os.replace(os.path.join(od, "X.npy.tmp"), os.path.join(od, "X.npy"))
    pd.DataFrame(meta).to_csv(os.path.join(od, "meta.csv"), index=False)
    json.dump(dict(split=split, fold=fold, seed=seed, train_groups=train_groups, n_per_class=n_per_class,
                   p_subband=P_SUBBAND, p_rfi=P_RFI), open(os.path.join(od, "info.json"), "w"), indent=1)
    print("saved", od, k)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else 12000)
