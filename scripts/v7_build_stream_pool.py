"""Stage 2 training option: STREAM-SAMPLED training pool (train files of one split only).

Each sample is one causal 256-bin window cut from a synthetic stream of a TRAIN file's real
(in-painted) background. With p=0.5 a burst (template from any TRAIN file) is placed so that
its first arrival lies anywhere in [-span, 256) relative to the window start, i.e. the window
can contain the trailing part, the whole sweep, or only the LEADING part (first-arriving
channels) -- exactly what a causal sliding window sees while a burst is arriving.
Orientation: 'physical' (high-frequency first) or 'stored' (low first), 50/50.
Label: 1 if the in-window part of the burst has S/N_in = S/N_total*sqrt(E_in/E_total) >= 5,
       0 if no burst energy is in the window; windows with 0 < S/N_in < 5 are dropped
       (ambiguous). meta.snr = S/N_in, meta.frac_band = fraction of live channels whose peak
       arrival is inside the window.
Same augmentation as mix_primary_rfi: sub-band masking p=0.35, synthetic RFI p=0.5 (both classes).
Usage: python scripts/v7_build_stream_pool.py <split> [n_per_class]
Output: augmented_dataset/v7/stream_<split>/{X.npy, meta.csv, info.json}
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from stream_sim import WIN, prepare_file  # noqa: E402
from v7_build_dataset import OUT, window  # noqa: E402
from v7_build_mixed import add_rfi  # noqa: E402
from v7_common import channel_normalise, delays_bins, shift_channels  # noqa: E402

SNR_TOT = (3.0, 50.0)
ADM = (0.5, 12.0)
SNR_IN_MIN = 5.0


def main(split, n_per_class=12000, seed=4242):
    rng = np.random.default_rng(seed)
    sp = json.load(open("results/splits/v7_splits.json"))
    g2f = {int(k): v for k, v in sp["group_to_file"].items()}
    groups = sorted(sp[split]["train"])
    P = {g: prepare_file(g2f[g], rng) for g in groups}
    od = os.path.join(OUT, f"stream_{split}")
    os.makedirs(od, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(od, "X.npy.tmp"), "w+", np.float16, (2 * n_per_class, WIN, 256))
    meta, k = [], 0
    counts = {0: 0, 1: 0}
    dropped = 0
    while min(counts.values()) < n_per_class:
        gb = int(rng.choice(groups)); B = P[gb]
        beam = int(rng.integers(0, 2))
        patch, vmask, st = window(B["bgs"][beam], B["valids"][beam], rng)
        if rng.integers(0, 2):
            patch, vmask = patch[::-1].copy(), vmask[::-1].copy()
        live = B["live"].copy()
        if rng.random() < 0.35:
            lc = np.where(live)[0]
            w = max(8, int(len(lc) * rng.uniform(0.15, 0.6)))
            a = int(rng.integers(0, max(1, len(lc) - w + 1)))
            keep = np.zeros_like(live); keep[lc[a:a + w]] = True
            vmask = vmask & keep[None, :]; live = live & keep
        live = live & (vmask.sum(0) >= 16)
        row = dict(bg_group=gb, tpl_group=-1, orientation="", snr=np.nan, snr_total=np.nan, frac_band=0.0, dm_res=np.nan)
        want_burst = rng.random() < 0.5
        label = 0
        if want_burst and live.any():
            gt = int(rng.choice(groups)); tpl = P[gt]["tpl"]
            orient = "physical" if rng.random() < 0.5 else "stored"
            dm = (1 if orient == "physical" else -1) * rng.uniform(*ADM)
            d = delays_bins(dm); d = d - d[live].min()
            span = int(np.ceil(d[live].max())) + tpl.shape[0] + 2
            t_a = int(rng.integers(-span + 1, WIN))  # first arrival relative to window start
            full = np.zeros((span, 256), np.float32)
            full[:tpl.shape[0]] = tpl
            full = shift_channels(full, d)
            full[:, ~live] = 0
            E_tot = (full ** 2).sum()
            # crop to window
            a0, a1 = max(0, -t_a), min(span, WIN - t_a)
            canvas = np.zeros((WIN, 256), np.float32)
            if a1 > a0:
                canvas[t_a + a0:t_a + a1] = full[a0:a1]
            canvas *= vmask
            E_in = (canvas ** 2).sum()
            snr_tot = float(np.exp(rng.uniform(*np.log(SNR_TOT))))
            if E_tot <= 0:
                continue
            snr_in = snr_tot * np.sqrt(E_in / E_tot)
            if E_in > 0 and snr_in < SNR_IN_MIN:
                dropped += 1
                continue
            if E_in > 0:
                patch = patch + canvas * (snr_tot / np.sqrt(E_tot))
                peak = int(np.argmax(tpl.sum(1)))
                arr = t_a + peak + d[live]
                label = 1
                row.update(tpl_group=gt, orientation=orient, snr=float(snr_in), snr_total=snr_tot,
                           frac_band=float(((arr >= 0) & (arr < WIN)).mean()), dm_res=float(dm))
        if counts[label] >= n_per_class:
            continue
        if rng.random() < 0.5:
            patch, _ = add_rfi(patch.copy(), vmask, live, rng)
        X[k] = channel_normalise(patch, live, vmask).clip(-20, 60)
        row.update(idx=k, label=label)
        meta.append(row)
        counts[label] += 1
        k += 1
    X.flush(); del X
    os.replace(os.path.join(od, "X.npy.tmp"), os.path.join(od, "X.npy"))
    m = pd.DataFrame(meta)
    m.to_csv(os.path.join(od, "meta.csv"), index=False)
    json.dump(dict(split=split, seed=seed, train_groups=groups, n_per_class=n_per_class, snr_in_min=SNR_IN_MIN,
                   dropped_ambiguous=dropped), open(os.path.join(od, "info.json"), "w"), indent=1)
    f = m[m.label == 1]
    print("saved", od, k, "dropped", dropped, "| positives with <50% of band in window:",
          round(float((f.frac_band < 0.5).mean()), 3), "| median frac_band", round(float(f.frac_band.median()), 3))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12000)
