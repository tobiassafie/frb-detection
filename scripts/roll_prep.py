"""Rolling-LSTM prep: recover per-sample burst onset/end on the ORIGINAL v6 data (as on disk).

v6 stores no burst position (only dm_*). But every v6 FRB sample is
    x = normalize(bg_k + burst)      with bg_k one of the 5 unique noise images,
and the 5 noise images are stored normalised: N_k = (bg_k - m_k)/s_k. Hence exactly
    x = a*N_k + c + burst/s          (a = s_k/s, c = (m_k - m)/s).
Per FRB sample: least-squares fit x ~ a*N_k + c for each k (trimmed refits so burst pixels do not bias
the fit), keep the best k, residual r = burst/s. Per time bin e_t = sum_f r_tf^2.
  onset = first t with e_t > 1 % of max_t e_t,   end = last such t   (time bins of 0.655 ms).
Samples whose residual is ~0 everywhere (max e_t < 1e-6) contain NO burst: these are the FRB-labelled
samples that are pixel-identical to a noise image (expected 36 train / 10 val / 5 test) -> flagged bad.

Orientation: v6 time axis = index 0 earliest; frequency axis index 0 = HIGHEST frequency
(frb_augmentation_v6_with_dm.ipynb: load_spectrogram transposes (freq, time) as stored, CHIME
index_map/freq is 800 -> 400 MHz; redisperse() uses freqs = linspace(freq_hi, freq_lo, n_freq) and
delays K*DM*(f^-2 - f_hi^-2) >= 0 growing with channel index). So high frequencies already arrive first
along increasing time -> the stored order IS the physical arrival order; no flip needed.

Output: augmented_dataset/v6bench/roll_meta.npz (per split: onset, end, best_k, bad, a, c, resid_max)
and results/fig_roll_onsets.png.
"""
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REL = 0.01


def fit_one(x, Ns):
    xf = x.ravel().astype(np.float64)
    best = None
    for k, N in enumerate(Ns):
        nf = N.ravel().astype(np.float64)
        keep = np.ones_like(xf, bool)
        for _ in range(3):
            A = np.stack([nf[keep], np.ones(keep.sum())], 1)
            (a, c), *_ = np.linalg.lstsq(A, xf[keep], rcond=None)
            r = xf - a * nf - c
            thr = max(np.percentile(np.abs(r), 90) * 5, 1e-7)
            keep = np.abs(r) < thr
        rss = float(np.sum(r[keep] ** 2) / keep.sum())
        if best is None or rss < best[0]:
            best = (rss, k, a, c, r.reshape(x.shape))
    return best


def track_fit(r, ks=np.linspace(0.0, 2.0, 81)):
    """Find the dispersive sweep in the STORED orientation (low freq first) as a straight track
    t(ch) = t0 + k*(C-1-ch): the lowest channel (C-1) arrives at t0, channel 0 (highest freq) at t0 + (C-1)k.
    Residual robust-normalised per channel (removes steady RFI rows), clipped to [-3, 10].
    Score = sum of z along the in-patch part of the track / sqrt(n in-patch channels), >= 16 channels.
    Returns (score, t0, k)."""
    T, C = r.shape
    med = np.median(r, 0)
    mad = np.median(np.abs(r - med), 0) * 1.4826 + 1e-9
    z = np.clip((r - med) / mad, -3, 10).astype(np.float32)
    ch = np.arange(C)
    t0s = np.arange(-int(2.0 * (C - 1)) - 1, T)            # every start that leaves >= 1 bin in the patch
    best = (-np.inf, 0, 0.0)
    for k in ks:
        sh = np.round(k * (C - 1 - ch)).astype(int)        # (C,)
        tt = t0s[:, None] + sh[None, :]                     # (n_t0, C) time index of the track per channel
        inside = (tt >= 0) & (tt < T)
        vals = np.where(inside, z[np.clip(tt, 0, T - 1), ch[None, :]], 0.0)
        n = inside.sum(1)
        score = np.where(n >= 16, vals.sum(1) / np.sqrt(np.maximum(n, 1)), -np.inf)
        a = int(np.argmax(score))
        if score[a] > best[0]:
            best = (float(score[a]), int(t0s[a]), float(k), vals[a].copy(), inside[a].copy(), tt[a].copy())
    return best


def signal_extent(vals, inside, tt, T, min_z=1.5, smooth=9):
    """Channels that actually carry the sweep: along-track z smoothed over `smooth` channels > min_z
    (and inside the patch). Returns (onset, end) in the PHYSICAL (time-reversed) frame, i.e. earliest
    and latest physical arrival among those channels; falls back to all in-patch channels if none."""
    sm = np.convolve(vals, np.ones(smooth) / smooth, mode="same")
    sig = inside & (sm > min_z)
    if not sig.any():
        sig = inside
    tphys = (T - 1) - tt[sig]
    return int(np.clip(tphys.min(), 0, T - 1)), int(np.clip(tphys.max(), 0, T - 1)), int(sig.sum())


def main():
    d = np.load(os.path.join(ROOT, "augmented_dataset", "lstm_dataset_v6.npz"))
    # the 5 unique normalised noise images
    allN = np.concatenate([d[f"X_{s}"][d[f"y_{s}"] == 0] for s in ("train", "val", "test")])
    _, ui = np.unique(allN.reshape(len(allN), -1), axis=0, return_index=True)
    Ns = allN[np.sort(ui)]
    print("unique noise images:", len(Ns))
    out = {}
    for s in ("train", "val", "test"):
        X, y = d[f"X_{s}"], d[f"y_{s}"]
        n = len(y)
        onset = np.full(n, -1); end = np.full(n, -1); bad = np.zeros(n, bool); bk = np.full(n, -1)
        rmax = np.zeros(n)
        score = np.full(n, np.nan); t0s = np.full(n, -9999); slope = np.full(n, np.nan); nsig = np.zeros(n, int)
        for i in np.where(y == 1)[0]:
            rss, k, a, c, r = fit_one(X[i], Ns)
            e = (r ** 2).sum(1)
            rmax[i] = e.max()
            bk[i] = k
            if e.max() < 1e-6:          # pixel-identical to a noise image: no burst at all
                bad[i] = True
                continue
            sc, t0, kk, vals, inside, tt = track_fit(X[i].astype(np.float64))
            score[i], t0s[i], slope[i] = sc, t0, kk
            onset[i], end[i], nsig[i] = signal_extent(vals, inside, tt, X.shape[1])   # PHYSICAL frame
        out.update({f"{s}_onset": onset, f"{s}_end": end, f"{s}_bad": bad, f"{s}_best_k": bk, f"{s}_resid_max": rmax,
                    f"{s}_track_score": score, f"{s}_track_t0_stored": t0s, f"{s}_track_slope": slope,
                    f"{s}_n_signal_channels": nsig})
        f = y == 1
        print(f"{s}: FRB {f.sum()}, bad (no burst) {bad.sum()}, onset median {np.median(onset[f & ~bad]):.0f}, "
              f"duration median {np.median((end - onset)[f & ~bad]):.0f} bins, track score median "
              f"{np.nanmedian(score[f & ~bad]):.1f}, score<8: {int(np.sum(score[f & ~bad] < 8))}")
    np.savez(os.path.join(ROOT, "augmented_dataset", "v6bench", "roll_meta.npz"), **out)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y = d["X_test"], d["y_test"]
    idx = [i for i in np.where((y == 1) & ~out["test_bad"])[0][:5]]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), dpi=90)
    for j, i in enumerate(idx):
        xp = X[i][::-1]                                           # physical order: time reversed
        for row, img, ttl in ((0, X[i], "stored (low-f first)"), (1, xp, "PHYSICAL (time-reversed)")):
            ax = axes[row, j]
            ax.imshow(img.T, aspect="auto", origin="upper", cmap="viridis",
                      vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
            if row == 1:
                for t, col in ((out["test_onset"][i], "#eb6834"), (out["test_end"][i], "#1baf7a")):
                    ax.axvline(t, color=col, lw=1.2)
            else:
                t0, kk = out["test_track_t0_stored"][i], out["test_track_slope"][i]
                ch = np.arange(256); ax.plot(t0 + kk * (255 - ch), ch, color="#eb6834", lw=0.8, alpha=0.7)
                ax.set_xlim(0, 255)
            ax.set_title(f"test {i} {ttl}\nonset {out['test_onset'][i]} end {out['test_end'][i]} "
                         f"score {out['test_track_score'][i]:.1f}", fontsize=8)
            ax.set_xlabel("time bin (0.655 ms)", fontsize=7); ax.set_ylabel("freq ch (0 = highest)", fontsize=7)
    fig.suptitle("top: fitted sweep track on the stored sample; bottom: physical order, onset (orange) / end (green)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "fig_roll_onsets.png"))
    print("figure saved")


if __name__ == "__main__":
    main()
