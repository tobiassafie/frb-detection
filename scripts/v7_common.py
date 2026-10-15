"""Shared helpers for the v7 dataset (grouped-by-source rebuild).

Conventions: arrays are (time, freq). The common frequency grid is CHIME's
1024 channels, 800 -> 400 MHz (index 0 = highest frequency), later binned to
256. Missing / flagged channels are NaN until the final per-channel
normalisation, where they become 0.
"""
import numpy as np

K_DM = 4.148808e3            # s MHz^2 / (pc cm^-3)
CACHE_DT = 2.56e-6 * 64      # cache resolution (s)
WORK_REBIN = 4               # cache -> working resolution
WORK_DT = CACHE_DT * WORK_REBIN   # 0.655 ms, same as v6
N_CHAN_FULL = 1024
FREQ_FULL = 800.0 - (np.arange(N_CHAN_FULL) + 0.5) * (400.0 / N_CHAN_FULL)  # centres, high->low
FREQ_BIN = 4                 # 1024 -> 256 channels
N_FREQ = N_CHAN_FULL // FREQ_BIN
FREQ_256 = FREQ_FULL.reshape(N_FREQ, FREQ_BIN).mean(1)
PATCH_T = 256
EDGE_TRIM = 6                # working-res bins trimmed at both file ends (edge spike artefact)


def rebin_time(p, k):
    n = (p.shape[0] // k) * k
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(p[:n].reshape(n // k, k, *p.shape[1:]), axis=1)


def to_common_grid(power, freq):
    """power (T, n_ch) on the file's own channel list -> (T, 256) on the common grid."""
    T = power.shape[0]
    full = np.full((T, N_CHAN_FULL), np.nan, np.float32)
    idx = np.clip(np.round((800.0 - freq) / (400.0 / N_CHAN_FULL) - 0.5).astype(int), 0, N_CHAN_FULL - 1)
    full[:, idx] = power
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(full.reshape(T, N_FREQ, FREQ_BIN), axis=2)


def live_mask(p, frac=0.5):
    good = np.isfinite(p) & (p != 0)
    m = good.mean(0) > frac
    std = np.nanstd(np.where(good, p, np.nan), axis=0)
    return m & (np.nan_to_num(std) > 0)


def channel_normalise(p, live, valid=None, min_valid=16):
    """Robust per-channel normalisation (median / 1.4826 MAD) over valid pixels only;
    dead channels, invalid pixels and channels with < min_valid valid samples -> 0."""
    import warnings
    q = p.astype(np.float32).copy()
    if valid is not None:
        q[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(q, axis=0)
        mad = np.nanmedian(np.abs(q - med), axis=0) * 1.4826
    n_ok = np.isfinite(q).sum(0)
    out = (q - med) / np.where(mad > 0, mad, 1.0)
    out[:, ~live | (n_ok < min_valid) | ~(mad > 0)] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def delays_bins(dm, freqs=FREQ_256, dt=WORK_DT, fref=800.0):
    """Arrival delay (bins) relative to fref. dm may be negative (residual sweep)."""
    return K_DM * dm * (freqs ** -2 - fref ** -2) / dt


def shift_channels(p, shifts):
    """Shift each channel later in time by shifts[c] (float -> linear interp), zero-fill."""
    T, C = p.shape
    out = np.zeros_like(p)
    t = np.arange(T)
    for c in range(C):
        if not np.any(p[:, c]):
            continue
        out[:, c] = np.interp(t - shifts[c], t, p[:, c], left=0.0, right=0.0)
    return out
