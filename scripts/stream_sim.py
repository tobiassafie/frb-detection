"""Stage 2: causal streaming simulation + streaming metrics (and a stream-sampled training pool).

STREAM (v7-style, real CHIME noise):
  * A stream is a sequence of per-file "sessions" (one instrument configuration per session).
    A session concatenates random segments (>= 64 bins) of that file's burst-in-painted real
    background (both pols, random time-reversal), at 0.655 ms x 256 channels, per-channel
    normalised units (same arrays as the v7 dataset).
  * Bursts are injected at random times (gaps U(1,3) s) with the SAME FILE's template
    (as v7 val/test), |residual DM| ~ U(0.5,12), optimal S/N ~ logU(3,50).
  * orientation='physical': high frequencies arrive first (delay grows toward 400 MHz).
    orientation='stored'  : the sign seen in the stored CHIME baseband arrays (low first),
                            i.e. the orientation every v6/v7 training sample had.
  * Causal sliding window: 256 bins, stride 64 (1/4 window); window ending at t uses only
    bins [t-256, t). Each window is normalised on its own (no look-ahead): per-channel robust
    (v7 models) or per-window z-score (v6 models), then block-mean pooled to the model input.
METRICS (per model, at a threshold chosen on a VAL-files stream):
  * per burst: detected?, time-to-detect = (end of first window with p >= thr) - (arrival of
    the first live channel), fraction of the live band arrived at detection time.
  * false alarms per hour = number of clusters of consecutive above-threshold windows that do
    not overlap any burst's [first arrival, last arrival + 1 window] interval, per hour of stream.
V6 STREAM (comparison only): v6 test patches tiled end-to-end (each FRB patch followed by 3-10
  noise patches drawn from v6 test noise); truth = the FRB patch span.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from v7_build_dataset import CACHE, make_template, burst_window, inpaint_track  # noqa: E402
from v7_common import WORK_DT, channel_normalise, delays_bins, shift_channels  # noqa: E402
from v7_fit_bursts import load_work  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WIN, STRIDE = 256, 64
SNR_RANGE, ADM_RANGE, GAP_S = (3.0, 50.0), (0.5, 12.0), (1.0, 3.0)


# ------------------------------------------------------------------ v7-style stream
def prepare_file(fn, rng):
    fits = json.load(open(os.path.join(CACHE, "burst_fits.json")))
    fit = fits[fn]
    path = os.path.join(CACHE, fn.replace(".h5", ".npz"))
    xs, vs, lives = [], [], []
    for b in (0, 1):
        x, live, valid = load_work(path, b, return_valid=True)
        xs.append(x); vs.append(valid); lives.append(live)
    live = lives[0] & lives[1]
    dd, lo, hi = burst_window(xs[0] + xs[1], live, fit)
    tpl = make_template(dd, live, lo, hi)
    bgs = [inpaint_track(xs[b], vs[b], live, fit, lo, hi, rng) for b in (0, 1)]
    return dict(bgs=bgs, valids=vs, live=live, tpl=tpl, fn=fn)


def session(P, n_bins, rng, orientation, t_offset):
    """Return (data[n_bins,256] float32, valid mask, bursts list) for one file session."""
    data = np.zeros((n_bins, 256), np.float32)
    vmask = np.zeros((n_bins, 256), bool)
    t = 0
    while t < n_bins:  # concatenate random background segments
        b = int(rng.integers(0, 2))
        bg, vv = P["bgs"][b], P["valids"][b]
        L = int(rng.integers(64, bg.shape[0] + 1))
        a = int(rng.integers(0, bg.shape[0] - L + 1))
        seg, sv = bg[a:a + L], vv[a:a + L]
        if rng.integers(0, 2):
            seg, sv = seg[::-1], sv[::-1]
        n = min(L, n_bins - t)
        data[t:t + n], vmask[t:t + n] = seg[:n], sv[:n]
        t += n
    live = P["live"] & (vmask.mean(0) > 0.3)
    bursts = []
    sign = 1.0 if orientation == "physical" else -1.0
    t = int(rng.uniform(*GAP_S) / WORK_DT)
    tpl = P["tpl"]
    while True:
        dm = sign * rng.uniform(*ADM_RANGE)
        d = delays_bins(dm)
        d = d - d[live].min()  # first live channel arrives at offset 0
        span = int(np.ceil(d[live].max())) + tpl.shape[0] + 2
        if t + span >= n_bins:
            break
        snr = float(np.exp(rng.uniform(*np.log(SNR_RANGE))))
        canvas = np.zeros((span, 256), np.float32)
        canvas[:tpl.shape[0]] = tpl
        canvas = shift_channels(canvas, d)
        canvas[:, ~live] = 0
        canvas *= vmask[t:t + span]
        e = np.sqrt((canvas ** 2).sum())
        if e > 0:
            data[t:t + span] += canvas * (snr / e)
            peak = int(np.argmax(tpl.sum(1)))
            arr = t + peak + d  # arrival (peak) time per channel
            bursts.append(dict(file=P["fn"], t_first=t_offset + int(arr[live].min()), t_last=t_offset + int(arr[live].max()),
                               arrivals=(t_offset + arr[live]).astype(np.float32), snr=snr, dm_res=float(dm)))
        t += span + int(rng.uniform(*GAP_S) / WORK_DT)
    return data, vmask, live, bursts


def window_batches(data, vmask, live, t_offset, bs=512):
    """Causal windows over one session: yields (end_times, raw windows, valid windows)."""
    ends = np.arange(WIN, data.shape[0] + 1, STRIDE)
    for i in range(0, len(ends), bs):
        e = ends[i:i + bs]
        W = np.stack([data[x - WIN:x] for x in e])
        V = np.stack([vmask[x - WIN:x] for x in e])
        yield t_offset + e, W, V


def norm_v7(W, V, live):
    return np.stack([channel_normalise(w, live, v).clip(-20, 60) for w, v in zip(W, V)])


def norm_v6(W):
    m = W.mean((1, 2), keepdims=True)
    s = W.std((1, 2), keepdims=True)
    return (W - m) / np.where(s > 0, s, 1)


def pool(W, tp, fp):
    n, T, F = W.shape
    return W.reshape(n, T // tp, tp, F // fp, fp).mean((2, 4))


# ------------------------------------------------------------------ scoring + metrics
class ModelSpec:
    def __init__(self, label, run, model, preproc):
        self.label, self.run, self.model, self.preproc = label, run, model, preproc
        c = run["cfg"]
        self.kind, self.tp, self.fp, self.clip = c["kind"], c["t_pool"], c["f_pool"], c.get("clip")

    def score(self, Wn):
        x = pool(Wn, self.tp, self.fp)
        if self.clip:
            x = np.clip(x, *self.clip)
        x = x[..., None] if self.kind == "cnn" else x
        return self.model.predict(x, batch_size=512, verbose=0).ravel()


def run_v7_stream(specs, groups, minutes_per_file, orientation, seed):
    rng = np.random.default_rng(seed)
    sp = json.load(open(os.path.join(ROOT, "results", "splits", "v7_splits.json")))
    g2f = {int(k): v for k, v in sp["group_to_file"].items()}
    ends_all, scores = [], {s.label: [] for s in specs}
    bursts = []
    t_off = 0
    n_bins = int(minutes_per_file * 60 / WORK_DT)
    for g in groups:
        P = prepare_file(g2f[g], rng)
        data, vmask, live, b = session(P, n_bins, rng, orientation, t_off)
        bursts += b
        for e, W, V in window_batches(data, vmask, live, t_off):
            ends_all.append(e)
            need6 = any(s.preproc == "v6" for s in specs)
            Wv7 = norm_v7(W, V, live)
            Wv6 = norm_v6(W) if need6 else None
            for s in specs:
                scores[s.label].append(s.score(Wv7 if s.preproc == "v7" else Wv6))
        t_off += n_bins + WIN  # gap between sessions: no window spans two files
    ends = np.concatenate(ends_all)
    return ends, {k: np.concatenate(v) for k, v in scores.items()}, bursts, len(groups) * n_bins * WORK_DT / 3600


def stream_metrics(ends, p, bursts, hours, thr):
    above = p >= thr
    starts = ends - WIN
    in_burst = np.zeros(len(ends), bool)
    det = []
    for b in bursts:
        lo, hi = b["t_first"], b["t_last"] + WIN
        ov = (ends > b["t_first"]) & (starts < hi)  # window overlaps the burst's arrival span
        in_burst |= ov
        hit = np.where(ov & above)[0]
        if len(hit):
            te = ends[hit[0]]
            det.append(dict(detected=True, ttd_ms=(te - b["t_first"]) * WORK_DT * 1e3,
                            frac_band_arrived=float((b["arrivals"] < te).mean()), snr=b["snr"]))
        else:
            det.append(dict(detected=False, ttd_ms=np.nan, frac_band_arrived=np.nan, snr=b["snr"]))
    fa = above & ~in_burst
    clusters = int(fa[0]) + int(np.sum(fa[1:] & ~fa[:-1]))
    d = pd.DataFrame(det)
    out = dict(n_bursts=len(d), stream_hours=round(hours, 4), threshold=float(thr),
               fa_per_hour=clusters / hours if hours > 0 else np.nan, fa_clusters=clusters,
               det_rate_all=float(d.detected.mean()) if len(d) else np.nan)
    for lo, hi in ((3, 7), (7, 50), (10, 50), (15, 50)):
        s = d[(d.snr >= lo) & (d.snr < hi)]
        out[f"det_rate_snr_{lo}_{hi}"] = float(s.detected.mean()) if len(s) else np.nan
    s = d[d.detected & (d.snr >= 7)]
    out["ttd_ms_median_snr_ge7"] = float(s.ttd_ms.median()) if len(s) else np.nan
    out["ttd_ms_p90_snr_ge7"] = float(s.ttd_ms.quantile(0.9)) if len(s) else np.nan
    out["frac_band_at_detect_median_snr_ge7"] = float(s.frac_band_arrived.median()) if len(s) else np.nan
    out["frac_detected_before_half_band_snr_ge7"] = float((s.frac_band_arrived < 0.5).mean()) if len(s) else np.nan
    return out


def pick_threshold(ends, p, bursts, hours, fa_target_per_hour):
    """Lowest threshold whose VAL-stream false-alarm rate is <= target."""
    for thr in np.unique(np.round(np.concatenate([np.linspace(0.05, 0.999, 200), [0.9999, 0.99999]]), 6)):
        m = stream_metrics(ends, p, bursts, hours, thr)
        if m["fa_per_hour"] <= fa_target_per_hour:
            return float(thr)
    return 1.0


# ------------------------------------------------------------------ v6 stream (comparison only)
def run_v6_stream(specs, split, seed, noise_between=(3, 10)):
    rng = np.random.default_rng(seed)
    meta = pd.read_csv(os.path.join(ROOT, "augmented_dataset", "v6bench", "meta.csv"))
    X = np.load(os.path.join(ROOT, "augmented_dataset", "v6bench", "X.npy"), mmap_mode="r")
    m = meta[meta.split == split]
    frb, noi = m.idx[m.label == 1].values, m.idx[m.label == 0].values
    order = []
    for f in rng.permutation(frb):
        order += [int(i) for i in rng.choice(noi, int(rng.integers(*noise_between)))] + [int(f)]
    order += [int(i) for i in rng.choice(noi, 5)]
    data = np.concatenate([np.asarray(X[i], np.float32) for i in order])
    bursts = []
    for k, i in enumerate(order):
        if meta.label[i] == 1:
            t0 = k * WIN
            bursts.append(dict(t_first=t0, t_last=t0, arrivals=np.array([t0], np.float32), snr=np.inf, dm_res=np.nan))
    vmask = np.ones_like(data, bool)
    live = np.ones(256, bool)
    ends_all, scores = [], {s.label: [] for s in specs}
    for e, W, V in window_batches(data, vmask, live, 0):
        ends_all.append(e)
        Wv7 = np.stack([channel_normalise(w, live) for w in W]).clip(-20, 60)
        Wv6 = norm_v6(W)
        for s in specs:
            scores[s.label].append(s.score(Wv7 if s.preproc == "v7" else Wv6))
    ends = np.concatenate(ends_all)
    return ends, {k: np.concatenate(v) for k, v in scores.items()}, bursts, len(data) * WORK_DT / 3600
