"""Classical reference detector on the same v7 splits: incoherent dedispersion
over residual-DM trials [-13, 0] + boxcar matched filter; statistic = max S/N.
Threshold chosen on val (max accuracy); reported on test. Gives the
'what does a standard single-pulse search get on this data' reference line.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import frb_train as ft  # noqa: E402
from v7_common import FREQ_256, delays_bins  # noqa: E402

F_POOL = 8
DMS = np.arange(-13.0, -0.39, 0.2)  # zero-DM excised as in real single-pulse searches
WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16]


def stat(Xb, freqs):
    N, T, C = Xb.shape
    best = np.full(N, -np.inf, np.float32)
    live = (np.abs(Xb).sum(1) > 0)  # (N, C)
    nlive = np.maximum(live.sum(1), 1)
    pad = T
    Xp = np.zeros((N, T + 2 * pad, C), np.float32)
    Xp[:, pad:pad + T] = Xb
    for dm in DMS:
        sh = np.clip(np.round(delays_bins(dm, freqs)).astype(int), -pad, pad)
        ts = np.zeros((N, T), np.float32)
        for c in range(C):  # slice views, no copies: channel c arrives sh[c] bins late
            ts += Xp[:, pad + sh[c]:pad + sh[c] + T, c]
        ts /= np.sqrt(nlive)[:, None]
        cs = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(ts, 1)], 1)
        for w in WIDTHS:
            sm = (cs[:, w:] - cs[:, :-w]) / np.sqrt(w)
            ok = sm != 0  # bins with no recorded data don't count as noise
            n = np.maximum(ok.sum(1, keepdims=True), 1)
            med = (sm * ok).sum(1, keepdims=True) / n
            sig = np.sqrt((((sm - med) * ok) ** 2).sum(1, keepdims=True) / n)
            ok2 = ok & (np.abs(sm - med) < 3 * sig)  # one 3-sigma clip (removes the burst itself)
            n2 = np.maximum(ok2.sum(1, keepdims=True), 1)
            med = (sm * ok2).sum(1, keepdims=True) / n2
            sig = np.sqrt((((sm - med) * ok2) ** 2).sum(1, keepdims=True) / n2) + 1e-3
            best = np.maximum(best, ((sm - med) / sig).max(1))
    return best


def main():
    X, meta = ft.load_data(1, F_POOL)
    freqs = FREQ_256.reshape(-1, F_POOL).mean(1)
    y = meta.label.values
    out = {}
    for split in sys.argv[1:] or ["primary"]:
        idx = ft.split_indices(meta, split)
        s = {k: stat(X[idx[k]] * np.sqrt(F_POOL), freqs) for k in ("val", "test", "test_all")}  # block mean -> unit noise
        thr_grid = np.unique(s["val"])
        accs = [((s["val"] >= t) == y[idx["val"]]).mean() for t in thr_grid]
        thr = float(thr_grid[int(np.argmax(accs))])
        m = ft.metrics(y[idx["test"]], (s["test"] >= thr).astype(float))
        from sklearn.metrics import roc_auc_score
        m["auc"] = float(roc_auc_score(y[idx["test"]], s["test"]))
        m["threshold_snr"] = thr
        m["val_acc"] = float(max(accs))
        m.update(ft.recall_by_snr(meta.iloc[idx["test_all"]], (s["test_all"] >= thr).astype(float)))
        m["allsnr_acc"] = float(((s["test_all"] >= thr) == y[idx["test_all"]]).mean())
        out[split] = m
        print(split, json.dumps(m, indent=0))
    path = "results/classical_baseline.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out)
    json.dump(old, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
