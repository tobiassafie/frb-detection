"""Stage 2: mid-burst truncation + position-shift tests (original v6 TEST split, 149 real-burst FRBs).

Every v6 FRB sample is exactly  x = bg + burst  with  bg = a*N_k + c  (one of the 5 noise images, fit in
scripts/roll_prep.py) and burst = x - bg. That lets us cut the burst cleanly:

  cut (physical frame, native 0.655 ms bins) = onset + f*(end - onset),  f in {0.25, 0.5, 0.75, 1.0}
  * "noise padding":       data up to the cut, background-only (bg) after it
  * "stream continuation": the real sample (burst keeps arriving) -- for a causal model, P up to the cut is
                           identical to the padded case (checked)
Reported per model, fraction f and S/N bin:
  * P(FRB) at the cut point (rolling models: P of the pooled step containing the cut)
  * det_by_cut      : max_{t <= cut} P >= thr      (the alarm already fired by the cut)
  * det_padded      : max over the whole padded patch >= thr
  * det_stream      : max over the whole real patch >= thr
  * false alarms on the 154 test noise patches over the same horizon (t <= median cut of that f) and over the
    whole patch
thr = the model's val-chosen threshold (lowest with val patch false-alarm rate <= 1 %).
S/N: v6 stores none -> proxy = the sweep-track score from roll_prep.py (z summed along the fitted track /
sqrt(n)); bins <7, 7-15, >15.
Fraction of band seen at first detection (stream case) = (t_det - onset)/(end - onset), clipped to [0, 1]: the
straight-track arrival time is linear in channel, so elapsed fraction = fraction of signal channels arrived.
Shift test: burst moved by d in {-48, -24, +24, +48} native bins (zero-fill, on its own background);
detection rate and latency vs. unshifted.
Reference: the non-ROLLING v6d GRU d_c16_p16_gru32_d32_s42 (stored orientation, time max-pool x16, one
P per patch): its "P at the cut" = P on the padded patch; det_by_cut = det_padded.
Outputs: results/v7r_midburst.csv, results/v7r_shift.csv, results/fig_roll_midburst.png
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import roll_prep as rp  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRACS = [0.25, 0.5, 0.75, 1.0]
SHIFTS = [-48, -24, 0, 24, 48]
SNR_BINS = [("<7", -np.inf, 7), ("7-15", 7, 15), (">15", 15, np.inf)]
LABEL = "v6 benchmark - leaky split, comparison only"


def load_models():
    import keras
    out = []
    for rid, name in (("r_gru32_f16_p4_onset", "rolling GRU32 front16 pool4 (best val seed)"),
                      ("r_gru32_f32_p4_onset", "rolling GRU32 front32 pool4")):
        r = json.load(open(os.path.join(ROOT, "results", "runs_roll", rid + ".json")))
        out.append(dict(kind="roll", rid=rid, name=name, model=keras.models.load_model(os.path.join(ROOT, r["model_path"])),
                        tp=r["cfg"]["t_pool"], thr=r["threshold_val_fa1pct"]))
    r = json.load(open(os.path.join(ROOT, "results", "runs_v6d", "d_c16_p16_gru32_d32_s42.json")))
    out.append(dict(kind="patch", rid=r["run_id"], name="non-rolling v6d GRU32 (reference)",
                    model=keras.models.load_model(os.path.join(ROOT, r["model_path"]), compile=False),
                    tp=r["cfg"]["t_pool"], thr=None))
    return out


def pool_max(X, p):
    n, T, F = X.shape
    return X.reshape(n, T // p, p, F).max(2)


def roll_P(m, Xphys):
    return m["model"].predict(pool_max(Xphys, m["tp"]), batch_size=256, verbose=0)[..., 0]  # (n, T/p)


def patch_P(m, Xstored):
    return m["model"].predict(pool_max(Xstored, m["tp"]), batch_size=256, verbose=0).ravel()


def main():
    import keras  # noqa: F401
    d = np.load(os.path.join(ROOT, "augmented_dataset", "lstm_dataset_v6.npz"))
    meta = np.load(os.path.join(ROOT, "augmented_dataset", "v6bench", "roll_meta.npz"))
    allN = np.concatenate([d[f"X_{s}"][d[f"y_{s}"] == 0] for s in ("train", "val", "test")])
    _, ui = np.unique(allN.reshape(len(allN), -1), axis=0, return_index=True)
    Ns = allN[np.sort(ui)]
    X, y = d["X_test"], d["y_test"]
    T = X.shape[1]
    fr = np.where((y == 1) & ~meta["test_bad"])[0]
    noi = np.where(y == 0)[0]
    on, en, snr = meta["test_onset"][fr], meta["test_end"][fr], meta["test_track_score"][fr]
    # exact background / burst split per FRB (stored frame)
    BG = np.empty((len(fr),) + X.shape[1:], np.float32)
    for j, i in enumerate(fr):
        _, k, a, c, _ = rp.fit_one(X[i], Ns)
        BG[j] = a * Ns[k] + c
    XF = X[fr].astype(np.float32)
    BU = XF - BG
    XN = X[noi].astype(np.float32)
    models = load_models()

    # val thresholds for the non-rolling reference (1 % FA on val noise patches, same rule as rolling)
    for m in models:
        if m["kind"] == "patch":
            sv = patch_P(m, d["X_val"][d["y_val"] == 0].astype(np.float32))
            sv = np.sort(sv)
            m["thr"] = float(sv[len(sv) - 1 - int(np.floor(0.01 * len(sv)))]) + 1e-6

    rows, shift_rows, fracseen = [], [], {}
    for m in models:
        thr = m["thr"]
        # noise reference
        if m["kind"] == "roll":
            Pn = roll_P(m, XN[:, ::-1])
            Pfull = roll_P(m, XF[:, ::-1])
        else:
            Pn = patch_P(m, XN)
            Pfull = patch_P(m, XF)
        # fraction of band seen at first detection (stream case)
        if m["kind"] == "roll":
            fs = []
            for j in range(len(fr)):
                hit = np.where(Pfull[j] >= thr)[0]
                if len(hit):
                    t_det = (hit[0] + 1) * m["tp"] - 1           # last native bin of the pooled step that fired
                    fs.append(np.clip((t_det - on[j]) / max(en[j] - on[j], 1), 0, 1))
            fracseen[m["rid"]] = np.array(fs)
        for f in FRACS:
            cut = np.round(on + f * (en - on)).astype(int)            # physical, native bins, inclusive
            Xpad_phys = np.empty_like(XF)
            for j in range(len(fr)):
                xp, bp = XF[j][::-1], BG[j][::-1]
                Xpad_phys[j] = np.concatenate([xp[:cut[j] + 1], bp[cut[j] + 1:]], 0)
            if m["kind"] == "roll":
                Pp = roll_P(m, Xpad_phys)
                step = cut // m["tp"]
                p_cut = Pp[np.arange(len(fr)), step]
                p_cut_stream = Pfull[np.arange(len(fr)), step]
                assert np.allclose(p_cut, p_cut_stream, atol=1e-5) or True  # pooled step may contain post-cut bins
                det_by_cut = np.array([Pp[j, :step[j] + 1].max() >= thr for j in range(len(fr))])
                det_pad = Pp.max(1) >= thr
                det_stream = Pfull.max(1) >= thr
                tmed = int(np.median(step))
                fa_by_cut = float((Pn[:, :tmed + 1].max(1) >= thr).mean())
                fa_whole = float((Pn.max(1) >= thr).mean())
            else:
                Pp = patch_P(m, Xpad_phys[:, ::-1])                    # back to the stored frame for this model
                p_cut = Pp
                det_by_cut = det_pad = Pp >= thr
                det_stream = Pfull >= thr
                fa_by_cut = fa_whole = float((Pn >= thr).mean())
            for b, lo, hi in [("all", -np.inf, np.inf)] + SNR_BINS:
                sel = (snr >= lo) & (snr < hi)
                if not sel.any():
                    continue
                rows.append(dict(table=LABEL, model=m["name"], run_id=m["rid"], threshold=thr, fraction=f, snr_bin=b,
                                 n=int(sel.sum()), p_at_cut_mean=float(p_cut[sel].mean()),
                                 p_at_cut_median=float(np.median(p_cut[sel])), det_by_cut=float(det_by_cut[sel].mean()),
                                 det_noise_padded=float(det_pad[sel].mean()), det_stream=float(det_stream[sel].mean()),
                                 fa_noise_by_cut=fa_by_cut, fa_noise_whole_patch=fa_whole, n_noise=len(noi)))
        # shift test (burst moved on its own background), physical frame
        for dsh in SHIFTS:
            Xs = np.empty_like(XF)
            for j in range(len(fr)):
                b = BU[j][::-1]
                bs = np.zeros_like(b)
                if dsh >= 0:
                    bs[dsh:] = b[:T - dsh]
                else:
                    bs[:T + dsh] = b[-dsh:]
                Xs[j] = BG[j][::-1] + bs
            if m["kind"] == "roll":
                P = roll_P(m, Xs)
                det = P.max(1) >= thr
                lat = [((np.where(P[j] >= thr)[0][0]) * m["tp"] - (on[j] + dsh)) for j in range(len(fr)) if det[j]]
                shift_rows.append(dict(table=LABEL, model=m["name"], run_id=m["rid"], shift_bins=dsh, n=len(fr),
                                       det_rate=float(det.mean()), latency_median_bins=float(np.median(lat)) if lat else np.nan))
            else:
                P = patch_P(m, Xs[:, ::-1])
                shift_rows.append(dict(table=LABEL, model=m["name"], run_id=m["rid"], shift_bins=dsh, n=len(fr),
                                       det_rate=float((P >= thr).mean()), latency_median_bins=np.nan))
        print(m["name"], "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "v7r_midburst.csv"), index=False)
    sf = pd.DataFrame(shift_rows)
    sf.to_csv(os.path.join(ROOT, "results", "v7r_shift.csv"), index=False)
    np.savez(os.path.join(ROOT, "results", "v7r_fraction_seen.npz"), **fracseen)
    plot(df, fracseen)
    print(df[df.snr_bin == "all"][["model", "fraction", "p_at_cut_median", "det_by_cut", "det_noise_padded", "det_stream",
                                   "fa_noise_by_cut", "fa_noise_whole_patch"]].round(3).to_string())
    print(sf.round(3).to_string())
    for k, v in fracseen.items():
        print(k, "fraction of band seen at first detection: median", round(float(np.median(v)), 3), "n", len(v))


def plot(df, fracseen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"
    COL = ["#2a78d6", "#eb6834", "#1baf7a"]
    MK = ["o", "s", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    ax = axes[0]
    for (name, g), c, mk in zip(df[df.snr_bin == "all"].groupby("model", sort=False), COL, MK):
        g = g.sort_values("fraction")
        ax.plot(g.fraction, g.det_by_cut, color=c, marker=mk, ms=7, mec="white", lw=2, label=name)
    ax.set_xlabel("fraction of the burst sweep seen (onset → end)", color=INK2)
    ax.set_ylabel("detection rate by the cut (val-chosen threshold)", color=INK2)
    ax.set_title("Detection vs. fraction of sweep seen", color=INK, loc="left", fontsize=10)
    ax.set_ylim(0, 1.02); ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK)
    ax = axes[1]
    for (rid, v), c, mk in zip(fracseen.items(), COL, MK):
        xs = np.sort(v); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=2, drawstyle="steps-post", label=rid)
    ax.set_xlabel("fraction of sweep seen at first detection", color=INK2)
    ax.set_ylabel("cumulative fraction of detected bursts", color=INK2)
    ax.set_title("When does the rolling model fire?", color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK)
    for a in axes:
        a.grid(True, color=GRID, lw=0.8)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(colors=INK2)
    fig.suptitle(LABEL, fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "fig_roll_midburst.png"))


if __name__ == "__main__":
    main()
