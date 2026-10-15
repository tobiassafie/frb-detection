"""v6 benchmark, step 1: put the ORIGINAL lstm_dataset_v6.npz (on-disk version = the one behind
fpga_handoff/X_test.npy and every 96-99 % number) into the harness format WITHOUT re-splitting,
and compute the no-learning baselines + the deduplicated test subset.

Output augmented_dataset/v6bench/{X.npy (float16, train|val|test concatenated in original
order), meta.csv (idx, split, orig_index, label, dm, dup_in_train)} and
results/v6_baselines.json.

Note on float16: v6 patches are per-patch z-scored floats (|x| <~ 40); float16 storage
(as for v7) changes values by <0.05 % relative -- far below anything the models resolve.
Exact-duplicate checks use the ORIGINAL float32 arrays.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "augmented_dataset", "v6bench")


def h(x):
    return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    d = np.load(os.path.join(ROOT, "augmented_dataset", "lstm_dataset_v6.npz"))
    os.makedirs(OUT, exist_ok=True)
    rows, Xs = [], []
    for s in ("train", "val", "test"):
        X, y = d[f"X_{s}"], d[f"y_{s}"]
        dm = d[f"dm_{s}"] if f"dm_{s}" in d.files else np.full(len(y), np.nan)
        Xs.append(X)
        for i in range(len(y)):
            rows.append(dict(split=s, orig_index=i, label=int(y[i]), dm=float(dm[i]), hash=h(X[i])))
    meta = pd.DataFrame(rows)
    meta.insert(0, "idx", np.arange(len(meta)))
    train_hash = set(meta.hash[meta.split == "train"])
    meta["dup_in_train"] = meta.hash.isin(train_hash) & (meta.split != "train")
    # harness compatibility: no per-sample S/N in v6 -> FRB rows get snr=inf so S/N filters keep them
    meta["snr"] = np.where(meta.label == 1, np.inf, np.nan)
    meta["group_id"] = -1
    meta["file"] = "v6"
    X = np.concatenate(Xs).astype(np.float16)
    np.save(os.path.join(OUT, "X.npy"), X)
    meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)

    # ---- no-learning baselines on the ORIGINAL v6 test split (float32 arrays) ----
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    th = meta[meta.split == "test"]
    lookup = np.where(th.hash.isin(set(meta.hash[(meta.split == "train") & (meta.label == 0)])), 0, 1)
    # 1-nearest-neighbour (Euclidean, full 256x256 resolution) over the whole train set
    A = Xtr.reshape(len(Xtr), -1).astype(np.float32)
    B = Xte.reshape(len(Xte), -1).astype(np.float32)
    d2 = (B ** 2).sum(1)[:, None] - 2 * B @ A.T + (A ** 2).sum(1)[None, :]
    nn = d2.argmin(1)
    nn_pred = ytr[nn]
    # score for AUC: distance to nearest noise minus distance to nearest FRB
    sc_nn = d2[:, ytr == 0].min(1) - d2[:, ytr == 1].min(1)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import frb_train as ft
    base = {"lookup_exact_noise": ft.metrics(yte, lookup.astype(float)),
            "nearest_neighbour_1nn": ft.metrics(yte, nn_pred.astype(float))}
    from sklearn.metrics import roc_auc_score
    base["nearest_neighbour_1nn"]["auc"] = float(roc_auc_score(yte, sc_nn))
    dedup = th[~th.dup_in_train]
    base["dedup_info"] = dict(n_test=int(len(th)), n_test_dup=int(th.dup_in_train.sum()),
                              n_dedup=int(len(dedup)), dedup_noise=int((dedup.label == 0).sum()),
                              dedup_frb=int((dedup.label == 1).sum()),
                              dup_noise=int((th.dup_in_train & (th.label == 0)).sum()),
                              dup_frb=int((th.dup_in_train & (th.label == 1)).sum()),
                              unique_noise_images_total=int(meta.hash[meta.label == 0].nunique()))
    dm = ~th.dup_in_train.values
    base["lookup_exact_noise_on_dedup"] = dict(recall=float(lookup[dm].mean()), n=int(dm.sum()))
    base["nearest_neighbour_1nn_on_dedup"] = dict(recall=float(nn_pred[dm].mean()), n=int(dm.sum()))
    json.dump(base, open(os.path.join(ROOT, "results", "v6_baselines.json"), "w"), indent=1)
    print(json.dumps(base, indent=1))


if __name__ == "__main__":
    main()
