"""Stage 0: re-evaluate the SAVED historical v6 models (exact weights) on the original v6 test set,
with the same context columns as scripts/v6a_train.py. Writes results/runs_v6a/saved__<name>.json
(stage = stage0_saved) so they land in results/v6a_runs.csv."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402

import v6a_train as va  # noqa: E402

SAVED = [  # name, path, kind, t_pool, f_pool, originally reported test acc, where it was reported
    ("cnn_v1", "models/cnn_v1/model.keras", "cnn", 1, 1, 0.9740, "models/cnn_v1/config.json test_acc"),
    ("seedcheck_baseline_s142", "ablation_results/models/seedcheck_baseline_s142.keras", "cnn", 1, 1, 0.9675,
     "frb_ablation_study.ipynb c17 (= fpga_handoff/group1)"),
    ("group1_f4", "ablation_results/models/group1_f4.keras", "cnn", 1, 1, 0.9643, "frb_ablation_study.ipynb c6"),
    ("group1_f64", "ablation_results/models/group1_f64.keras", "cnn", 1, 1, 0.9740, "frb_ablation_study.ipynb c6/c24"),
    ("group1_f16", "ablation_results/models/group1_f16.keras", "cnn", 1, 1, 0.974, "frb_ablation_study.ipynb c6"),
    ("group2_b_front_loaded", "ablation_results/models/group2_b_front_loaded.keras", "cnn", 1, 1, 0.9643,
     "frb_ablation_study.ipynb c9 (= fpga_handoff/group2)"),
    ("lstm_v3", "models/lstm_v3/model.keras", "rnn", 1, 1, 0.9643, "models/lstm_v3/config.json; lstm_training_2.ipynb c14"),
    ("lstm_v2", "models/lstm_v2/model.keras", "rnn", 1, 1, 0.9578, "models/lstm_v2/config.json"),
    ("lstm_v1", "models/lstm_v1/model.keras", "rnn", 1, 1, 0.9351, "models/lstm_v1/config.json"),
    ("v6_sw_r128_max_flatten_d16_seed1", "results/models/v6_sw_r128_max_flatten_d16_seed1.keras", "cnn", 2, 2, 0.9838,
     "previous session v6 sweep (results/runs/)"),
    ("v6_sw_r128_max_flatten_d16", "results/models/v6_sw_r128_max_flatten_d16.keras", "cnn", 2, 2, 0.9773,
     "previous session v6 sweep"),
    ("v6_sw_r128_max_flatten_d16_seed2", "results/models/v6_sw_r128_max_flatten_d16_seed2.keras", "cnn", 2, 2, 0.9773,
     "previous session v6 sweep"),
]


def main():
    import keras
    import frb_train as ft
    from sklearn.metrics import roc_auc_score
    D = va.data()
    os.makedirs(va.RUNS, exist_ok=True)
    for name, path, kind, tp, fp, rep, where in SAVED:
        m = keras.models.load_model(os.path.join(va.ROOT, path), compile=False)
        prep = (lambda a: a[..., None]) if kind == "cnn" else (lambda a: a)
        Xte = va.pool(D["X_test"], tp, fp)
        p = ft.predict(m, prep(Xte))
        r = dict(run_id=f"saved__{name}", stage="stage0_saved", kind=kind, params=int(m.count_params()),
                 macs=va.macs(m), input_shape=list(Xte.shape[1:]), model_path=path, reported_test_acc=rep,
                 reported_where=where, n_attempts=0, cfg=dict(kind=kind, t_pool=tp, f_pool=fp, source="saved weights"))
        r.update({f"test_{k}": v for k, v in ft.metrics(D["y_test"], p).items()})
        r.update({f"val_{k}": v for k, v in ft.metrics(D["y_val"], ft.predict(m, prep(va.pool(D["X_val"], tp, fp)))).items()})
        r["lookup_baseline_acc"] = va.LOOKUP_ACC
        r["dedup_recall"] = float((p[D["dedup"]] >= 0.5).mean())
        r["dedup_n"] = int(D["dedup"].sum())
        pn = ft.predict(m, prep(Xte + np.random.default_rng(0).normal(0, 0.05, Xte.shape).astype(np.float32)))
        r["noise_acc_perturbed_0p05"] = float((pn[D["y_test"] == 0] < 0.5).mean())
        r["frb_recall_perturbed_0p05"] = float((pn[D["y_test"] == 1] >= 0.5).mean())
        r["v6test_timereversed_acc"] = float(((ft.predict(m, prep(Xte[:, ::-1])) >= 0.5) == D["y_test"]).mean())
        X7, y7 = va.v7_test(tp, fp)
        p7 = ft.predict(m, prep(X7))
        r["v7test_acc"] = float(((p7 >= 0.5) == y7).mean())
        r["v7test_auc"] = float(roc_auc_score(y7, p7))
        r["attempts"] = []
        json.dump(r, open(os.path.join(va.RUNS, r["run_id"] + ".json"), "w"), indent=1, default=float)
        print(f"{name:36s} params={r['params']:7d} reported={rep:.4f} reproduced={r['test_acc']:.4f} "
              f"auc={r['test_auc']:.4f} dedup={r['dedup_recall']:.3f} rev={r['v6test_timereversed_acc']:.3f} "
              f"v7={r['v7test_acc']:.3f}", flush=True)
    va.aggregate()


if __name__ == "__main__":
    main()
