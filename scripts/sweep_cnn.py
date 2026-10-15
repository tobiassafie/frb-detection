"""Stage 1: Optuna (TPE) CNN sweep on the v7 primary (file-grouped) split.

Track A: best accuracy regardless of size (BatchNorm allowed, params <= 300k).
Track B: synthesis-friendly (Conv+ReLU+MaxPool only, no BatchNorm, params <= 10k).
Objective = val accuracy (val files only; test is never used for selection).

Usage: python scripts/sweep_cnn.py <track A|B> <n_trials_this_worker> <threads>
Launch several workers per track; they share results/optuna/cnn.journal.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import optuna  # noqa: E402
from optuna.storages import JournalStorage  # noqa: E402
from optuna.storages.journal import JournalFileBackend  # noqa: E402

import frb_train as ft  # noqa: E402

FILTER_CHOICES = [2, 4, 8, 16]


def sample_cfg(trial, track):
    pool_f = trial.suggest_categorical("res_pool", [2, 4])   # input 128 / 64 (256 checked separately: 4x slower, not better in the N=200 pilot)
    res = 256 // pool_f
    nb = trial.suggest_int("n_blocks", 2, 5)
    f0 = trial.suggest_categorical("f0", FILTER_CHOICES)
    filters = [f0]
    for i in range(1, nb):  # non-decreasing per block
        step = trial.suggest_int(f"fstep_{i}", 0, 1)
        filters.append(min(16, filters[-1] * (2 if step else 1)))
    sched = trial.suggest_categorical("pool_sched", ["2x2", "4x4_early"])
    pools = [4 if (sched == "4x4_early" and i == 0) else 2 for i in range(nb)]
    prod = 1
    for p in pools:
        prod *= p
    if res // prod < 1:
        raise optuna.TrialPruned("spatial dims < 1")
    cfg = dict(kind="cnn", t_pool=pool_f, f_pool=pool_f, filters=filters, pools=pools,
               kernel=trial.suggest_categorical("kernel", [3, 5]),
               pool_type=trial.suggest_categorical("pool_type", ["max", "avg"]),
               head=trial.suggest_categorical("head", ["flatten", "gap"]),
               dense=trial.suggest_categorical("dense", [8, 16, 32]),
               dropout=trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
               lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
               batch=trial.suggest_categorical("batch", [32, 64, 128]),
               class_weight=trial.suggest_categorical("class_weight", [False, True]),
               train_snr_min=trial.suggest_categorical("train_snr_min", [3.0, 5.0, 7.0]),
               time_flip=trial.suggest_categorical("time_flip", [False, True]),
               epochs=30, patience=8, seed=0, split="primary", track=track, stage="stage1_optuna_mix",
               train_pool="mix_primary")
    if track == "A":
        cfg["batchnorm"] = trial.suggest_categorical("batchnorm", [False, True])
        cfg["max_params"] = 300_000
    else:
        cfg["batchnorm"] = False
        cfg["max_params"] = 10_000
    return cfg


def objective(trial, track, threads):
    cfg = sample_cfg(trial, track)
    r = ft.train_eval(cfg, run_id=f"s1m{track}_t{trial.number:04d}", threads=threads)
    if r["status"] != "ok":
        raise optuna.TrialPruned(r["status"])
    for k in ("params", "macs", "test_acc", "test_auc", "val_auc"):
        trial.set_user_attr(k, r[k])
    return r["val_acc"]


if __name__ == "__main__":
    track, n, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("results/optuna", exist_ok=True)
    storage = JournalStorage(JournalFileBackend("results/optuna/cnn.journal"))
    study = optuna.create_study(study_name=f"cnn_mix_track{track}", storage=storage, direction="maximize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=None, n_startup_trials=16))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda t: objective(t, track, threads), n_trials=n, catch=(Exception,))
