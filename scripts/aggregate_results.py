"""Collect results/runs/*.json (+ hardware JSONs) into results/sweep_results.csv."""
import glob
import json
import os

import pandas as pd

COLS = ["run_id", "stage", "track", "kind", "status", "split", "fold", "seed", "params", "macs", "flops",
        "input_shape", "val_acc", "val_auc", "val_f1", "test_acc", "test_bal_acc", "test_precision", "test_recall",
        "test_f1", "test_auc", "test_recall_at_1pct_fpr", "test_allsnr_acc", "test_tn", "test_fp", "test_fn",
        "test_tp", "test_recall_snr_3_5", "test_recall_snr_5_7", "test_recall_snr_7_10", "test_recall_snr_10_15",
        "test_recall_snr_15_25", "test_recall_snr_25_50", "epochs_run", "best_epoch", "train_seconds",
        "quant", "model_path", "cfg_json"]


def main():
    rows = []
    for f in sorted(glob.glob("results/runs/*.json")):
        r = json.load(open(f))
        cfg = r.get("cfg", {})
        row = {k: r.get(k) for k in COLS}
        for k in ("stage", "track", "kind", "split", "fold", "seed"):
            row[k] = cfg.get(k, r.get(k))
        row["quant"] = cfg.get("quant", "float32")
        row["cfg_json"] = json.dumps(cfg, sort_keys=True)
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv("results/sweep_results.csv", index=False)
    print(f"{len(df)} rows -> results/sweep_results.csv")
    return df


if __name__ == "__main__":
    main()
