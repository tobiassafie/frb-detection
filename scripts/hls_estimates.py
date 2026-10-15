"""ANALYTIC hardware estimates for every written hls4ml project (no Vitis here).

  mult_total   : sum over layers of hls4ml's multiplier_limit (parameters.h) = number of
                 multipliers instantiated; an UPPER BOUND on DSPs (narrow ones map to LUTs).
  weight_kbits : total weight+bias storage bits / 1000 (from defines.h widths x param counts).
  est_latency_cycles (CNN, io_stream dataflow): max over conv/pool layers of
                 in_height*in_width*max(3, reuse_factor), +10 %. Calibrated on the ONLY
                 design ever synthesised here (frb_pooled_hls: 64x64 input, RF=1 ->
                 12,299 cycles = 3.0 cycles per input pixel of the first conv).
  est_latency_cycles (RNN, io_parallel): timesteps * (reuse_factor + 12)  [rough; not calibrated]
These are labelled `est_*` everywhere; csynth numbers from EC2 replace them in the summary.
Writes results/hls_estimates.csv and merges est_* columns into results/hls_projects_index.csv.
"""
import glob
import json
import os
import re

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(ROOT, os.environ.get("HLS_DIR", "hls_projects"))
CLK_NS = 5.0


def parse(d):
    par = open(os.path.join(d, "firmware", "parameters.h")).read()
    dfs = open(os.path.join(d, "firmware", "defines.h")).read()
    meta = json.load(open(os.path.join(d, "frb_export.json")))
    blocks = re.split(r"\nstruct ", par)
    mult_total = 0
    conv_px = []
    for b in blocks:
        def g(k):
            m = re.search(rf"static const unsigned {k}\s*=\s*(\d+)", b)
            return int(m.group(1)) if m else None
        rf = g("reuse_factor") or 1
        head = b.split("{", 1)[0]
        if "dense_config" in head:  # conv per-pixel kernels (config*_mult) and dense layers
            n_in, n_out = g("n_in"), g("n_out")
            if n_in and n_out:
                mult_total += -(-(n_in * n_out) // rf)
        if "conv2d_config" in head or "pooling2d_config" in head:
            h, w = g("in_height"), g("in_width")
            if h and w:
                conv_px.append(h * w * max(3, rf))
    wb = [int(x) for x in re.findall(r"typedef ap_u?fixed<(\d+)[^>]*>\s+\w*(?:weight|bias)\w*_t;", dfs)]
    wbit = max(wb) if wb else 16
    rp = next(p for p in (os.path.join(ROOT, "results", d, f"{meta['run_id']}.json")
                          for d in ("runs", "runs_v6a", "runs_v6c")) if os.path.exists(p))
    run = json.load(open(rp))
    if meta["kind"] == "cnn":
        cyc = int(1.1 * max(conv_px)) if conv_px else None
    else:
        T = run["input_shape"][0]
        cyc = int(T * (meta["variant"]["rf"] + 12))
    return dict(project=os.path.basename(d), est_mult_total=mult_total,
                est_weight_kbits=round(run["params"] * wbit / 1000, 1), est_latency_cycles=cyc,
                est_latency_us=round(cyc * CLK_NS / 1000, 2) if cyc else None)


if __name__ == "__main__":
    rows = [parse(d) for d in sorted(glob.glob(os.path.join(P, "*"))) if os.path.isdir(d)
            and not os.path.basename(d).startswith("_") and os.path.exists(os.path.join(d, "frb_export.json"))]
    est = pd.DataFrame(rows)
    tag = os.environ.get("HLS_TAG", "")
    est.to_csv(os.path.join(ROOT, "results", f"hls_estimates{tag}.csv"), index=False)
    idx_p = os.path.join(ROOT, "results", os.environ.get("EXPORT_INDEX", "v6a_hls_index.csv") if tag else "hls_projects_index.csv")
    idx = pd.read_csv(idx_p)
    idx = idx[[c for c in idx.columns if not c.startswith("est_")]].merge(est, on="project", how="left")
    idx.to_csv(idx_p, index=False)
    print(est.to_string())
