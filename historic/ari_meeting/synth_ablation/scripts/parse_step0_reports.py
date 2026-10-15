"""
Step 0: parse the 23 existing per-module csynth.xml reports under
frb_pooled_hls/frb_pooled_hls/frb_classifier_prj/solution1/syn/report/
into one table. Read-only -- does not touch frb_pooled_hls.
"""
import csv
import os
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
REPORT_DIR = os.path.join(
    ROOT, "frb_pooled_hls", "frb_pooled_hls", "frb_classifier_prj",
    "solution1", "syn", "report",
)
OUT_DIR = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation")

# Map each report's TopModelName / filename to the pooled model's Keras layer.
# Derived from frb_pooled_hls's hls4ml config + parameters.h/defines.h (see 00_inventory.md /
# 01_basics.md): config2/config2_mult=conv2d, config3=conv2d_relu, config4=max_pooling2d,
# config5/config5_mult=conv2d_1, config6=conv2d_1_relu, config7=max_pooling2d_1,
# config9=dense, config10=dense_softmax. shift_line_buffer / compute_output_buffer are
# internal line-buffer helper submodules of the conv/pool layers, not separate Keras layers.
LAYER_MAP = {
    "conv_2d_cl_array_ap_fixed_1u_array_ap_fixed_37_17_5_3_0_2u_config2_s": "conv2d (conv)",
    "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_37_17_5_3_0_config2_mult_s": "conv2d (mult submodule)",
    "shift_line_buffer_array_ap_fixed_16_6_5_3_0_1u_config2_s": "conv2d (line buffer)",
    "relu_array_ap_fixed_2u_array_ap_fixed_16_6_5_3_0_2u_relu_config3_s": "conv2d_relu",
    "pooling2d_cl_array_ap_fixed_2u_array_ap_fixed_16_6_5_3_0_2u_config4_s": "max_pooling2d",
    "shift_line_buffer_array_ap_fixed_16_6_5_3_0_2u_config4_s": "max_pooling2d (line buffer)",
    "conv_2d_cl_array_ap_fixed_2u_array_ap_fixed_38_18_5_3_0_2u_config5_s": "conv2d_1 (conv)",
    "dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_38_18_5_3_0_config5_mult_s": "conv2d_1 (mult submodule)",
    "shift_line_buffer_array_ap_fixed_16_6_5_3_0_2u_config5_s": "conv2d_1 (line buffer)",
    "relu_array_ap_fixed_2u_array_ap_fixed_16_6_5_3_0_2u_relu_config6_s": "conv2d_1_relu",
    "pooling2d_cl_array_ap_fixed_2u_array_ap_fixed_16_6_5_3_0_2u_config7_s": "max_pooling2d_1",
    "shift_line_buffer_array_ap_fixed_16_6_5_3_0_2u_config7_s": "max_pooling2d_1 (line buffer)",
    "dense_array_array_ap_fixed_42_22_5_3_0_2u_config9_Pipeline_DataPrepare": "dense (data prepare)",
    "dense_latency_wrapper_ap_fixed_ap_fixed_42_22_5_3_0_config9_s": "dense (wrapper)",
    "dense_array_ap_fixed_2u_array_ap_fixed_42_22_5_3_0_2u_config9_s": "dense (top)",
    "init_exp_table_ap_fixed_42_22_5_3_0_softmax_config10_s": "dense_softmax (exp table init)",
    "softmax_array_array_ap_fixed_16_6_5_3_0_2u_softmax_config10_s": "dense_softmax",
    "softmax_stable_array_array_ap_fixed_16_6_5_3_0_2u_softmax_config10_s": "dense_softmax (stable variant)",
    "compute_output_buffer_2d_array_array_ap_fixed_37_17_5_3_0_2u_config2_s": "conv2d (output buffer)",
    "compute_output_buffer_2d_array_array_ap_fixed_38_18_5_3_0_2u_config5_s": "conv2d_1 (output buffer)",
    "frb_classifier": "TOP (whole model)",
}


def parse_one(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    top_name = root.findtext("UserAssignments/TopModelName", default="?")

    perf = root.find("PerformanceEstimates")
    lat = perf.find("SummaryOfOverallLatency") if perf is not None else None
    clock = perf.findtext("SummaryOfTimingAnalysis/EstimatedClockPeriod") if perf is not None else None

    def gi(node, tag):
        if node is None:
            return None
        v = node.findtext(tag)
        try:
            return int(v)
        except (TypeError, ValueError):
            return v

    area = root.find("AreaEstimates/Resources")
    avail = root.find("AreaEstimates/AvailableResources")

    def ga(node, tag):
        v = node.findtext(tag) if node is not None else None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    row = {
        "report_file": os.path.basename(xml_path),
        "module": top_name,
        "layer": LAYER_MAP.get(top_name, "?"),
        "latency_cycles_best": gi(lat, "Best-caseLatency"),
        "latency_cycles_worst": gi(lat, "Worst-caseLatency"),
        "latency_us_best": lat.findtext("Best-caseRealTimeLatency") if lat is not None else None,
        "II_min": gi(lat, "Interval-min"),
        "II_max": gi(lat, "Interval-max"),
        "est_clock_ns": clock,
        "DSP": ga(area, "DSP"),
        "LUT": ga(area, "LUT"),
        "FF": ga(area, "FF"),
        "BRAM_18K": ga(area, "BRAM_18K"),
        "DSP_avail": ga(avail, "DSP"),
        "LUT_avail": ga(avail, "LUT"),
        "FF_avail": ga(avail, "FF"),
        "BRAM_avail": ga(avail, "BRAM_18K"),
    }
    for res in ("DSP", "LUT", "FF", "BRAM_18K"):
        v, a = row[res], row.get(f"{res.replace('_18K','')}_avail" if res != "BRAM_18K" else "BRAM_avail")
        row[f"{res}_pct"] = round(100 * v / a, 2) if (v is not None and a) else None
    return row


def main():
    rows = []
    for fname in sorted(os.listdir(REPORT_DIR)):
        if fname.endswith("_csynth.xml") or fname == "csynth.xml":
            if fname == "csynth.xml":
                continue  # duplicate of frb_classifier_csynth.xml (top-level), skip to avoid double count
            rows.append(parse_one(os.path.join(REPORT_DIR, fname)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "step0_report_table.csv")
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Parsed {len(rows)} reports -> {out_csv}")
    for r in sorted(rows, key=lambda r: -(r["DSP"] or 0)):
        print(f"{r['layer']:<35s} DSP={r['DSP']:>4} LUT={r['LUT']:>6} FF={r['FF']:>6} "
              f"BRAM={r['BRAM_18K']:>2} lat_best={r['latency_cycles_best']:>6} II={r['II_min']}")


if __name__ == "__main__":
    main()
