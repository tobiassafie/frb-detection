"""Parse every project's Vitis csynth report into synth_results.csv.
Latency/II from <proj>/frb_prj/solution1/syn/report/frb_csynth.xml; status/time from synth_status.csv.
If synthesis failed, the last ERROR lines of synthesis.log are recorded (useful for the meeting)."""
import csv
import glob
import os
import xml.etree.ElementTree as ET

rows = []
for d in sorted(p for p in glob.glob("*/") if not p.startswith("_")):
    name = d.rstrip("/\\")
    row = dict(project=name, status="not_run", synth_seconds="")
    st = os.path.join(d, "synth_status.csv")
    if os.path.exists(st):
        _, row["status"], row["synth_seconds"] = open(st).read().strip().split(",")
    rpt = glob.glob(os.path.join(d, "*_prj", "solution1", "syn", "report", "*_csynth.xml"))
    rpt = [r for r in rpt if os.path.basename(r) in ("frb_csynth.xml", "csynth.xml")] or rpt
    if rpt:
        r = ET.parse(rpt[0]).getroot()
        g = lambda p: (r.find(p).text if r.find(p) is not None else "")  # noqa: E731
        row.update(target_clk_ns=g("UserAssignments/TargetClockPeriod"),
                   est_clk_ns=g("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
                   latency_best=g("PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
                   latency_worst=g("PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"),
                   interval_min=g("PerformanceEstimates/SummaryOfOverallLatency/Interval-min"),
                   interval_max=g("PerformanceEstimates/SummaryOfOverallLatency/Interval-max"),
                   BRAM_18K=g("AreaEstimates/Resources/BRAM_18K"), DSP=g("AreaEstimates/Resources/DSP"),
                   FF=g("AreaEstimates/Resources/FF"), LUT=g("AreaEstimates/Resources/LUT"),
                   URAM=g("AreaEstimates/Resources/URAM"),
                   avail_DSP=g("AreaEstimates/AvailableResources/DSP"), avail_LUT=g("AreaEstimates/AvailableResources/LUT"))
        try:
            row["fmax_mhz_est"] = round(1000.0 / float(row["est_clk_ns"]), 1)
            row["latency_us"] = round(float(row["latency_worst"]) * float(row["target_clk_ns"]) / 1000, 3)
        except (ValueError, ZeroDivisionError):
            pass
    elif os.path.exists(os.path.join(d, "synthesis.log")):
        errs = [l.strip() for l in open(os.path.join(d, "synthesis.log"), errors="ignore") if "ERROR" in l]
        row["error_tail"] = " | ".join(errs[-3:])[:500]
    rows.append(row)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "project", k != "status", k))
with open("synth_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} projects -> synth_results.csv")
