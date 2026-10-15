"""
Step F: make a COPY of the unmodified frb_pooled_hls project (never touching the
original), drop in real testbench vectors, and write a cosim TCL/shell runner.

This copies the *entire* project including the already-completed
frb_classifier_prj/solution1 (with its cached csynth database), so cosim_design can
reuse the existing synthesized RTL rather than re-running csynth_design -- i.e. this
really is "the unmodified baseline project", just relocated.
"""
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "frb_pooled_hls", "frb_pooled_hls")
DST = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation", "F_cosim", "frb_pooled_hls_copy")
TB_SRC_DIR = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation", "F_cosim", "tb_data_generated")

RUN_COSIM_TCL = """\
open_project frb_classifier_prj
set_top frb_classifier

add_files firmware/frb_classifier.cpp -cflags "-std=c++0x"
add_files -tb frb_classifier_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data

open_solution "solution1"
set_part {xcvu9p-flga2104-2L-e}
create_clock -period 5 -name default

# solution1 already has completed C-synthesis (see syn/report/*csynth*) from the
# original run copied in with this project. Uncomment the next line only if Vitis
# reports it can't find valid RTL to co-simulate against:
# csynth_design

puts "Running C simulation (sanity check against tb_output_predictions.dat)..."
csim_design

puts "Running C/RTL co-simulation..."
cosim_design -trace_level all -rtl verilog

puts "Co-simulation complete. See solution1/sim/report/ for the cosim latency report."
exit
"""

RUN_COSIM_SH = """\
#!/bin/bash
# Co-simulates the UNMODIFIED baseline pooled-model project (this is a copy;
# frb_pooled_hls/ itself was never touched). Reports measured RTL latency to
# compare against the csynth estimate of 12,299 cycles / 61.495 us.
source /opt/Xilinx/Vitis/2025.1/settings64.sh 2>/dev/null || source /opt/Xilinx/2025.1/Vitis/settings64.sh 2>/dev/null || true
cd "$(dirname "$0")"
/usr/bin/time -v vitis-run --mode hls --tcl run_cosim.tcl 2>&1 | tee cosim.log
echo "EXIT_CODE=$?" >> cosim.log
echo "--- csim results (tb_data/csim_results.log) ---"
cat frb_classifier_prj/solution1/tb_data/csim_results.log 2>/dev/null || echo "(not found -- check csim_design output above)"
echo "--- RTL cosim latency report ---"
find frb_classifier_prj/solution1/sim/report -iname "*.rpt" -exec echo {} \\; -exec cat {} \\; 2>/dev/null
"""


def main():
    if os.path.exists(DST):
        print(f"Removing stale copy at {DST}")
        shutil.rmtree(DST)
    print(f"Copying {SRC} -> {DST} (this can take a minute, ~173MB) ...")
    shutil.copytree(SRC, DST)

    tb_dst = os.path.join(DST, "tb_data")
    os.makedirs(tb_dst, exist_ok=True)
    for fname in ("tb_input_features.dat", "tb_output_predictions.dat"):
        shutil.copy(os.path.join(TB_SRC_DIR, fname), os.path.join(tb_dst, fname))

    with open(os.path.join(DST, "run_cosim.tcl"), "w") as f:
        f.write(RUN_COSIM_TCL)
    sh_path = os.path.join(DST, "run_cosim.sh")
    with open(sh_path, "w") as f:
        f.write(RUN_COSIM_SH)
    os.chmod(sh_path, 0o755)

    print(f"Done. Copy + tb_data + cosim runner ready at {DST}")


if __name__ == "__main__":
    main()
