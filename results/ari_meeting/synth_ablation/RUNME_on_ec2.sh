#!/bin/bash
# Run this on the EC2 instance that has Vitis HLS 2025.1 installed, from inside
# this synth_ablation/ directory (or adjust the paths below).
#
# Runs, in order:
#   1. E1_parallelization_factor  (csynth only)
#   2. E2_reusefactor3            (csynth only)
#   3. F_cosim/frb_pooled_hls_copy (csim + cosim on the UNMODIFIED baseline)
#
# Each run is capped at 60 minutes; a timeout is recorded as a result, not hidden.
# Nothing under the real frb_pooled_hls/ project is touched by any of this.

set -uo pipefail
TIMEOUT_SEC=3600
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/run_summary.csv"
echo "run,start,end,wall_seconds,exit_status" > "$LOG"

run_one () {
    name="$1"
    dir="$2"
    script="$3"
    echo "=== [$name] starting in $dir ==="
    start=$(date +%s)
    start_iso=$(date -Iseconds)
    ( cd "$dir" && timeout "$TIMEOUT_SEC" bash "$script" )
    status=$?
    end=$(date +%s)
    end_iso=$(date -Iseconds)
    wall=$((end - start))
    if [ "$status" -eq 124 ]; then
        echo "=== [$name] TIMED OUT after ${TIMEOUT_SEC}s -- recording as a result, not hiding it ==="
    fi
    echo "$name,$start_iso,$end_iso,$wall,$status" >> "$LOG"
    echo "=== [$name] done: exit=$status wall=${wall}s ==="
}

run_one "E1_parallelization_factor" "$HERE/E_fixes/E1_parallelization_factor" "run_synthesis.sh"
run_one "E2_reusefactor3"           "$HERE/E_fixes/E2_reusefactor3"           "run_synthesis.sh"
run_one "F_cosim_baseline"          "$HERE/F_cosim/frb_pooled_hls_copy"       "run_cosim.sh"

echo ""
echo "All done. Summary: $LOG"
echo "Send back (or point me at):"
echo "  - E_fixes/E1_parallelization_factor/frb_classifier_E1_pf_prj/solution1/syn/report/*csynth*"
echo "  - E_fixes/E1_parallelization_factor/synthesis.log"
echo "  - E_fixes/E2_reusefactor3/frb_classifier_E2_rf3_prj/solution1/syn/report/*csynth*"
echo "  - E_fixes/E2_reusefactor3/synthesis.log"
echo "  - F_cosim/frb_pooled_hls_copy/cosim.log"
echo "  - F_cosim/frb_pooled_hls_copy/frb_classifier_prj/solution1/sim/report/  (cosim latency report)"
echo "  - F_cosim/frb_pooled_hls_copy/frb_classifier_prj/solution1/tb_data/csim_results.log and rtl_cosim_results.log"
echo "  - $LOG"
