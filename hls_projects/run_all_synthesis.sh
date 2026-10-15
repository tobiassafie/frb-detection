#!/bin/bash
# Run on the EC2 box with Vitis HLS 2025.1 (same setup as fpga_handoff/*/run_synthesis.sh).
#   bash run_all_synthesis.sh [parallel_jobs]      (default: nproc/4)
# Every project's run_vitis_synthesis.sh does csynth only and is killed after 20 min ("timeout").
# Then:  python3 parse_synth_reports.py   -> synth_results.csv  (copy back to results/)
#        python3 check_accuracy.py <proj>  -> bit-accurate C-sim accuracy (needs g++ + hls4ml)
cd "$(dirname "$0")"
J=${1:-$(( $(nproc) / 4 ))}; [ "$J" -lt 1 ] && J=1
ls -d */ | grep -v '^_' | sed 's#/##' | xargs -P "$J" -I{} bash -c 'echo "[start] {}"; bash {}/run_vitis_synthesis.sh; echo "[done] {} $(cat {}/synth_status.csv)"'
python3 parse_synth_reports.py
