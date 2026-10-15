"""
Step E: generate two candidate hls4ml projects that each try one real, wired
config-level fix for conv2d's achieved-II=3 vs target-II=1 shortfall (see
step0_results.md / step_E_diagnosis.md for the diagnosis).

This ONLY runs hls4ml's Python conversion (`convert_from_keras_model(...).write()`),
which generates the C++ project files -- it does NOT call `.build()` and does NOT
invoke Vitis HLS. Actual synthesis is done afterwards on the EC2 box via the
generated run_synthesis.sh/.tcl, matching how frb_pooled_hls itself was built
(see run_synthesis.sh: `vitis-run --mode hls --tcl run_synthesis.tcl`).

Investigated and rejected as a candidate: `ConvImplementation: Encoded`. The
installed hls4ml's Vitis-backend io_stream Conv2D template
(hls4ml/templates/vitis/nnet_utils/nnet_conv2d_stream.h) has
`assert(CONFIG_T::implementation == conv_implementation::linebuffer && ...)` --
'Encoded' would fail to compile for io_stream Conv2D on this backend, so it's not
a usable fix here.

Read-only w.r.t. frb_pooled_hls: only ever reads classifier/frb_classifier_pooled.h5
and frb_pooled_hls/frb_pooled_hls/hls4ml_config.yml for reference. Writes new,
independent project directories under synth_ablation/E_fixes/.
"""
import os
import tensorflow as tf
import hls4ml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_BASE = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation", "E_fixes")
MODEL_PATH = os.path.join(ROOT, "results", "ari_meeting", "synth_ablation", "rebuilt_pooled_model.keras")

COMMON_KWARGS = dict(
    backend="Vitis",
    io_type="io_stream",
    clock_period=5,
    clock_uncertainty="27%",
    part="xcvu9p-flga2104-2L-e",
)

RUN_SYNTHESIS_TCL_TEMPLATE = """\
open_project {project_name}_prj
set_top frb_classifier

add_files firmware/frb_classifier.cpp -cflags "-std=c++0x"
add_files -tb frb_classifier_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data

open_solution "solution1"
set_part {{xcvu9p-flga2104-2L-e}}
create_clock -period 5 -name default

config_compile -pipeline_loops 64

puts "Starting C synthesis..."
csynth_design

puts "Synthesis complete!"
exit
"""

RUN_SYNTHESIS_SH_TEMPLATE = """\
#!/bin/bash
# Same invocation style as frb_pooled_hls/frb_pooled_hls/run_synthesis.sh --
# uses the custom TCL workflow via `vitis-run`, NOT hls4ml's hls_model.build().
source /opt/Xilinx/Vitis/2025.1/settings64.sh 2>/dev/null || source /opt/Xilinx/2025.1/Vitis/settings64.sh 2>/dev/null || true
cd "$(dirname "$0")"
/usr/bin/time -v vitis-run --mode hls --tcl run_synthesis.tcl 2>&1 | tee synthesis.log
echo "EXIT_CODE=$?" >> synthesis.log
"""


def build_base_config(model):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", backend="Vitis")
    cfg["Model"]["Precision"]["default"] = "ap_fixed<16,6>"
    cfg["Model"]["ReuseFactor"] = 1
    cfg["Model"]["Strategy"] = "Latency"
    cfg["Model"]["BramFactor"] = 1000000000
    cfg["Model"]["TraceOutput"] = False
    return cfg


def write_project(model, cfg, out_dir, project_name):
    os.makedirs(out_dir, exist_ok=True)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, output_dir=out_dir, project_name=project_name, hls_config=cfg, **COMMON_KWARGS
    )
    hls_model.write()  # generate project files only -- NOT .build(), no synthesis invoked

    with open(os.path.join(out_dir, "run_synthesis.tcl"), "w") as f:
        f.write(RUN_SYNTHESIS_TCL_TEMPLATE.format(project_name=project_name))
    sh_path = os.path.join(out_dir, "run_synthesis.sh")
    with open(sh_path, "w") as f:
        f.write(RUN_SYNTHESIS_SH_TEMPLATE)
    os.chmod(sh_path, 0o755)
    return hls_model


def main():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    assert model.count_params() == 844, f"unexpected param count {model.count_params()}"

    # --- E1: ParallelizationFactor on conv2d/conv2d_1 (real, wired hls4ml knob --
    # confirmed in hls4ml/backends/vivado/vivado_backend.py:init_conv2d, which the
    # Vitis backend inherits). Values are the largest hls4ml-valid divisor of each
    # conv layer's output H*W, per get_valid_conv_partition_splits():
    #   conv2d output 62x62=3844 -> valid {1,2,4,31,62,124,961,1922,3844}, pick 4
    #   conv2d_1 output 29x29=841 -> valid {1,29,841}, pick 29
    cfg_e1 = build_base_config(model)
    cfg_e1["LayerName"]["conv2d"]["ParallelizationFactor"] = 4
    cfg_e1["LayerName"]["conv2d_1"]["ParallelizationFactor"] = 29
    write_project(model, cfg_e1, os.path.join(OUT_BASE, "E1_parallelization_factor"), "frb_classifier_E1_pf")

    # --- E2: explicit ReuseFactor=3 on conv2d/conv2d_1, matching the achieved II=3
    # found in Step 0/E diagnosis, instead of leaving RF=1 (which asks for an
    # unattainable II=1 that Vitis silently downgrades to 3 anyway).
    cfg_e2 = build_base_config(model)
    cfg_e2["LayerName"]["conv2d"]["ReuseFactor"] = 3
    cfg_e2["LayerName"]["conv2d_1"]["ReuseFactor"] = 3
    write_project(model, cfg_e2, os.path.join(OUT_BASE, "E2_reusefactor3"), "frb_classifier_E2_rf3")

    print("Wrote E1_parallelization_factor/ and E2_reusefactor3/ under", OUT_BASE)


if __name__ == "__main__":
    main()
