# Step 0: Parsing the pooled model's existing csynth reports

Read-only — parsed the 23 report pairs already on disk under
`frb_pooled_hls/frb_pooled_hls/frb_classifier_prj/solution1/syn/report/`. Nothing synthesized,
nothing in `frb_pooled_hls/` touched. Script: [`scripts/parse_step0_reports.py`](scripts/parse_step0_reports.py),
full per-module data: [`step0_report_table.csv`](step0_report_table.csv) (21 rows — the 23 files
minus `csynth_design_size.xml`, which carries no latency/area data, and `csynth.xml`, a duplicate
of `frb_classifier_csynth.xml`).

Vitis reports several internal submodules per Keras layer (a "mult" submodule, a "line buffer",
an "output buffer", etc.) in addition to the top-level instance for that layer. The table below
uses only the **top-level instance per layer** — the same rows the whole-model `frb_classifier_csynth.rpt`
itself reports under `Instance` — so the DSP/LUT/FF/BRAM columns sum exactly to the model total
(580/35,459/14,706/17). Submodule-level detail is in the CSV if you need it.

## Per-layer table

| Layer | Latency (cycles) | II | DSP | LUT | FF | BRAM_18K | DSP % | LUT % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conv2d (1st conv) | 12,291 | 12,288 | 16 | 1,312 | 631 | 0 | 0.23% | 0.11% |
| conv2d_relu | 3,845 | 3,844 | 0 | 214 | 15 | 0 | 0% | 0.02% |
| max_pooling2d | 7,690 | 7,688 | 0 | 936 | 348 | 0 | 0% | 0.08% |
| conv2d_1 (2nd conv) | 2,885 | 2,883 | 31 | 1,935 | 1,016 | 0 | 0.45% | 0.16% |
| conv2d_1_relu | 842 | 841 | 0 | 212 | 13 | 0 | 0% | 0.02% |
| max_pooling2d_1 | 1,684 | 1,682 | 0 | 932 | 346 | 0 | 0% | 0.08% |
| **dense** | **201** | **201** | **524** | **25,363** | **10,270** | 0 | **90.3%** | **71.5%** |
| dense_softmax | 1,047 | 1,047 | 9 | 3,770 | 856 | 4 | 1.55% | 10.6% |
| **TOP (whole model)** | **12,299** | **12,288** | **580** | **35,459** | **14,706** | **17** | **100%** | **100%** |

(% columns are of the model total, not of the FPGA's available resources — see `01_basics.md`/`02_compute` for device-utilization %.)

## Confirm or refute

**(a) "The dense layer accounts for most DSPs" — CONFIRMED.** Dense alone is 524 of 580 total
DSPs (90.3%). Every other layer combined uses only 56 DSPs (16 + 31 + 9). This lines up with
Phase 1's parameter finding: dense holds 786 of the model's 844 parameters (93%) — the DSP count
tracks the dense layer's weight/multiply count almost exactly, not the conv layers' spatial extent.

**(b) "The first conv layer / input streaming dominates latency" — CONFIRMED, dominantly.**
`conv2d`'s own instance latency is 12,291 cycles out of the model's 12,299 total (99.9%). Under
`io_stream` + `dataflow`, most layers execute concurrently as a pipeline rather than strictly
sequentially, so the whole-model latency is not literally the sum of the per-layer numbers above
(they'd sum to ~30,485 cycles if added serially) — it's gated by the slowest stage. `conv2d` is
that stage: at 64×64 input with `ReuseFactor=1` and `Strategy=Latency`, it streams and reads
every one of the 4,096 input pixels with `II=3` per the `ReadInputHeight_ReadInputWidth` loop
report (`TripCount=4096`, `PipelineII=3` → ≈12,288 cycles), and every downstream layer's dataflow
stage has to wait on that pipeline to drain. The second conv layer, by contrast, only takes 2,885
cycles (its input is already the pooled 31×31 output of the first conv+pool stage) — an order of
magnitude less. So it isn't "input streaming" in the generic sense; it's specifically the
**first conv layer's raw pixel-by-pixel read of the full 64×64 input** that sets the latency floor.

---

Stopping here per your instructions. Ready for Step 1 (RF / precision / input-size / flatten-width
sweeps) on your go-ahead.
