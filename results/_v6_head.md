# v6 benchmark + streaming simulation — summary (2026-10-01)

Companion to `results/summary.md` (v7). Every number is next to the `run_id` it comes from.
Per-run rows: `results/v6_benchmark.csv` (v6 / dedup / cross-dataset) and
`results/stream_results.csv` (streaming). Audit, assumptions and incidents: `results/NOTES_v6.md`.

## TL;DR

1. **Audit (NOTES_v6 Stage 0): nothing in the repo evaluated a real stream.**
   * v6/v7 training and eval use independent, fixed-size patches containing the whole
     (residual-)dispersed burst. 97 % of each v7 sweep lies inside its patch.
   * `frb_streaming_pipeline.ipynb` only masks frequency channels of a fixed 0.66 ms patch, using
     a random, unrelated DM. Its extraction crashed on 29/31 files (2 positive patches), and its
     training/streaming cells have no recorded outputs.
   * No detection latency or false-alarm-per-time metric existed.
2. **v6 benchmark (leaky split, comparison only).** Retrained on the original v6 split, the
   v7 architectures score **0.94–0.98** on the v6 test set:
   * best: the sweep's 128² max-pool/flatten CNN, 0.979 ± 0.003 over 3 seeds;
   * the v7 synthesis CNN `s1mB_t0041_rfi`: 0.938 ± 0.005.
   * **None beats the no-learning exact-match lookup (0.984).** Every model has precision 1.000
     only because all 154 v6 test noise images are verbatim training images.
   * On the deduplicated test (149 FRBs, 0 noise) recall is 0.91–0.99. The old v6 "96–99 %"
     numbers measure memorisation of 5 noise images plus detection of FRBs on those same 5
     backgrounds.
3. **Cross-dataset: no transfer either way.**
   * v6-trained models on the v7 test: 0.41–0.59.
   * v7-trained models on the v6 test: 0.47–0.53.
   * The two datasets contain different "FRBs" (v6: DM 100–2500 slivers in the top channels;
     v7: real residual sweeps).
4. **Streaming (new): orientation is the dominant effect.**
   * In a causal 1-hour stream with the sweep in the orientation every training sample had,
     the v7 CNN `s1mB_t0041_rfi` detects 82.7 % of S/N ≥ 7 bursts (96.2 % at S/N ≥ 15).
   * With physical high-frequency-first arrival it detects **1.9 %**. The patch-trained models
     learned the sweep direction.
   * The **stream-trained CNN (`strm_t0041_s1`) detects 77.5 % / 78.2 % (physical / stored) at
     S/N ≥ 7** at the same val-chosen operating point, triggering after ~81–86 % of the band
     has arrived. Its false-alarm rate on the test files is higher, though: 494 vs 80 per hour.
   * At that operating point the RNNs and the v6-trained models detect **≤ 17 %** of S/N ≥ 7
     bursts. At a fixed 0.5 threshold the v7 RNNs detect 70–85 %, but at 4,000–5,000 FA/h.
5. **False-alarm rates do not transfer from val files to test files.** A threshold giving
   8 FA/h on the val stream gives 80–500 FA/h on the test stream. With only 5–6 files per split,
   RFI differs a lot between files. The FA/h column should be read as "order of magnitude".

