
## Reading the streaming table

* Every v6-trained model is either saturated in a causal stream of real CHIME noise (threshold
  1.000: no threshold reaches ≤ 10 FA/h on the val stream) or fires independently of bursts.
  The "≈ 0.3 detected" rows have identical rates in both sweep orientations, at 560–3,900 FA/h.
* None is usable as a streaming trigger, including the 0.984 ceiling model.
* The v6 tiled stream (v6 test patches butted end-to-end; see NOTES_v6) is comparison only.

## Bottom line

* **CNN:** on this split, `s1_H3_res256*` (22.6k params) reaches the **mathematical ceiling of
  0.9838** on 3/3 seeds. `s1_H4_half` (2.6k params) is the smallest within 0.5 pt (0.977).
* **RNN:** the lstm_v3 recipe is best (0.963 ± 0.003, 30k params). Smaller-input RNNs are
  1–2 pt lower.
* The CNNs are hls4ml-ready (emulated at ap_fixed<16,6> without loss). The RNNs are written but
  large for io_parallel.
* None of these numbers transfers off v6. Use the v7 / streaming results (`summary.md`,
  `summary_v6.md`) for any claim about detecting new bursts.
