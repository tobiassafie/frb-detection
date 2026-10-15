

## Table 3 — streaming metrics (causal 256-bin window, stride 64; 1.0 h v7-style test stream, 1,716 bursts per orientation)

Rows with threshold **1.000** mean the model could not meet ≤ 10 FA/h on the val stream at any
threshold, because its output is saturated. Their detection rates are not an operating point,
e.g. `v6_loc_t0041_A_w8k5bn_s0` fires continuously on v7 data (2,201 FA/h, so "100 %"
detection). The v6 tiled stream is comparison only; see NOTES_v6 Stage 2.

