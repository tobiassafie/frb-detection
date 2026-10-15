# Best rolling model — stats sheet

The model is `r_gru32_f16_p4_onset` (file: `models/best/rolling/rolling_gru32_best_r_gru32_f16_p4_onset.keras`).
* Two more seeds exist: `r_gru32_f16_p4_onset_s142` and `_s242` (`results/models_roll/`).
* All numbers are on the original v6 split, a leaky benchmark (comparison only).
* "Measured" = timed on the development machine (AMD Ryzen 9 7900X, CPU only).
* `est_` = analytic estimate, not measured on hardware.

## Architecture
* **Front (upstream, no multipliers):** causal time max-pool ×4 over the time-reversed v6 patch.
  Each step is 4 native bins of 0.655 ms = **2.62 ms of data**, with 256 frequency channels.
* **Model, applied per step:**
  1. Conv1D(16, kernel 1) + ReLU, i.e. 256 → 16;
  2. GRU(32), unidirectional, carried state;
  3. Conv1D(16, kernel 1) + ReLU;
  4. Conv1D(1, kernel 1) + sigmoid → P(FRB | data up to t).
* One 168 ms patch = 64 steps.

## Parameters: 9,457
| layer | params |
|---|---|
| Conv1D front 256 → 16 | 4,112 |
| GRU 32 (reset_after, 3 gates) | 4,800 |
| head Conv1D 32 → 16 | 528 |
| output Conv1D 16 → 1 | 17 |

Weights at 16 bits ≈ **18.9 KB**; float32 ≈ 37.8 KB.

## Compute
| | MACs | FLOPs (≈ 2 × MACs) |
|---|---|---|
| per step (2.62 ms of data) | **9,232** | **≈ 18.5k** |
| per 168 ms patch (64 steps) | 590,848 | ≈ 1.18M |
| sustained rate in real time | 3.52M MAC/s | **≈ 7.0 MFLOP/s** |

* Per-step breakdown: front 4,096 + GRU 4,608 (3 × 32 × (16 + 32)) + head 512 + output 16 MACs.
* Not in the table: 96 sigmoid/tanh evaluations per step (gates) + 1 output sigmoid, and the
  upstream max-pool (256 channels × 3 comparisons = 768 comparisons per step).

## Time / latency
| | value | type |
|---|---|---|
| real-time budget per step | 2.62 ms (4 × 0.655 ms bins) | definition |
| one step, plain numpy (batch 1) | **0.035 ms** (≈ 75× faster than real time); matches Keras to 3.6e-7 | measured |
| one step, Keras stateful call (batch 1) | 8.2 ms; slower than real time, all Python / framework overhead | measured |
| whole patch, Keras batch (303 patches together) | 1.1 ms per patch | measured |
| decision delay from data | ≤ 1 step (2.62 ms): P_t updates once each 4-bin block is complete | by construction |
| detection latency after burst onset (test, 3 seeds) | median 0 / −4 / +4 bins (≈ 0 ms ± 2.6 ms); p90 8–28 bins (5–18 ms) | measured |
| FPGA (hls4ml, io_parallel, static, RF 16) | est_ ≈ 30–40 cycles per step ≈ 0.2 µs at 200 MHz; est_ multipliers ≈ 580 | **est_, not synthesised** |

Detection-latency caveat: on v6, 61 % of the injected burst energy precedes the visible sweep onset (a
v6 augmentation artefact), so "≈ 0 bins after onset" partly reflects detecting that precursor content
(see `results/summary_roll.md`, Stage 2).

## Training data (v6, original split; the 36/10/5 FRB-labelled samples with no burst are excluded)
| split | patches | FRB | noise | input tensor (after ×4 pooling) | size float32 | labelled steps | positive steps |
|---|---|---|---|---|---|---|---|
| train | 1,400 | 682 | 718 | 1,400 × 64 × 256 | 91.8 MB | 89,600 | 31,110 |
| val | 298 | 144 | 154 | 298 × 64 × 256 | 19.5 MB | 19,072 | 6,777 |
| test | 303 | 149 | 154 | 303 × 64 × 256 | 19.9 MB | 19,392 | 6,750 |

* Raw patches are 256 × 256 float32 (256 KB each), so the train split is 367 MB before pooling.
* Unique content: the noise class is only **5 distinct images**; the FRB class derives from 27 real
  CHIME bursts.

## Training cost
* Recipe: Adam 1e-3, batch 128 (11 steps/epoch), 150 epochs (1,650 optimizer steps),
  ReduceLROnPlateau, best-val restore.
* Time: **≈ 70–75 s per run** on CPU (measured: 75.3 / 71.0 / 69.6 s for the 3 seeds).
* Seeds s142 and s242 restored epoch 150 (still improving), so a longer run might help a little.
