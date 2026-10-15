# NOTES_v6a — targeted v6 experiments (2026-10-07)

Every number here is on the **original v6 split** (`augmented_dataset/lstm_dataset_v6.npz` as on
disk, float32, not re-split) — **v6 benchmark - leaky split, comparison only**. Per-run rows:
`results/v6a_runs.csv` (from `results/runs_v6a/*.json`). Scripts: `scripts/v6a_train.py`
(trainer + context columns), `scripts/v6a_eval_saved.py` (saved historical weights).

Context columns on every run:
* lookup baseline (0.984);
* dedup recall (149 FRBs, 0 noise);
* v7 test accuracy;
* time-reversed v6 test accuracy;
* **memorisation probe**: noise / FRB accuracy on the v6 test set with N(0, 0.05) added per
  pixel. v6 pixels are z-scored, so 0.05 σ is invisible by eye.

## Stage 0 — what already worked

### Search
* `train_cnn(2).py` does not exist anywhere in the working tree or git history
  (`git log --all -- "*train_cnn*"`). The only script is `train_cnn.py` (3-conv [4,8,8]).
* No ~99 % model or notebook exists for v6. I scanned every notebook's saved outputs. The best
  reported v6 test accuracies are:
  * 0.9740: `cnn_training_v3/v4.ipynb` / `models/cnn_v1`, and ablation `group1_f16`/`group1_f64`;
  * 0.9675: `seedcheck_baseline_s142` = `fpga_handoff/group1`;
  * 0.9643: `lstm_training_2.ipynb` / `models/lstm_v3`;
  * 0.9838: my v6 sweep (`v6_sw_r128_max_flatten_d16_seed1`).
* The 0.952–0.966 numbers in `classifier/*.ipynb` are on FETCH data (`classifier/fetch/*.hdf5`),
  not v6.

### Saved weights reproduce exactly
All 12 saved models score exactly their reported test accuracy on the original v6 test set
(`stage0_saved` rows).

### Retrained from scratch with each original recipe
Recipe (`scripts/ablation_arch.py`, `train_cnn.py`, notebook CONFIGs): Adam 1e-3, batch 128,
class weights, ES(val_loss, patience 30), RLROP(0.5, patience 10), ≤ 150 epochs; full
256 × 256 input, no extra normalisation.

| run | reproduces | reported | reproduced (test) | why different |
|---|---|---|---|---|
| `s0_baseline_2_4_4_4_4_s142` | seedcheck_baseline_s142 | 0.9675 | **0.9773** | Same recipe and seed; TF/Keras version and CPU op order differ, so ±1 pt is ordinary run-to-run variance on 308 samples (1 pt = 3 samples). |
| `s0_train_cnn_py_4_8_8` | train_cnn.py | (none recorded) | 0.9773 | — |
| `s0_lstm_v3_recipe` | lstm_v3 | 0.9643 | **0.7662** | Not reproduced at seed 42. The original training log (lstm_training_2.ipynb cell 11) ran all 150 epochs (best epoch 145, val acc still rising). This run plateaued, so ES(patience 10) stopped it at epoch 49 (val 0.78). Rerun as `s0_lstm_v3_recipe_full150`, same recipe without the early stop. |

No seed collapsed: the collapse-retry rule (restart if best val < 0.65) never fired.

### Finding that changes how to read v6 numbers
The memorisation probe splits the historical models into two groups (noise accuracy on
perturbed noise):

* robust: `cnn_v1` 0.994, `group2_b_front_loaded` 0.981, `s0_baseline_2_4_4_4_4_s142` 1.000,
  `lstm_v3` 0.948;
* lookup-like:
  * my v6 sweep `v6_sw_r128_max_flatten_d16` × 3 seeds: 0.08 / 0.31 / 0.25;
  * `group1_f64` (134k): 0.461;
  * `seedcheck_baseline_s142`: 0.734.

The sweep models that top the v6 test (0.977–0.984) recognise the 5 noise images by their
exact pixels. Add invisible noise and they call most noise "FRB". Their extra ~1 pt over the
old recipe is memorisation, not detection. Both groups are reported; the probe column is beside
every number.

## Stage 1 — CNN hypotheses (pre-registered before the runs)

**Base** = best reproduced model by v6 **val** accuracy: the `v6_sw_r128_max_flatten_d16`
architecture, i.e.:
* 128 × 128 input (2 × 2 block mean); Conv[4,8,16,16,16] k3 + ReLU + MaxPool2 ×5; Flatten;
  Dense16 + dropout 0.2; 10,273 params;
* recipe: lr 1.09e-3, batch 64, ≤ 400 epochs, ES patience 60, RLROP patience 30, no class
  weights;
* 3 existing seeds: val 0.958 / 0.961 / 0.964, test 0.977 / 0.984 / 0.977 (mean 0.979).

Each experiment changes **one** thing, at seed 0, and is compared with that 3-seed spread:
* inside the spread (test 0.977–0.984) → inconclusive;
* outside it → confirmed / rejected.

**Step budget for H3.** Batch, epochs and patience stay identical, so every resolution gets the
same number of optimizer steps (23 steps/epoch × ≤ 400 epochs). That keeps the LR schedule
unchanged, as asked, without oversampling.

| id | hypothesis | single change | prediction |
|---|---|---|---|
| H1 | flatten beats GAP (burst position is informative on v6) | head → GAP | GAP lower (prior sweep: max/GAP 0.971 vs max/flatten 0.979) |
| H2 | max-pool beats avg-pool on short bright sweeps | pool → avg | avg ≤ max (prior sweep: avg/flatten 0.976) |
| H3 | resolution | input 256 (t,f pool 1) and 64 (pool 4) | 256 ≥ 128 > 64 |
| H4 | shrinking within 0.5 pt | filters & dense ×½ ([2,4,8,8,8], d8) and ×¼ ([1,2,4,4,4], d4) | ½ within 0.5 pt, ¼ drops |
| H5 | the last 1–2 pts are variance | (a) L2 1e-4 + dropout 0.3; (b) augmentation: random circular time shift ±16 bins + Gaussian noise σ = 0.1 | test ±, memorisation probe much more robust for (b) |

**LSTM reproduction resolved.** `s0_lstm_v3_recipe_full150` (same recipe, all 150 epochs, no
early stop) scores test **0.9610** vs reported 0.9643 (1 sample). The 0.766 run was an early stop
on a plateau. Stage 2 therefore uses the lstm_v3 recipe with all 150 epochs:
* Adam 1e-3, batch 128, class weights, RLROP(0.5, patience 5);
* the best-val epoch is restored at the end.

## Stage 2 — many-to-one LSTM/GRU (pre-registered)

**How lstm_dataset_v6 was meant to be read.** Each sample is (256 time, 256 freq): the
notebook's own summary says "LSTM input shape: (batch, 256, 256)". lstm_training_2 feeds
**time steps as the sequence** and all 256 frequency channels as features, reduced per step by a
1×1 Conv1D to 64. So the frequency-time ridge appears as a sequence of spectra.

The v7 RNNs (64 t × 8 f) threw away 31/32 of the frequency information. v6's burst lives in a
handful of edge channels, so heavy frequency pooling should hurt even more here.

Baseline: `s0_lstm_v3_recipe_full150` (256 t × 256 f → Conv1D(64, k1) → LSTM(32) → Dense32 →
dropout 0.3; 29,953 params; test 0.961).

| id | hypothesis | runs | prediction |
|---|---|---|---|
| H6 | more frequency channels per step with fewer steps keeps accuracy at a fraction of the cost | LSTM(32), no 1×1 reduction, input (T,F) = (64,64), (32,128), (64,128) | (64,128) ≈ baseline; (64,64) lower |
| H7 | 32/64 hidden units beat 16; GRU vs LSTM | at the best H6 shape: LSTM(64), GRU(32), LSTM(16) | 64 ≥ 32 > 16; GRU ≈ LSTM |
| H8 | a small Conv1D front end (over time, frequency as channels) helps if H6/H7 plateau below the CNN | Conv1D(32,k5)+MaxPool2 → Conv1D(32,k5)+MaxPool2 → LSTM(32) at the best shape | small gain |

## Stage 1 — outcomes

Base spread (3 seeds of the base config): test 0.977–0.984, val 0.958–0.964. All runs seed 0;
`noise@0.05` = v6 test noise accuracy with invisible N(0, 0.05) added.

| id | run | params | val | test | noise@0.05 | verdict |
|---|---|---|---|---|---|---|
| H1 | `s1_H1_gap` | 6,433 | 0.942 | 0.968 | 0.18 | **confirmed**: flatten > GAP (below the spread on val and test) |
| H2 | `s1_H2_avgpool` | 10,273 | 0.961 | 0.977 | 0.77 | **inconclusive**: inside the spread; avg-pool memorises somewhat less |
| H3 | `s1_H3_res256` | 22,561 | 0.964 | 0.984 | 0.24 | inconclusive at seed 0 (top of spread) → 3-seed check below |
| H3 | `s1_H3_res64` | 7,201 | 0.958 | 0.981 | 0.50 | **inconclusive**: 64 is as good as 128 |
| H4 | `s1_H4_half` [2,4,8,8,8], d8 | 2,601 | 0.951 | 0.977 | 0.17 | **within 0.5 pt** of the base mean (0.979): smallest acceptable |
| H4 | `s1_H4_quarter` [1,2,4,4,4], d4 | 667 | 0.925 | 0.951 | 0.31 | **confirmed drop** (−2.8 pt) |
| H5a | `s1_H5a_l2_drop` L2 1e-4 + dropout 0.3 | 10,273 | 0.948 | 0.974 | 0.25 | **rejected**: no gain |
| H5b | `s1_H5b_aug` shift ±16 + noise 0.1 | 10,273 | 0.909 | 0.938 | **0.994** | **rejected for v6 accuracy** (−4 pt), but the only variant that stops memorising |

**Reading.** Each single change moves v6 test by ≤ 1 pt (inside seed noise), except GAP, the
¼-size model and augmentation. But every variant on this recipe (no class weights, batch 64,
≤ 400 epochs) is lookup-like on the probe.

Meanwhile the old v6 recipe (`s0_baseline_2_4_4_4_4_s142`: 256², [2,4,4,4,4], Dense32/drop 0.3,
class weights, batch 128, ES 30, ≤ 150 epochs) reaches **0.977** with noise@0.05 = **1.000**.

On this split, test accuracy above ~0.977 is bought with memorising the 5 noise images.
Augmentation that blocks the memorisation costs ~4 pt of v6 test. That is the size of the leak.

**Top-3 configs for 3 seeds (by val):**
1. `s1_H3_res256` (0.964);
2. the base (0.961 mean; already has 3 seeds);
3. a 3-way tie at 0.961 between `s1_H2_avgpool`, the base and `s0_baseline`. Tie-break, stated
   before seeing seeds: the memorisation probe → `s0_baseline` (old recipe).

### Top-3 CNN configs, 3 seeds (mean ± std; v6 test)

| config | runs | params | val | test | noise@0.05 |
|---|---|---|---|---|---|
| 256 input, base architecture (H3) | `s1_H3_res256`, `_seed1`, `_seed2` | 22,561 | 0.964 ± 0.003 | **0.984 ± 0.000** | 0.31 ± 0.08 |
| base (128, max, flatten) | `v6_sw_r128_max_flatten_d16[_seed1/2]` (previous session) | 10,273 | 0.961 ± 0.002 | 0.979 ± 0.003 | 0.21 ± 0.09 |
| old recipe [2,4,4,4,4] 256² | `s0_baseline_2_4_4_4_4_s142`, `_seed1`, `_seed2` | 8,797 | 0.948 ± 0.010 | 0.966 ± 0.008 | **0.998 ± 0.003** |

H3 verdict over 3 seeds: 256 ≥ 128 by 0.5 pt on test and 0.3 pt on val. That is **weakly
confirmed**, but it is also the most memorising.

The best v6 CNN reaches **0.984**: precision 1.000 and 149/154 FRBs, with every deduplicated FRB
detected.

**Its 5 misses are not model errors. They are label noise in v6, and they set a hard ceiling.**
* The 5 missed test "FRBs" (test idx 135, 153, 212, 231, 255) are **byte-identical to noise
  images**: the injected burst fell outside the patch, so the "FRB" is a pure noise image with
  label 1. Each of them appears ~130–150× in train labelled noise.
* Counted over the whole dataset, 36/718 train, 10/154 val and 5/154 test FRB-labelled samples
  are pixel-identical to a noise image.
* Any deterministic classifier gives identical inputs identical outputs, so the **maximum
  achievable accuracy is 303/308 = 0.9838 on test and 0.9675 on val**.
* `s1_H3_res256_seed2` hits **both ceilings exactly** (val 0.9675, test 0.9838).

**≥ 98 % is achieved, and 99 % is mathematically impossible on this test set.** This includes
the old 97.4 % models: they sit 1 pt (3 samples) below the ceiling.

## Stage 2 — outcomes (seed 42 unless noted)

| id | run | input (T × F) | params | val | test | noise@0.05 | verdict |
|---|---|---|---|---|---|---|---|
| baseline | `s0_lstm_v3_recipe_full150` (+2 seeds) | 256 × 256 → Conv1D(64,k1) → LSTM32 | 29,953 | 0.947 ± 0.004 | **0.963 ± 0.003** | 0.89 | best RNN |
| H6 | `s2_H6_T64_F64` | 64 × 64 | 13,505 | 0.916 | 0.948 | 0.73 | |
| H6 | `s2_H6_T32_F128` | 32 × 128 | 21,697 | 0.912 | 0.942 | 0.76 | |
| H6 | `s2_H6_T64_F128` | 64 × 128 | 21,697 | 0.938 | 0.945 | 0.77 | best H6 shape |
| H7 | `s2_H7_lstm64` (+2 seeds) | 64 × 128, LSTM64 | 51,521 | 0.938 ± 0.008 | 0.953 ± 0.007 | 0.81 | |
| H7 | `s2_H7_gru32` | 64 × 128, GRU32 | 16,641 | 0.942 | 0.955 | — | |
| H8 | `s2_H8_conv1d_front` | 64 × 128, 2×[Conv1D(32,k5)+pool2] → LSTM32 | 35,073 | 0.912 | 0.945 | — | |

* **H6:** confirmed that frequency resolution matters: all three ≫ the v7-style 64 × 8 RNNs.
  But fewer time steps do not keep the baseline's accuracy (−1.5 to −2 pt vs 256 × 256 with a
  1×1 reduction), so **rejected** as "same accuracy at lower cost".
* **H7:** LSTM64 > LSTM32 at 64 × 128 (+1.6 pt at seed 42; 3-seed 0.953) — **weakly confirmed**.
  GRU32 ≈ LSTM32/64 (0.955) — **inconclusive**.
* **H8:** the Conv1D front end did not help (0.945) — **rejected**.

Bidirectional was not run: H8's Conv1D variant already showed no gain, and the run budget was
used for seeds.

## Stage 4 — synthesis readiness

`which vitis_hls vitis-run g++ gcc clang cl` → nothing. No WSL distro, no `C:\Xilinx`. hls4ml C-sim
needs a C++ compiler, so it was **not run**. Projects are written to `hls_projects_v6a/`
(`scripts/v6a_export.py`); index in `results/v6a_hls_index.csv`:
* ap_fixed<16,6>, Resource, RF 4; CNN io_stream; RNN io_parallel with static=True;
* `emu_test_acc_v6` = Python fixed-point emulation, not C-sim;
* `est_*` = analytic estimates, not measured.

RNN synthesis risk:
* lstm_v3 takes a 256 × 256 io_parallel input (65,536 values at once) plus a 256-step LSTM:
  est. 7.4k multipliers at RF 4.
* LSTM64 at 64 × 128 is est. 12.8k multipliers.

Both are large for io_parallel; the smaller H6 shapes (64 × 64 LSTM32, 13.5k params) are the safer
synthesis candidates, at −1.5 pt.

## Stage 3 — context columns

Context columns on every run (`results/v6a_runs.csv`):
* (a) lookup baseline 0.984;
* (b) dedup FRB recall;
* (c) v7 test accuracy (v7 patches per-patch z-scored = v6 convention, then pooled);
* (d) time-reversed v6 test accuracy;
* the +0.05 σ memorisation probe.

Streaming (`results/stream_results_v6a.csv`, reusing `scripts/stream_sim.py`/`stream_eval.py`):
* evaluated for 5 CNNs and the best LSTM, in stored and physical (high-frequency-first)
  orientation;
* v6 models get per-window z-score preprocessing, as in v6 training;
* all are unusable as triggers (see summary_v6a.md).

**v6 has no consistent sweep direction.** Measured on 60 v6 FRB test patches: median ridge
slope ≈ 0 and 47 % positive. The signal is a near-vertical sliver at the band edges, so "physical
orientation" is only defined for the v7-style stream.
