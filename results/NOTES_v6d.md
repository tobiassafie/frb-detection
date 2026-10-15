# NOTES_v6d — small-RNN sweep for synthesis cost (2026-10-07, start 22:32, cap 75 min)

**v6 benchmark - leaky split, comparison only.**
* Data: `lstm_dataset_v6.npz` as on disk, original split.
* Runs: `results/runs_v6d/` → `results/v6d_runs.csv` (`V6A_TAG=v6d`, `scripts/v6a_train.py`).
* Reference: lstm_v3 = Conv1D(64,k1) → LSTM(32) → Dense32, 256 steps, 7.34M MACs, test
  0.963 ± 0.003.
* Lookup baseline 0.984; v6 test ceiling 0.9838.

## Recipe (every run)
* Exact lstm_v3: raw v6, Adam 1e-3, batch 128, class weights, RLROP(0.5, patience 5),
  150 epochs.
* No early stopping; the best-val-loss epoch is restored at the end.
* Retry if best val < 0.65 or val-prediction std < 1e-3, with seed + 1000·k. Attempts are logged;
  a collapsed seed is rerun with a new seed, never reported.
* Seeds 42 / 142 / 242.

## Architecture grid
input (256 t × 256 f) → upstream time pooling ×p (avg or max over p bins, frequency untouched) →
Conv1D(c, k1, relu) over time with frequency as channels → LSTM/GRU(u) → Dense(d, relu) →
dropout 0.3 → sigmoid.
* c ∈ {16, 32, 64}, p ∈ {2, 4, 8, 16}, cell ∈ {LSTM, GRU}, u ∈ {32, 64}, d ∈ {32, 64}
  → 96 configs.

## Budget plan
96 configs × 3 seeds = 288 runs will not fit 75 min on this CPU (~1–3 min per run). So:
* **A.** avg vs max pooling at one setting (c64, p4, LSTM32, d32), 3 seeds each (6 runs). The
  winner (by mean val) is reused everywhere.
* **B.** Seed 42 of all 96 configs with the chosen pooling.
* **Prune rules, applied after seed 42:**
  * est. MACs > lstm_v3's 7.34M (none expected: every config pools time ≥ 2×);
  * test < 0.915, i.e. > 3 pt below 0.945 (rule as given; note that it prunes on test).
* **C.** Seeds 142 and 242 for the survivors, highest seed-42 val first, until the cap.
  Configs that did not get 3 seeds are reported as such (n shown beside every mean).

## Outcomes (stopped at 23:32 to finish inside the 75-min cap)

* **A. avg vs max time pooling** (c64, p4, LSTM32, d32; seeds 42/142/242):
  * max: val 0.952, test 0.964 ± 0.003 (`d_A_max_s*`);
  * avg: val 0.920, test 0.943 ± 0.019 (`d_A_avg_s*`).

  **Max chosen** and reused everywhere. v6 bursts are short, bright slivers; a mean over p bins
  dilutes them, a max keeps them.
* **B. seed 42, all 96 configs:** done (no retries or collapses needed in any run;
  `n_attempts = 1` everywhere).
  * MAC prune removed nothing: every config is ≤ 6.30M < 7.34M.
  * Test-acc prune (< 0.915) removed 4: `c32_p4_lstm32_d32`, `c16_p2_lstm32_d64`,
    `c64_p4_gru64_d32`, `c32_p2_lstm64_d32`.
* **C. seeds 142/242:** 114 of the 184 queued runs finished before the cutoff.
  * **58 configs have all 3 seeds.**
  * **34 configs have seed 42 only (not reached).** They are queued by seed-42 val, so the
    unreached ones are the lower-val ones.
  * The 4 pruned configs have seed 42 only by design.
  * Every number in `results/v6d_configs.csv` carries `n_seeds`.

### Named picks (3 seeds each)
* **Cheapest within 1 pt of 0.963: `c16_p16_gru32_d32`.**
  * Max-pool ×16 (16 time steps) → Conv1D(16) → GRU(32) → Dense32.
  * **10,001 params, 140,320 MACs (52× fewer than lstm_v3)**.
  * test **0.965 ± 0.002**, val 0.957 ± 0.005, probe 0.93, v7 0.52.
  * run_ids `d_c16_p16_gru32_d32_s42/_s142/_s242`.
* **Best under 1M MACs: `c32_p8_lstm32_d64`.**
  * Max-pool ×8 → Conv1D(32) → LSTM(32) → Dense64.
  * 18,721 params, 526,400 MACs.
  * test **0.975 ± 0.002**, val 0.958 ± 0.000, probe 0.84, v7 0.53.
  * It ties `c32_p16_gru64_d64` (0.975 ± 0.002, 430k MACs, but 31k params). Tie-broken by val
    (0.958 vs 0.957).
  * It is also the highest-accuracy 3-seed config in the whole grid.
* 56 of the 58 three-seed configs land within 1 pt of 0.963. On v6, time pooling + small
  front-ends cost nothing: the expensive part of lstm_v3 (Conv1D 64 over 256 steps) was not
  buying accuracy.

### Caveats
* v6 numbers, leaky split: lookup baseline 0.984, ceiling 0.9838. Every config has precision
  1.000.
* v7 accuracy for the picks is 0.49–0.53 (chance). The probe (0.80–0.93) shows partial
  memorisation of the 5 noise images.
* The prune rule uses test accuracy, as specified. Picks are made on 3-seed **test** means as
  the task asked; val agrees (both picks have top-tier val).
* MACs are exact counts from the built model (`v6a_train.macs`), not hardware estimates.
* Time pooling is upstream (max over p bins of the 256 × 256 patch). An FPGA design needs that
  max-binning stage in front of the RNN; it is cheap (comparators, no multipliers).
