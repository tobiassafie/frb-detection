
## What changed vs the v7 results (`results/summary.md`)

* **The v6 numbers look better than v7 (0.94–0.98 vs 0.92–0.94), but only because of leakage.**
  * On v6 the same architectures score higher than on v7, yet still below a lookup table that
    has learned nothing (0.984).
  * Their precision is 1.000 for free, since all v6 test noise images are memorised training
    images.
  * They do not transfer to v7 or to any stream (≈ chance / ≈ 0 % detection at a usable
    threshold).
  * The v7 file-grouped numbers remain the ones to quote for patch classification.
* **v6 prefers different architectures.** The sweep on v6 prefers flatten heads (0.976–0.979)
  over the v7 winner's GAP head (0.938). Flatten keeps position, and position is what you
  need to memorise 5 fixed backgrounds. Avg-pool + GAP models sometimes fail to train at all
  on v6 (dead seeds).
* **Patch accuracy ≠ streaming performance.** v7 patch test accuracy (0.94) corresponds to
  83 % detection of S/N ≥ 7 bursts at ~80 FA/h in a causal stream — and only when the sweep
  direction matches training.
  * With physical high-frequency-first arrival, every v7 patch-trained CNN is ≤ 5 % at the
    val-chosen threshold (≤ 28 % at a fixed 0.5).
  * The stream-trained CNN fixes the orientation problem (77–78 % in both orientations, about
    5 points below the standard CNN in the stored orientation).
  * It triggers earlier: median time-to-detect 42–48 ms vs 58 ms, at ~81–86 % of the band.
  * It costs a higher test false-alarm rate (494 vs 80 /h at the same val-chosen operating
    point).
  * The RNNs, which looked acceptable on patches (0.78–0.84), are not usable as streaming
    triggers: they detect ≤ 17 % at ≤ 10 FA/h on val.
* **Open items:**
  * the FA/h calibration (val→test transfer is poor with 5–6 files per split);
  * whether the deployed system sees stored-orientation (baseband-dump) or
    physical-orientation data;
  * the 256×256 and filter-width v6 sweep configs that did not finish in the 2 h budget.
