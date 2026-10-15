| | **Best CNN - accuracy track (any size; best 3-seed val)** | **Best CNN - synthesis track (<=10k, exported to HLS)** | **Best LSTM/GRU** | **Largest CNN tried (68k, BN)** | **8-bit QAT of the synthesis CNN** | classical dedispersion search |
|---|---|---|---|---|---|---|
| run_id | `loc_t0041_b32` | `s1mB_t0041_rfi` | `s3r_lstm_u16_T64_F8_lr0.003_do0.2_d0` | `loc_t0041_A_w8k5bn` | `qat_s1mB_t0041_rfi_8b` | `scripts/classical_baseline.py` |
| architecture | 128² in; Conv[4, 8, 16, 16, 16] k3, avg-pool [2, 2, 2, 2, 2], gap, Dense16 | 128² in; Conv[4, 8, 16, 16, 16] k3, avg-pool [2, 2, 2, 2, 2], gap, Dense16 | LSTM(16) on 64 t × 8 f | 128² in; Conv[8, 16, 32, 32, 32] k5, avg-pool [2, 2, 2, 2, 2], gap, Dense16, BN | 128² in; Conv[4, 8, 16, 16, 16] k3, avg-pool [2, 2, 2, 2, 2], gap, Dense16 | DM −13…−0.4, boxcar 1–16 |
| params | 6,433 | 6,433 | 1,617 | 68,425 | 6,433 | — |
| MACs / inference | 3,686,672 | 3,686,672 | 98,320 | 37,683,728 | 3,686,672 | — |
| **test acc** (seed 0; noise + FRB S/N ≥ 7) | **0.9155** | **0.9443** | **0.8354** | **0.9236** | **0.9425** | 0.8453 |
| test acc, 3 training seeds | 0.915 ± 0.002 (n=3) | 0.921 ± 0.017 (n=3) | 0.826 ± 0.006 (n=3) | 0.924 | 0.943 | — |
| test acc, 4-fold file-grouped CV | — | 0.885 ± 0.026 (n=4) | 0.732 ± 0.040 (n=4) | — | — | — |
| test acc, held-out June-2019 family | — | 0.810 | 0.560 | — | — | 0.744 |
| precision / recall | 0.893 / 0.905 | 0.994 / 0.871 | 0.840 / 0.744 | 0.974 / 0.838 | 0.963 / 0.895 | 0.874 / 0.732 |
| F1 | 0.8986 | 0.9283 | 0.7891 | 0.9008 | 0.9280 | 0.7966 |
| ROC-AUC | 0.9622 | 0.9716 | 0.8734 | 0.9683 | 0.9710 | 0.8932 |
| recall @ 1 % FPR | 0.853 | 0.888 | 0.593 | 0.831 | 0.870 | 0.000 |
| specificity (noise samples correct) | 0.9231 | 0.9961 | 0.9000 | 0.9842 | 0.9758 | — |
| test acc, FRB S/N ≥ 10 | 0.9316 | 0.9712 | 0.8670 | 0.9552 | 0.9637 | — |
| test acc, FRB S/N ≥ 15 | 0.9360 | 0.9830 | 0.8963 | 0.9694 | 0.9710 | — |
| test acc, FRB S/N ≥ 20 | 0.9347 | 0.9861 | 0.9073 | 0.9749 | 0.9718 | — |
| test acc, v6-brightness-equivalent FRBs (peak 0.5–5 σ) | 0.8150 | 0.8291 | 0.7443 | 0.8070 | 0.8365 | — |
| test acc, all S/N 3–50 | 0.8190 | 0.8268 | 0.7460 | 0.8036 | 0.8350 | 0.7336 |