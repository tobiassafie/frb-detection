# Results index
| file | content |
|---|---|
| `summary.md` | **v7** leak-free CNN/LSTM sweep, QAT, hls4ml status (est_ only) |
| `summary_v6.md` | v6 benchmark vs lookup baseline, cross-dataset tests, streaming simulation |
| `summary_v6a.md` | winning models on the original v6 split; reproduction of the historic models; the 0.9838 ceiling |
| `summary_v6c.md` | shrinking lstm_v3; LSTM/GRU hls4ml projects |
| `summary_v6d.md` | small-RNN sweep (96 configs) + Pareto plot `fig_v6d_pareto.png` |
| `summary_roll.md` | **rolling causal GRU/LSTM**, mid-burst truncation and shift tests |
| `NOTES*.md` | reasoning, assumptions and incidents for each stage |
| `*_runs.csv`, `sweep_results.csv` | one row per training run (`run_id` → `runs*/<run_id>.json`) |
| `models*/` | all trained weights by stage (the best ones are copied to `/models/best/`) |
