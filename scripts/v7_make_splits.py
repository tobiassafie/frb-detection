"""Stage 0d: file-level (= source-burst-level) splits for v7. Saved to results/splits/.

primary        : train/val/test by file, stratified so the June-2019 cluster
                 (8 files sharing one 441-799 MHz band config) is spread across splits.
heldout_family : train+val on every non-June-2019 file, test on all 8 June-2019 files
                 (band-configuration shift -> held-out-source/family test).
cv4            : 4 grouped folds over all 28 files (each fold = test files).
"""
import json
import os

import numpy as np
import pandas as pd

SEED = 42
meta = pd.read_csv("augmented_dataset/v7/meta.csv")
files = meta.groupby("group_id").file.first().to_dict()
june = sorted(g for g, f in files.items() if f.startswith(("FRB201906",)) and not f.startswith("FRB20190627")
              and not f.startswith("FRB20190630"))
other = sorted(g for g in files if g not in june)
rng = np.random.default_rng(SEED)
j, o = list(rng.permutation(june)), list(rng.permutation(other))
primary = dict(test=j[:2] + o[:4], val=j[2:3] + o[4:8], train=j[3:] + o[8:])

rest = list(rng.permutation(other))
heldout_family = dict(test=june, val=rest[:4], train=rest[4:])

allg = list(rng.permutation(sorted(files)))
cv4 = [dict(test=allg[k::4]) for k in range(4)]
for k, fold in enumerate(cv4):
    remaining = [g for g in allg if g not in fold["test"]]
    fold["val"] = remaining[:4]
    fold["train"] = remaining[4:]

to_int = lambda d: {k: [int(x) for x in v] for k, v in d.items()}  # noqa: E731
out = dict(seed=SEED, group_to_file={int(k): v for k, v in files.items()}, june_2019_cluster=[int(x) for x in june],
           primary=to_int(primary), heldout_family=to_int(heldout_family), cv4=[to_int(f) for f in cv4])
os.makedirs("results/splits", exist_ok=True)
json.dump(out, open("results/splits/v7_splits.json", "w"), indent=1)
for name in ("train", "val", "test"):
    idx = meta.idx[meta.group_id.isin(primary[name])].values
    np.save(f"results/splits/v7_primary_{name}_idx.npy", idx)
    print(name, sorted(primary[name]), len(idx))
print("heldout_family test", june)
