"""Make run configs that re-train chosen configs for robustness:
  seeds   : seeds 1, 2 on the primary split (seed 0 = the original run)  -> stage=<prefix>_seeds
  heldout : train on non-June-2019 files, test on all 8 June-2019 files  -> stage=<prefix>_heldout
  cv      : 4 grouped folds over all 28 files                              -> stage=<prefix>_cv
Usage: python scripts/robustness_cfgs.py out.json seeds,heldout,cv run_id [run_id ...]
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out, kinds, rids = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
cfgs = []
for rid in rids:
    base = json.load(open(os.path.join(ROOT, "results", "runs", f"{rid}.json")))["cfg"]
    pre = base["stage"].split("_")[0]
    common = dict(base, parent=rid)
    sfx = "_rfi" if str(base.get("train_pool", "")).endswith("_rfi") else ""
    if "seeds" in kinds:
        for s in (1, 2):
            cfgs.append(dict(common, run_id=f"{rid}_seed{s}", seed=s, stage=f"{pre}_seeds"))
    if "heldout" in kinds:
        cfgs.append(dict(common, run_id=f"{rid}_heldout", split="heldout_family", fold=None,
                         train_pool=f"mix_heldout_family{sfx}" if base.get("train_pool") else None, stage=f"{pre}_heldout"))
    if "cv" in kinds:
        for k in range(4):
            cfgs.append(dict(common, run_id=f"{rid}_cv{k}", split="cv4", fold=k,
                             train_pool=f"mix_cv4_f{k}{sfx}" if base.get("train_pool") else None, stage=f"{pre}_cv"))
json.dump(cfgs, open(out, "w"), indent=1)
print(len(cfgs), "configs ->", out)
