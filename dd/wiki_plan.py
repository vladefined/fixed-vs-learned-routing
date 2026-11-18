"""Wikipedia, step 2: the grid from A_wiki's calibration (the TinyStories recipe of run_r6ts.py: one step per state,
delta_l = RMS of A's state l / 3.2, levels -K..K with K = 40, the embedding multiplied by sqrt(d) before quantisation
and delta_0 by the same factor) -> runs_vast/A_wiki_s0/grid.json and the plan vast_plans/wiki_main.json:
Q0, QG25-static, QG25 (1 seed each, same data, no-repeat sampler, evaluated against A_wiki_s0).
python -m dd.wiki_plan [runs_vast/A_wiki_s0]"""
from __future__ import annotations

import json
import math
import os
import sys


def main(a_run=os.path.join("runs_vast", "A_wiki_s0"), K=40, d=512):
    c = json.load(open(os.path.join(a_run, "calib.json")))
    es = math.sqrt(d)
    prof = [float(f"{s['rms'] / 3.2:.3g}") for s in c["per_state"]]
    prof[0] = float(f"{prof[0] * es:.3g}")
    grid = ["--q_delta_layers", ",".join(map(str, prof)), "--q_levels", str(K), "--emb_scale", f"{es:.6f}"]
    json.dump(dict(scheme="profile", delta_layers=prof, K=K, emb_scale=es, calib=c),
              open(os.path.join(a_run, "grid.json"), "w"), indent=1)
    common = ["--seed", "0", "--data", "data_wiki", "--sampler", "perm"]
    ref = a_run.replace("\\", "/")
    plan = {"data": "wiki", "wiki_shards": 4, "expect": os.path.join("runs_vast", "_data_wiki", "hashes.json"),
            "jobs": [dict(name="QG25static_wiki_s0", args=["--variant", "QGstatic", "--mask_p", "0.25"] + common + grid,
                          eval_ref=ref),
                     dict(name="QG25_wiki_s0", args=["--variant", "QG", "--target", "0.25"] + common + grid, eval_ref=ref),
                     dict(name="Q0_wiki_s0", args=["--variant", "Q0"] + common + grid, eval_ref=ref)]}
    json.dump(plan, open(os.path.join("vast_plans", "wiki_main.json"), "w"), indent=1)
    print("grid:", prof, "K", K, "emb_scale", round(es, 6))
    print("TinyStories grid for comparison:", json.load(open(os.path.join("runs_ts", "grid.json")))["delta_layers"])


if __name__ == "__main__":
    main(*sys.argv[1:])
