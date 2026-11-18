"""Round 11: evaluate finished runs with a substituted mask (no training) -- the first k groups of the queue
static_groups(l, G) open in every block, k fixed (k = 2: exactly the static mask; k = 8: everything open), vs the
run's own mask.  Same 608 analysis windows as dd.ts_eval / dd.r8_map; randk's own mask = mask seed 0.
    python -m dd.r11_swap runs_vast/QG25randk_s0 runs_vast/QG25ordered_s0 ...   -> report/r11_swap.json (merged)"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from .r8_map import windows
from .ts_eval import load_run


@torch.no_grad()
def ce(model, W, fn=None, bs=16):
    tot = n = 0.0
    torch.manual_seed(0)
    for i in range(0, W.shape[0], bs):
        idx = W[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(idx, ctl={"hc_gate_fn": fn} if fn else None)
        tot += float(F.cross_entropy(lg[:, :-1].float().flatten(0, 1), idx[:, 1:].flatten(), reduction="sum"))
        n += idx[:, 1:].numel()
    return tot / n


def run(paths, out="report/r11_swap.json"):
    W = windows()
    res = json.load(open(out)) if os.path.exists(out) else {}
    for r in paths:
        model, cfg = load_run(r)
        G = cfg.depth_groups
        rank = {}
        for b in model.blocks:
            rk = torch.zeros(G, device="cuda")
            rk[cfg.static_groups(b.l, G)] = torch.arange(G, dtype=torch.float32, device="cuda")
            rank[b.l] = rk

        def fixed(k):
            return lambda blk, m, zg: (rank[blk.l] < k).float().expand_as(m)
        row = dict(own=ce(model, W), k2=ce(model, W, fixed(2)), k8=ce(model, W, fixed(8)))
        res[r] = row
        print(r, {k: round(v, 4) for k, v in row.items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1:])
