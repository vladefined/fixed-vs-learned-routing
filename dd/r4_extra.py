"""Round-4 comparison measures that also work for the quantized state and for the round-3 references.
Per block input (layer 0 = embedding ... L = output), real tokens of the first 800 test rows:
  norm_layer            mean ||x|| per token
  maxabs_over_rms_layer mean over tokens of max_i|x_i| / RMS_i(x)  (defined even when the median is 0, as in Q)
  zero_median_tok_layer share of tokens whose median |x_i| is 0 (makes eval5's max/median meaningless)
  sparse                dd.evaluate_r4.sparse_coords (share of coordinates with median|x_i| < 0.1 mean|x_i|, ...)
python -m dd.r4_extra runs5/A_s0 runs6/Q0_s0 ...  -> <run>/r4_extra.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .evaluate import DEV, load_run, load_split
from .evaluate_r4 import sparse_coords


@torch.no_grad()
def extra(run, data="data5", n_rows=800):
    model, cfg = load_run(run)
    te = load_split(data, "test")
    rows = torch.tensor(te["tok"][:n_rows].astype(np.int64), device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, aux = model(rows, collect=True)
    m = rows != 0
    out = dict(norm_layer=[], maxabs_over_rms_layer=[], zero_median_tok_layer=[])
    for x in aux["xs"]:
        xm = x.float()[m]
        ax = xm.abs()
        rms = xm.pow(2).mean(-1).sqrt().clamp_min(1e-12)
        out["norm_layer"].append(float(xm.norm(dim=-1).mean()))
        out["maxabs_over_rms_layer"].append(float((ax.max(-1).values / rms).mean()))
        out["zero_median_tok_layer"].append(float((ax.median(-1).values == 0).float().mean()))
    out["sparse"] = sparse_coords(model, cfg, rows)
    return out


def main(*runs):
    for r in runs:
        res = extra(r)
        json.dump(res, open(os.path.join(r, "r4_extra.json"), "w"), indent=1)
        print(r, "max/rms", np.round(res["maxabs_over_rms_layer"], 1).tolist(), "sparse",
              np.round(res["sparse"]["sparse_layer"], 2).tolist(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
