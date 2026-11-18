"""Heavy-tail loophole check for the relative-change cost (round 3).
For every block input state x and update dx (test tokens):
  tail        = median_i  RMS_tokens(x_i) / median_tokens |x_i|      (~1.2 for Gaussian-like coordinates)
  frac_gt3med = fraction of (token, coordinate) with |x_i| > 3 * median_tokens |x_i|  (~0.03 for Gaussian)
  cost_rel    = mean |dx_i| / s_i       (s_i = RMS over tokens of x_i: what the I/J cost sees)
  tok_rel     = mean_g RMS(dx_g) / RMS(x_g) per token group (the real relative change)
  ratio       = cost_rel / tok_rel  (< 1: the cost under-charges changes relative to the token's own state)
  frozen_own  = fraction of (token, coordinate) with |dx_i| / (|x_i| + 0.1 s_i) < 0.01
  frozen_s    = fraction with |dx_i| / s_i < 0.01  (what looks frozen to the cost)
Zeroing test (I/J): zero every coordinate update that is frozen by the own-value measure -> dCE, answer acc.
python -m dd.tails runs5/I_s0 runs5/H_s0 ...  -> <run>/tails.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .evaluate import DEV, ce_of, load_run, load_split
from .evaluate5 import answer_eval

THR = 0.01


@torch.no_grad()
def check(run, data="data5", n_rows=800, zero_test=True):
    model, cfg = load_run(run)
    te = load_split(data, "test")
    rows_all = torch.tensor(te["tok"].astype(np.int64), device=DEV)
    rows = rows_all[:n_rows]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, aux = model(rows, collect=True)
    m = rows != 0
    G = cfg.n_heads
    out = {k: [] for k in ("tail", "frac_gt3med", "cost_rel", "tok_rel", "ratio", "frozen_own", "frozen_s")}
    for l in range(cfg.n_layers):
        x = aux["xs"][l].float()[m]
        dx = (aux["xs"][l + 1].float() - aux["xs"][l].float())[m]
        rms = x.pow(2).mean(0).sqrt()
        med = x.abs().median(0).values.clamp_min(1e-12)
        out["tail"].append(float((rms / med).median()))
        out["frac_gt3med"].append(float((x.abs() > 3 * med).float().mean()))
        blk = model.blocks[l]
        s = blk.s_floor() if hasattr(blk, "s_ema") else torch.maximum(rms, 1e-3 * rms.mean())
        c = (dx.abs() / s).mean(-1)
        xg = x.view(-1, G, cfg.d // G).pow(2).mean(-1).sqrt().clamp_min(1e-12)
        t = (dx.view(-1, G, cfg.d // G).pow(2).mean(-1).sqrt() / xg).mean(-1)
        out["cost_rel"].append(float(c.mean()))
        out["tok_rel"].append(float(t.mean()))
        out["ratio"].append(float(c.mean() / t.mean().clamp_min(1e-12)))
        out["frozen_own"].append(float((dx.abs() / (x.abs() + 0.1 * s) < THR).float().mean()))
        out["frozen_s"].append(float((dx.abs() / s < THR).float().mean()))
    if zero_test and cfg.budget:
        va = load_split(data, "val")
        vrows = torch.tensor(va["tok"][:1500].astype(np.int64), device=DEV)
        base_ce = ce_of(model, vrows)
        out["dce_zero_own"] = ce_of(model, vrows, ctl={"zero_own_below": THR}) - base_ce
        b = answer_eval(model, rows_all, te, n_rows=2500)
        z = answer_eval(model, rows_all, te, n_rows=2500, ctl={"zero_own_below": THR})
        out["answer_base"], out["answer_zero_own"] = b["acc"], z["acc"]
        out["answer_base_by_m"], out["answer_zero_own_by_m"] = b["acc_by_m"], z["acc_by_m"]
    json.dump(out, open(os.path.join(run, "tails.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for r in sys.argv[1:]:
        o = check(r)
        print(os.path.basename(r), {k: ([round(v, 3) for v in vs] if isinstance(vs, list) else
                                        (round(vs, 4) if isinstance(vs, float) else vs))
                                    for k, vs in o.items() if not k.endswith("_by_m")})
