"""K extra: (1) does the gate head still get gradient at the end of training?  (2) real savings in the
current (shared-FFN) architecture: fraction of (token, block) with NO shifted group (whole FFN skippable).
python -m dd.k_extra runs5/K25_s0 ...  -> report/k_extra.json"""
import json
import os
import sys

import numpy as np
import torch

from .evaluate import DEV, load_run, load_split
from .model import lm_loss

te = load_split("data5", "test")
rows = torch.tensor(te["tok"].astype(np.int64), device=DEV)
res = {}
for run in sys.argv[1:]:
    model, cfg = load_run(run)
    # (1) gradient on 4 batches in training mode (sampled gates), same loss as in training
    model.train()
    gh, ff, frac_cont = [], [], []
    for i in range(4):
        idx = rows[i * 64:(i + 1) * 64]
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, cost, _, aux = lm_loss(model, idx)
        ce.backward()                                      # CE part only: what shapes the gates besides mu
        gh.append(np.mean([float(b.gate_head.weight.grad.norm() / b.gate_head.weight.detach().norm()) for b in model.blocks]))
        ff.append(np.mean([float(b.w1.weight.grad.norm() / b.w1.weight.detach().norm()) for b in model.blocks]))
        msk = idx != 0
        frac_cont.append(np.mean([float(((d[msk] > 0) & (d[msk] < 1)).float().mean()) for d in aux["dd"]]))
    model.eval()
    # (2) whole-token skip with hard gates
    L = cfg.n_layers
    none_shift = np.zeros(L); n = 0
    with torch.no_grad():
        for i in range(0, 1500, 128):
            idx = rows[i:i + 128]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, aux = model(idx, collect=True)
            msk = idx != 0
            n += int(msk.sum())
            for l in range(L):
                none_shift[l] += float(((aux["dd"][l][msk] > 0).sum(-1) == 0).float().sum())
    r = dict(grad_rel_gate_head=float(np.mean(gh)), grad_rel_w1=float(np.mean(ff)),
             frac_fractional_train=float(np.mean(frac_cont)), token_no_shift_layer=(none_shift / n).tolist())
    res[os.path.basename(run)] = r
    print(os.path.basename(run), {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 5)) for k, v in r.items()})
json.dump(res, open(os.path.join("report", "k_extra.json"), "w"), indent=1)
