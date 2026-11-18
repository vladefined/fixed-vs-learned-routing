"""G (and K): learned attention depth-bias slopes per block/head, and the effect of zeroing them.
python -m dd.g_slopes runs5/G_s0 runs5/G_s1 ...  -> <run>/slopes.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .evaluate import DEV, ce_of, load_run, load_split
from .evaluate5 import answer_eval


@torch.no_grad()
def main(runs, data="data5"):
    te, va = load_split(data, "test"), load_split(data, "val")
    rows = torch.tensor(te["tok"].astype(np.int64), device=DEV)
    vrows = torch.tensor(va["tok"][:1500].astype(np.int64), device=DEV)
    for run in runs:
        model, cfg = load_run(run)
        slopes = [b.slope.detach().cpu().tolist() for b in model.blocks]
        base = answer_eval(model, rows, te, n_rows=2500)
        base["val_ce"] = ce_of(model, vrows)
        saved = [b.slope.detach().clone() for b in model.blocks]
        for b in model.blocks:
            b.slope.zero_()
        zero = answer_eval(model, rows, te, n_rows=2500)
        zero["val_ce"] = ce_of(model, vrows)
        for b, s in zip(model.blocks, saved):
            b.slope.copy_(s)
        res = dict(slopes=slopes, base=base, zeroed=zero)
        json.dump(res, open(os.path.join(run, "slopes.json"), "w"), indent=1)
        print(os.path.basename(run), "slopes", [[round(x, 2) for x in s] for s in slopes])
        print(f"   answer {base['acc']:.4f} -> {zero['acc']:.4f} | val CE {base['val_ce']:.4f} -> {zero['val_ce']:.4f}"
              f" | m=6 {base['acc_by_m'].get(6, float('nan')):.3f} -> {zero['acc_by_m'].get(6, float('nan')):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
