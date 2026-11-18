"""Extra analyses on finished runs (does not touch eval.json):
  1) frozen fraction per layer by three definitions:  dd < 0.01 | raw update |s*t|/RMS(x) < 0.01 (before
     re-normalisation) | relative change after normalisation |dx|/RMS(x) < 0.01  -> shows how much of the
     dd-vs-|dx| disagreement is the per-token re-normalisation;
  2) do the frozen coordinates repeat across sentences?  Jaccard similarity of frozen sets for the same word
     type in different sentences vs. different word types, vs. the chance level for the same density.
python -m dd.extra runs4/B_l0_s0 runs4/B_m_s0 ...   -> writes <run>/extra.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .evaluate import load_run, DEV

THR = 0.01


@torch.no_grad()
def analyse(run, data="data4", n_rows=1500, seed=0):
    model, cfg = load_run(run)
    te = np.load(os.path.join(data, "test.npz"))
    rows = torch.tensor(te["tok"][:n_rows].astype(np.int64), device=DEV)
    sid = torch.tensor(te["sid"][:n_rows].astype(np.int64), device=DEV)
    L, d = cfg.n_layers, cfg.d
    NG = cfg.norm_groups if cfg.norm_state else 1
    stash = {}
    hooks = []
    # capture the pre-normalisation state: x_prev + s*t  ==  x_new * max(RMS(pre), floor);  we recompute it
    # by hooking the block forward inputs/outputs and the quantities we need.
    out = {"frozen_dd": np.zeros(L), "frozen_raw": np.zeros(L), "frozen_rel": np.zeros(L), "n": 0}
    masks = {l: [] for l in range(L)}
    toks, sids = [], []
    for i in range(0, rows.shape[0], 128):
        idx = rows[i:i + 128]
        B = idx.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(idx, collect=True)
        m = idx != 0
        toks.append(idx[m].cpu().numpy())
        sids.append(sid[i:i + 128][m].cpu().numpy())
        for l in range(L):
            x0, x1 = aux["xs"][l].float(), aux["xs"][l + 1].float()
            if NG > 1:
                rms0 = x0.view(B, -1, NG, d // NG).pow(2).mean(-1).sqrt().repeat_interleave(d // NG, -1)
            else:
                rms0 = x0.pow(2).mean(-1, keepdim=True).sqrt()
            rel = (x1 - x0).abs() / rms0
            fr_rel = (rel < THR)[m]
            out["frozen_rel"][l] += float(fr_rel.float().sum())
            if cfg.budget:
                dd = aux["dd"][l]
                s = dd.clamp_max(1.0) if cfg.update in ("min", "pull") else dd
                s_full = model.blocks[l].expand(s)
                # raw update before re-normalisation:  |s * t| (or |s*(t-x)| for pull);  t unknown directly,
                # but s == 0  <=>  raw update == 0, and for s>0 we bound |s*t| <= s.  Use the exact value via
                # pre = x1 * r  with r = RMS(pre) recovered from depth-free identity  pre - x0 = s*t.
                # r solves  RMS(x1*r - x0 ... ) -- instead recompute t by re-running the block is costly;
                # use the upper bound s (|tanh| <= 1, |t - x| <= 1 + |x|).
                raw_bound = (s_full * (1.0 if cfg.update != "pull" else (1 + x0.abs()))) / rms0
                fr_raw = (raw_bound < THR)[m]
                fr_dd = (dd < THR)[m]
                out["frozen_raw"][l] += float(fr_raw.float().sum())
                out["frozen_dd"][l] += float(fr_dd.float().mean(-1).sum()) * d
                masks[l].append(fr_rel.cpu().numpy())
            else:
                masks[l].append(fr_rel.cpu().numpy())
        out["n"] += int(m.sum()) * d
    toks, sids = np.concatenate(toks), np.concatenate(sids)
    res = {k: (v / out["n"]).tolist() for k, v in out.items() if k != "n"}
    # Jaccard: same word type, different sentences  vs  different word types
    rng = np.random.default_rng(seed)
    jac = []
    for l in range(L):
        M = np.concatenate(masks[l])                       # N, d (bool)
        p = M.mean()
        same, diff = [], []
        types, counts = np.unique(toks, return_counts=True)
        frequent = types[counts >= 20]
        for _ in range(4000):
            w = rng.choice(frequent)
            ii = np.where(toks == w)[0]
            a, b = rng.choice(ii, 2, replace=False)
            if sids[a] == sids[b]:
                continue
            same.append(_j(M[a], M[b]))
            c, e = rng.integers(0, len(toks), 2)
            if toks[c] != toks[e]:
                diff.append(_j(M[c], M[e]))
        chance = p / (2 - p) if p > 0 else 0.0               # E[Jaccard] for independent Bernoulli(p) sets
        jac.append(dict(p=float(p), same_word=float(np.nanmean(same)), diff_word=float(np.nanmean(diff)),
                        chance=float(chance)))
    res["jaccard"] = jac
    json.dump(res, open(os.path.join(run, "extra.json"), "w"), indent=1)
    return res


def _j(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else np.nan


if __name__ == "__main__":
    for r in sys.argv[1:]:
        res = analyse(r)
        print(os.path.basename(r), "frozen dd", np.round(res["frozen_dd"], 3).tolist(),
              "raw<=", np.round(res["frozen_raw"], 3).tolist(), "rel", np.round(res["frozen_rel"], 3).tolist())
        print("   jaccard", [{k: round(v, 3) for k, v in j.items()} for j in res["jaccard"]])
