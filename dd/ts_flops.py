"""Round 6, part 2: realisable compute savings of a TinyStories QG run from its gate decisions (tok.npz, written by
dd.ts_eval).  Same accounting as round 5 (dd.evaluate_r4.qg_flops): per-token, per-block FLOPs; attention of all
heads always runs (the gate reads rmsnorm(x + a)); shared FFN (QG): a closed group skips its rows of w2, a token
with no open group in the block skips the whole FFN; split FFN (QG-split): a closed group skips its FFN block; the
gate head is added.  Attention scores use the mean attended context of the analysed tokens (document mask: keys
from the story start, at most the window)."""
from __future__ import annotations

import os

import numpy as np
import torch

from .evaluate_r4 import qg_flops
from .model import Cfg


def mean_context(cur):
    """cur: (N_windows * CTX,) current token ids, windows of CTX; keys attended = positions since the last <eot>
    (id 1) in the window, inclusive, or since the window start."""
    return float(attended(cur).mean())


def attended(cur, ctx=512):
    W = cur.reshape(-1, ctx)
    pos = np.broadcast_to(np.arange(ctx), W.shape)
    last = np.where(W == 1, pos, 0)
    last = np.maximum.accumulate(last, axis=1)
    return (pos - last + 1).ravel()


def savings(run):
    ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
    cfg = Cfg(**ck["cfg"])
    z = np.load(os.path.join(run, "tok.npz"))
    L, G = cfg.n_layers, cfg.depth_groups
    opn = np.unpackbits(z["open_bits"].reshape(-1, L)[..., None], axis=-1, bitorder="little")[..., :G]   # N, L, G
    ctx = mean_context(z["cur"].astype(np.int64))
    per = []
    for l in range(L):
        o = opn[:, l].astype(np.float32)
        per.append(qg_flops(cfg, ctx, float(1 - o.mean()), float((o.sum(1) == 0).mean())))
    blk = float(np.mean([p["realisable_block"] for p in per]))
    ffn = float(np.mean([p["realisable_ffn"] for p in per]))
    return dict(realisable_block=blk, realisable_ffn=ffn, ctx=ctx, ffn_share_of_block=per[0]["ffn_share_of_block"],
                realisable_block_layer=[p["realisable_block"] for p in per],
                realisable_ffn_layer=[p["realisable_ffn"] for p in per],
                open_share=float(opn.mean()), token_none_open=float((opn.sum(2) == 0).mean()))
