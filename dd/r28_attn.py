"""Round 28: attention with a weighted K / V gradient (PairAttn) and running sums of the own / foreign split of
output gradients (GradAcc). Shared by dd.r28_measure and dd.r28_stage0."""
import math

import torch


class PairAttn(torch.autograd.Function):
    """softmax attention with a causal (or given) mask; forward = ordinary attention; backward: the K and V gradient of
    the pair (reader t, read s) times W[t, s] (B, T, T), the Q gradient untouched. Internals in fp32."""

    @staticmethod
    def forward(ctx, q, k, v, mask, W, scale):
        qf, kf, vf = q.float(), k.float(), v.float()
        s = (qf @ kf.transpose(-1, -2)) * scale
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, -1)
        ctx.save_for_backward(qf, kf, vf, p, W)
        ctx.scale = scale
        ctx.dt = (q.dtype, k.dtype, v.dtype)
        return (p @ vf).to(q.dtype)

    @staticmethod
    def backward(ctx, gy):
        qf, kf, vf, p, W = ctx.saved_tensors
        gy = gy.float()
        Wh = W[:, None].float() if W is not None else None
        gp = gy @ vf.transpose(-1, -2)
        pw = p * Wh if Wh is not None else p
        gv = pw.transpose(-1, -2) @ gy
        gs = p * (gp - (gp * p).sum(-1, keepdim=True))
        gq = (gs @ kf) * ctx.scale
        gsw = gs * Wh if Wh is not None else gs
        gk = (gsw.transpose(-1, -2) @ qf) * ctx.scale
        return gq.to(ctx.dt[0]), gk.to(ctx.dt[1]), gv.to(ctx.dt[2]), None, None, None


class GradAcc:
    """running sums per layer of the own / foreign split of output gradients (positions pooled)"""

    def __init__(self, n):
        self.s = [dict(f2=0.0, g2=0.0, o2=0.0, of=0.0, ratio=0.0, cos=0.0, n=0) for _ in range(n)]

    def add(self, i, gf, go):
        gf, go = gf.double(), go.double()
        fo = gf - go
        nf, no, ng = fo.norm(dim=-1), go.norm(dim=-1), gf.norm(dim=-1)
        ok = (ng > 0) & (no > 0) & (nf > 0)
        a = self.s[i]
        a["f2"] += float(fo.pow(2).sum())
        a["g2"] += float(gf.pow(2).sum())
        a["o2"] += float(go.pow(2).sum())
        a["of"] += float((go * fo).sum())
        a["ratio"] += float((nf[ok] / ng[ok]).sum())
        a["cos"] += float(((go[ok] * fo[ok]).sum(-1) / (no[ok] * nf[ok])).sum())
        a["n"] += int(ok.sum())

    def stats(self):
        out = []
        for a in self.s:
            out.append(dict(share_energy=math.sqrt(a["f2"] / max(a["g2"], 1e-300)),
                            own_energy=math.sqrt(a["o2"] / max(a["g2"], 1e-300)),
                            share_mean=a["ratio"] / max(a["n"], 1), cos=a["cos"] / max(a["n"], 1),
                            cos_pooled=a["of"] / max(math.sqrt(a["o2"] * a["f2"]), 1e-300), n=a["n"]))
        return out
