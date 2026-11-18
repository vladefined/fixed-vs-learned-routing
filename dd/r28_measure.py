"""Round 28: measurements of a stage-1 model at a save (2k / 5k / 10k) and the foreign-gradient share.

measure_all (val windows a.m_windows, calibration windows a.fit_windows):
  spec   - per MoE block, the CE rise on the tokens of expert i when they get: every other expert k (foreign), the FFN
           of the same block of dense-cont, a zero FFN output (and for «add» the shared part only = additions off);
           normalised specialisation = (foreign - own) / (zero - own) with own = 0 (the model itself), the same with
           dense-cont instead of foreign; summaries weighted by the token shares of the experts;
  refit  - for every expert i two foreign experts k (fixed random): the down refitted by ridge on the calibration
           tokens of i (inputs of k's down frozen) to i's output; share of the gap removed = (raw - refit) / raw;
  div    - per expert the update since step 0 (all its own tensors): relative norm, cosine between experts' updates.
foreign_share: per MoE block the share |full - own| / |full| of the gradient of the experts' own tensors (fp32): natural
  (ordinary attention gradient) and as trained (the variant's weighting, if any).
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import apply_rope
from .r21_moe import CTX, E, load_base, load_moe
from .r28_attn import PairAttn

_REF = {}


def ratio_se(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    ok = den > 0
    num, den = num[ok], den[ok]
    if len(den) < 2:
        return float("nan"), float("nan")
    r = num.sum() / den.sum()
    res = num - r * den
    return float(r), float(math.sqrt((res ** 2).sum() / (len(den) - 1) / len(den)) / den.mean())


def refs(model, a, dev):
    """A-mix and dense-cont FFN weights of the MoE blocks (cached)"""
    if "dense" not in _REF:
        dn = load_moe(a.dense, dev=dev)
        _REF["dense"] = [tuple(getattr(dn.base.blocks[b], w).weight.detach().float() for w in ("w1", "w3", "w2"))
                         for b in range(model.N, model.L)]
        del dn
    return _REF["dense"]


def swiglu(z, w):
    return (F.silu(z @ w[0].t()) * (z @ w[1].t())) @ w[2].t()


class Cache:
    def __init__(self, model, idx_all, bs=32):
        self.m, self.bs = model, bs
        n = idx_all.shape[0]
        self.idx = idx_all
        self.x = [torch.empty(n, CTX + 1, model.d, device=idx_all.device) for _ in range(model.M)]
        self.e = [torch.empty(n, CTX + 1, dtype=torch.int8, device=idx_all.device) for _ in range(model.M)]
        self.ce = torch.empty(n, CTX, device=idx_all.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for w0 in range(0, n, bs):
                idx = idx_all[w0:w0 + bs]
                x, es, xs = model(idx, keep=True, gradw=False)
                for j in range(model.M):
                    self.x[j][w0:w0 + bs] = xs[j].float()
                    self.e[j][w0:w0 + bs] = es[j].to(torch.int8)
                self.ce[w0:w0 + bs] = self.token_ce(x, idx)

    def token_ce(self, x, idx, sel=None):
        base = self.m.base
        h = base.nf(x[:, :-1])
        tgt = idx[:, 1:]
        out = torch.zeros(tgt.shape, device=x.device)
        sel = (torch.ones_like(tgt, dtype=torch.bool) if sel is None else sel) & (tgt != 0)
        lg = h[sel] @ base.emb.weight.t().to(h.dtype)
        out[sel] = F.cross_entropy(lg.float(), tgt[sel], reduction="none")
        return out

    @torch.no_grad()
    def rise(self, j, mods, sel_tok):
        num, den = [], []
        m = self.m
        for w0 in range(0, self.idx.shape[0], self.bs):
            idx = self.idx[w0:w0 + self.bs]
            sel = sel_tok(w0, w0 + self.bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, _, _ = m(idx, b0=m.N + j, x=self.x[j][w0:w0 + self.bs], mods=mods, gradw=False)
                ce = self.token_ce(x, idx, sel)
            sel = sel & (idx[:, 1:] != 0)
            num.append(((ce - self.ce[w0:w0 + self.bs]) * sel).double().sum(1).cpu())
            den.append(sel.double().sum(1).cpu())
        return ratio_se(torch.cat(num).numpy(), torch.cat(den).numpy())


def calib(path, dev, n):
    arr = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    s = np.arange(0, arr.size - CTX - 1, CTX)
    s = s[np.linspace(0, s.size - 1, min(n, s.size)).round().astype(int)]
    return torch.from_numpy(np.stack([arr[x:x + CTX + 1] for x in s])).to(dev)


def ridge(A, Y, held, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    Af, Yf, Ah, Yh = A[~held], Y[~held], A[held], Y[held]
    XtX, XtY = (Af.t() @ Af).double(), (Af.t() @ Yf).double()
    sc = float(torch.diagonal(XtX).mean()) + 1e-12
    eye = torch.eye(XtX.shape[0], dtype=torch.float64, device=A.device)
    best = None
    for lam in lams:
        W = torch.linalg.solve(XtX + lam * sc * eye, XtY).float()
        sse = float((Ah @ W - Yh).pow(2).sum())
        if best is None or sse < best[0]:
            best = (sse, W)
    return best[1]                                                   # (F, d)


def measure_all(model, D, a):
    dev = next(model.parameters()).device
    t0 = time.time()
    dense = refs(model, a, dev)
    idx_all = D.win(D.starts(a.m_windows))
    C = Cache(model, idx_all)
    M = model.M
    out = dict(spec={}, refit={}, div={})
    load = [[float((C.e[j][:, :-1] == i).float().mean()) for i in range(E)] for j in range(M)]
    out["load"] = load
    rng = np.random.default_rng(28)
    npair = getattr(a, "n_pairs", 2)
    pairs = {j: {i: sorted(int(k) for k in rng.choice([k for k in range(E) if k != i], npair, replace=False))
                 for i in range(E)} for j in range(M)}
    for j in range(M):
        rows = {}
        for i in range(E):
            sel = (lambda w0, w1, i=i: C.e[j][w0:w1, :-1] == i)
            si = (lambda e, i=i: e == i)
            r = {}
            for k in range(E):
                if k != i:
                    r[str(k)] = C.rise(j, {j: [(si, (lambda z, k=k: model.expert_out(j, k, z)))]}, sel)
            r["dense"] = C.rise(j, {j: [(si, (lambda z: swiglu(z, dense[j])))]}, sel)
            r["zero"] = C.rise(j, {j: [(si, (lambda z: torch.zeros(z.shape[0], model.d, device=z.device, dtype=z.dtype)))]}, sel)
            if model.v == "add":
                r["addoff"] = C.rise(j, {j: [(si, (lambda z: model.shared(j, z)))]}, sel)
            rows[str(i)] = r
        out["spec"][str(j)] = rows
    if getattr(a, "all_blocks", False):                            # every MoE block at once (as stage 0)
        rows = {}
        for i in range(E):
            def sel(w0, w1, i=i):
                s_ = None
                for jj in range(M):
                    t = C.e[jj][w0:w1, :-1] == i
                    s_ = t if s_ is None else (s_ | t)
                return s_
            si = (lambda e, i=i: e == i)
            r = {}
            for k in range(E):
                if k != i:
                    r[str(k)] = C.rise(0, {jj: [(si, (lambda z, k=k, jj=jj: model.expert_out(jj, k, z)))] for jj in range(M)}, sel)
            r["dense"] = C.rise(0, {jj: [(si, (lambda z, jj=jj: swiglu(z, dense[jj])))] for jj in range(M)}, sel)
            r["zero"] = C.rise(0, {jj: [(si, (lambda z: torch.zeros(z.shape[0], model.d, device=z.device, dtype=z.dtype)))]
                                   for jj in range(M)}, sel)
            rows[str(i)] = r
        out["spec_all"] = rows
    # refit of the down: calibration
    cal = calib(a.calib, dev, a.fit_windows)
    held_w = (torch.arange(cal.shape[0], device=dev) % 5) == 4
    for j in range(M):
        Z, EI, HW = [], [], []
        with torch.no_grad():
            for w0 in range(0, cal.shape[0], 32):
                idx = cal[w0:w0 + 32]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, es, xs = model(idx, keep=True, gradw=False)
                    blk = model.base.blocks[model.N + j]
                    x = xs[j]
                    rope = model.base.rope(idx.shape[1], dev)
                    from .model import doc_causal_mask
                    y = x + blk.attn(blk.n1(x), rope, mask=doc_causal_mask(idx))
                    z = blk.n2(y)
                Z.append(z[:, :-1].reshape(-1, model.d).float())
                EI.append(es[j][:, :-1].reshape(-1))
                HW.append(held_w[w0:w0 + 32][:, None].expand(-1, CTX).reshape(-1))
            Z, EI, HW = torch.cat(Z), torch.cat(EI), torch.cat(HW)
            Y = torch.empty(Z.shape[0], model.d, device=dev)
            for i in range(E):
                s = EI == i
                if s.any():
                    Y[s] = model.expert_out(j, i, Z[s])
            rows = {}
            for i in range(E):
                s = EI == i
                if int(s.sum()) < 50:
                    continue
                r = {}
                for k in pairs[j][i]:
                    A = model.expert_feats(j, k, Z[s])
                    Wd = ridge(A, Y[s], HW[s])
                    rep = (lambda z, k=k, Wd=Wd: model.expert_feats(j, k, z.float()) @ Wd)
                    sel = (lambda w0, w1, i=i: C.e[j][w0:w1, :-1] == i)
                    fit = C.rise(j, {j: [((lambda e, i=i: e == i), rep)]}, sel)
                    raw = out["spec"][str(j)][str(i)][str(k)]
                    r[str(k)] = dict(raw=raw, fit=fit)
                rows[str(i)] = r
            del Z, Y, EI, HW
            torch.cuda.empty_cache()
        out["refit"][str(j)] = rows
    # divergence of the updates since step 0
    for j in range(M):
        cur = [torch.cat([p[k].reshape(-1).float() for p in model.expert_params(j)]) for k in range(E)]
        ini = model.init_vecs[j]
        U = [c - i_.to(c.device) for c, i_ in zip(cur, ini)]
        nrm = float(torch.stack(ini).norm(dim=1).mean())
        cs = [float(F.cosine_similarity(U[x], U[y], dim=0)) for x in range(E) for y in range(E) if x != y]
        out["div"][str(j)] = dict(upd_rel=[float(u.norm()) / nrm for u in U], cos_upd_mean=float(np.mean(cs)),
                                  cos_upd_min=float(np.min(cs)), cos_upd_max=float(np.max(cs)))
    out["summary"] = summary(out, M)
    out["seconds"] = time.time() - t0
    return out


def summary(out, M):
    s = {}
    for j in range(M):
        w = np.asarray(out["load"][j])
        rows = out["spec"][str(j)]
        fo = np.array([np.mean([rows[str(i)][str(k)][0] for k in range(E) if k != i]) for i in range(E)])
        de = np.array([rows[str(i)]["dense"][0] for i in range(E)])
        ze = np.array([rows[str(i)]["zero"][0] for i in range(E)])
        ok = ~np.isnan(fo) & ~np.isnan(ze)
        wm = lambda v: float((v[ok] * w[ok]).sum() / w[ok].sum())                 # noqa: E731
        d = dict(foreign=wm(fo), dense=wm(de), zero=wm(ze))
        d["spec_foreign"] = d["foreign"] / d["zero"]
        d["spec_dense"] = d["dense"] / d["zero"]
        if "addoff" in rows["0"]:
            ao = np.array([rows[str(i)]["addoff"][0] for i in range(E)])
            d["addoff"] = wm(ao)
        rf = out["refit"].get(str(j), {})
        raw = [v["raw"][0] for r in rf.values() for v in r.values()]
        fit = [v["fit"][0] for r in rf.values() for v in r.values()]
        d["refit_share"] = float((np.sum(raw) - np.sum(fit)) / np.sum(raw)) if raw else float("nan")
        d["cos_upd"] = out["div"][str(j)]["cos_upd_mean"]
        d["upd_rel"] = float(np.mean(out["div"][str(j)]["upd_rel"]))
        s[str(j)] = {k: round(v, 4) for k, v in d.items()}
    return s


def foreign_share(model, D, n_win=64):
    """per MoE block: |full - own| / |full| of the gradient of the experts' own tensors (mean over experts), natural
    and as trained (gradw variants: the variant's weighting of the expert-write path), fp32"""
    dev = next(model.parameters()).device
    was = model.training
    model.eval()
    offs = D.starts(n_win)
    params = [p for j in range(model.M) for p in model.expert_params(j)]
    req = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    per = [len(model.expert_params(j)) for j in range(model.M)]
    acc = {c: [torch.zeros_like(p) for p in params] for c in ("full", "own", "trained")}

    def attn_w(W):
        def attn(b, blk, z, rope, mask):
            if b <= model.N:
                return blk.attn(z, rope, mask=mask)
            B, T, d = z.shape
            q, k, v = blk.qkv(z).view(B, T, 3, blk.H, blk.hd).permute(2, 0, 3, 1, 4)
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
            y = PairAttn.apply(q, k, v, mask, W, 1.0 / math.sqrt(blk.hd))
            return blk.o(y.transpose(1, 2).reshape(B, T, d))
        return attn
    for w0 in range(0, offs.numel(), 8):
        idx = D.win(offs[w0:w0 + 8])
        B, T = idx.shape
        tgt = idx[:, 1:]
        for c in ("full", "own", "trained"):
            if c == "trained" and model.v not in ("own", "hier", "flat", "shuf"):
                continue
            if c == "trained":
                x, _, _ = model(idx, keep=True, gradw=True)
            else:
                W = None if c == "full" else torch.eye(T, device=dev)[None].expand(B, -1, -1)
                x, _, _ = model(idx, attn=attn_w(W), gradw=False)
            h = model.base.nf(x[:, :-1])
            loss = F.cross_entropy((h @ model.base.emb.weight.t()).float().flatten(0, 1), tgt.flatten(),
                                   ignore_index=0, reduction="sum")
            gr = torch.autograd.grad(loss, params)
            for a_, g_ in zip(acc[c], gr):
                a_ += g_.float()
    for p, r in zip(params, req):
        p.requires_grad_(r)
    model.train(was)
    out = {}
    i0 = 0
    for c in ("natural", "trained"):
        out[c] = []
    for j in range(model.M):
        sl = slice(i0, i0 + per[j])
        i0 += per[j]

        def vec(c, k):
            return torch.cat([t[k].reshape(-1) for t in acc[c][sl]])
        nat, tr = [], []
        for k in range(E):
            f, o = vec("full", k), vec("own", k)
            nat.append(float((f - o).norm() / max(1e-30, f.norm())))
            if model.v in ("own", "hier", "flat", "shuf"):
                t = vec("trained", k)
                tr.append(float((t - o).norm() / max(1e-30, t.norm())))
        out["natural"].append(float(np.mean(nat)))
        out["trained"].append(float(np.mean(tr)) if tr else out["natural"][-1])
    return out
