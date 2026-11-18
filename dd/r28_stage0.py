"""Round 28, stage 0 (home GPU, no training): what keeps the experts of r21 alike. Checkpoints of r21: MoE perlayer s0
and MoE hash s0 (mix data, A model d 512, 8 blocks, experts in blocks 5..8, E = 8, every expert started as a copy of
the A-mix FFN of its block).

  swap  - tokens of expert i (in one MoE block, by the model's own routing) go through expert k instead; all 8 x 8
          pairs, in every MoE block alone and in all MoE blocks at once (there: in every block the tokens of that
          block's expert i go through that block's expert k; CE on the tokens of expert i in any block). References for
          the tokens of i: the A-mix FFN of the block, the FFN of the same block of dense-cont (A-mix trained further,
          r21 «dense»), a zero output. Rise of CE on those tokens against the model itself, +- 2 SE over windows.
  fit   - the same with the output refitted: gate and up of expert k frozen, down fitted by ridge regression on the
          calibration tokens of expert i to the output of expert i (lambda by held-out windows, every 5th); evaluated
          on val. References: the A-mix FFN's gate / up, expert i's own gate / up, random gate / up (the norms of
          expert i's rows). Share of the own-vs-foreign gap removed by the refit = (swap - fit) / swap.
  div   - weights: cosine and relative norm of the difference between experts and against the A-mix FFN; the same
          for the updates (expert - A-mix), with the dense-cont FFN update as a reference.
  grad  - for every expert the gradient of its weights from the loss at the positions it processed (own) and from
          the losses at other positions (foreign: everything that reaches it through the K/V of later attention);
          foreign split by the topic of the reading token (its expert in block 5): 8 x 8 per block. Forward unchanged,
          the gradient of a (reader t, read s) pair through K and V is weighted (1 on the diagonal).

  python -m dd.r28_stage0 swap|fit|div|grad --ckpt perlayer|hash [--smoke]
  python -m dd.r28_stage0 report
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import apply_rope, doc_causal_mask, normalize
from .r21_moe import CTX, E, SRC, Data, load_base, load_moe, nearest

OUT = os.path.join("runs_home", "r28")
CKPT = {"perlayer": "runs_vast/mix_perlayer", "hash": "runs_vast/mix_hash"}
AMIX = "runs_vast/Amix_s0/model.pt"
DENSE = "runs_vast/mix_dense"
CALIB = "data_mix/calib_train.bin"
N_EVAL, N_CAL, N_GRAD = 1024, 1024, 256
BS = 32
HOLD = 5
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SMOKE = False


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def jsave(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p + ".tmp", "w", encoding="utf-8"), indent=1)
    os.replace(p + ".tmp", p)


def opath(name):
    return os.path.join(OUT, ("smoke_" if SMOKE else "") + name)


def swiglu(z, w1, w3, w2):
    return (F.silu(z @ w1.t()) * (z @ w3.t())) @ w2.t()


def route(m, j, x, idx):
    """expert of every token in MoE block j from the block input x (as r21 MoE.forward)"""
    B, T, d = x.shape
    if m.route == "hash":
        return m.hash_table[j][idx]
    if m.route == "perlayer":
        return m.pl_table[j][nearest(normalize(x.float()).reshape(-1, d), m.pl_books[j]).view(B, T)]
    raise ValueError(m.route)


def ratio_se(num, den):
    """ratio estimator over windows: mean = sum num / sum den, SE by the delta method"""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    ok = den > 0
    num, den = num[ok], den[ok]
    n = len(den)
    if n < 2 or den.sum() == 0:
        return float("nan"), float("nan")
    r = num.sum() / den.sum()
    res = num - r * den
    se = math.sqrt((res ** 2).sum() / (n - 1) / n) / den.mean()
    return float(r), float(se)


# ------------------------------------------------------------------------------------------------ models and cache
class Ctx:
    """the model, its references (A-mix FFN, dense-cont FFN) and the cached states of the eval windows"""

    def __init__(self, ckpt, dev):
        self.m = load_moe(CKPT[ckpt], dev=dev)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.N, self.L = self.m.N, self.m.base.cfg.n_layers
        a = load_base(AMIX, dev)
        dn = load_moe(DENSE, dev=dev)
        self.amix = [tuple(getattr(a.blocks[b], w).weight.detach().float() for w in ("w1", "w3", "w2"))
                     for b in range(self.N, self.L)]
        self.dense = [tuple(getattr(dn.base.blocks[b], w).weight.detach().float() for w in ("w1", "w3", "w2"))
                      for b in range(self.N, self.L)]
        del a, dn
        self.dev = dev

    def expert(self, j, k):
        ex = self.m.experts[j]
        return ex.w1[k], ex.w3[k], ex.w2[k]

    @torch.no_grad()
    def cache(self, D, offs):
        """block inputs of the MoE blocks, routes, the model's own per-token CE (positions 0..T-2) and sources"""
        m, base = self.m, self.m.base
        n = offs.numel()
        T = CTX + 1
        self.offs = offs
        self.x = [torch.empty(n, T, base.cfg.d, device=self.dev) for _ in range(self.L - self.N)]     # fp32: exact resume
        self.e = [torch.empty(n, T, dtype=torch.int8, device=self.dev) for _ in range(self.L - self.N)]
        self.ce = torch.empty(n, T - 1, device=self.dev)
        self.src = torch.empty(n, T - 1, dtype=torch.int8, device=self.dev)
        self.idx = torch.empty(n, T, dtype=torch.long, device=self.dev)
        for w0 in range(0, n, BS):
            o = offs[w0:w0 + BS]
            idx = D.win(o)
            self.idx[w0:w0 + BS] = idx
            self.src[w0:w0 + BS] = D.src(o)[:, 1:].to(torch.int8)
            with torch.autocast("cuda", dtype=torch.bfloat16):            # the same precision as every modified run
                x, es, xs = self.forward(idx, b0=0, keep=True)
                for j in range(self.L - self.N):
                    self.x[j][w0:w0 + BS] = xs[j].float()
                    self.e[j][w0:w0 + BS] = es[j].to(torch.int8)
                self.ce[w0:w0 + BS] = self.token_ce(x, idx)

    def token_ce(self, x, idx, sel=None):
        """per-token CE of positions 0..T-2 (B, T-1); sel (B, T-1) bool -> only there (else 0)"""
        base = self.m.base
        h = base.nf(x[:, :-1])
        tgt = idx[:, 1:]
        out = torch.zeros(tgt.shape, device=x.device)
        if sel is None:
            sel = torch.ones_like(tgt, dtype=torch.bool)
        sel = sel & (tgt != 0)
        lg = h[sel] @ base.emb.weight.t().to(h.dtype)
        out[sel] = F.cross_entropy(lg.float(), tgt[sel], reduction="none")
        return out

    def forward(self, idx, b0=0, x=None, mods=None, keep=False, attn=None, outs=None):
        """blocks b0..L-1 from x (the input of block b0; the embedding when b0 = 0); mods {j: [(sel_fn, repl)]}:
        tokens where sel_fn(e) is True get repl(z) instead of their expert; attn(b, blk, z, rope, mask) -> attention
        output (gradient experiment); returns the last state, routes, block inputs of the MoE blocks"""
        m, base = self.m, self.m.base
        B, T = idx.shape
        rope = base.rope(T, idx.device)
        mask = doc_causal_mask(idx) if base.cfg.doc_mask else None
        if x is None:
            x = base.emb(idx)
        es, xs = {}, {}
        for b in range(b0, self.L):
            blk = base.blocks[b]
            if b >= self.N and keep:
                xs[b - self.N] = x
            z1 = blk.n1(x)
            a = attn(b, blk, z1, rope, mask) if attn is not None else blk.attn(z1, rope, mask=mask)
            y = x + a
            z = blk.n2(y)
            if b < self.N:
                x = y + blk.w2(F.silu(blk.w1(z)) * blk.w3(z))
                continue
            j = b - self.N
            e = route(m, j, x, idx)
            es[j] = e
            uf = m.experts[j](z, e)
            for sel_fn, repl in (mods or {}).get(j, []):
                s = sel_fn(e)
                if s.any():
                    uf = uf.clone()
                    uf[s] = repl(z[s]).to(uf.dtype)
            if outs is not None:
                outs.append(uf)
            x = y + uf
        return x, es, xs


# ------------------------------------------------------------------------------------------------ swap / fit runs
def run_mod(C, j0, mods, sel_tok):
    """CE rise on the selected tokens: per-window sums (rise) and counts; sel_tok(w0, w1) -> (B, T-1) bool"""
    num, den = [], []
    n = C.offs.numel()
    for w0 in range(0, n, BS):
        idx = C.idx[w0:w0 + BS]
        sel = sel_tok(w0, w0 + BS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, _, _ = C.forward(idx, b0=C.N + j0, x=C.x[j0][w0:w0 + BS].float(), mods=mods)
            ce = C.token_ce(x, idx, sel)
        sel = sel & (idx[:, 1:] != 0)
        d = (ce - C.ce[w0:w0 + BS]) * sel
        num.append(d.double().sum(1).cpu())
        den.append(sel.double().sum(1).cpu())
    return ratio_se(torch.cat(num).numpy(), torch.cat(den).numpy())


def sel_expert(C, j, i):
    return lambda w0, w1: (C.e[j][w0:w1, :-1] == i)


def sel_any(C, i):
    def f(w0, w1):
        s = None
        for j in range(C.L - C.N):
            t = C.e[j][w0:w1, :-1] == i
            s = t if s is None else (s | t)
        return s
    return f


def repl_w(w):
    return lambda z: swiglu(z, *w)                       # under the same autocast as the experts' own dispatch


def eval_ckpt(ckpt, dev):
    D = Data(dev=dev)
    C = Ctx(ckpt, dev)
    offs = D.starts(16 if SMOKE else N_EVAL)
    C.cache(D, offs)
    return D, C


@torch.no_grad()
def swap_cmd(ckpt):
    p = opath(f"swap_{ckpt}.json")
    res = jload(p)
    dev = torch.device("cuda")
    D, C = eval_ckpt(ckpt, dev)
    M = C.L - C.N
    res.setdefault("load", [[float((C.e[j][:, :-1] == i).float().mean()) for i in range(E)] for j in range(M)])
    res["resume_check"] = [run_mod(C, j, {}, lambda w0, w1: torch.ones_like(C.e[0][w0:w1, :-1], dtype=torch.bool))
                           for j in range(M)]                    # no modification -> rise must be exactly 0
    print(f"swap {ckpt}: resume check (rise without modification) {res['resume_check']}", flush=True)
    t0 = time.time()
    blocks = list(range(M)) + ["all"]
    for j in blocks:
        key = str(j)
        if key in res.get("done", []):
            continue
        out = {}
        js = list(range(M)) if j == "all" else [j]
        j0 = 0 if j == "all" else j
        for i in range(E):
            sel = sel_any(C, i) if j == "all" else sel_expert(C, j, i)
            row = {}
            refs = [(str(k), lambda jj, k=k: C.expert(jj, k)) for k in range(E) if k != i]
            refs += [("amix", lambda jj: C.amix[jj]), ("dense", lambda jj: C.dense[jj])]
            for name, wf in refs:
                mods = {jj: [((lambda e, i=i: e == i), repl_w(wf(jj)))] for jj in js}
                row[name] = run_mod(C, j0, mods, sel)
            mods = {jj: [((lambda e, i=i: e == i), (lambda z: torch.zeros_like(z)))] for jj in js}
            row["zero"] = run_mod(C, j0, mods, sel)
            out[str(i)] = row
        res.setdefault("blocks", {})[key] = out
        res.setdefault("done", []).append(key)
        jsave(p, res)
        print(f"swap {ckpt} block {key}: done ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ refit of the down
def calib_windows(dev, n):
    a = np.fromfile(CALIB, dtype=np.uint16).astype(np.int64)
    s = np.arange(0, a.size - CTX - 1, CTX)
    s = s[np.linspace(0, s.size - 1, min(n, s.size)).round().astype(int)]
    return torch.from_numpy(np.stack([a[x:x + CTX + 1] for x in s])).to(dev)


@torch.no_grad()
def fit_cmd(ckpt):
    p = opath(f"fit_{ckpt}.json")
    res = jload(p)
    dev = torch.device("cuda")
    D, C = eval_ckpt(ckpt, dev)
    M = C.L - C.N
    cal = calib_windows(dev, 32 if SMOKE else N_CAL)
    held_w = (torch.arange(cal.shape[0], device=dev) % HOLD) == HOLD - 1
    # inputs z, routes and own outputs of every MoE block on the calibration windows (dense inputs of the model)
    g = torch.Generator(device=dev).manual_seed(28)
    t0 = time.time()
    for j in range(M):
        if str(j) in res.get("done", []):
            continue
        Z, EI, HW = [], [], []
        for w0 in range(0, cal.shape[0], BS):
            idx = cal[w0:w0 + BS]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, es, xs = C.forward(idx, keep=True)
                blk = C.m.base.blocks[C.N + j]
                x = xs[j]
                rope = C.m.base.rope(idx.shape[1], dev)
                mask = doc_causal_mask(idx)
                y = x + blk.attn(blk.n1(x), rope, mask=mask)
                z = blk.n2(y)
            Z.append(z[:, :-1].reshape(-1, z.shape[-1]).float())
            EI.append(es[j][:, :-1].reshape(-1))
            HW.append(held_w[w0:w0 + BS][:, None].expand(-1, CTX).reshape(-1))
        Z, EI, HW = torch.cat(Z), torch.cat(EI), torch.cat(HW)
        Y = torch.empty(Z.shape[0], Z.shape[1], device=dev)
        for i in range(E):
            s = EI == i
            Y[s] = swiglu(Z[s], *C.expert(j, i))
        h = C.amix[j][0].shape[0]
        srcs = {str(k): C.expert(j, k)[:2] for k in range(E)}
        srcs["amix"] = C.amix[j][:2]
        fits = {}
        for name, (w1, w3) in srcs.items():
            for i in range(E):
                s = EI == i
                if name == "rand":
                    continue
                ww1, ww3 = w1, w3
                fits[(name, i)] = ridge(Z[s], Y[s], HW[s], ww1, ww3)
        for i in range(E):                                          # random gate / up with expert i's row norms
            w1, w3 = C.expert(j, i)[:2]
            r1 = torch.randn(w1.shape, generator=g, device=dev) * w1.norm(dim=1, keepdim=True) / math.sqrt(w1.shape[1])
            r3 = torch.randn(w3.shape, generator=g, device=dev) * w3.norm(dim=1, keepdim=True) / math.sqrt(w3.shape[1])
            s = EI == i
            fits[("rand", i)] = ridge(Z[s], Y[s], HW[s], r1, r3)
        del Z, Y, EI, HW
        torch.cuda.empty_cache()
        out = {}
        for i in range(E):
            row = {}
            for name in [str(k) for k in range(E)] + ["amix", "rand"]:
                w1, w3, w2, lam, expl = fits[(name, i)]
                mods = {j: [((lambda e, i=i: e == i), repl_w((w1, w3, w2)))]}
                r = run_mod(C, j, mods, sel_expert(C, j, i))
                row[name] = dict(rise=r, lam=lam, explained=expl)
            out[str(i)] = row
        res.setdefault("blocks", {})[str(j)] = out
        res.setdefault("done", []).append(str(j))
        jsave(p, res)
        print(f"fit {ckpt} block {j}: done ({time.time() - t0:.0f}s)", flush=True)


def ridge(Z, Y, held, w1, w3):
    """down (d, h) for gate / up (w1, w3) frozen: min |act(Z) W^T - Y|^2 + lam |W|^2, lambda by the held-out rows;
    returns (w1, w3, w2, lam, explained share of Y on the held-out rows)"""
    A = F.silu(Z @ w1.t()) * (Z @ w3.t())
    Af, Yf, Ah, Yh = A[~held].double(), Y[~held].double(), A[held].double(), Y[held].double()
    XtX, XtY = Af.t() @ Af, Af.t() @ Yf
    sc = float(torch.diagonal(XtX).mean()) + 1e-12
    eye = torch.eye(XtX.shape[0], dtype=torch.float64, device=Z.device)
    best = None
    for lam in LAMS:
        W = torch.linalg.solve(XtX + lam * sc * eye, XtY)                # (h, d)
        sse = float((Ah @ W - Yh).pow(2).sum())
        if best is None or sse < best[0]:
            best = (sse, lam, W)
    yy = float((Yh - Yh.mean(0)).pow(2).sum()) + 1e-12
    return w1, w3, best[2].t().float(), best[1], 1 - best[0] / yy


# ------------------------------------------------------------------------------------------------ weights
@torch.no_grad()
def div_cmd(ckpt):
    p = opath(f"div_{ckpt}.json")
    dev = torch.device("cuda")
    C = Ctx(ckpt, dev)
    M = C.L - C.N

    def flat(w):
        return torch.cat([t.reshape(-1).float() for t in w])

    def cos(a, b):
        return float(F.cosine_similarity(a, b, dim=0))
    out = {}
    for j in range(M):
        A = flat(C.amix[j])
        W = [flat(C.expert(j, k)) for k in range(E)]
        U = [w - A for w in W]
        Ud = flat(C.dense[j]) - A
        out[str(j)] = dict(
            cos_w=[[cos(W[a], W[b]) for b in range(E)] for a in range(E)],
            rel_diff=[[float((W[a] - W[b]).norm() / A.norm()) for b in range(E)] for a in range(E)],
            cos_upd=[[cos(U[a], U[b]) for b in range(E)] for a in range(E)],
            upd_rel=[float(u.norm() / A.norm()) for u in U],
            cos_amix=[cos(w, A) for w in W],
            cos_upd_dense=[cos(u, Ud) for u in U],
            dense_upd_rel=float(Ud.norm() / A.norm()))
    jsave(p, out)
    print(f"div {ckpt}: done", flush=True)


# ------------------------------------------------------------------------------------------------ foreign gradient
def make_attn(Wfn):
    """attention of a block with the gradient of the (reader t, read s) pair through K and V weighted by W[t, s]
    (B, T, T); the forward value is that of the ordinary attention"""
    def attn(b, blk, z, rope, mask):
        W = Wfn(b)
        if W is None:
            return blk.attn(z, rope, mask=mask)
        B, T, d = z.shape
        H, hd = blk.H, blk.hd
        q, k, v = blk.qkv(z).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        q, k, v = q.float(), k.float(), v.float()                      # fp32: the split sums round alike
        sc = 1.0 / math.sqrt(hd)
        Wh = W[:, None].float()
        with torch.autocast("cuda", enabled=False):
            s = (q @ k.transpose(-1, -2)) * sc
            s_sg = (q @ k.detach().transpose(-1, -2)) * sc
            s = Wh * s + (1 - Wh) * s_sg
            s = s.masked_fill(~mask, float("-inf"))
        with torch.autocast("cuda", enabled=False):
            pr = torch.softmax(s, -1)
            y = (pr * Wh) @ v + (pr * (1 - Wh)) @ v.detach()
            return F.linear(y.transpose(1, 2).reshape(B, T, d), blk.o.weight.float())
    return attn


def grad_cmd(ckpt):
    p = opath(f"grad_{ckpt}.json")
    if os.path.exists(p):
        print(f"grad {ckpt}: done before, skipped", flush=True)
        return
    dev = torch.device("cuda")
    D = Data(dev=dev)
    C = Ctx(ckpt, dev)
    M = C.L - C.N
    ex = [C.m.experts[j] for j in range(M)]
    params = [w for e in ex for w in (e.w1, e.w3, e.w2)]
    for w in params:
        w.requires_grad_(True)
    offs = D.starts(16 if SMOKE else N_GRAD)
    configs = ["full", "own"] + [f"g{g}" for g in range(E)]
    acc = {c: [torch.zeros_like(w) for w in params] for c in configs}
    ntok = 0
    tcount = torch.zeros(E, device=dev)
    gb = 8
    t0 = time.time()
    for w0 in range(0, offs.numel(), gb):
        idx = D.win(offs[w0:w0 + gb])
        with torch.no_grad():                                          # fp32 throughout: bf16 rounding noise
            _, es, _ = C.forward(idx)                                  # would leak into own vs full
        topic = es[0]                                                  # the reader's topic: its expert in block 5
        tcount += torch.bincount(topic[:, :-1].reshape(-1), minlength=E).float()
        T = idx.shape[1]
        eye = torch.eye(T, dtype=torch.bool, device=dev)[None]
        tgt = idx[:, 1:]
        ntok += int((tgt != 0).sum())
        for c in configs:
            if c == "full":                                            # the same explicit attention, weights 1
                Wm = torch.ones(idx.shape[0], T, T, device=dev)
                Wfn = lambda b, Wm=Wm: Wm if b > C.N else None         # noqa: E731
            elif c == "own":
                Wm = eye.expand(idx.shape[0], -1, -1).float()
                Wfn = lambda b, Wm=Wm: Wm if b > C.N else None         # noqa: E731
            else:
                g_ = int(c[1:])
                Wm = (eye | (topic == g_)[:, :, None]).float()
                Wfn = lambda b, Wm=Wm: Wm if b > C.N else None         # noqa: E731
            x, _, _ = C.forward(idx, attn=make_attn(Wfn))
            h = C.m.base.nf(x[:, :-1])
            lg = h @ C.m.base.emb.weight.t()
            loss = F.cross_entropy(lg.float().flatten(0, 1), tgt.flatten(), ignore_index=0, reduction="sum")
            gr = torch.autograd.grad(loss, params)
            for a_, g2 in zip(acc[c], gr):
                a_ += g2.float()
        print(f"grad {ckpt}: windows {w0 + gb} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for j in range(M):
        r = {}
        G = {c: [torch.cat([acc[c][3 * j + t][i].reshape(-1) for t in range(3)]) / ntok for i in range(E)]
             for c in configs}
        own = G["own"]
        fo = [G["full"][i] - own[i] for i in range(E)]
        r["own"] = [float(own[i].norm()) for i in range(E)]
        r["foreign"] = [float(fo[i].norm()) for i in range(E)]
        r["full"] = [float(G["full"][i].norm()) for i in range(E)]
        r["cos_own_foreign"] = [float(F.cosine_similarity(own[i], fo[i], dim=0)) for i in range(E)]
        r["by_topic"] = [[float((G[f"g{g}"][i] - own[i]).norm()) for g in range(E)] for i in range(E)]
        r["sum_topics_vs_foreign"] = [float(sum(((G[f"g{g}"][i] - own[i]) for g in range(E))).norm() / max(1e-30, fo[i].norm()))
                                      for i in range(E)]
        out[str(j)] = r
    out["reader_topic_share"] = (tcount / tcount.sum()).tolist()
    jsave(p, out)
    print(f"grad {ckpt}: done ({time.time() - t0:.0f}s)", flush=True)


def ograd_cmd(ckpt, win):
    """stage 0b comparison: gradient with respect to the expert output of every MoE block (the block's FFN output),
    own position vs later positions, as in the OLMoE measurement of this round (the same autograd attention), windows of `win` tokens,
    fp32"""
    from .r28_attn import GradAcc, PairAttn
    p = opath(f"ograd_{ckpt}_w{win}.json")
    if os.path.exists(p):
        print(f"ograd {ckpt} w{win}: done before, skipped", flush=True)
        return
    dev = torch.device("cuda")
    D = Data(dev=dev)
    C = Ctx(ckpt, dev)
    M = C.L - C.N
    offs = D.starts(16 if SMOKE else N_GRAD)

    def pair_attn(W):
        def attn(b, blk, z, rope, mask):
            B, T, d = z.shape
            q, k, v = blk.qkv(z).view(B, T, 3, blk.H, blk.hd).permute(2, 0, 3, 1, 4)
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
            y = PairAttn.apply(q, k, v, mask, W, 1.0 / math.sqrt(blk.hd))
            return blk.o(y.transpose(1, 2).reshape(B, T, d))
        return attn
    acc = GradAcc(M)
    t0 = time.time()
    gb = 8
    for w0 in range(0, offs.numel(), gb):
        idx = D.win(offs[w0:w0 + gb])[:, :win + 1]
        B, T = idx.shape
        tgt = idx[:, 1:]
        got = {}
        for c in ("full", "own"):
            W = None if c == "full" else torch.eye(T, device=dev)[None].expand(B, -1, -1)
            x0 = C.m.base.emb(idx).detach().requires_grad_(True)
            outs = []
            x, _, _ = C.forward(idx, x=x0, attn=pair_attn(W), outs=outs)
            h = C.m.base.nf(x[:, :-1])
            loss = F.cross_entropy((h @ C.m.base.emb.weight.t()).float().flatten(0, 1), tgt.flatten(), ignore_index=0,
                                   reduction="sum")
            gr = torch.autograd.grad(loss, outs)
            got[c] = [g_[:, :-1][tgt != 0].float() for g_ in gr]
        for j in range(M):
            acc.add(j, got["full"][j], got["own"][j])
    jsave(p, dict(win=win, n_win=int(offs.numel()), blocks=acc.stats()))
    print(f"ograd {ckpt} w{win}: done ({time.time() - t0:.0f}s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["swap", "fit", "div", "grad", "ograd", "report"])
    ap.add_argument("--ckpt", default="perlayer", choices=list(CKPT))
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    globals()["SMOKE"] = a.smoke
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.cmd == "report":
        from .r28_report import report0
        return report0()
    if a.cmd == "ograd":
        return ograd_cmd(a.ckpt, a.win)
    dict(swap=swap_cmd, fit=fit_cmd, div=div_cmd, grad=grad_cmd)[a.cmd](a.ckpt)


if __name__ == "__main__":
    main()
