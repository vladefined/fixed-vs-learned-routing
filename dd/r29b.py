"""Round 29, addition (home): routing to the new parameters by a frozen linear classifier "new / old" on the input of
blocks 5-8, gated LoRA, the same 1000 steps and windows as dd.r29, both sources.
  zclf   з) new experts by the classifier (perlayer 40k): a token marked at block j goes to the copy of its own expert
            (the copies and the fallback of ж: owners of the codes of в); only the copies learn
  iclf   и) LoRA by the classifier (perlayer 40k): LoRA on qkv / o / FFN of blocks 5-8, the addition is switched on only
            for marked tokens; the FFN addition is shared by the experts of a block; parameters as з
  kclf   к) the same on dense-cont 40k (its own classifier, trained on dense-cont states)
  llabel л) LoRA by the document label (perlayer 40k): the honest pair of ж
Classifier (per block, per base model, per source): logistic regression on the RMS-normalised block input of the base
model, standardised; trained on train windows (new: 256 windows of its train stream; old: wiki / code / ts train,
equal token shares), balanced classes, threshold 0; frozen during training.
LoRA: two lr (3e-4, 1e-3), both reported, the better one by the new source chosen. Measures as dd.r29 plus the share of
old tokens marked as new and of new tokens not marked, per block, at step 0 (base states) and after training.
    python -m dd.r29b clf
    python -m dd.r29b train --src ru --variant iclf --lr 1e-3
    python -m dd.r29b report
"""
import argparse
import json
import math
import os
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .model import Cfg, Model, doc_causal_mask, normalize  # noqa: E402
from .r21_moe import CTX, E, load_base, nearest  # noqa: E402
from .r28_train import R28, expert_down  # noqa: E402
from .r29 import BOOKS, INIT, N_EVAL, NEW, OLD, OUT, PL40, RU_SRC, Streams, jsave  # noqa: E402

DENSE40 = "runs_home/r28s3/dense/model_40000_bf16.pt"
VARIANTS = ("zclf", "iclf", "kclf", "llabel", "zclf5", "iclf5")
KIND = dict(zclf="pl", iclf="pl", kclf="dense", llabel="pl", zclf5="pl", iclf5="pl")
EXP = ("zclf", "zclf5")
RU_V = dict(zclf="з) новые эксперты по классификатору", iclf="и) LoRA по классификатору, perlayer",
            kclf="к) LoRA по классификатору, dense-cont", llabel="л) LoRA по метке, perlayer",
            zclf5="з5) новые эксперты, одна отметка от блока 5", iclf5="и5) LoRA, одна отметка от блока 5")
N_CLF, N_TOK = 256, 120_000


# ------------------------------------------------------------------------------------------------ models
def load_core(kind, dev):
    if kind == "pl":
        books = torch.load(BOOKS, map_location="cpu", weights_only=False)
        m = R28(load_base(INIT, dev), "base", books).to(dev)
        st = torch.load(PL40, map_location="cpu", weights_only=False)["state"]
        m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in st.items()})
        return m.eval()
    ck = torch.load(DENSE40, map_location="cpu", weights_only=False)
    base = Model(Cfg(**ck["cfg"]))
    base.load_state_dict({k[5:]: v.float() for k, v in ck["state"].items() if k.startswith("base.")})
    return base.to(dev).eval()


class GLin(nn.Module):
    """nn.Linear + LoRA addition switched on per token by ctx["m"] (B, T, 1) or off (None)"""

    def __init__(self, lin, r, ctx):
        super().__init__()
        self.lin, self.ctx = lin, ctx
        dev = lin.weight.device
        self.A = nn.Parameter(torch.randn(r, lin.in_features, device=dev) / math.sqrt(lin.in_features))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=dev))

    def forward(self, x):
        y = self.lin(x)
        m = self.ctx.get("m")
        if m is None:
            return y
        return y + ((x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)) * m.to(x.dtype)


class FFNLoRA(nn.Module):
    def __init__(self, d, h, r, dev):
        super().__init__()
        self.A1 = nn.Parameter(torch.randn(r, d, device=dev) / math.sqrt(d))
        self.B1 = nn.Parameter(torch.zeros(h, r, device=dev))
        self.A3 = nn.Parameter(torch.randn(r, d, device=dev) / math.sqrt(d))
        self.B3 = nn.Parameter(torch.zeros(h, r, device=dev))
        self.A2 = nn.Parameter(torch.randn(r, h, device=dev) / math.sqrt(h))
        self.B2 = nn.Parameter(torch.zeros(d, r, device=dev))

    def d(self, k, x):
        A, B = getattr(self, f"A{k}"), getattr(self, f"B{k}")
        return (x @ A.t().to(x.dtype)) @ B.t().to(x.dtype)


def pre_acts(z, eid, w1, w3):
    """per-token pre-activations of the token's expert (gate, up), original order"""
    order = torch.argsort(eid)
    counts = torch.bincount(eid, minlength=w1.shape[0]).tolist()
    gs, us = [], []
    for e, c in enumerate(z[order].split(counts)):
        gs.append(c @ w1[e].t())
        us.append(c @ w3[e].t())
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return torch.cat(gs)[inv], torch.cat(us)[inv]


class G29(nn.Module):
    """perlayer 40k (core = R28) or dense-cont 40k (core = Model) with marks per MoE-side block (classifier or
    label); mode none (the base), exp (з: marked tokens -> copies of their experts), lora (gated LoRA, blocks 5-8)"""

    def __init__(self, kind, core, mode, mark="clf", clf=None, sel=None, codesel=None, r=0):
        super().__init__()
        self.kind, self.core, self.mode, self.mark = kind, core, mode, mark
        self.base = core.base if kind == "pl" else core
        cfg = self.base.cfg
        self.L, self.N = cfg.n_layers, 4
        self.M, self.d, self.h = self.L - self.N, cfg.d, cfg.ffn_hidden
        dev = self.base.emb.weight.device
        self.ctx = {}
        if clf is not None:
            for k in ("W", "b", "mu", "sd"):
                self.register_buffer("clf_" + k, clf[k].to(dev).float())
        self.codesel = codesel                                              # (M, 64) bool: codes of в (pl only)
        for p in self.parameters():
            p.requires_grad_(False)
        if mode == "exp":
            self.nw1, self.nw3, self.nw2 = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
            self.newmap, self.fallback = [], []
            for j in range(self.M):
                owners = sel[str(j)]["owners"]
                mp = torch.full((E,), -1, dtype=torch.long, device=dev)
                for k, o in enumerate(owners):
                    mp[o] = E + k
                self.newmap.append(mp)
                self.fallback.append(E + int(np.argmax([sel[str(j)]["owner_mass"][str(o)] for o in owners])))
                self.nw1.append(nn.Parameter(core.w1[j][owners].detach().clone()))
                self.nw3.append(nn.Parameter(core.w3[j][owners].detach().clone()))
                self.nw2.append(nn.Parameter(core.w2[j][owners].detach().clone()))
        elif mode == "lora":
            self.lffn = nn.ModuleList()
            for j in range(self.M):
                blk = self.base.blocks[self.N + j]
                blk.qkv = GLin(blk.qkv, r, self.ctx)
                blk.o = GLin(blk.o, r, self.ctx)
                self.lffn.append(FFNLoRA(self.d, self.h, r, dev))

    def trainable(self):
        if self.mode == "exp":
            return list(self.nw1) + list(self.nw3) + list(self.nw2)
        if self.mode == "lora":
            ps = list(self.lffn.parameters())
            for j in range(self.M):
                blk = self.base.blocks[self.N + j]
                ps += [blk.qkv.A, blk.qkv.B, blk.o.A, blk.o.B]
            return ps
        return []

    def marks(self, j, x, label):
        if self.mark == "label":
            return label
        if self.mark == "clf5":                     # one mark per token: the classifier of block 5 (its input comes
            if j == 0:                              # from blocks 1-4, untouched by training), reused in blocks 6-8
                xn = normalize(x.float())
                self._mk5 = ((xn - self.clf_mu[0]) / self.clf_sd[0]) @ self.clf_W[0] + self.clf_b[0] > 0
            return self._mk5
        xn = normalize(x.float())
        return ((xn - self.clf_mu[j]) / self.clf_sd[j]) @ self.clf_W[j] + self.clf_b[j] > 0

    def forward(self, idx, label=None, rec=None):
        base = self.base
        B, T = idx.shape
        rope = base.rope(T, idx.device)
        mask = doc_causal_mask(idx) if base.cfg.doc_mask else None
        x = base.emb(idx)
        self.ctx["m"] = None
        for b in range(self.L):
            blk = base.blocks[b]
            j = b - self.N
            mk = None
            if j >= 0:
                mk = self.marks(j, x, label) if (self.mark == "label" or hasattr(self, "clf_W")) else None
                if rec is not None:
                    if mk is not None:
                        rec.setdefault(f"mk{j}", []).append(mk.sum(1))
                    if rec.get("keep_x"):
                        rec.setdefault(f"x{j}", []).append(normalize(x.float()).reshape(-1, self.d).half())
                    if self.kind == "pl" and self.codesel is not None:
                        c = nearest(normalize(x.float()).reshape(-1, self.d), self.core.pl_books[j]).view(B, T)
                        rec.setdefault(f"cm{j}", []).append(self.codesel[j][c].sum(1))
                if self.mode == "lora" and mk is not None and bool(mk.any()):
                    self.ctx["m"] = mk[..., None].float()
            z1 = blk.n1(x)
            a = blk.attn(z1, rope, mask=mask)
            y = x + a
            z = blk.n2(y)
            m = self.ctx["m"]
            if j < 0 or self.kind == "dense":
                if m is not None:
                    lf = self.lffn[j]
                    mm = m.to(z.dtype)
                    g = blk.w1(z) + mm * lf.d(1, z)
                    u = blk.w3(z) + mm * lf.d(3, z)
                    a2 = F.silu(g) * u
                    x = y + blk.w2(a2) + mm * lf.d(2, a2)
                else:
                    x = y + blk.w2(F.silu(blk.w1(z)) * blk.w3(z))
            else:
                core = self.core
                e = core.route(j, x, idx)
                if self.mode == "exp" and mk is not None and bool(mk.any()):
                    mp = self.newmap[j][e]
                    mp = torch.where(mp < 0, torch.full_like(mp, self.fallback[j]), mp)
                    e = torch.where(mk, mp, e)
                if self.mode == "exp":
                    w1 = torch.cat([core.w1[j], self.nw1[j]])
                    w3 = torch.cat([core.w3[j], self.nw3[j]])
                    w2 = torch.cat([core.w2[j], self.nw2[j]])
                    zf, ef = z.reshape(-1, self.d), e.reshape(-1)
                    from .r28_train import acts
                    uf = expert_down(acts(zf, ef, w1, w3), ef, w2).view(B, T, -1)
                elif m is not None:
                    lf = self.lffn[j]
                    zf, ef = z.reshape(-1, self.d), e.reshape(-1)
                    mf = m.reshape(-1, 1).to(zf.dtype)
                    g, u = pre_acts(zf, ef, core.w1[j], core.w3[j])
                    g = g + mf * lf.d(1, zf)
                    u = u + mf * lf.d(3, zf)
                    a2 = F.silu(g) * u
                    uf = (expert_down(a2, ef, core.w2[j]) + mf * lf.d(2, a2)).view(B, T, -1)
                else:
                    uf = core.ffn(j, z, e)
                x = y + uf
            self.ctx["m"] = None
        return base.nf(x) @ base.emb.weight.t().to(x.dtype)


def lora_rank(n_target, d, h, M):
    per = M * (d * 4 + d * 2 + 3 * (d + h))          # qkv r(d+3d), o r(2d), FFN 3 r(d+h) per block
    return max(1, round(n_target / per)), per


# ------------------------------------------------------------------------------------------------ classifier
def clf_path(kind, src):
    return os.path.join(OUT, f"clf_{kind}_{src}.pt")


@torch.no_grad()
def states(G, S, src, n, split, bs=16):
    offs = S.offs(src, n, split)
    rec = {"keep_x": True}
    for i in range(0, offs.numel(), bs):
        idx = S.win(src, offs[i:i + bs], split)[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            G(idx, rec=rec)
    return [torch.cat(rec[f"x{j}"]) for j in range(G.M)]


def fit_logreg(Xn, Xo, steps=400):
    X = torch.cat([Xn, Xo]).float()
    y = torch.cat([torch.ones(Xn.shape[0]), torch.zeros(Xo.shape[0])]).to(X.device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, y) + 1e-4 * w.pow(2).sum()
        loss.backward()
        opt.step()
    acc = float((((Z @ w + b) > 0) == (y > 0)).float().mean())
    return w.detach(), b.detach()[0], mu, sd, acc


def clf_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    S = Streams(dev)
    g = torch.Generator(device=dev).manual_seed(29)
    for kind in ("pl", "dense"):
        G = G29(kind, load_core(kind, dev), "none").to(dev)
        old = {o: states(G, S, o, N_CLF, "train") for o in OLD}
        for src in NEW:
            if os.path.exists(clf_path(kind, src)):
                continue
            new = states(G, S, src, N_CLF, "train")
            out = dict(W=[], b=[], mu=[], sd=[], acc=[])
            for j in range(G.M):
                Xn = new[j][torch.randperm(new[j].shape[0], generator=g, device=dev)[:N_TOK]]
                Xo = torch.cat([old[o][j][torch.randperm(old[o][j].shape[0], generator=g, device=dev)[:N_TOK // 3]]
                                for o in OLD])
                with torch.enable_grad():
                    w, b, mu, sd, acc = fit_logreg(Xn, Xo)
                for k, v in zip(("W", "b", "mu", "sd", "acc"), (w, b, mu, sd, acc)):
                    out[k].append(v)
            ck = {k: torch.stack(v).cpu() for k, v in out.items() if k != "acc"}
            ck["train_acc"] = out["acc"]
            torch.save(ck, clf_path(kind, src))
            print(f"clf {kind} {src}: train acc " + " ".join(f"{x:.4f}" for x in out["acc"]), flush=True)
        del G
        torch.cuda.empty_cache()
    # forward checks: G29 base == R28 forward (perlayer), dense == the r21 MoE dense of dense-cont
    from .r21_moe import load_moe
    G = G29("dense", load_core("dense", dev), "none").to(dev)
    md = load_moe("runs_home/r28s3/dense", dev=dev, fname="model_40000_bf16.pt")
    idx = S.win("wiki", S.offs("wiki", 4))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l1, l2 = G(idx).float(), md(idx)[0].float()
    print(f"check dense forward: max |logit diff| {float((l1 - l2).abs().max()):.2e}", flush=True)


# ------------------------------------------------------------------------------------------------ eval / train
@torch.no_grad()
def eval_marks(G, S, src, bs=32):
    offs = S.offs(src, N_EVAL, "val")
    sums, cnt = [], []
    rec = {}
    G.eval()
    for i in range(0, offs.numel(), bs):
        idx = S.win(src, offs[i:i + bs], "val")
        lab = torch.full(idx.shape, src in NEW, dtype=torch.bool, device=idx.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = G(idx, label=lab, rec=rec)
        tgt = idx[:, 1:]
        lo = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="none",
                             ignore_index=0).view(tgt.shape).double()
        sums.append(lo.sum(1))
        cnt.append((tgt != 0).double().sum(1))
    n_tok = offs.numel() * (CTX + 1)
    mk = {k: float(torch.cat(v).sum()) / n_tok for k, v in rec.items() if k.startswith(("mk", "cm"))}
    return torch.cat(sums).cpu().numpy(), torch.cat(cnt).cpu().numpy(), mk


def eval_all(G, S, src, tag, out):
    res, marks = {}, {}
    for s in (src,) + OLD:
        sm, ct, mk = eval_marks(G, S, s)
        res[s] = float(sm.sum() / ct.sum())
        marks[s] = mk
        np.savez(os.path.join(out, f"eval_{tag}_{s}.npz"), sum=sm, cnt=ct)
    return res, marks


def codesel_of(sel, dev):
    cs = torch.zeros(4, 64, dtype=torch.bool, device=dev)
    for j in range(4):
        cs[j, sel[str(j)]["codes"]] = True
    return cs


def train_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(29)
    dev = torch.device("cuda")
    S = Streams(dev)
    out = os.path.join(OUT, a.src)
    s0 = json.load(open(os.path.join(OUT, "stage0.json"), encoding="utf-8"))
    sel = s0["select"][a.src]
    n_new = sum(len(v["owners"]) for v in sel.values())
    res_p = os.path.join(out, "r29b.json")
    res = json.load(open(res_p, encoding="utf-8")) if os.path.exists(res_p) else {}
    t0 = time.time()
    for kind in ("pl", "dense"):                                  # base evals with step-0 marks (once)
        key = f"base_{kind}"
        if key not in res:
            clf = torch.load(clf_path(kind, a.src), map_location="cpu", weights_only=False)
            G = G29(kind, load_core(kind, dev), "none", clf=clf,
                    codesel=codesel_of(sel, dev) if kind == "pl" else None).to(dev)
            ce, mk = eval_all(G, S, a.src, f"b_{kind}", out)
            res[key] = dict(ce=ce, marks=mk)
            if kind == "pl":                                      # the same function as the dd.r29 base
                res[key]["equal_r29_base"] = {s: bool(np.array_equal(
                    np.load(os.path.join(out, f"eval_b_pl_{s}.npz"))["sum"],
                    np.load(os.path.join(out, f"eval_base_{s}.npz"))["sum"])) for s in (a.src,) + OLD}
            jsave(res_p, res)
            print(f"{key}: {ce} marks {json.dumps(mk)} ({time.time() - t0:.0f}s)", flush=True)
            del G
            torch.cuda.empty_cache()
    if a.variant == "base":
        return
    kind = KIND[a.variant]
    tag = a.variant if a.variant == "zclf" else f"{a.variant}_lr{a.lr:g}"
    clf = torch.load(clf_path(kind, a.src), map_location="cpu", weights_only=False)
    n_target = n_new * 3 * 512 * 1408
    r = 0
    if a.variant not in EXP:
        r, _ = lora_rank(n_target, 512, 1408, 4)
    mark = "label" if a.variant == "llabel" else "clf5" if a.variant.endswith("5") else "clf"
    G = G29(kind, load_core(kind, dev), "exp" if a.variant in EXP else "lora", mark=mark, clf=clf, sel=sel,
            r=r).to(dev)
    params = G.trainable()
    for p in params:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in params)
    print(f"{a.src} {tag}: trainable {n_tr / 1e6:.2f}M (target {n_target / 1e6:.2f}M, r {r})", flush=True)
    groups = {}
    for p in params:
        groups.setdefault(a.wd if p.dim() >= 2 else 0.0, []).append(p)
    opt = torch.optim.AdamW([dict(params=v, weight_decay=w) for w, v in groups.items()], lr=a.lr,
                            betas=(0.9, 0.95), fused=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    st = S.train[a.src]
    n_win = (st.numel() - 1) // (CTX + 1)
    perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(29))[:a.steps * a.bs].to(dev) * (CTX + 1)
    log, mk_train = [], []
    G.train()
    for step in range(a.steps):
        idx = st[perm[step * a.bs:(step + 1) * a.bs, None] + torch.arange(CTX + 1, device=dev)]
        lab = torch.ones(idx.shape, dtype=torch.bool, device=dev)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        rec = {} if step % 100 == 0 else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = G(idx, label=lab, rec=rec)
            ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), idx[:, 1:].flatten(), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if rec is not None or step == a.steps - 1:
            frac = [float(torch.cat(rec[f"mk{j}"]).sum()) / idx.numel() for j in range(4)] if rec else None
            log.append(dict(step=step, ce=float(ce.detach()), gn=float(gn), marked=frac))
            print(f"{step:5d} ce {float(ce):.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} marked {frac} "
                  f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
    ce, mk = eval_all(G, S, a.src, tag, out)
    res[tag] = dict(variant=a.variant, kind=kind, ce=ce, marks=mk, trainable=n_tr, lora_r=r, lr=a.lr, log=log,
                    time=time.time() - t0)
    if a.variant == "llabel":
        res[tag]["bit_exact_old"] = {s: bool(np.array_equal(np.load(os.path.join(out, f"eval_{tag}_{s}.npz"))["sum"],
                                                            np.load(os.path.join(out, f"eval_b_pl_{s}.npz"))["sum"]))
                                     for s in OLD}
    jsave(res_p, res)
    print(f"{a.src} {tag}: {ce} marks {json.dumps(mk)} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ report
def load_ev(out, tag, s):
    z = np.load(os.path.join(out, f"eval_{tag}_{s}.npz"))
    return z["sum"], z["cnt"]


def pd(out, t1, t2, s):
    a_, c = load_ev(out, t1, s)
    b_, c2 = load_ev(out, t2, s)
    assert np.array_equal(c, c2)
    n = len(c)
    q = (a_.sum() - b_.sum()) / c.sum()
    se = math.sqrt((((a_ - b_) - q * c) ** 2).sum() / (n - 1) / n) / c.mean()
    return float(q), float(se)


def row(out, tag, base, src):
    dn = pd(out, tag, base, src)
    do = [pd(out, tag, base, s) for s in OLD]
    mo = float(np.mean([x[0] for x in do]))
    seo = math.sqrt(sum(x[1] ** 2 for x in do)) / 3
    return dn, do, mo, seo, -dn[0] - mo, math.sqrt(dn[1] ** 2 + seo ** 2)


def fmt_marks(mk, src, key="mk"):
    if not mk or f"{key}0" not in mk.get(src, {}):
        return "—", "—"
    old = [np.mean([mk[o][f"{key}{j}"] for o in OLD]) for j in range(4)]
    new = [1 - mk[src][f"{key}{j}"] for j in range(4)]
    return " / ".join(f"{x:.3f}" for x in old), " / ".join(f"{x:.3f}" for x in new)


def report_cmd(a):
    L = ["# r29, дополнение: маршрут по классификатору, LoRA по классификатору и по метке\n",
         "Те же 1000 шагов и окна, что в r29; 512 окон на источник; Δ к базе своей модели (perlayer 40k или "
         "dense-cont 40k) ± 2 SE парно по окнам. Отметки: доля старых токенов, отмеченных как новые (среднее по "
         "wiki / code / ts), и доля новых, не отмеченных; по блокам 5 / 6 / 7 / 8, после обучения; отметки на "
         "состояниях базы (шаг 0) — строкой над таблицей.\n"]
    for src in NEW:
        out = os.path.join(OUT, src)
        p = os.path.join(out, "r29b.json")
        if not os.path.exists(p):
            continue
        res = json.load(open(p, encoding="utf-8"))
        old = json.load(open(os.path.join(out, "r29.json"), encoding="utf-8"))
        bp, bd = res["base_pl"]["ce"], res["base_dense"]["ce"]
        L.append(f"\n## {RU_SRC[src]}\n")
        L.append(f"База perlayer 40k: новый {bp[src]:.4f}, wiki {bp['wiki']:.4f}, code {bp['code']:.4f}, ts "
                 f"{bp['ts']:.4f}. База dense-cont 40k: новый {bd[src]:.4f}, wiki {bd['wiki']:.4f}, code "
                 f"{bd['code']:.4f}, ts {bd['ts']:.4f}.\n")
        o0, n0 = fmt_marks(res["base_pl"]["marks"], src, "mk")
        oc, nc = fmt_marks(res["base_pl"]["marks"], src, "cm")
        od, nd = fmt_marks(res["base_dense"]["marks"], src, "mk")
        L.append(f"Отметки на состояниях базы (шаг 0): классификатор perlayer — старые как новые {o0}, новые "
                 f"пропущены {n0}; коды в) — старые как новые {oc}, новые пропущены {nc}; классификатор dense-cont — "
                 f"старые как новые {od}, новые пропущены {nd}.\n")
        L.append("| вариант | lr | обучаемых | новый Δ | wiki | code | ts | старые в среднем | итог | старые → новые, "
                 "блоки 5–8 | новые пропущены, блоки 5–8 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = [("в) по кодам (r29)", "codes", "base", "3e-4", f"{old['codes']['trainable'] / 1e6:.2f}M", (oc, nc)),
                ("ж) по метке (r29)", "label", "base", "3e-4", f"{old['label']['trainable'] / 1e6:.2f}M", ("0", "0")),
                ("д) LoRA везде (r29)", "lora", "base", "3e-4", f"{old['lora']['trainable'] / 1e6:.2f}M", ("—", "—"))]
        best = {}
        for v in VARIANTS:
            tags = [t for t in res if t != "base_pl" and t != "base_dense" and res[t]["variant"] == v]
            if v != "zclf" and tags:
                best[v] = min(tags, key=lambda t: res[t]["ce"][src])
            for t in sorted(tags, key=lambda t: res[t]["lr"]):
                r = res[t]
                nm = RU_V[v] + (" ← лучший lr" if v != "zclf" and len(tags) > 1 and best.get(v) == t else "")
                mk = fmt_marks(r["marks"], src) if v != "llabel" else ("0", "0")
                rows.append((nm, t, "b_" + r["kind"], f"{r['lr']:g}",
                             f"{r['trainable'] / 1e6:.2f}M" + (f" (r {r['lora_r']})" if r["lora_r"] else ""), mk))
        for nm, tag, base, lr, ntr, mk in rows:
            dn, do, mo, seo, net, sen = row(out, tag, base, src)
            L.append(f"| {nm} | {lr} | {ntr} | {dn[0]:+.4f} ± {2 * dn[1]:.4f} | " +
                     " | ".join(f"{x[0]:+.4f}" for x in do) + f" | {mo:+.4f} ± {2 * seo:.4f} | "
                     f"**{net:+.4f} ± {2 * sen:.4f}** | {mk[0]} | {mk[1]} |")
        pairs = []
        if "zclf" in res:
            pairs.append(("з − ж (новый)", "zclf", "label", src))
            pairs.append(("з − в (новый)", "zclf", "codes", src))
        if "llabel" in best:
            pairs.append(("л − ж (новый)", best["llabel"], "label", src))
            if "llabel_lr0.0003" in res:
                pairs.append(("л при 3e-4 − ж (новый, тот же lr)", "llabel_lr0.0003", "label", src))
        if "iclf" in best and "zclf" in res:
            pairs.append(("и − з (новый)", best["iclf"], "zclf", src))
        if "iclf" in best and "llabel" in best:
            pairs.append(("и − л (новый)", best["iclf"], best["llabel"], src))
        for nm, t1, t2 in (("з5 − з (новый; lr 1e-3 против 3e-4)", "zclf5_lr0.001", "zclf"),
                           ("з5 − ж (новый)", "zclf5_lr0.001", "label"),
                           ("з5 − в (новый)", "zclf5_lr0.001", "codes"),
                           ("и5 − и, оба 1e-3 (новый)", "iclf5_lr0.001", "iclf_lr0.001"),
                           ("и5 − л, оба 1e-3 (новый)", "iclf5_lr0.001", "llabel_lr0.001"),
                           ("и5 − з5, оба 1e-3 (новый)", "iclf5_lr0.001", "zclf5_lr0.001")):
            if t1 in res and (t2 in res or t2 in ("label", "codes")):
                pairs.append((nm, t1, t2, src))
        if "kclf" in best and "iclf" in best:
            pairs.append(("к − и, абсолютная CE (новый)", best["kclf"], best["iclf"], src))
            for s in OLD:
                pairs.append((f"к − и, абсолютная CE ({s})", best["kclf"], best["iclf"], s))
        if pairs:
            L.append("\nПарные разности: " + "; ".join(
                f"{nm}: {q:+.4f} ± {2 * se:.4f}" for nm, (q, se) in
                ((nm, pd(out, t1, t2, s)) for nm, t1, t2, s in pairs)) + ".")
        for t in res:
            if t.startswith("llabel") and "bit_exact_old" in res[t]:
                L.append(f"\n{t}: побитовое совпадение с базой на старых {res[t]['bit_exact_old']}.")
        L.append(f"\nКонтроль: прямой проход G29 = база r29 по каждому окну: {res['base_pl'].get('equal_r29_base')}.")
    open("report/r29b_tables.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clf", "train", "report"])
    ap.add_argument("--src", default="ru", choices=NEW)
    ap.add_argument("--variant", default="zclf", choices=VARIANTS + ("base",))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.1)
    a = ap.parse_args(argv)
    dict(clf=clf_cmd, train=train_cmd, report=report_cmd)[a.cmd](a)


if __name__ == "__main__":
    main()
