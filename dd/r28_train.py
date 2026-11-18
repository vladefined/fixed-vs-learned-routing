"""Round 28, stage 1: what keeps experts alike -- training variants (as r21 perlayer: start from A-mix, 10k steps,
the same windows and lr plan, the same per-layer routing; vast.ai).

Only the gradient that reaches the experts' writes from other positions changes in the variants a..d (gradw): the
forward is unchanged; in the attention of the upper blocks (after the first MoE block) the keys and values are taken
twice -- from the state x and from x - e + stopgrad(e), e = the sum of the expert outputs written so far -- and the
gradient of the pair (reader t, read s) goes to the first copy with weight W[t, s], to the second with 1 - W[t, s].
So the rest of the state gets its gradient as usual and only the expert-write path is weighted. W from the topics of
reader and read token = their experts in block 5 (the first MoE block):
  base  (а) ordinary perlayer (the r21 baseline, retrained to save 2k / 5k)
  own   (б) W = 1 for the own topic, 0 for the others
  hier  (в) own topic 1, broad topic 0.5 (experts paired by the nearest centres -> 4 broad topics), others 0.1
  flat  (г) own topic 1, all others 0.3
  shuf  (д) as hier, but the topic of a token from a shuffled code -> topic table of block 5 (same group sizes)
  sdown (е) one down per block (shared), gate and up per expert
  add   (ж) shared FFN of A-mix, its 1056 most important neurons (lr x 0.1), always on + per expert an addition of
            352 neurons (gate / up = copies of the other 352 A-mix neurons, down = 0 at the start); 1408 per token
Step 0: every variant except add computes the A-mix function (checked); add starts at the A-mix FFN cut to 1056.
Saves at 2k / 5k / 10k with the measurements of dd.r28_measure (specialisation, refit share, divergence) at every save;
the foreign-gradient share at the start and the end.

  python -m dd.r28_train train --variant hier [--steps 10000 --name r28_hier]
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

from .model import apply_rope, doc_causal_mask, normalize  # noqa: E402
from .r21_moe import CTX, E, K, SRC, Data, eval_ce, load_base, nearest  # noqa: E402
from .ts_train import windows  # noqa: E402

VARIANTS = ("base", "own", "hier", "flat", "shuf", "sdown", "add")
GRADW = ("own", "hier", "flat", "shuf")
# stage 4 (routing by the output): tables from dd.r28_route prep
ROUTE4 = ("out", "inout", "mixed", "oracle")
ROUTE_TABLES = "runs_home/r28s4/route_tables.pt"
RU = dict(base="а) обычный perlayer", own="б) только своя тема", hier="в) иерархия", flat="г) ровное ослабление 0.3",
          shuf="д) контроль: иерархия, темы перемешаны", sdown="е) общий down", add="ж) общий + добавки",
          out="а) по выходу", inout="б) вход × выход", mixed="в) смешанный: хеш + выход", oracle="г) подсмотренный ответ",
          hash="hash")
SAVES = (2000, 5000, 10000)
W_SHARED, W_ADD = 1056, 352


# ------------------------------------------------------------------------------------------------ attention with pair gradients
class PairAttnSplit(torch.autograd.Function):
    """forward: ordinary softmax attention of q over (k, v); backward: the K / V gradient of the pair (t, s) goes to
    (k, v) with weight W[t, s] and to (ks, vs) with 1 - W[t, s] (the two copies have the same values); Q untouched"""

    @staticmethod
    def forward(ctx, q, k, ks, v, vs, mask, W, scale):
        up = (lambda t: t if t.dtype == torch.float64 else t.float())               # noqa: E731  fp32 (fp64 in tests)
        qf, kf, vf = up(q), up(k), up(v)
        s = (qf @ kf.transpose(-1, -2)) * scale
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, -1)
        ctx.save_for_backward(qf, kf, vf, p, W)
        ctx.scale, ctx.dt = scale, (q.dtype, k.dtype, ks.dtype, v.dtype, vs.dtype)
        return (p @ vf).to(q.dtype)

    @staticmethod
    def backward(ctx, gy):
        qf, kf, vf, p, W = ctx.saved_tensors
        gy = gy.to(qf.dtype)
        Wh = W[:, None].to(qf.dtype)
        gp = gy @ vf.transpose(-1, -2)
        gs = p * (gp - (gp * p).sum(-1, keepdim=True))
        gq = (gs @ kf) * ctx.scale
        pw, gw = p * Wh, gs * Wh
        gv, gvs = pw.transpose(-1, -2) @ gy, (p - pw).transpose(-1, -2) @ gy
        gk, gks = (gw.transpose(-1, -2) @ qf) * ctx.scale, ((gs - gw).transpose(-1, -2) @ qf) * ctx.scale
        dt = ctx.dt
        return gq.to(dt[0]), gk.to(dt[1]), gks.to(dt[2]), gv.to(dt[3]), gvs.to(dt[4]), None, None, None


def acts(z, eid, w1, w3):
    """per-token SwiGLU activations of the token's expert (n, h), original order"""
    order = torch.argsort(eid)
    counts = torch.bincount(eid, minlength=w1.shape[0]).tolist()
    outs = []
    for e, c in enumerate(z[order].split(counts)):
        outs.append(F.silu(c @ w1[e].t()) * (c @ w3[e].t()))
    a = torch.cat(outs)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return a[inv]


def swiglu(z, w1, w3, w2):
    return (F.silu(z @ w1.t()) * (z @ w3.t())) @ w2.t()


# ------------------------------------------------------------------------------------------------ the model
class R28(nn.Module):
    """A model whose blocks N+1..L get E experts routed by the r21 per-layer books (perlayer), with the variant's FFN
    form and gradient weighting"""

    def __init__(self, base, variant, books, imp=None, seed=0):
        super().__init__()
        cfg = base.cfg
        self.base, self.v, self.N = base, variant, books["N"]
        self.L = cfg.n_layers
        self.M = self.L - self.N
        d, h = cfg.d, cfg.ffn_hidden
        self.d, self.h = d, h
        self.register_buffer("pl_books", books["pl"].clone().float())
        self.register_buffer("pl_table", books["pl_sem"].clone().long())
        if variant in ("hash", "mixed"):                           # stage 3 / 4: r21 MoE-hash routing (token id -> expert)
            self.register_buffer("hash_table", books["hash"].clone().long())
        if variant in ROUTE4:                                      # stage 4: frozen A-mix lens + group tables (rebuilt
            rt = books["route"]                                    # from A-mix / the tables file, never saved)
            self.G = int(rt["out"].shape[1])
            self.register_buffer("tok_group", rt["tok_group"].clone().long(), persistent=False)
            self.register_buffer("out_table", rt["out"].clone().long(), persistent=False)       # (M, G)
            self.register_buffer("pair_table", rt["pair"].clone().long(), persistent=False)     # (M, G * G)
            self.register_buffer("next_table", rt["next"].clone().long(), persistent=False)     # (M, G)
            self.register_buffer("lens_w", base.nf.w.detach().clone().float(), persistent=False)
            self.register_buffer("lens_emb", base.emb.weight.detach().clone().to(torch.bfloat16), persistent=False)
            self.lens_eps = base.nf.eps
        tt = self.pl_table[0].clone()
        if variant == "shuf":                                       # same group sizes, membership shuffled
            g = torch.Generator().manual_seed(1000 + seed)
            tt = tt[torch.randperm(tt.numel(), generator=g)]
        self.register_buffer("topic_table", tt)
        self.register_buffer("Wt", self.topic_weights(variant))
        self.w1 = nn.ParameterList()
        self.w3 = nn.ParameterList()
        self.w2 = nn.ParameterList()
        self.sw1, self.sw3, self.sw2 = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for j in range(self.M):
            blk = base.blocks[self.N + j]
            A1, A3, A2 = blk.w1.weight.detach(), blk.w3.weight.detach(), blk.w2.weight.detach()
            if variant == "add":
                order = imp[j].argsort(descending=True)
                keep, rest = order[:W_SHARED], order[W_SHARED:W_SHARED + W_ADD]
                self.sw1.append(nn.Parameter(A1[keep].clone()))
                self.sw3.append(nn.Parameter(A3[keep].clone()))
                self.sw2.append(nn.Parameter(A2[:, keep].clone()))
                self.w1.append(nn.Parameter(A1[rest].expand(E, -1, -1).clone()))
                self.w3.append(nn.Parameter(A3[rest].expand(E, -1, -1).clone()))
                self.w2.append(nn.Parameter(torch.zeros(E, d, W_ADD)))
            else:
                self.w1.append(nn.Parameter(A1.expand(E, -1, -1).clone()))
                self.w3.append(nn.Parameter(A3.expand(E, -1, -1).clone()))
                if variant == "sdown":
                    self.w2.append(nn.Parameter(A2.clone()))                # (d, h): one down per block
                else:
                    self.w2.append(nn.Parameter(A2.expand(E, -1, -1).clone()))
            del blk.w1, blk.w3, blk.w2

    def topic_weights(self, variant):
        W = torch.ones(E, E)
        if variant in ("own",):
            W = torch.eye(E)
        elif variant == "flat":
            W = torch.full((E, E), 0.3)
            W.fill_diagonal_(1.0)
        elif variant in ("hier", "shuf"):
            broad = self.broad_topics()
            W = torch.full((E, E), 0.1)
            for a in range(E):
                for b in range(E):
                    if broad[a] == broad[b]:
                        W[a, b] = 0.5
            W.fill_diagonal_(1.0)
        return W

    def broad_topics(self):
        """the 8 experts of block 5 paired greedily by the distance of their centres (mean of their book codes)"""
        C = torch.stack([self.pl_books[0][self.pl_table[0] == e].mean(0) for e in range(E)])
        D = torch.cdist(C, C)
        D.fill_diagonal_(float("inf"))
        broad, left, g = [-1] * E, set(range(E)), 0
        while left:
            pairs = sorted((float(D[a, b]), a, b) for a in left for b in left if a < b)
            if not pairs:
                broad[left.pop()] = g
                break
            _, a, b = pairs[0]
            broad[a] = broad[b] = g
            left -= {a, b}
            g += 1
        self.broad = broad
        return broad

    # --- routing and the FFN of a MoE block
    @torch.no_grad()
    def lens_argmax(self, x, chunk=8):
        """the most probable token of the A-mix output layer (final norm + tied embedding, frozen) at the state x"""
        out = torch.empty(x.shape[:2], dtype=torch.long, device=x.device)
        for b0 in range(0, x.shape[0], chunk):
            xf = x[b0:b0 + chunk].float()
            h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.lens_eps) * self.lens_w
            out[b0:b0 + chunk] = (h.to(torch.bfloat16) @ self.lens_emb.t()).argmax(-1)
        return out

    def route(self, j, x, idx=None):
        if self.v == "hash" or (self.v == "mixed" and j < 2):
            return self.hash_table[j][idx]
        if self.v in ROUTE4:
            if self.v == "oracle":                                 # the true next token (last position: its own token)
                nxt = torch.cat([idx[:, 1:], idx[:, -1:]], 1)
                return self.next_table[j][self.tok_group[nxt]]
            gp = self.tok_group[self.lens_argmax(x)]
            if self.v == "inout":
                return self.pair_table[j][self.tok_group[idx] * self.G + gp]
            return self.out_table[j][gp]                          # out; mixed blocks 7-8
        B, T, d = x.shape
        return self.pl_table[j][nearest(normalize(x.float()).reshape(-1, d), self.pl_books[j]).view(B, T)]

    def topic(self, x):
        B, T, d = x.shape
        return self.topic_table[nearest(normalize(x.float()).reshape(-1, d), self.pl_books[0]).view(B, T)]

    def shared(self, j, z):
        return swiglu(z, self.sw1[j], self.sw3[j], self.sw2[j])

    def ffn(self, j, z, e):
        sh = z.shape
        zf, ef = z.reshape(-1, sh[-1]), e.reshape(-1)
        if self.v == "sdown":
            y = acts(zf, ef, self.w1[j], self.w3[j]) @ self.w2[j].t()
        else:
            y = expert_down(acts(zf, ef, self.w1[j], self.w3[j]), ef, self.w2[j])
        if self.v == "add":
            y = y + self.shared(j, zf)
        return y.view(*sh[:-1], -1)

    # experts as functions (measurements)
    def expert_out(self, j, k, z):
        if self.v == "sdown":
            return swiglu(z, self.w1[j][k], self.w3[j][k], self.w2[j])
        y = swiglu(z, self.w1[j][k], self.w3[j][k], self.w2[j][k])
        return y + self.shared(j, z) if self.v == "add" else y

    def expert_feats(self, j, k, z):
        """the inputs of the down that produce expert k's output (n, F)"""
        a = F.silu(z @ self.w1[j][k].t()) * (z @ self.w3[j][k].t())
        if self.v == "add":
            return torch.cat([F.silu(z @ self.sw1[j].t()) * (z @ self.sw3[j].t()), a], -1)
        return a

    def expert_params(self, j):
        """per-expert parameter tensors (E, ...) of block j (for gradients / divergence)"""
        ps = [self.w1[j], self.w3[j]]
        if self.v != "sdown":
            ps.append(self.w2[j])
        return ps

    # --- forward
    def forward(self, idx, ctl=None, b0=0, x=None, mods=None, keep=False, attn=None, outs=None, gradw=None):
        """gradw: weight the expert-write gradient of the attention (default: on while training for the gradw
        variants); mods {j: [(sel_fn(e), repl(z))]}: the FFN output of block j replaced where sel_fn; attn(b, blk, z,
        rope, mask) -> attention output (measurements); returns logits (or the last state if keep / b0 given), aux"""
        base = self.base
        B, T = idx.shape
        rope = base.rope(T, idx.device)
        mask = doc_causal_mask(idx) if base.cfg.doc_mask else None
        if x is None:
            x = base.emb(idx)
        if gradw is None:
            gradw = self.training and self.v in GRADW
        es, xs = {}, {}
        ew = torch.zeros_like(x) if gradw else None                          # the experts' writes so far
        Wp = None
        for b in range(b0, self.L):
            blk = base.blocks[b]
            if b >= self.N and keep:
                xs[b - self.N] = x
            if b == self.N and gradw:
                tp = self.topic(x)
                Wp = self.Wt[tp[:, :, None], tp[:, None, :]]                 # (B, T, T)
            z1 = blk.n1(x)
            if attn is not None:
                a = attn(b, blk, z1, rope, mask)
            elif gradw and b > self.N:
                a = self.split_attn(blk, z1, blk.n1(x - ew + ew.detach()), rope, mask, Wp)
            else:
                a = blk.attn(z1, rope, mask=mask)
            y = x + a
            z = blk.n2(y)
            if b < self.N:
                x = y + blk.w2(F.silu(blk.w1(z)) * blk.w3(z))
                continue
            j = b - self.N
            e = self.route(j, x, idx)
            es[j] = e
            uf = self.ffn(j, z, e)
            for sel_fn, repl in (mods or {}).get(j, []):
                s = sel_fn(e)
                if s.any():
                    uf = uf.clone()
                    uf[s] = repl(z[s]).to(uf.dtype)
            if outs is not None:
                outs.append(uf)
            if gradw:
                ew = ew + uf
            x = y + uf
        if keep or b0 > 0 or attn is not None or outs is not None:
            return x, es, xs
        logits = base.nf(x) @ base.emb.weight.t().to(x.dtype)
        return logits, dict(eid=[es[j] for j in range(self.M)])

    def split_attn(self, blk, z, zs, rope, mask, W):
        B, T, d = z.shape
        H, hd = blk.H, blk.hd
        q, k, v = blk.qkv(z).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        _, ks, vs = blk.qkv(zs).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, ks = apply_rope(q, *rope), apply_rope(k, *rope), apply_rope(ks, *rope)
        y = PairAttnSplit.apply(q, k, ks, v, vs, mask, W, 1.0 / math.sqrt(hd))
        return blk.o(y.transpose(1, 2).reshape(B, T, d))


def expert_down(a, eid, w2):
    """per-token down of its expert: a (n, h), w2 (E, d, h) -> (n, d)"""
    order = torch.argsort(eid)
    counts = torch.bincount(eid, minlength=w2.shape[0]).tolist()
    outs = [c @ w2[e].t() for e, c in enumerate(a[order].split(counts))]
    y = torch.cat(outs)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return y[inv]


def loss_fn(model, idx):
    logits, aux = model(idx)
    tgt = idx[:, 1:]
    ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), tgt.flatten(), ignore_index=0)
    return ce, aux


@torch.no_grad()
def amix_importance(base, D, N, dev, n_win=64):
    """mean |a| x down column norm of every A-mix FFN neuron of the MoE blocks on train windows"""
    g = torch.Generator(device=dev).manual_seed(77)
    offs = torch.randint(0, D.train.numel() - CTX - 1, (n_win,), device=dev, generator=g)
    imp = torch.zeros(base.cfg.n_layers - N, base.cfg.ffn_hidden, dtype=torch.float64, device=dev)
    cn = [base.blocks[b].w2.weight.float().norm(dim=0) for b in range(N, base.cfg.n_layers)]
    for w0 in range(0, n_win, 16):
        idx = windows(D.train, offs[w0:w0 + 16], CTX)
        B, T = idx.shape
        rope = base.rope(T, dev)
        mask = doc_causal_mask(idx)
        x = base.emb(idx)
        for b, blk in enumerate(base.blocks):
            y = x + blk.attn(blk.n1(x), rope, mask=mask)
            z = blk.n2(y)
            a = F.silu(blk.w1(z)) * blk.w3(z)
            if b >= N:
                imp[b - N] += (a.abs().float() * cn[b - N]).reshape(-1, a.shape[-1]).sum(0).double()
            x = y + blk.w2(a)
    return imp.float().cpu()


def build(variant, init_from, books_path, dev, D=None, seed=0):
    books = torch.load(books_path, map_location="cpu", weights_only=False)
    if variant in ROUTE4:
        books["route"] = torch.load(ROUTE_TABLES, map_location="cpu", weights_only=False)
    base = load_base(init_from, dev)
    imp = amix_importance(base, D, books["N"], dev) if variant == "add" else None
    return R28(base, variant, books, imp=imp, seed=seed).to(dev), books


# ------------------------------------------------------------------------------------------------ training
def train_cmd(a):
    from .r28_measure import foreign_share, measure_all
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    D = Data(a.data, dev, train=True)
    model, books = build(a.variant, a.init_from, a.books, dev, D, a.seed)
    model.init_vecs = [[torch.cat([p[k].detach().reshape(-1).float() for p in model.expert_params(j)]).cpu()
                        for k in range(E)] for j in range(model.M)]
    gv = torch.Generator(device=dev).manual_seed(1234)                       # as r21: the same val windows
    voffs = torch.randint(0, D.val.numel() - CTX - 1, (a.val_windows,), device=dev, generator=gv)
    from .r21_moe import MoE
    dense0 = MoE(load_base(a.init_from, dev), "dense", books["N"], books=books).to(dev)
    ce_amix, _ = eval_ce(dense0, D, voffs)
    del dense0
    model.eval()
    ce0, src0 = eval_ce(model, D, voffs)
    print(f"{a.name}: variant {a.variant} ({RU[a.variant]}); step-0 val CE {ce0:.5f}, A-mix {ce_amix:.5f}"
          + (f"; broad topics {model.broad}" if hasattr(model, "broad") else ""), flush=True)
    if a.variant != "add":
        assert abs(ce0 - ce_amix) < 2e-3, "the variant does not compute the A-mix function at step 0"
    res_path = os.path.join(out, "r28.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res.update(variant=a.variant, name=a.name, seed=a.seed, ce0=ce0, ce_amix=ce_amix, ce0_src=src0,
               broad=getattr(model, "broad", None), Wt=model.Wt.tolist())
    json.dump(res, open(res_path, "w"), indent=1)
    if "grad_start" not in res:
        res["grad_start"] = foreign_share(model, D, n_win=a.grad_windows)
        json.dump(res, open(res_path, "w"), indent=1)
        print("  [grad] start: " + json.dumps({k: [round(x, 3) for x in v] for k, v in res["grad_start"].items()}),
              flush=True)
    start = {id(p) for p in model.base.blocks[:model.N].parameters()}
    shared = {id(p) for p in list(model.sw1) + list(model.sw3) + list(model.sw2)}
    groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            m = a.start_mult if (id(p) in start or id(p) in shared) else 1.0
            groups.setdefault((m, a.wd if p.dim() >= 2 else 0.0), []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=a.lr * m, weight_decay=w, mult=m) for (m, w), v in groups.items()],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    n_win = (D.train.numel() - 1) // (CTX + 1)
    need = (a.skip_steps + a.steps) * a.bs
    assert n_win >= need
    perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(a.data_seed))[a.skip_steps * a.bs:need]
    if a.seed:
        perm = perm[torch.randperm(perm.numel(), generator=torch.Generator().manual_seed(a.seed))]
    perm = perm.to(dev) * (CTX + 1)
    step_fn = torch.compile(lambda idx: loss_fn(model, idx)) if a.compile else (lambda idx: loss_fn(model, idx))
    log = res.setdefault("val_log", [dict(step=0, val_ce=ce0, val_ce_src=src0)])
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        idx = windows(D.train, perm[step * a.bs:(step + 1) * a.bs], CTX)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step) * gr["mult"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, aux = step_fn(idx)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"{step:6d} ce {float(ce):.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} {time.time() - t0:.0f}s mem "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            model.eval()
            vce, vsrc = eval_ce(model, D, voffs)
            model.train()
            log.append(dict(step=step + 1, val_ce=vce, val_ce_src=vsrc, t=time.time() - t0))
            json.dump(res, open(res_path, "w"), indent=1)
            print(f"  [val] step {step + 1} ce {vce:.4f} " + " ".join(f"{s} {v:.4f}" for s, v in zip(SRC, vsrc)), flush=True)
        if step + 1 in SAVES:
            model.eval()
            torch.save(dict(variant=a.variant, state={k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                                                      for k, v in model.state_dict().items()}),
                       os.path.join(out, f"model_{step + 1}_bf16.pt"))
            ts = time.time()
            res.setdefault("measure", {})[str(step + 1)] = measure_all(model, D, a)
            res["s_per_step"] = (time.time() - t0) / (step + 1)
            json.dump(res, open(res_path, "w"), indent=1)
            print(f"  [measure] step {step + 1} ({time.time() - ts:.0f}s): "
                  + json.dumps({k: v for k, v in res["measure"][str(step + 1)].get("summary", {}).items()}), flush=True)
            model.train()
    model.eval()
    res["grad_end"] = foreign_share(model, D, n_win=a.grad_windows)
    full = eval_ce(model, D, D.starts(), per_window=False)
    res["val_full"] = dict(ce=full[0], src=full[1])
    res["train_time"] = time.time() - t0
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"{a.name}: done; full val CE {full[0]:.4f} " + " ".join(f"{s} {v:.4f}" for s, v in zip(SRC, full[1])), flush=True)


def load_trained(a, dev, D, step=10000):
    """the model of a run at a save (bf16 weights), with init_vecs of its step 0"""
    model, _ = build(a.variant, a.init_from, a.books, dev, D, a.seed)
    model.init_vecs = [[torch.cat([p[k].detach().reshape(-1).float() for p in model.expert_params(j)]).cpu()
                        for k in range(E)] for j in range(model.M)]
    st = torch.load(os.path.join(a.out, a.name, f"model_{step}_bf16.pt"), map_location="cpu", weights_only=False)["state"]
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in st.items()})
    return model.eval()


def stage2_cmd(a):
    """stage 2 of a run: full val CE per window (paired SE against the baseline and dense-cont), the full swap and
    refit matrices (all pairs, every block and all blocks at once, 1024 val / 1024 calibration windows, as stage 0)"""
    from .r28_measure import measure_all
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    p = os.path.join(a.out, a.name, "r28.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    if "stage2" in res:
        print(f"{a.name}: stage 2 done before, skipped", flush=True)
        return
    D = Data(a.data, dev, train=a.variant == "add")
    model = load_trained(a, dev, D, step=a.steps)
    t0 = time.time()
    tot, src, S, Cn = eval_ce(model, D, D.starts() if a.steps >= 10000 else D.starts(16), per_window=True)
    np.savez(os.path.join(a.out, a.name, "val_full_windows.npz"), S=S, C=Cn)
    if a.steps >= 10000:                                         # smoke: keeps its small windows
        a.m_windows, a.fit_windows = 1024, 1024
    a.n_pairs, a.all_blocks = 7, True
    meas = measure_all(model, D, a)
    res = json.load(open(p)) if os.path.exists(p) else {}
    res["stage2"] = dict(val_full=dict(ce=tot, src=src), measure=meas, seconds=time.time() - t0)
    json.dump(res, open(p, "w"), indent=1)
    print(f"{a.name}: stage 2 done ({time.time() - t0:.0f}s); full val CE {tot:.4f}", flush=True)


def ref_cmd(a):
    """per-window full val CE of the r21 references (dense-cont, perlayer s0, A-mix) for paired differences"""
    from .r21_moe import load_moe
    dev = torch.device("cuda")
    D = Data(a.data, dev)
    out = os.path.join(a.out, "refs")
    os.makedirs(out, exist_ok=True)
    for nm, run in (("dense", "runs_vast/mix_dense"), ("perlayer_r21", "runs_vast/mix_perlayer")):
        f = os.path.join(out, f"{nm}_val_full_windows.npz")
        if os.path.exists(f):
            continue
        m = load_moe(run, dev=dev)
        tot, src, S, Cn = eval_ce(m, D, D.starts(), per_window=True)
        np.savez(f, S=S, C=Cn)
        print(f"ref {nm}: full val CE {tot:.4f}", flush=True)
        del m
        torch.cuda.empty_cache()


def seed2_cmd(a):
    """the best variant (lowest full val CE of stage 2, the baseline excluded) trained again with seed 1 + stage 2"""
    best = None
    for v in VARIANTS:
        if v == "base":
            continue
        p = os.path.join(a.out, f"r28_{v}", "r28.json")
        if os.path.exists(p):
            r = json.load(open(p))
            ce = r.get("stage2", {}).get("val_full", {}).get("ce")
            if ce is not None and (best is None or ce < best[0]):
                best = (ce, v)
    if best is None:
        print("seed2: no finished variant", flush=True)
        return
    a.variant, a.seed = best[1], 1
    a.name = f"r28_{a.variant}_s1"
    print(f"seed2: best variant {a.variant} (full val CE {best[0]:.4f}) -> {a.name}", flush=True)
    p = os.path.join(a.out, a.name, "r28.json")
    if not (os.path.exists(p) and "val_full" in json.load(open(p))):
        train_cmd(a)
    stage2_cmd(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "stage2", "ref", "seed2"])
    ap.add_argument("--variant", default="base", choices=VARIANTS + ROUTE4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs_vast")
    ap.add_argument("--data", default="data_mix")
    ap.add_argument("--init_from", default="runs_vast/Amix_s0/model.pt")
    ap.add_argument("--books", default="runs_vast/r21_books/books.pt")
    ap.add_argument("--dense", default="runs_vast/mix_dense", help="dense-cont run (its FFN is a reference)")
    ap.add_argument("--calib", default="data_mix/calib_train.bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--skip_steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--start_mult", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--val_windows", type=int, default=256)
    ap.add_argument("--m_windows", type=int, default=256, help="val windows of the specialisation measure")
    ap.add_argument("--fit_windows", type=int, default=512, help="calibration windows of the refit")
    ap.add_argument("--grad_windows", type=int, default=64)
    ap.add_argument("--compile", type=int, default=0)
    a = ap.parse_args(argv)
    a.name = a.name or f"r28_{a.variant}" + (f"_s{a.seed}" if a.seed else "")
    global SAVES
    if a.steps < 10000:                                           # smoke: saves scaled to the run
        SAVES = tuple(sorted({max(1, a.steps // 5), max(1, a.steps // 2), a.steps}))
    dict(train=train_cmd, stage2=stage2_cmd, ref=ref_cmd, seed2=seed2_cmd)[a.cmd](a)


if __name__ == "__main__":
    main()
