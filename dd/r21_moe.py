"""Round 21 (the user's block «r20»; the name r20 is taken by the depth-needed round): experts without a trained
gate -- one decision after a dense start, by the codes of fixed books and fixed tables.

Data: dd.mix_data (Wikipedia / Python code / TinyStories, GPT-2 BPE, equal token shares, source of every token).
Model: A (d 512, 8 blocks, 8 heads, SwiGLU 1408, tied head, document mask).  A mixture variant replaces the FFN of
blocks N+1..8 by E = 8 experts of the same size, top-1 (compute per token as dense, more parameters); every expert
starts as a copy of the block's A-mix FFN, so at step 0 every variant computes exactly the A-mix function.

Stage 1: A-mix = dd.ts_train --variant A --data data_mix --sampler perm (20k steps).
books:  probe -- states after blocks 1..8 on 2000 val windows, RMS-normalised, k-means on 64 codes; NMI of the code with
        the source and with the token type; N = argmax over blocks 2..4 of NMI(source); M = 8 - N.
        Books from the A-mix states on ~5M train tokens, never updated afterwards:
          RVQ: M stages of 64 codes on u = RMS-normalised state after block N:  r = u; c_j = nearest(r, book_j);
               r -= book_j[c_j]; the code of stage j picks the expert of block N+j through a table;
          semantic tables: k-means of the 64 centres of book_j into 8 superclusters, expert = supercluster;
          random tables: random assignment of the 64 codes, balanced by their load on the 5M tokens (best of 5000);
          per-layer books: one 64-code book of the RMS-normalised state before every MoE block, semantic tables;
          hash tables: token id -> expert, balanced by the train frequencies (greedy, largest first, random ties).
Stage 2 (10k steps each, the same lr plan, the same train windows: those after the 20k x 32 of stage 1):
  dense            A-mix trained further (control for the extra training)
  learned          trained router top-1 per MoE block (Switch: gate p / stopgrad(p), so the forward value is 1 and
                   the upcycled start is kept; balance loss alpha * E * sum_i f_i P_i per block)
  hash             expert = balanced hash of the current token id, fixed
  rvq_sem          RVQ codes, semantic tables              } + closeness penalty beta * |u - sum_j book_j[c_j]|^2
  rvq_rand         RVQ codes, random balanced tables       } (u RMS-normalised, the books fixed: the gradient goes
  rvq_shared       rvq_sem + a shared expert; shared and routed experts of half size (704), so the compute per token
                   stays the same; init = the two halves of the A-mix FFN (shared + routed = A-mix FFN exactly)
  perlayer         a new decision in every MoE block: per-layer book of the state before the block, semantic table
  rvq_sem_frozen   rvq_sem with blocks 1..N frozen
  Blocks 1..N train with lr x 0.1 (the embedding is tied to the output head and trains with the full lr everywhere).

Commands (python -m dd.r21_moe <cmd> ...):
  books   --run runs_vast/Amix_s0 --data data_mix --name r21_books --out runs_vast
          -> probe.json, books.pt, books.json, amix_analysis.json (+ routes / window losses of A-mix)
  train   --variant V --init_from .../model.pt --books .../books.pt [--beta 1e-3|auto --gate gate.json] --name ...
          -> log.json (val CE per source, loads, drift of the codes every 1k steps), model.pt, model_bf16.pt
  gate    --probes n1,n2,... --name r21_gate -> gate.json: checks of the 300-step probes, beta (rc 1 = stop)
  analyze --run runs_vast/<name>  -> analysis.json, routes.npz, winloss.npz
  report  -> report/r21_tables.md, report/r21_*.png"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .model import Cfg, Model, doc_causal_mask, normalize  # noqa: E402
from .ts_train import load_stream, windows  # noqa: E402

E, K, CTX, BS = 8, 64, 512, 32
CH = 1 << 18                         # rows per chunk for distances
SRC = ("wiki", "code", "ts")
TYPES = ("пробелы", "пунктуация", "числа", "служебные", "имена", "начало слова", "продолжение слова")
TABLE = {"rvq_rand": "rvq_rand", "rvq_semw": "rvq_semw"}       # RVQ variants: which table (default rvq_sem)
ROUTE = {"dense": None, "learned": "learned", "hash": "hash", "rvq_sem": "rvq", "rvq_semw": "rvq", "rvq_rand": "rvq",
         "rvq_shared": "rvq",
         "perlayer": "perlayer", "rvq_sem_frozen": "rvq"}
FUNC = set("the of and in to a is was for on as by with that from at it an be this which or are his her were has have "
           "had also its their he she they not but who after first one two been into i you we me my your our them him "
           "if then so when what there all no can will would could do did does said up out about than only very just "
           "some other more most any each".split())


# ------------------------------------------------------------------------------------------------ small tools
def nmi(a, b):
    """normalised mutual information, arithmetic mean of the entropies (the sklearn default)"""
    a = np.unique(np.asarray(a).ravel(), return_inverse=True)[1]
    b = np.unique(np.asarray(b).ravel(), return_inverse=True)[1]
    na, nb = a.max() + 1, b.max() + 1
    p = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)
    p /= p.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum())
    ha, hb = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum()), float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    if ha + hb == 0:
        return 1.0
    return max(0.0, 2 * mi / (ha + hb))


def entropy_bits(f):
    f = np.asarray(f, np.float64)
    f = f[f > 0] / f.sum()
    return float(-(f * np.log2(f)).sum())


def nearest(x, C):
    """index of the nearest row of C (squared Euclidean), fp32 without autocast; no gradient"""
    with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
        x, C = x.float(), C.float()
        return (C.pow(2).sum(1)[None] - 2 * (x @ C.t())).argmin(1)


def sq_sum(X, chunk=1 << 18):
    """sum of squares of a large (n, d) tensor, fp32 per chunk (no full-size fp32 copy)"""
    return sum(float(X[s:s + chunk].float().pow(2).sum()) for s in range(0, X.shape[0], chunk))


def nearest_chunked(X, C, chunk=1 << 18):
    return torch.cat([nearest(X[s:s + chunk], C) for s in range(0, X.shape[0], chunk)])


@torch.no_grad()
def kmeans(X, k, iters=30, seed=0, chunk=1 << 18, init_sample=1 << 16):
    """Lloyd's k-means (squared Euclidean), k-means++ init on a sample; X (n, d) any float dtype, sums in fp32.
    Returns centres (k, d) fp32, counts (k,), inertia (mean squared distance)."""
    dev = X.device
    g = torch.Generator(device=dev).manual_seed(seed)
    n = X.shape[0]
    S = X[torch.randperm(n, generator=g, device=dev)[:init_sample]].float()
    C = S[torch.randint(0, S.shape[0], (1,), generator=g, device=dev)]
    dmin = (S - C).pow(2).sum(1)
    trials = 2 + int(math.log(k))                         # greedy k-means++ (as sklearn): best of a few candidates
    for _ in range(1, k):
        cand = torch.multinomial(dmin / dmin.sum().clamp_min(1e-30), trials, replacement=True, generator=g)
        dc = torch.minimum(dmin[None], torch.cdist(S[cand], S).pow(2))
        b = int(dc.sum(1).argmin())
        C = torch.cat([C, S[cand[b:b + 1]]])
        dmin = dc[b]
    for it in range(iters + 1):
        sums = torch.zeros(k, X.shape[1], device=dev)
        cnt = torch.zeros(k, device=dev)
        inert = 0.0
        for s in range(0, n, chunk):
            x = X[s:s + chunk].float()
            with torch.autocast(dev.type, enabled=False):
                d2 = x.pow(2).sum(1, keepdim=True) - 2 * (x @ C.t()) + C.pow(2).sum(1)[None]
            m, a = d2.min(1)
            inert += float(m.clamp_min(0).sum())
            sums.index_add_(0, a, x)
            cnt += torch.bincount(a, minlength=k).float()
        if it == iters:
            break
        newC = sums / cnt.clamp_min(1)[:, None]
        empty = cnt == 0
        if empty.any():                                   # re-seed empty clusters with random points
            newC[empty] = S[torch.randint(0, S.shape[0], (int(empty.sum()),), generator=g, device=dev)]
        C = newC
    return C, cnt, inert / n


def superclusters(C, n=E, restarts=50):
    """k-means of the k centres of a book into n superclusters (unweighted), best of `restarts`"""
    best = None
    for r in range(restarts):
        S, cnt, inert = kmeans(C, n, iters=50, seed=1000 + r, init_sample=C.shape[0])
        if (cnt > 0).all() and (best is None or inert < best[1]):
            best = (nearest(C, S), inert)
    assert best is not None, "no restart gave 8 non-empty superclusters"
    return best[0]


def weighted_superclusters(C, w, n=E, restarts=50, iters=100, seed=0):
    """k-means of the k centres of a book into n superclusters, every centre weighted by its token share (round 21
    addition: unweighted superclusters of the residual books put ~99% of the tokens into one expert); weighted
    k-means++ init, best of `restarts` by weighted inertia among solutions with n non-empty superclusters"""
    X = np.asarray(C, np.float64)
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        idx = [int(rng.choice(len(X), p=w))]
        for _ in range(1, n):
            d2 = ((X[:, None] - X[idx][None]) ** 2).sum(-1).min(1)
            q = w * d2
            idx.append(int(rng.choice(len(X), p=q / q.sum())) if q.sum() > 0 else int(rng.integers(len(X))))
        S = X[idx].copy()
        for _ in range(iters):
            lab = ((X[:, None] - S[None]) ** 2).sum(-1).argmin(1)
            new = np.stack([(w[lab == c, None] * X[lab == c]).sum(0) / w[lab == c].sum() if (lab == c).any() else S[c]
                            for c in range(n)])
            if np.allclose(new, S):
                break
            S = new
        lab = ((X[:, None] - S[None]) ** 2).sum(-1).argmin(1)
        inert = float((w * ((X - S[lab]) ** 2).sum(-1)).sum())
        if len(np.unique(lab)) == n and (best is None or inert < best[1]):
            best = (lab, inert)
    assert best is not None, "no restart gave n non-empty weighted superclusters"
    return torch.from_numpy(best[0].astype(np.int64))


def random_balanced_table(freq, n=E, tries=5000, seed=0):
    """random table code -> expert with an even load: random order, each code to the least loaded expert so far;
    the best of `tries` by MaxVio"""
    rng = np.random.default_rng(seed)
    f = np.asarray(freq, np.float64) / max(1e-30, float(np.sum(freq)))
    best = None
    for _ in range(tries):
        load, tab = np.zeros(n), np.zeros(len(f), np.int64)
        for c in rng.permutation(len(f)):
            e = int(np.argmin(load))
            tab[c] = e
            load[e] += f[c]
        mv = load.max() * n - 1
        if best is None or mv < best[0]:
            best = (mv, tab)
    return best[1]


def hash_table(freq, n=E, seed=0):
    """balanced hash of the token id: tokens by frequency (largest first, random ties), each to the least loaded
    expert (random among equal loads); tokens never seen in train -> a random expert"""
    rng = np.random.default_rng(seed)
    freq = np.asarray(freq, np.float64)
    order = np.lexsort((rng.random(freq.size), -freq))
    load, tab = np.zeros(n), rng.integers(0, n, freq.size)
    for t in order:
        if freq[t] <= 0:
            break
        m = load.min()
        cand = np.flatnonzero(load == m)
        e = int(cand[rng.integers(cand.size)]) if cand.size > 1 else int(cand[0])
        tab[t] = e
        load[e] += freq[t]
    return tab


def load_stats(eid, n=E):
    f = np.bincount(np.asarray(eid).ravel(), minlength=n) / max(1, np.asarray(eid).size)
    return dict(load=f.tolist(), maxvio=float(f.max() * n - 1), entropy_bits=entropy_bits(f))


# ------------------------------------------------------------------------------------------------ token types
def vocab_props(tok_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    n = tok.get_vocab_size()
    s = [tok.decode([i]) if i >= 2 else "" for i in range(n)]
    st = [x.strip() for x in s]
    letters = np.array([any(c.isalpha() for c in x) for x in st])
    digits = np.array([any(c.isdigit() for c in x) for x in st])
    ws = np.array([x == "" for x in st])
    disp = list(s)
    disp[:2] = ["<pad>", "<eot>"]
    return dict(strs=s, disp=disp, ws=ws, word=letters, num=digits & ~letters, punct=~ws & ~letters & ~digits,
                func=np.array([x.isalpha() and x.lower() in FUNC for x in st]),
                cap=np.array([bool(x) and x[0].isupper() for x in st]),
                lead=np.array([x[:1].isspace() for x in s]),
                tail_word=np.array([bool(x) and (x[-1].isalnum() or x[-1] == "_") for x in s]))


def token_types(P, prev, cur):
    """type of the current token (TYPES): whitespace (and document start), punctuation, numbers, function words,
    names (a capitalised word start), word start, word continuation (no leading space after a letter/digit)"""
    prev, cur = np.asarray(prev, np.int64), np.asarray(cur, np.int64)
    word = P["word"][cur]
    cont = word & ~P["lead"][cur] & P["tail_word"][prev] & (prev > 1)
    start = word & ~cont
    t = np.full(cur.shape, 6, np.int8)
    t[start] = 5
    t[start & P["cap"][cur]] = 4
    t[start & P["func"][cur]] = 3
    t[P["num"][cur]] = 2
    t[P["punct"][cur]] = 1
    t[P["ws"][cur]] = 0
    return t


# ------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self, data="data_mix", dev="cuda", train=False):
        self.dir, self.dev = data, torch.device(dev)
        self.meta = json.load(open(os.path.join(data, "meta.json")))
        self.val = load_stream(os.path.join(data, "val.bin"), self.dev)
        src = np.fromfile(os.path.join(data, "val_src.bin"), dtype=np.uint8).astype(np.int64)
        assert src.size == self.val.numel()
        self.val_src = torch.from_numpy(src).to(self.dev)
        self.train = load_stream(os.path.join(data, "train.bin"), self.dev) if train else None

    def starts(self, n=None):
        """non-overlapping val windows of CTX + 1 tokens; n -> n of them, evenly spread over the whole val stream
        (the stream is [wiki][code][ts], so every source is covered in proportion)"""
        s = torch.arange(0, self.val.numel() - CTX - 1, CTX, device=self.dev)
        if n is not None and n < s.numel():
            s = s[torch.linspace(0, s.numel() - 1, n, device=self.dev).round().long()]
        return s

    def win(self, offs):
        return windows(self.val, offs, CTX)

    def src(self, offs):
        return self.val_src[offs[:, None] + torch.arange(CTX + 1, device=self.dev)]


@torch.no_grad()
def base_states(base, idx, layers):
    """A model: residual states after `layers` blocks (0 = embedding), as a dict"""
    B, T = idx.shape
    rope = base.rope(T, idx.device)
    mask = doc_causal_mask(idx) if base.cfg.doc_mask else None
    x = base.emb(idx)
    out = {0: x} if 0 in layers else {}
    depth = torch.zeros(B, T, 1, device=idx.device)
    for b, blk in enumerate(base.blocks[:max(layers)]):
        x = blk(x, depth, rope, None, mask)[0]
        if b + 1 in layers:
            out[b + 1] = x
    return out


# ------------------------------------------------------------------------------------------------ the model
@torch.compiler.disable
def dispatch(z, eid, w1, w3, w2, off=None):
    """top-1 experts: tokens sorted by expert, one SwiGLU per expert; off = expert whose output is 0 (ablation)"""
    order = torch.argsort(eid)
    counts = torch.bincount(eid, minlength=w1.shape[0]).tolist()
    outs = []
    for e, c in enumerate(z[order].split(counts)):
        y = (F.silu(c @ w1[e].t()) * (c @ w3[e].t())) @ w2[e].t()
        outs.append(torch.zeros_like(y) if off == e else y)
    y = torch.cat(outs)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return y[inv]


class Experts(nn.Module):
    def __init__(self, n, d, h):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(n, h, d))
        self.w3 = nn.Parameter(torch.zeros(n, h, d))
        self.w2 = nn.Parameter(torch.zeros(n, d, h))

    def forward(self, z, eid, off=None):
        sh = z.shape
        return dispatch(z.reshape(-1, sh[-1]), eid.reshape(-1), self.w1, self.w3, self.w2, off).view(*sh[:-1], -1)


class MoE(nn.Module):
    """A model (dd.model.Model, variant A) whose blocks N+1..L get E top-1 experts; routing by ROUTE[variant].
    books (dict from the `books` command) fill the fixed books / tables and start the experts from the A FFN."""

    def __init__(self, base, variant, N, books=None):
        super().__init__()
        cfg = base.cfg
        self.base, self.variant, self.N, self.M = base, variant, N, cfg.n_layers - N
        self.route = ROUTE[variant]
        d, h, M = cfg.d, cfg.ffn_hidden, self.M
        self.register_buffer("rvq_books", torch.zeros(M, K, d))
        self.register_buffer("rvq_table", torch.zeros(M, K, dtype=torch.long))
        self.register_buffer("pl_books", torch.zeros(M, K, d))
        self.register_buffer("pl_table", torch.zeros(M, K, dtype=torch.long))
        self.register_buffer("hash_table", torch.zeros(M, cfg.vocab, dtype=torch.long))
        if books is not None:
            assert books["N"] == N and books["rvq"].shape == self.rvq_books.shape
            self.rvq_books.copy_(books["rvq"])
            self.rvq_table.copy_(books[TABLE.get(variant, "rvq_sem")])
            self.pl_books.copy_(books["pl"])
            self.pl_table.copy_(books["pl_sem"])
            self.hash_table.copy_(books["hash"])
        self.shared = variant == "rvq_shared"
        self.experts = nn.ModuleList()
        if self.route:
            hh = h // 2 if self.shared else h                # routed experts: the last hh hidden units of the FFN
            for j in range(M):
                blk = base.blocks[N + j]
                ex = Experts(E, d, hh)
                with torch.no_grad():
                    ex.w1.copy_(blk.w1.weight[h - hh:].expand(E, -1, -1))
                    ex.w3.copy_(blk.w3.weight[h - hh:].expand(E, -1, -1))
                    ex.w2.copy_(blk.w2.weight[:, h - hh:].expand(E, -1, -1))
                self.experts.append(ex)
                if self.shared:                          # shared expert: the first h - hh hidden units
                    w1, w3 = nn.Linear(d, h - hh, bias=False), nn.Linear(d, h - hh, bias=False)
                    w2 = nn.Linear(h - hh, d, bias=False)
                    with torch.no_grad():
                        w1.weight.copy_(blk.w1.weight[:h - hh])
                        w3.weight.copy_(blk.w3.weight[:h - hh])
                        w2.weight.copy_(blk.w2.weight[:, :h - hh])
                    blk.w1, blk.w3, blk.w2 = w1, w3, w2
                else:
                    del blk.w1, blk.w3, blk.w2
        if self.route == "learned":
            self.router = nn.ModuleList([nn.Linear(d, E, bias=False) for _ in range(M)])
            for r in self.router:
                nn.init.normal_(r.weight, 0.0, 0.02)

    def forward(self, idx, ctl=None):
        """ctl: codes=True -> aux['codes'] (RVQ codes of the state after block N) for every variant;
        off=(j, e) -> expert e of MoE block j outputs 0;
        states=True -> aux['s_in']: RMS-normalised input of every MoE block (the first one = the state after block N);
        route_ahead=f -> f(RMS-normalised state after block N, (B, T, d)) returns M expert tensors (B, T) that replace
        the variant's own routing in blocks N+1..L (round 23: all experts chosen right after block N);
        off_where={j: (mask (B, T) bool, experts)} -> the output of MoE block j is 0 where mask and its expert is one of
        `experts` (dd.r21_ws: the «spaces» experts switched off only at space positions);
        x_in=True -> aux['x_in']: the residual before every block; ffn_io=True -> aux['z'], aux['uf'], aux['y'] per MoE
        block: the expert input RMSNorm(y), the expert output (before off_where), the stream before the FFN"""
        ctl = ctl or {}
        base, N, M = self.base, self.N, self.M
        B, T = idx.shape
        d = base.cfg.d
        rope = base.rope(T, idx.device)
        mask = doc_causal_mask(idx) if base.cfg.doc_mask else None
        x = base.emb(idx)
        aux = {"eid": [], "pen": x.new_zeros((), dtype=torch.float32), "bal": x.new_zeros((), dtype=torch.float32)}
        eids = None
        off = ctl.get("off")
        ahead = None
        for b, blk in enumerate(base.blocks):
            if ctl.get("x_in"):
                aux.setdefault("x_in", []).append(x)
            if b == N and ctl.get("route_ahead") is not None:
                ahead = ctl["route_ahead"](normalize(x.float()))
            if b >= N and ctl.get("states"):
                aux.setdefault("s_in", []).append(normalize(x.float()))
            if b == N and (self.route == "rvq" or ctl.get("codes")):
                u = normalize(x.float())
                codes, rec = rvq_encode(u.detach().reshape(-1, d), self.rvq_books)
                aux["codes"] = codes.view(B, T, M)
                if ctl.get("u"):
                    aux["u"] = u.detach()
                if self.route == "rvq":
                    aux["pen"] = (u - rec.view_as(u)).pow(2).sum(-1).mean()
                    eids = torch.gather(self.rvq_table, 1, codes.t()).t().reshape(B, T, M)
            a = blk.attn(blk.n1(x), rope, mask=mask)
            y = x + a
            z = blk.n2(y)
            if b < N or not self.route:
                x = y + blk.w2(F.silu(blk.w1(z)) * blk.w3(z))
                continue
            j = b - N
            gate = None
            if ahead is not None:
                e = ahead[j]
                if self.route == "perlayer":                   # own codes still logged (true state)
                    aux.setdefault("pl_codes", []).append(
                        nearest(normalize(x.float()).reshape(-1, d), self.pl_books[j]).view(B, T))
            elif self.route == "rvq":
                e = eids[..., j]
            elif self.route == "hash":
                e = self.hash_table[j][idx]
            elif self.route == "perlayer":
                c = nearest(normalize(x.float()).reshape(-1, d), self.pl_books[j]).view(B, T)
                aux.setdefault("pl_codes", []).append(c)
                e = self.pl_table[j][c]
            else:                                            # learned router (Switch), forward gate value 1
                p = F.softmax(self.router[j](z).float(), -1)
                e = p.argmax(-1)
                pe = p.gather(-1, e[..., None])
                gate = pe / pe.detach()
                f = F.one_hot(e, E).float().mean((0, 1))
                aux["bal"] = aux["bal"] + E * (f * p.mean((0, 1))).sum()
            aux["eid"].append(e)
            uf = self.experts[j](z, e, off[1] if off is not None and off[0] == j else None)
            if gate is not None:
                uf = uf * gate
            if self.shared:
                uf = uf + blk.w2(F.silu(blk.w1(z)) * blk.w3(z))
            if ctl.get("ffn_io"):
                for k, v in (("z", z), ("uf", uf), ("y", y)):
                    aux.setdefault(k, []).append(v)
            ow = ctl.get("off_where")
            if ow is not None and j in ow:
                tm, exs = ow[j]
                kill = tm & torch.isin(e, torch.as_tensor(exs, device=e.device, dtype=e.dtype))
                uf = uf.masked_fill(kill[..., None], 0)
            x = y + uf
        logits = base.nf(x) @ base.emb.weight.t().to(x.dtype)
        return logits, aux


def rvq_encode(u, books):
    """residual VQ of u (n, d) with fixed books (M, K, d): codes (n, M), reconstruction sum_j book_j[c_j]"""
    with torch.no_grad():
        r = u.float().clone()
        codes = []
        for j in range(books.shape[0]):
            c = nearest(r, books[j])
            codes.append(c)
            r = r - books[j][c]
        return torch.stack(codes, 1), u.float() - r


def moe_loss(model, idx, beta=0.0, alpha=0.0):
    logits, aux = model(idx)
    tgt = idx[:, 1:]
    ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), tgt.flatten(), ignore_index=0)
    return ce + beta * aux["pen"] + alpha * aux["bal"], ce, aux


def load_base(path, dev="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    base = Model(Cfg(**ck["cfg"]))
    base.load_state_dict(ck["state"])
    return base.to(dev)


def load_moe(run, books=None, dev="cuda", fname=None):
    """a stage-2 run (model.pt or model_bf16.pt with variant / N), or A-mix (dd.ts_train) as a dense MoE with books"""
    for f in ([fname] if fname else ["model.pt", "model_bf16.pt"]):
        if os.path.exists(os.path.join(run, f)):
            break
    ck = torch.load(os.path.join(run, f), map_location="cpu", weights_only=False)
    base = Model(Cfg(**ck["cfg"]))
    if "variant" in ck:
        m = MoE(base, ck["variant"], ck["N"])
        m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state"].items()})
    else:
        base.load_state_dict(ck["state"])
        if isinstance(books, str):
            books = torch.load(books, map_location="cpu", weights_only=False)
        m = MoE(base, "dense", books["N"], books=books)
    return m.to(dev).eval()


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_ce(model, D, offs, bs=BS, ctl=None, per_window=False):
    """val CE (mean over predicted tokens) and per source of the predicted token; per_window -> sums, counts (nw, 3)"""
    was = model.training
    model.eval()
    S, C = [], []
    for i in range(0, offs.numel(), bs):
        o = offs[i:i + bs]
        idx = D.win(o)
        with torch.autocast(D.dev.type, dtype=torch.bfloat16):
            logits, _ = model(idx, ctl)
        tgt = idx[:, 1:]
        lo = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="none",
                             ignore_index=0).view(tgt.shape).double()
        m = (tgt != 0).double()
        oh = F.one_hot(D.src(o)[:, 1:], len(SRC)).double() * m[..., None]
        S.append((oh * lo[..., None]).sum(1))
        C.append(oh.sum(1))
    model.train(was)
    S, C = torch.cat(S).cpu().numpy(), torch.cat(C).cpu().numpy()
    tot = float(S.sum() / C.sum())
    src = (S.sum(0) / np.maximum(C.sum(0), 1)).tolist()
    return (tot, src, S, C) if per_window else (tot, src)


@torch.no_grad()
def snapshot(model, D, offs, bs=BS):
    """routing of the fixed drift windows: RVQ codes of the state after block N (every variant), the variant's own
    experts, per-layer codes (perlayer)"""
    was = model.training
    model.eval()
    out = {"codes": [], "eid": [], "plc": [], "rn": [], "mg": []}
    for i in range(0, offs.numel(), bs):
        idx = D.win(offs[i:i + bs])
        with torch.autocast(D.dev.type, dtype=torch.bfloat16):
            _, aux = model(idx, {"codes": True, "u": True})
        out["codes"].append(aux["codes"][:, :-1].reshape(-1, model.M))
        rn, mg = rvq_residuals(aux["u"][:, :-1].reshape(-1, aux["u"].shape[-1]), out["codes"][-1], model.rvq_books)
        out["rn"].append(rn)
        out["mg"].append(mg)
        if aux["eid"]:
            out["eid"].append(torch.stack(aux["eid"], -1)[:, :-1].reshape(-1, model.M))
        if aux.get("pl_codes"):
            out["plc"].append(torch.stack(aux["pl_codes"], -1)[:, :-1].reshape(-1, model.M))
    model.train(was)
    return {k: torch.cat(v).cpu() for k, v in out.items() if v}


@torch.no_grad()
def rvq_residuals(u, codes, books):
    """per token: norm of the residual at the input of every RVQ stage (stage 1 input = u) and after the last stage
    -> (n, M + 1); margin of the code choice per stage = distance to the 2nd nearest book point minus distance to the
    chosen one -> (n, M); fp32 without autocast, the given codes define the residual chain"""
    with torch.autocast(u.device.type, enabled=False):
        r = u.float().clone()
        books = books.float()
        rn, mg = [], []
        for j in range(books.shape[0]):
            C = books[j]
            rn.append(r.norm(dim=1))
            dist = (r.pow(2).sum(1, keepdim=True) - 2 * r @ C.t() + C.pow(2).sum(1)[None]).clamp_min(0).sqrt()
            two = dist.topk(2, dim=1, largest=False).values
            mg.append(two[:, 1] - dist.gather(1, codes[:, j:j + 1].long())[:, 0])
            r = r - C[codes[:, j].long()]
        rn.append(r.norm(dim=1))
        return torch.stack(rn, 1), torch.stack(mg, 1)


def book_spacing(books):
    """mean Euclidean distance between the points of each book (all pairs i < k) -> list of M floats"""
    out = []
    for C in books.float():
        dd = torch.cdist(C[None], C[None])[0]
        iu = torch.triu_indices(C.shape[0], C.shape[0], 1)
        out.append(float(dd[iu[0], iu[1]].mean()))
    return out


def drift_rec(step, s, s0, sp):
    r = dict(step=step)
    for k in ("codes", "eid", "plc"):
        if k not in s:
            continue
        ch0, chp = (s[k] != s0[k]), (s[k] != sp[k])
        r[f"{k}_any_vs0"] = float(ch0.any(1).float().mean())
        r[f"{k}_per_vs0"] = ch0.float().mean(0).tolist()
        r[f"{k}_any_vsprev"] = float(chp.any(1).float().mean())
        r[f"{k}_per_vsprev"] = chp.float().mean(0).tolist()
    if "rn" in s:                                            # RVQ residual diagnostics (stages 1..M, then after M)
        r["resid_norm"] = s["rn"].mean(0).tolist()
        r["resid_rms"] = s["rn"].pow(2).mean(0).sqrt().tolist()
        r["margin"] = s["mg"].mean(0).tolist()
        r["margin_median"] = s["mg"].median(0).values.tolist()
        c = s["codes"]
        ent = []
        for j in range(c.shape[1]):
            f = torch.bincount(c[:, j].long(), minlength=K).double()
            f = f[f > 0] / f.sum()
            ent.append(float(-(f * f.log2()).sum()))
        r["code_entropy_bits"] = ent
    return r


# ------------------------------------------------------------------------------------------------ books (+ probe)
def books_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    base = load_base(os.path.join(a.run, "model.pt"), dev).eval()
    L = base.cfg.n_layers
    D = Data(a.data, dev, train=True)
    P = vocab_props(os.path.join(a.data, "tokenizer.json"))
    t0 = time.time()
    # probe: 2000 val windows, states after blocks 1..L
    offs = D.starts(a.probe_windows)
    Xs = {l: [] for l in range(1, L + 1)}
    cur, src = [], []
    for i in range(0, offs.numel(), BS):
        idx = D.win(offs[i:i + BS])
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            st = base_states(base, idx, list(range(1, L + 1)))
        for l in Xs:
            Xs[l].append(normalize(st[l][:, :-1].float()).reshape(-1, base.cfg.d).to(torch.bfloat16))
        cur.append(idx[:, :-1].cpu())
        src.append(D.src(offs[i:i + BS])[:, :-1].cpu())
    cur, src = torch.cat(cur).numpy(), torch.cat(src).numpy().ravel()
    prev = np.concatenate([np.ones((cur.shape[0], 1), np.int64), cur[:, :-1]], 1)
    typ = token_types(P, prev.ravel(), cur.ravel())
    probe = dict(windows=int(offs.numel()), tokens=int(cur.size), blocks={})
    for l in range(1, L + 1):
        X = torch.cat(Xs[l])
        Xs[l] = None
        C, cnt, inert = kmeans(X, K, seed=l)
        code = nearest_chunked(X, C).cpu().numpy()
        probe["blocks"][l] = dict(nmi_source=nmi(code, src), nmi_type=nmi(code, typ), inertia=inert,
                                  used=int((np.bincount(code, minlength=K) > 0).sum()))
        print(f"probe block {l}: NMI source {probe['blocks'][l]['nmi_source']:.3f} type "
              f"{probe['blocks'][l]['nmi_type']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del X
    probe["nmi_type_vs_source"] = nmi(typ, src)
    N = max(a.n_range, key=lambda l: probe["blocks"][l]["nmi_source"]) if a.force_n is None else a.force_n
    M = L - N
    probe.update(N=N, M=M, rule=f"N = argmax over blocks {a.n_range} of NMI(code, source)")
    json.dump(probe, open(os.path.join(out, "probe.json"), "w"), indent=1)
    print(f"N = {N}, M = {M}", flush=True)
    del Xs
    torch.cuda.empty_cache()
    # books: ~5M train tokens (random windows of the train stream, fixed seed)
    g = torch.Generator(device=dev).manual_seed(7)
    toffs = torch.randint(0, D.train.numel() - CTX - 1, (a.book_windows,), device=dev, generator=g)

    def states(layer):                                       # preallocated: no list + cat copy (2 x 5 GB)
        X = torch.empty(toffs.numel() * CTX, base.cfg.d, dtype=torch.bfloat16, device=dev)
        for i in range(0, toffs.numel(), BS):
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                x = base_states(base, windows(D.train, toffs[i:i + BS], CTX), [layer])[layer]
            x = normalize(x[:, :-1].float()).reshape(-1, base.cfg.d)
            X[i * CTX:i * CTX + x.shape[0]] = x.to(torch.bfloat16)
        return X
    U = states(N)
    info = dict(N=N, M=M, book_tokens=int(U.shape[0]), stages=[], perlayer=[], hash=[])
    R = U.clone()
    books_rvq = []
    for j in range(M):
        C, cnt, inert = kmeans(R, K, seed=100 + j)
        books_rvq.append(C)
        for s in range(0, R.shape[0], CH):
            r = R[s:s + CH].float()
            R[s:s + CH] = (r - C[nearest(r, C)]).to(R.dtype)
        print(f"RVQ stage {j + 1}: inertia {inert:.1f} ({time.time() - t0:.0f}s)", flush=True)
    del R
    books_rvq = torch.stack(books_rvq)
    codes, err = [], torch.zeros(M, dtype=torch.float64)
    for s in range(0, U.shape[0], CH):                  # exact codes (fp32), as in training
        u = U[s:s + CH].float()
        r = u.clone()
        cs = []
        for j in range(M):
            c = nearest(r, books_rvq[j])
            cs.append(c)
            r = r - books_rvq[j][c]
            err[j] += float(r.pow(2).sum())
        codes.append(torch.stack(cs, 1).cpu())
    codes = torch.cat(codes).numpy()
    nu = sq_sum(U)
    sem, semw, rnd = [], [], []
    for j in range(M):
        fq = np.bincount(codes[:, j], minlength=K)
        sem.append(superclusters(books_rvq[j]).cpu())
        semw.append(weighted_superclusters(books_rvq[j].cpu().numpy(), fq, seed=700 + j))
        rnd.append(torch.from_numpy(random_balanced_table(fq, seed=500 + j)))
        info["stages"].append(dict(stage=j + 1, rel_err_after=float(err[j] / nu), used=int((fq > 0).sum()),
                                   code_freq=(fq / fq.sum()).tolist(),
                                   sem=load_stats(sem[-1].numpy()[codes[:, j]]),
                                   rand=load_stats(rnd[-1].numpy()[codes[:, j]]),
                                   semw=load_stats(semw[-1].numpy()[codes[:, j]]),
                                   semw_sizes=np.bincount(semw[-1].numpy(), minlength=E).tolist(),
                                   sem_sizes=np.bincount(sem[-1].numpy(), minlength=E).tolist()))
        print(f"stage {j + 1}: rel. error {info['stages'][-1]['rel_err_after']:.3f}, sem MaxVio "
              f"{info['stages'][-1]['sem']['maxvio']:.2f}, weighted sem MaxVio {info['stages'][-1]['semw']['maxvio']:.2f}, "
              f"rand MaxVio {info['stages'][-1]['rand']['maxvio']:.3f}", flush=True)
    info["penalty_at_init"] = float(err[M - 1] / U.shape[0])          # |u - rec|^2 per token (beta multiplies it)
    del U
    torch.cuda.empty_cache()
    pl, pls = [], []
    for j in range(M):                                       # per-layer books: state before block N+1+j
        X = states(N + j)
        C, cnt, inert = kmeans(X, K, seed=200 + N + j)
        pl.append(C)
        pls.append(superclusters(C).cpu())
        c = nearest_chunked(X, C).cpu().numpy()
        info["perlayer"].append(dict(block=N + j + 1, rel_err=inert / (sq_sum(X) / X.shape[0]),
                                     sem=load_stats(pls[-1].numpy()[c])))
        del X
        torch.cuda.empty_cache()
        print(f"per-layer book for block {N + j + 1}: MaxVio {info['perlayer'][-1]['sem']['maxvio']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    freq = np.load(os.path.join(a.data, "freq.npy"))
    hs = []
    tr_ids = windows(D.train, toffs[:2048], CTX)[:, :-1].cpu().numpy().ravel()
    for j in range(M):
        hs.append(torch.from_numpy(hash_table(freq, seed=300 + j)))
        info["hash"].append(dict(block=N + j + 1, **load_stats(hs[-1].numpy()[tr_ids])))
    books = dict(N=N, M=M, rvq=books_rvq.cpu(), rvq_sem=torch.stack(sem), rvq_semw=torch.stack(semw),
                 rvq_rand=torch.stack(rnd),
                 pl=torch.stack(pl).cpu(), pl_sem=torch.stack(pls), hash=torch.stack(hs))
    torch.save(books, os.path.join(out, "books.pt"))
    json.dump(info, open(os.path.join(out, "books.json"), "w"), indent=1)
    print(f"books saved ({time.time() - t0:.0f}s)", flush=True)
    del D
    torch.cuda.empty_cache()
    model = MoE(base, "dense", N, books=books).to(dev).eval()
    analyze_model(model, Data(a.data, dev), P, out, prefix="amix_", ablate=False)


# ------------------------------------------------------------------------------------------------ stage 2
def train_cmd(a):
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False            # exact fp32 distances for the codes
    dev = torch.device("cuda")
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    books = torch.load(a.books, map_location="cpu", weights_only=False)
    N = books["N"]
    beta = float(json.load(open(a.gate))["beta"]) if a.beta == "auto" else float(a.beta)
    if ROUTE[a.variant] != "rvq":
        beta = 0.0
    alpha = a.alpha if ROUTE[a.variant] == "learned" else 0.0
    base = load_base(a.init_from)
    D = Data(a.data, dev, train=True)
    gv = torch.Generator(device=dev).manual_seed(1234)                   # as dd.ts_train: same val windows
    voffs = torch.randint(0, D.val.numel() - CTX - 1, (a.val_windows,), device=dev, generator=gv)
    doffs = D.starts(a.drift_windows)
    dense0 = MoE(base, "dense", N, books=books).to(dev)
    ce_dense0, src_dense0 = eval_ce(dense0, D, voffs)
    del dense0
    model = MoE(base, a.variant, N, books=books).to(dev)
    ce0, src0 = eval_ce(model, D, voffs)
    print(f"{a.name}: variant {a.variant}, N {N}, beta {beta}, alpha {alpha}; val CE at step 0: {ce0:.5f} "
          f"(A-mix {ce_dense0:.5f})", flush=True)
    assert abs(ce0 - ce_dense0) < 2e-3, "the converted model does not compute the A-mix function"
    if a.variant.endswith("_frozen"):
        for p in model.base.blocks[:N].parameters():
            p.requires_grad_(False)
    start = {id(p) for p in model.base.blocks[:N].parameters()}
    groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            key = (a.start_mult if id(p) in start else 1.0, a.wd if p.dim() >= 2 else 0.0)
            groups.setdefault(key, []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=a.lr * m, weight_decay=w, mult=m) for (m, w), v in groups.items()],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_params / 1e6:.2f}M params ({n_train / 1e6:.2f}M trained)", flush=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    n_win = (D.train.numel() - 1) // (CTX + 1)
    need = (a.skip_steps + a.steps) * a.bs
    assert n_win >= need, f"{n_win} windows < {need}: the data would repeat"
    # the permutation of dd.ts_train --sampler perm --seed data_seed (stage 1 used its first skip_steps * bs windows);
    # stage 2 takes the next steps * bs windows; seed != 0 only changes their order (and the router init)
    perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(a.data_seed))[a.skip_steps * a.bs:need]
    if a.seed:
        perm = perm[torch.randperm(perm.numel(), generator=torch.Generator().manual_seed(a.seed))]
    perm = perm.to(dev) * (CTX + 1)
    s0 = snapshot(model, D, doffs)
    sp = s0
    drift = [drift_rec(0, s0, s0, s0)]
    fn = (lambda idx: moe_loss(model, idx, beta, alpha))
    step_fn = torch.compile(fn) if a.compile else fn
    log, hist, t0 = [dict(step=0, val_ce=ce0, val_ce_src=src0, t=0.0)], [], time.time()
    t_warm, WARM, t_side = None, 20, 0.0                     # s/step without compile, evals and drift snapshots
    last = a.stop_at or a.steps
    for step in range(last):
        idx = windows(D.train, perm[step * a.bs:(step + 1) * a.bs], CTX)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step) * gr["mult"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, aux = step_fn(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == WARM - 1:
            torch.cuda.synchronize()
            t_warm = time.time() - t0
        if step % 100 == 0 or step == last - 1:
            rec = dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()), pen=float(aux["pen"]),
                       bal=float(aux["bal"]), gn=float(gn), lr=lr_at(step), t=time.time() - t0)
            if aux["eid"]:
                rec["maxvio"] = [float(torch.bincount(e.flatten(), minlength=E).max()) * E / e.numel() - 1
                                 for e in aux["eid"]]
            hist.append(rec)
            if step % 500 == 0:
                print(f"{step:6d} loss {rec['loss']:.4f} ce {rec['ce']:.4f} pen {rec['pen']:.1f} bal {rec['bal']:.3f} "
                      f"gn {rec['gn']:.2f} lr {rec['lr']:.2e} {rec['t']:.0f}s mem "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G" +
                      (" maxvio " + ",".join(f"{v:.2f}" for v in rec["maxvio"]) if "maxvio" in rec else ""),
                      flush=True)
        ts = time.time()
        if (step + 1) % a.eval_every == 0 or step == last - 1:
            vce, vsrc = eval_ce(model, D, voffs)
            log.append(dict(step=step + 1, val_ce=vce, val_ce_src=vsrc, t=time.time() - t0))
            print(f"  [val] step {step + 1} ce {vce:.4f} " + " ".join(f"{s} {v:.4f}" for s, v in zip(SRC, vsrc)),
                  flush=True)
        if (step + 1) % a.drift_every == 0 or step == last - 1:
            s = snapshot(model, D, doffs)
            drift.append(drift_rec(step + 1, s, s0, sp))
            sp = s
            print("  [drift] " + " ".join(f"{k}: {v:.4f}" for k, v in drift[-1].items() if k.endswith("any_vs0")),
                  flush=True)
        if step >= WARM:
            t_side += time.time() - ts
    train_time = time.time() - t0
    steady = ((train_time - t_warm - t_side) / (last - WARM) if t_warm is not None and last > 2 * WARM
              else train_time / last)
    res = dict(name=a.name, variant=a.variant, N=N, M=model.M, beta=beta, alpha=alpha, seed=a.seed, steps=last,
               tokens=last * a.bs * CTX, train_time=train_time, s_per_step=train_time / last, s_per_step_steady=steady,
               t_compile_warm=t_warm, mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, val_ce=log[-1]["val_ce"],
               val_ce_src=log[-1]["val_ce_src"], init_ce=ce0, init_dense_ce=ce_dense0, init_dense_ce_src=src_dense0,
               n_params=n_params, n_trained=n_train, args=vars(a))
    state = model.state_dict()
    if a.save:
        save_model(model, a.variant, N, state, out)
    json.dump(dict(result=res, val_log=log, train_log=hist, drift=drift), open(os.path.join(out, "log.json"), "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)


def save_model(model, variant, N, state, out):
    torch.save(dict(cfg=model.base.cfg.to_dict(), variant=variant, N=N, state=state), os.path.join(out, "model.pt"))
    pn = {n for n, _ in model.named_parameters()}
    torch.save(dict(cfg=model.base.cfg.to_dict(), variant=variant, N=N,
                    state={k: (v.to(torch.bfloat16) if k in pn else v) for k, v in state.items()}),
               os.path.join(out, "model_bf16.pt"))


def gate_cmd(a):
    """checks of the 300-step probes; beta for the main runs.  rc 1 = do not start the main runs."""
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    R = {}
    for n in a.probes.split(","):
        p = os.path.join(a.out, n, "log.json")
        R[n] = json.load(open(p))["result"] if os.path.exists(p) else None
    dense = R.get(a.dense)
    checks, ok = {}, True
    for n, r in R.items():
        if r is None:
            checks[n] = dict(ok=False, why="no result")
            ok = False
            continue
        c = dict(val_ce=r["val_ce"], init_diff=r["init_ce"] - r["init_dense_ce"],
                 s_per_step=r.get("s_per_step_steady", r["s_per_step"]))
        c["finite"] = bool(math.isfinite(r["val_ce"]))
        c["init_ok"] = abs(c["init_diff"]) < 2e-3
        if dense:
            c["vs_dense"] = r["val_ce"] - dense["val_ce"]
            c["ce_ok"] = c["vs_dense"] < a.max_worse
            c["speed_ok"] = c["s_per_step"] < a.max_slower * dense.get("s_per_step_steady", dense["s_per_step"])
        c["ok"] = all(v for k, v in c.items() if k.endswith(("_ok", "finite")))
        ok &= c["ok"]
        checks[n] = c
    beta, why = float(a.beta), "default"
    rb, r0 = R.get(a.beta_run), R.get(a.beta0_run)
    if rb and r0:
        dce = rb["val_ce"] - r0["val_ce"]
        if dce > a.beta_tol:
            beta, why = a.beta_low, f"CE with beta {a.beta} is worse than without the penalty by {dce:+.4f} > {a.beta_tol}"
        else:
            why = f"CE with beta {a.beta} vs without the penalty: {dce:+.4f} <= {a.beta_tol}"
    res = dict(ok=bool(ok and dense is not None), beta=beta, beta_why=why, checks=checks, dense=a.dense)
    json.dump(res, open(os.path.join(out, "gate.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    sys.exit(0 if res["ok"] else 1)


# ------------------------------------------------------------------------------------------------ analysis
def lru2_hits(eid):
    """eid (nw, T): per window, decoding order, a cache of 2 experts (least recently used out); share of hits"""
    eid = torch.as_tensor(eid)
    mru = torch.full((eid.shape[0],), -1, dtype=eid.dtype)
    sec = mru.clone()
    hits = 0
    for t in range(eid.shape[1]):
        e = eid[:, t]
        h = (e == mru) | (e == sec)
        hits += int(h.sum())
        new = e != mru
        sec = torch.where(new, mru, sec)
        mru = e
    return hits / eid.numel()


@torch.no_grad()
def analyze_model(model, D, P, out, prefix="", ablate=True, n_route=1024, n_ablate=512, seed=0):
    t0 = time.time()
    M = model.M
    res = dict(variant=model.variant, N=model.N, M=M)
    tot, src, S, C = eval_ce(model, D, D.starts(), per_window=True)
    res.update(val_ce=tot, val_ce_src=dict(zip(SRC, src)), val_windows=int(S.shape[0]), val_tokens=int(C.sum()))
    np.savez_compressed(os.path.join(out, prefix + "winloss.npz"), sums=S, counts=C)
    print(f"val CE {tot:.4f} " + " ".join(f"{s} {v:.4f}" for s, v in zip(SRC, src)) + f" ({time.time() - t0:.0f}s)",
          flush=True)
    offs = D.starts(n_route)
    cur, srcs, codes, eid, plc = [], [], [], [], []
    for i in range(0, offs.numel(), BS):
        o = offs[i:i + BS]
        idx = D.win(o)
        with torch.autocast(D.dev.type, dtype=torch.bfloat16):
            _, aux = model(idx, {"codes": True})
        cur.append(idx[:, :-1].cpu())
        srcs.append(D.src(o)[:, :-1].cpu())
        codes.append(aux["codes"][:, :-1].cpu())
        if aux["eid"]:
            eid.append(torch.stack(aux["eid"], -1)[:, :-1].cpu())
        if aux.get("pl_codes"):
            plc.append(torch.stack(aux["pl_codes"], -1)[:, :-1].cpu())
    cur, srcs, codes = torch.cat(cur).numpy(), torch.cat(srcs).numpy(), torch.cat(codes).numpy()
    eid = torch.cat(eid).numpy() if eid else None
    prev = np.concatenate([np.ones((cur.shape[0], 1), np.int64), cur[:, :-1]], 1)
    typ = token_types(P, prev, cur)
    sf, tf = srcs.ravel(), typ.ravel()
    newdoc = cur[:, 1:] == 1                                  # pair (t, t+1) crosses a document start
    res["codes"] = [dict(stage=j + 1, nmi_source=nmi(codes[..., j], sf), nmi_type=nmi(codes[..., j], tf),
                         used=int(len(np.unique(codes[..., j])))) for j in range(M)]
    npz = dict(cur=cur.astype(np.uint16), src=srcs.astype(np.int8), typ=typ, codes=codes.astype(np.int8))
    if eid is not None:
        npz["eid"] = eid.astype(np.int8)
        blocks = []
        for j in range(M):
            e = eid[..., j]
            ef = e.ravel()
            b = dict(block=model.N + j + 1, **load_stats(ef), nmi_source=nmi(ef, sf), nmi_type=nmi(ef, tf))
            b["src_by_expert"] = [np.bincount(sf[ef == k], minlength=len(SRC)).tolist() for k in range(E)]
            b["type_by_expert"] = [np.bincount(tf[ef == k], minlength=len(TYPES)).tolist() for k in range(E)]
            same = (e[:, 1:] == e[:, :-1])[~newdoc]
            b["neighbour_same"] = float(same.mean())
            b["lru2_hit"] = lru2_hits(e)
            rng = np.random.default_rng(seed + j)
            b["lru2_hit_shuffled"] = lru2_hits(np.stack([rng.permutation(r) for r in e]))
            rare = int(np.argmin(b["load"]))
            m = ef == rare
            ids, cn = np.unique(cur.ravel()[m], return_counts=True)
            top = np.argsort(-cn)[:15]
            b["rarest"] = dict(expert=rare, load=b["load"][rare],
                               top_tokens=[(P["disp"][int(ids[i])], int(cn[i])) for i in top],
                               src=np.bincount(sf[m], minlength=len(SRC)).tolist(),
                               type=np.bincount(tf[m], minlength=len(TYPES)).tolist())
            blocks.append(b)
        res["blocks"] = blocks
        path = np.zeros(eid.shape[:2], np.int64)
        for j in range(M):
            path = path * E + eid[..., j]
        _, pc = np.unique(path.ravel(), return_counts=True)
        pc = np.sort(pc)[::-1]
        cs = np.cumsum(pc) / pc.sum()
        res["paths"] = dict(distinct=int(pc.size), possible=E ** M, entropy_bits=entropy_bits(pc),
                            for_50=int(np.searchsorted(cs, 0.5) + 1), for_90=int(np.searchsorted(cs, 0.9) + 1),
                            for_99=int(np.searchsorted(cs, 0.99) + 1), nmi_source=nmi(path.ravel(), sf))
    if plc:
        npz["plc"] = torch.cat(plc).numpy().astype(np.int8)
    np.savez_compressed(os.path.join(out, prefix + "routes.npz"), **npz)
    print(f"routes ({time.time() - t0:.0f}s)", flush=True)
    if ablate and eid is not None:
        ao = D.starts(n_ablate)
        b_tot, b_src = eval_ce(model, D, ao)
        abl = []
        for j in range(M):
            row = []
            for e in range(E):
                t_, s_ = eval_ce(model, D, ao, ctl={"off": (j, e)})
                row.append(dict(d_ce=t_ - b_tot, d_ce_src=[x - y for x, y in zip(s_, b_src)]))
            abl.append(row)
            print(f"ablation block {model.N + j + 1} ({time.time() - t0:.0f}s)", flush=True)
        res["ablation"] = dict(windows=int(ao.numel()), base_ce=b_tot, base_ce_src=b_src, blocks=abl)
    json.dump(res, open(os.path.join(out, prefix + "analysis.json"), "w"), indent=1)
    print(f"analysis -> {out} ({time.time() - t0:.0f}s)", flush=True)
    return res


def analyze_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_moe(a.run, a.books, dev)
    data = a.data
    lp = os.path.join(a.run, "log.json")
    if os.path.exists(lp):
        data = json.load(open(lp))["result"]["args"].get("data", data)
    analyze_model(model, Data(data, dev), vocab_props(os.path.join(data, "tokenizer.json")), a.run,
                  ablate=not a.no_ablate)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["books", "train", "gate", "analyze", "report"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs_vast")
    ap.add_argument("--run", default=None)
    ap.add_argument("--data", default="data_mix")
    ap.add_argument("--books", default=None)
    # books
    ap.add_argument("--probe_windows", type=int, default=2000)
    ap.add_argument("--book_windows", type=int, default=9728, help="9728 x 512 = 4.98M train tokens")
    ap.add_argument("--n_range", default="2,3,4")
    ap.add_argument("--force_n", type=int, default=None)
    # train
    ap.add_argument("--variant", default="dense", choices=list(ROUTE))
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--seed", type=int, default=0, help="router init and the order of the stage-2 windows")
    ap.add_argument("--data_seed", type=int, default=0, help="the stage-1 permutation (dd.ts_train --seed of A-mix)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--skip_steps", type=int, default=20000, help="stage-1 steps: their windows are not used again")
    ap.add_argument("--stop_at", type=int, default=None)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--start_mult", type=float, default=0.1)
    ap.add_argument("--beta", default="1e-3")
    ap.add_argument("--gate", default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--drift_every", type=int, default=1000)
    ap.add_argument("--drift_windows", type=int, default=64)
    ap.add_argument("--val_windows", type=int, default=256)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--save", type=int, default=1, help="0: no model files (300-step probes)")
    # gate
    ap.add_argument("--probes", default="")
    ap.add_argument("--dense", default=None)
    ap.add_argument("--beta_run", default=None)
    ap.add_argument("--beta0_run", default=None)
    ap.add_argument("--beta_tol", type=float, default=0.01)
    ap.add_argument("--beta_low", type=float, default=1e-4)
    ap.add_argument("--max_worse", type=float, default=0.1)
    ap.add_argument("--max_slower", type=float, default=3.0)
    # analyze
    ap.add_argument("--no_ablate", action="store_true")
    a = ap.parse_args(argv)
    a.n_range = tuple(int(v) for v in a.n_range.split(","))
    if a.cmd == "books":
        books_cmd(a)
    elif a.cmd == "train":
        train_cmd(a)
    elif a.cmd == "gate":
        gate_cmd(a)
    elif a.cmd == "analyze":
        analyze_cmd(a)
    else:
        from .r21_report import report
        report()


if __name__ == "__main__":
    main()
