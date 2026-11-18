"""Round 29 (quick, home): a new source for the r21 mix model perlayer 40k (runs_home/r28s3/pl, stage 3 of r28).
New sources (dd.r29_data): med = PubMed abstracts (close: English, narrow domain), ru = Russian Wikipedia (far).

stage0 (no training), MoE blocks 5-8 (inputs of blocks N+1..L, N = 4):
  - the new source over the perlayer codes (64 per block) and experts; experts of hash 40k; how many codes / experts
    cover 80% of the new source's tokens
  - distance of the RMS-normalised state to the nearest code centre, new vs old sources
  - linear probe new / old on the state (logistic regression, balanced, split by windows); embedding as a reference
  - per code: density of the new source vs the old mix (train windows) -> the codes of variant в
train (1000 steps, windows of the new source only, the same windows and lr for every variant):
  all    а) every weight
  codes  в) the codes with the highest new-source share rewritten in the table to new experts (copy of the owner,
            one per distinct owner); only the new experts learn
  label  ж) the tokens of the new source go to new experts in blocks 5-8 (copy of their own expert; the same new
            experts as в); only they learn; old sources must be bit-identical to the base
  lora   д) LoRA on every linear layer (attention qkv / o, FFN of blocks 1-4, every expert), parameters = в's
eval: 512 windows of each source (new val + the three old val sources of data_mix), CE sums per window, paired
differences to the base +- 2 SE.
    python -m dd.r29 stage0
    python -m dd.r29 train --src ru --variant codes
    python -m dd.r29 report
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

from .model import normalize  # noqa: E402
from .r21_moe import CTX, E, load_base  # noqa: E402
from .r28_train import R28, acts, expert_down  # noqa: E402

OUT = "runs_home/r29"
NEW, OLD = ("med", "ru"), ("wiki", "code", "ts")
RU_SRC = dict(med="PubMed (близкий)", ru="ру-Википедия (далёкий)", wiki="wiki", code="code", ts="ts")
PL40, HASH40 = "runs_home/r28s3/pl/model_40000_bf16.pt", "runs_home/r28s3/hash/model_40000_bf16.pt"
BOOKS, INIT = "runs_vast/r21_books/books.pt", "runs_vast/Amix_s0/model.pt"
N_EVAL, N_S0, N_SEL = 512, 256, 256
VARIANTS = ("all", "codes", "label", "lora")
RU_V = dict(base="база", all="а) все веса", codes="в) новые эксперты по кодам", label="ж) новые эксперты по метке",
            lora="д) LoRA")
SHARE_MIN, COVER = 0.5, 0.8          # в: codes with new-source share >= 0.5 (descending), at most until 80% coverage


def jsave(p, o):
    json.dump(o, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ data
class Streams:
    def __init__(self, dev):
        self.dev = dev
        meta = json.load(open("data_mix/meta.json"))
        val = np.fromfile("data_mix/val.bin", dtype=np.uint16)
        tr = np.memmap("data_mix/train.bin", dtype=np.uint16, mode="r")
        vb, tb = meta["val_bounds"], meta["train_bounds"]
        self.val, self.train = {}, {}
        for i, s in enumerate(OLD):
            self.val[s] = torch.from_numpy(val[vb[i]:vb[i + 1]].astype(np.int64)).to(dev)
            # old train for the code statistics: 2M tokens from the END of each source (the base saw data_mix_ext
            # after data_mix; these are ordinary train documents, only used for counting)
            self.train[s] = torch.from_numpy(np.asarray(tr[tb[i + 1] - 2_000_000:tb[i + 1]]).astype(np.int64)).to(dev)
        for s in NEW:
            self.val[s] = torch.from_numpy(np.fromfile(f"data_r29/{s}/val.bin", dtype=np.uint16).astype(np.int64)).to(dev)
            self.train[s] = torch.from_numpy(np.fromfile(f"data_r29/{s}/train.bin", dtype=np.uint16)
                                             .astype(np.int64)).to(dev)

    def offs(self, s, n, split="val"):
        st = (self.val if split == "val" else self.train)[s]
        o = torch.arange(0, st.numel() - CTX - 1, CTX, device=self.dev)
        return o[torch.linspace(0, o.numel() - 1, n, device=self.dev).round().long()] if n < o.numel() else o

    def win(self, s, offs, split="val"):
        st = (self.val if split == "val" else self.train)[s]
        return st[offs[:, None] + torch.arange(CTX + 1, device=self.dev)]


def load_r28(kind, dev):
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    m = R28(load_base(INIT, dev), "base" if kind == "pl" else "hash", books).to(dev)
    st = torch.load(PL40 if kind == "pl" else HASH40, map_location="cpu", weights_only=False)["state"]
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in st.items()})
    return m.eval(), books


# ------------------------------------------------------------------------------------------------ stage 0
@torch.no_grad()
def collect(m, S, src, n, split, keep_states=0, bs=16):
    """per block j: codes, experts, distance to the nearest centre (RMS-normalised state), sampled states"""
    offs = S.offs(src, n, split)
    out = {k: [[] for _ in range(m.M)] for k in ("code", "eid", "dist", "x")}
    out["emb"], out["win"] = [], []
    for i in range(0, offs.numel(), bs):
        idx = S.win(src, offs[i:i + bs], split)[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, es, xs = m(idx, keep=True)
        keepm = (idx != 0).reshape(-1)
        for j in range(m.M):
            xn = normalize(xs[j].float()).reshape(-1, m.d)
            C = m.pl_books[j]
            d2 = (xn.pow(2).sum(1, keepdim=True) + C.pow(2).sum(1)[None] - 2 * xn @ C.t()).clamp_min(0)
            dm, c = d2.min(1)
            out["code"][j].append(c[keepm])
            out["eid"][j].append(es[j].reshape(-1)[keepm])
            out["dist"][j].append(dm.sqrt()[keepm])
            if keep_states:
                out["x"][j].append(xn[keepm].half())
        if keep_states:
            out["emb"].append(normalize(m.base.emb(idx).float()).reshape(-1, m.d)[keepm].half())
        out["win"].append(torch.arange(i, i + idx.shape[0], device=idx.device).repeat_interleave(idx.shape[1])[keepm])
    res = {k: [torch.cat(v) for v in out[k]] if out[k][0] else None for k in ("code", "eid", "dist", "x")}
    res["emb"] = torch.cat(out["emb"]) if keep_states else None
    res["win"] = torch.cat(out["win"])
    return res


def ncover(p, q=COVER):
    s = np.sort(np.asarray(p))[::-1]
    return int(np.searchsorted(np.cumsum(s), q - 1e-12) + 1)


@torch.no_grad()
def probe(Xn, Xo, wn, wo, steps=300):
    """logistic regression new (1) vs old (0); balanced; train on windows with id % 5 != 4, test on the rest"""
    g = torch.Generator(device=Xn.device).manual_seed(29)
    k = min(Xn.shape[0], Xo.shape[0], 60000)
    a = torch.randperm(Xn.shape[0], generator=g, device=Xn.device)[:k]
    b = torch.randperm(Xo.shape[0], generator=g, device=Xn.device)[:k]
    X = torch.cat([Xn[a], Xo[b]]).float()
    y = torch.cat([torch.ones(k), torch.zeros(k)]).to(X.device)
    te = torch.cat([wn[a] % 5 == 4, wo[b] % 5 == 4])
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    X = (X - mu) / sd
    with torch.enable_grad():
        w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
        b0 = torch.zeros(1, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([w, b0], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X[~te] @ w + b0, y[~te]) + 1e-4 * w.pow(2).sum()
            loss.backward()
            opt.step()
    pr = (X @ w + b0) > 0
    return float((pr[te] == (y[te] > 0)).float().mean()), float((pr[~te] == (y[~te] > 0)).float().mean())


def stage0_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.time()
    S = Streams(dev)
    res = {}
    for kind in ("pl", "hash"):
        m, books = load_r28(kind, dev)
        V = {s: collect(m, S, s, N_S0, "val", keep_states=(kind == "pl")) for s in NEW + OLD}
        R = res[kind] = {}
        for j in range(m.M):
            blk = {}
            for s in NEW + OLD:
                e = torch.bincount(V[s]["eid"][j], minlength=E).double()
                c = torch.bincount(V[s]["code"][j], minlength=64).double()
                pe, pc = (e / e.sum()).tolist(), (c / c.sum()).tolist()
                dd_ = V[s]["dist"][j]
                blk[s] = dict(experts=pe, n_exp80=ncover(pe), codes=pc, n_code80=ncover(pc),
                              ent_codes=float(-(torch.tensor(pc).clamp_min(1e-12).log2() * torch.tensor(pc)).sum()),
                              dist_mean=float(dd_.mean()), dist_q=[float(x) for x in torch.quantile(
                                  dd_[torch.randperm(dd_.numel(), device=dev)[:100000]],
                                  torch.tensor([0.1, 0.5, 0.9], device=dev))])
            R[str(j + 5)] = blk
        if kind == "pl":
            # probe new vs old (old = equal shares of wiki / code / ts)
            P = res["probe"] = {}
            for s in NEW:
                P[s] = {}
                for j in list(range(m.M)) + ["emb"]:
                    if j == "emb":
                        Xn, Xo = V[s]["emb"], [V[o]["emb"] for o in OLD]
                    else:
                        Xn, Xo = V[s]["x"][j], [V[o]["x"][j] for o in OLD]
                    k = min(x.shape[0] for x in Xo)
                    Xo_ = torch.cat([x[:k] for x in Xo])
                    wo = torch.cat([V[o]["win"][:k] for o in OLD])
                    te, trn = probe(Xn, Xo_, V[s]["win"], wo)
                    P[s]["emb" if j == "emb" else str(j + 5)] = dict(test=te, train=trn)
                print(f"probe {s}: " + " ".join(f"{k} {v['test']:.3f}" for k, v in P[s].items()), flush=True)
            # selection for в from TRAIN windows: per code density of the new source vs the old mix
            T = {s: collect(m, S, s, N_SEL, "train") for s in NEW + OLD}
            sel = res["select"] = {}
            for s in NEW:
                sel[s] = {}
                for j in range(m.M):
                    pn = torch.bincount(T[s]["code"][j], minlength=64).double()
                    pn = pn / pn.sum()
                    po = sum((lambda c: c / c.sum())(torch.bincount(T[o]["code"][j], minlength=64).double())
                             for o in OLD) / len(OLD)
                    share = pn / (pn + po).clamp_min(1e-12)
                    order = share.argsort(descending=True).tolist()
                    chosen, cov = [], 0.0
                    for c in order:
                        if share[c] < SHARE_MIN and chosen:
                            break
                        chosen.append(c)
                        cov += float(pn[c])
                        if cov >= COVER:
                            break
                    owners = sorted({int(m.pl_table[j][c]) for c in chosen})
                    om = {str(o): float(pn[m.pl_table[j] == o].sum()) for o in range(E)}
                    sel[s][str(j)] = dict(codes=chosen, share=[float(share[c]) for c in chosen], owners=owners,
                                          owner_mass=om, label_direct=float(sum(om[str(o)] for o in owners)),
                                          cover_new=cov, cover_old=float(sum(po[c] for c in chosen)),
                                          share_all=share.tolist())
                print(f"select {s}: " + "; ".join(f"b{int(j) + 5} codes {len(v['codes'])} owners {v['owners']} "
                                                  f"new {v['cover_new']:.2f} old {v['cover_old']:.3f}"
                                                  for j, v in sel[s].items()), flush=True)
        del m
        torch.cuda.empty_cache()
        print(f"{kind} done ({time.time() - t0:.0f}s)", flush=True)
    jsave(os.path.join(OUT, "stage0.json"), res)
    write_stage0(res)


def write_stage0(res):
    L = ["# r29, этап 0 (без обучения): новый источник в perlayer 40k\n",
         f"Окна: {N_S0} окон val на источник (по {CTX} токенов). Блоки 5–8 = входы MoE-блоков (то, что видит "
         "маршрутизация). Коды — 64 центра книги блока (r21 perlayer), эксперты — 8.\n"]
    for kind, nm in (("pl", "perlayer 40k"), ("hash", "hash 40k")):
        L.append(f"\n## Сколько экспертов покрывают 80% токенов источника — {nm}\n")
        L.append("| источник | блок 5 | блок 6 | блок 7 | блок 8 | доли экспертов, блок 5 (по убыванию) |")
        L.append("|---|---|---|---|---|---|")
        for s in NEW + OLD:
            r = [res[kind][str(b)][s] for b in (5, 6, 7, 8)]
            top = " ".join(f"{x:.2f}" for x in sorted(r[0]["experts"], reverse=True))
            L.append(f"| {RU_SRC[s]} | " + " | ".join(str(x["n_exp80"]) for x in r) + f" | {top} |")
    L.append("\n## Коды perlayer: сколько кодов покрывают 80% токенов, энтропия (бит, максимум 6)\n")
    L.append("| источник | блок 5 | блок 6 | блок 7 | блок 8 |")
    L.append("|---|---|---|---|---|")
    for s in NEW + OLD:
        L.append(f"| {RU_SRC[s]} | " + " | ".join(
            f"{res['pl'][str(b)][s]['n_code80']} ({res['pl'][str(b)][s]['ent_codes']:.1f})" for b in (5, 6, 7, 8)) + " |")
    L.append("\n## Расстояние нормированного состояния до ближайшего центра (среднее; медиана [10%, 90%])\n")
    L.append("| источник | блок 5 | блок 6 | блок 7 | блок 8 |")
    L.append("|---|---|---|---|---|")
    for s in NEW + OLD:
        L.append(f"| {RU_SRC[s]} | " + " | ".join(
            (lambda r: f"{r['dist_mean']:.2f}; {r['dist_q'][1]:.2f} [{r['dist_q'][0]:.2f}, {r['dist_q'][2]:.2f}]")(
                res['pl'][str(b)][s]) for b in (5, 6, 7, 8)) + " |")
    L.append("\n## Линейный классификатор «новый / старый» по состоянию (точность на отложенных окнах, классы "
             "поровну, старые = wiki/code/ts поровну)\n")
    L.append("| источник | эмбеддинг | блок 5 | блок 6 | блок 7 | блок 8 |")
    L.append("|---|---|---|---|---|---|")
    for s in NEW:
        p = res["probe"][s]
        L.append(f"| {RU_SRC[s]} | " + " | ".join(f"{p[k]['test']:.3f}" for k in ("emb", "5", "6", "7", "8")) + " |")
    L.append("\n## Коды для в) (по train-окнам: доля нового = p_new / (p_new + p_old), p_old — смесь трёх старых "
             f"поровну; коды по убыванию доли, пока доля ≥ {SHARE_MIN} и покрытие нового < {COVER:.0%})\n")
    L.append("| источник | блок | кодов | владельцы (новые эксперты) | покрытие нового | доля старых токенов в этих кодах |")
    L.append("|---|---|---|---|---|---|")
    for s in NEW:
        for j, v in res["select"][s].items():
            L.append(f"| {RU_SRC[s]} | {int(j) + 5} | {len(v['codes'])} | {v['owners']} | {v['cover_new']:.2f} | "
                     f"{v['cover_old']:.3f} |")
    open("report/r29_stage0.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ variants
class LoRALin(nn.Module):
    """y = x W^T + s (x A^T) B^T for an nn.Linear W (frozen)"""

    def __init__(self, lin, r, alpha=None):
        super().__init__()
        self.lin = lin
        self.A = nn.Parameter(torch.randn(r, lin.in_features, device=lin.weight.device) / math.sqrt(lin.in_features))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))
        self.s = 1.0

    def forward(self, x):
        return self.lin(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.s

    @property
    def weight(self):
        return self.lin.weight


class R29(R28):
    """perlayer 40k + new experts (codes / label) or LoRA; label: bool (B, T) of the new source, set before forward"""

    def setup(self, variant, sel=None, r=0):
        self.v29 = variant
        self.label = None
        self.nw1, self.nw3, self.nw2 = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        self.newmap = []
        if variant in ("codes", "label"):
            for j in range(self.M):
                owners = sel[str(j)]["owners"]
                mp = torch.full((E,), -1, dtype=torch.long)
                for k, o in enumerate(owners):
                    mp[o] = E + k
                self.newmap.append(mp)
                self.nw1.append(nn.Parameter(self.w1[j][owners].detach().clone()))
                self.nw3.append(nn.Parameter(self.w3[j][owners].detach().clone()))
                self.nw2.append(nn.Parameter(self.w2[j][owners].detach().clone()))
                if variant == "codes":
                    for c in sel[str(j)]["codes"]:
                        self.pl_table[j][c] = mp[int(self.pl_table[j][c])]
            fb = []                                          # label: a token whose own expert has no copy goes to the
            for j in range(self.M):                          # copy with the most new-source tokens (the first owner of
                owners = sel[str(j)]["owners"]               # the selection = largest coverage)
                fb.append(E + int(np.argmax([sel[str(j)]["owner_mass"][str(o)] for o in owners])))
            self.fallback = fb
            for p in self.parameters():
                p.requires_grad_(False)
            for p in list(self.nw1) + list(self.nw3) + list(self.nw2):
                p.requires_grad_(True)
        elif variant == "all":
            for p in self.parameters():
                p.requires_grad_(True)
        return self

    def route(self, j, x, idx=None):
        e = super().route(j, x, idx)
        if getattr(self, "v29", None) == "label" and self.label is not None and self.label.any():
            mp = self.newmap[j].to(e.device)[e]
            mp = torch.where(mp < 0, torch.full_like(mp, self.fallback[j]), mp)
            e = torch.where(self.label, mp, e)
        return e

    def ffn(self, j, z, e):
        if getattr(self, "v29", None) in ("codes", "label"):
            w1 = torch.cat([self.w1[j], self.nw1[j]])
            w3 = torch.cat([self.w3[j], self.nw3[j]])
            w2 = torch.cat([self.w2[j], self.nw2[j]])
        elif getattr(self, "v29", None) == "lora":
            w1 = self.w1[j] + self.lB1[j] @ self.lA1[j]
            w3 = self.w3[j] + self.lB3[j] @ self.lA3[j]
            w2 = self.w2[j] + self.lB2[j] @ self.lA2[j]
        else:
            return super().ffn(j, z, e)
        sh = z.shape
        zf, ef = z.reshape(-1, sh[-1]), e.reshape(-1)
        y = expert_down(acts(zf, ef, w1, w3), ef, w2)
        return y.view(*sh[:-1], -1)


def wrap_lora(m, r):
    """LoRA on every nn.Linear of the base blocks (attention qkv / o, FFN of the dense blocks)"""
    n = 0
    for b, blk in enumerate(m.base.blocks):
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, nn.Linear):
                parent = blk
                parts = name.split(".")
                for p_ in parts[:-1]:
                    parent = getattr(parent, p_)
                setattr(parent, parts[-1], LoRALin(mod, r))
                n += 1
    return n


def build29(variant, sel, dev, r=0):
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    m = R29(load_base(INIT, dev), "base", books).to(dev)
    st = torch.load(PL40, map_location="cpu", weights_only=False)["state"]
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in st.items()})
    if variant == "lora":
        for p in m.parameters():
            p.requires_grad_(False)
        wrap_lora(m, r)
        m.v29 = "lora"
        m.label = None
        d, h = m.d, m.h
        for nm in ("lA1", "lB1", "lA3", "lB3", "lA2", "lB2"):
            setattr(m, nm, nn.ParameterList())
        for j in range(m.M):
            m.lA1.append(nn.Parameter(torch.randn(E, r, d, device=dev) / math.sqrt(d)))
            m.lB1.append(nn.Parameter(torch.zeros(E, h, r, device=dev)))
            m.lA3.append(nn.Parameter(torch.randn(E, r, d, device=dev) / math.sqrt(d)))
            m.lB3.append(nn.Parameter(torch.zeros(E, h, r, device=dev)))
            m.lA2.append(nn.Parameter(torch.randn(E, r, h, device=dev) / math.sqrt(h)))
            m.lB2.append(nn.Parameter(torch.zeros(E, d, r, device=dev)))
    elif variant == "base":
        m.v29 = "base"
        m.label = None
    else:
        m.setup(variant, sel)
    return m.to(dev)


def lora_rank(m, n_target):
    """rank whose parameter count is closest to n_target"""
    per = 0
    for blk in m.base.blocks:
        for mod in blk.modules():
            if isinstance(mod, nn.Linear):
                per += mod.in_features + mod.out_features
    per += m.M * E * 3 * (m.d + m.h)
    return max(1, round(n_target / per)), per


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def eval_sums(m, S, src, bs=32):
    """CE sum and count per window, the 512 val windows of a source"""
    offs = S.offs(src, N_EVAL, "val")
    sums, cnt = [], []
    m.eval()
    for i in range(0, offs.numel(), bs):
        idx = S.win(src, offs[i:i + bs], "val")
        m.label = torch.full(idx.shape, src in NEW, dtype=torch.bool, device=idx.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(idx)
        tgt = idx[:, 1:]
        lo = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="none",
                             ignore_index=0).view(tgt.shape).double()
        sums.append(lo.sum(1))
        cnt.append((tgt != 0).double().sum(1))
    m.label = None
    return torch.cat(sums).cpu().numpy(), torch.cat(cnt).cpu().numpy()


def eval_all(m, S, src_new, tag, out):
    res = {}
    for s in (src_new,) + OLD:
        sm, ct = eval_sums(m, S, s)
        res[s] = float(sm.sum() / ct.sum())
        np.savez(os.path.join(out, f"eval_{tag}_{s}.npz"), sum=sm, cnt=ct)
    return res


# ------------------------------------------------------------------------------------------------ train
def train_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(29)
    dev = torch.device("cuda")
    S = Streams(dev)
    out = os.path.join(OUT, a.src)
    os.makedirs(out, exist_ok=True)
    s0 = json.load(open(os.path.join(OUT, "stage0.json"), encoding="utf-8"))
    sel = s0["select"][a.src]
    n_new = sum(len(v["owners"]) for v in sel.values())
    res_p = os.path.join(out, "r29.json")
    res = json.load(open(res_p, encoding="utf-8")) if os.path.exists(res_p) else {}
    if "base" not in res:
        mb = build29("base", sel, dev)
        res["base"] = dict(ce=eval_all(mb, S, a.src, "base", out))
        jsave(res_p, res)
        print(f"base: {res['base']['ce']}", flush=True)
        del mb
    if a.variant == "base":
        return
    t0 = time.time()
    r = 0
    if a.variant == "lora":
        tmp = build29("base", sel, dev)
        n_target = n_new * 3 * tmp.d * tmp.h
        r, per = lora_rank(tmp, n_target)
        del tmp
    m = build29(a.variant, sel, dev, r)
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    ce0 = eval_all(m, S, a.src, f"{a.variant}_step0", out) if a.variant in ("label", "codes") else None
    print(f"{a.src} {a.variant}: trainable {n_tr / 1e6:.2f}M (new experts {n_new}, lora r {r}); step-0 {ce0} "
          f"({time.time() - t0:.0f}s)", flush=True)
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
    assert n_win >= a.steps * a.bs
    perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(29))[:a.steps * a.bs].to(dev) * (CTX + 1)
    log = []
    m.train()
    for step in range(a.steps):
        idx = st[perm[step * a.bs:(step + 1) * a.bs, None] + torch.arange(CTX + 1, device=dev)]
        m.label = torch.ones(idx.shape, dtype=torch.bool, device=dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(idx)
            ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), idx[:, 1:].flatten(), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            log.append(dict(step=step, ce=float(ce), gn=float(gn)))
            print(f"{step:5d} ce {float(ce):.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
    m.label = None
    ce = eval_all(m, S, a.src, a.variant, out)
    res[a.variant] = dict(ce=ce, ce0=ce0, trainable=n_tr, n_new_experts=n_new, lora_r=r, log=log,
                          time=time.time() - t0, lr=a.lr, steps=a.steps, bs=a.bs)
    if a.variant == "label":                                    # bit-exactness on the old sources
        res[a.variant]["bit_exact_old"] = {
            s: bool(np.array_equal(np.load(os.path.join(out, f"eval_label_{s}.npz"))["sum"],
                                   np.load(os.path.join(out, f"eval_base_{s}.npz"))["sum"])) for s in OLD}
    jsave(res_p, res)
    print(f"{a.src} {a.variant}: {ce} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ report
def ratio_diff(out, tag, s):
    a_, b_ = np.load(os.path.join(out, f"eval_{tag}_{s}.npz")), np.load(os.path.join(out, f"eval_base_{s}.npz"))
    assert np.array_equal(a_["cnt"], b_["cnt"])
    n = len(a_["cnt"])
    q = (a_["sum"].sum() - b_["sum"].sum()) / a_["cnt"].sum()
    se = math.sqrt((((a_["sum"] - b_["sum"]) - q * a_["cnt"]) ** 2).sum() / (n - 1) / n) / a_["cnt"].mean()
    return float(q), float(se)


def pdiff(out, t1, t2, src):
    x, y = np.load(os.path.join(out, f"eval_{t1}_{src}.npz")), np.load(os.path.join(out, f"eval_{t2}_{src}.npz"))
    c = x["cnt"]
    q = (x["sum"].sum() - y["sum"].sum()) / c.sum()
    se = math.sqrt((((x["sum"] - y["sum"]) - q * c) ** 2).sum() / (len(c) - 1) / len(c)) / c.mean()
    return f"{q:+.4f} ± {2 * se:.4f}"


def report_cmd(a):
    L = ["# r29: новый источник, 1000 шагов дообучения perlayer 40k\n"]
    for src in NEW:
        out = os.path.join(OUT, src)
        p = os.path.join(out, "r29.json")
        if not os.path.exists(p):
            continue
        res = json.load(open(p, encoding="utf-8"))
        L.append(f"\n## {RU_SRC[src]}\n")
        b = res["base"]["ce"]
        L.append(f"База (perlayer 40k): {RU_SRC[src]} {b[src]:.4f}; wiki {b['wiki']:.4f}, code {b['code']:.4f}, "
                 f"ts {b['ts']:.4f}.\n")
        L.append("| вариант | обучаемых параметров | новый: Δ к базе ± 2 SE | wiki | code | ts | старые в среднем | "
                 "выигрыш на новом − потеря на старых |")
        L.append("|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            if v not in res:
                continue
            r = res[v]
            dn = ratio_diff(out, v, src)
            do = [ratio_diff(out, v, s) for s in OLD]
            mo = float(np.mean([x[0] for x in do]))
            seo = math.sqrt(sum(x[1] ** 2 for x in do)) / 3
            extra = f" (r {r['lora_r']})" if v == "lora" else f" ({r['n_new_experts']} эксп.)" if v in ("codes", "label") else ""
            sen = math.sqrt(dn[1] ** 2 + seo ** 2)
            L.append(f"| {RU_V[v]} | {r['trainable'] / 1e6:.2f}M{extra} | {dn[0]:+.4f} ± {2 * dn[1]:.4f} | " +
                     " | ".join(f"{x[0]:+.4f} ± {2 * x[1]:.4f}" for x in do) +
                     f" | {mo:+.4f} ± {2 * seo:.4f} | {-dn[0] - mo:+.4f} ± {2 * sen:.4f} |")
        pairs = [(x, y) for x, y in (("label", "codes"), ("label", "lora"), ("codes", "lora"), ("all", "label"))
                 if x in res and y in res]
        if pairs:
            L.append("\nПарные разности на новом источнике (те же окна): " + "; ".join(
                f"{RU_V[x]} − {RU_V[y]}: {pdiff(out, x, y, src)}" for x, y in pairs) + ".")
        for v in ("codes", "label"):
            if v in res and res[v].get("ce0"):
                L.append(f"\nШаг 0, {RU_V[v]}: новый {res[v]['ce0'][src]:.4f} (база {b[src]:.4f}).")
        if "label" in res:
            L.append(f"\nж: побитовое совпадение с базой на старых: {res['label'].get('bit_exact_old')}.")
    open("report/r29_tables.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage0", "train", "report"])
    ap.add_argument("--src", default="ru", choices=NEW)
    ap.add_argument("--variant", default="codes", choices=VARIANTS + ("base",))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.1)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    dict(stage0=stage0_cmd, train=train_cmd, report=report_cmd)[a.cmd](a)


if __name__ == "__main__":
    main()
