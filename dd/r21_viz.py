"""Page «эксперты perlayer» for the trained MoE-perlayer of round 21 (home, no training, the whole val).

  python -m dd.r21_viz data    # forward over all 8789 val windows -> runs_home/r21_viz/routes.npz (codes, distances to
                               #   the chosen code centre, per token) and report/r21_perlayer_data.json
  python -m dd.r21_viz page    # report/r21_perlayer_template.html + data (+ report/r21_ws.json, the card «пробелы в
                               #   коде» of dd.r21_ws, if present) -> report/r21_perlayer.html

Experts tab: load, sources, token types, top-20 tokens, 10 typical contexts (tokens nearest to the centres of the
expert's most used codes), CE rise when the expert is switched off (from the round-21 analysis, 512 val windows).
Texts tab: 3 fragments per source, every token with its expert and code in every MoE block (the fragment is run from
its document start, up to 512 tokens of context). Routes tab: the most frequent routes (experts of blocks N+1..L).
Diversity tab: NMI between the experts of adjacent blocks, routes covering 50 / 90 / 99% of tokens, perlayer and hash."""
import argparse
import json
import os

import numpy as np
import torch

from .model import normalize  # noqa: F401  (states are normalised inside the model)
from .r21_moe import BS, CTX, E, Data, load_moe, nmi, token_types, vocab_props

RUN = os.path.join("runs_vast", "mix_perlayer")
BOOKS = os.path.join("runs_vast", "r21_books", "books.pt")
OUT = os.path.join("runs_home", "r21_viz")
DATA_JSON = os.path.join("report", "r21_perlayer_data.json")
TEMPLATE = os.path.join("report", "r21_perlayer_template.html")
PAGE = os.path.join("report", "r21_perlayer.html")
SRC_RU = ["Википедия", "код", "сказки"]
TYPES_RU = ["пробелы", "пунктуация", "числа", "служебные", "имена", "начало слова", "продолжение слова"]
TYPES_SHORT = ["пробелы", "пунктуация", "числа", "служебные", "имена", "начала слов", "продолжения"]
BEFORE, AFTER = 8, 3


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join("data_mix", "tokenizer.json"))


def pieces(tok, ids):
    """text of every token of a run of ids (prefix-difference decoding keeps multi-byte characters whole)"""
    out, prev = [], ""
    ids = [int(i) for i in ids]
    for k in range(len(ids)):
        cur = tok.decode([i for i in ids[:k + 1] if i >= 2])
        out.append(cur[len(prev):] if cur.startswith(prev) else tok.decode([ids[k]]) if ids[k] >= 2 else "")
        prev = cur
    return out


@torch.no_grad()
def collect(model, D):
    """all val windows: per recorded token its stream position, code and distance to the code centre per MoE block"""
    offs = D.starts()
    M = model.M
    pos, codes, dist = [], [], []
    ar = torch.arange(CTX, device=D.dev)
    for i in range(0, offs.numel(), BS):
        o = offs[i:i + BS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(D.win(o), {"states": True})
        cs, ds = [], []
        for j in range(M):
            s = aux["s_in"][j][:, :-1].float()
            c = aux["pl_codes"][j][:, :-1]
            ds.append((s - model.pl_books[j][c]).norm(dim=-1))
            cs.append(c)
        pos.append((o[:, None] + ar[None]).flatten().cpu())
        codes.append(torch.stack(cs, -1).reshape(-1, M).to(torch.uint8).cpu())
        dist.append(torch.stack(ds, -1).reshape(-1, M).half().cpu())
    return torch.cat(pos).numpy(), torch.cat(codes).numpy(), torch.cat(dist).numpy()


def context(tok, ids, p, eot_pos):
    """BEFORE tokens + the token + AFTER tokens around stream position p, cut at document boundaries"""
    i = int(np.searchsorted(eot_pos, p, side="left"))           # eots before p: eot_pos[:i]
    lo = int(eot_pos[i - 1]) + 1 if i > 0 else 0
    hi = int(eot_pos[i]) if i < eot_pos.size and eot_pos[i] > p else (int(eot_pos[i + 1]) if i + 1 < eot_pos.size
                                                                       else ids.size)
    a, b = max(p - BEFORE, lo), min(p + AFTER + 1, max(hi, p + 1))
    pc = pieces(tok, ids[a:b])
    t = pc[p - a] if ids[p] >= 2 else "⟨конец текста⟩"
    return dict(pre="".join(pc[:p - a]), tok=t, post="".join(pc[p - a + 1:]))


def short_label(src, typ, top_tok, top_share):
    s = np.array(src)
    order = np.argsort(-s)
    if s[order[0]] >= 0.6:
        lab = SRC_RU[order[0]]
    elif s[order[0]] + s[order[1]] >= 0.8:
        lab = f"{SRC_RU[order[0]]} + {SRC_RU[order[1]]}"
    else:
        lab = "все источники"
    t = np.array(typ)
    lab += " · " + TYPES_SHORT[int(t.argmax())]
    if top_share >= 0.25:
        lab += f" «{top_tok.strip() or top_tok}»"
    return lab


def disp(t):
    """visible form of a token string for chips: leading space -> ␣, newline -> ↵, tab -> ⇥"""
    return t.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␣") if t.strip() == "" else \
        ("␣" + t[1:] if t.startswith(" ") else t).replace("\n", "↵").replace("\t", "⇥")


def route_stats(R, n_top=None):
    """R: (n, M) experts -> distinct routes, routes for 50 / 90 / 99% of tokens, entropy (bits)"""
    key = np.zeros(R.shape[0], np.int64)
    for j in range(R.shape[1]):
        key = key * E + R[:, j]
    cnt = np.bincount(key, minlength=E ** R.shape[1]).astype(np.float64)
    p = np.sort(cnt[cnt > 0])[::-1] / cnt.sum()
    cum = np.cumsum(p)
    return dict(distinct=int((cnt > 0).sum()), for50=int(np.searchsorted(cum, 0.5) + 1),
                for90=int(np.searchsorted(cum, 0.9) + 1), for99=int(np.searchsorted(cum, 0.99) + 1),
                entropy_bits=float(-(p * np.log2(p)).sum()))


def coverage(R, independent=False):
    """share of tokens covered by the k most frequent routes, k on a log grid (1 .. 4096)"""
    if independent:
        q = np.ones(1)
        for j in range(R.shape[1]):
            q = np.outer(q, np.bincount(R[:, j], minlength=E) / R.shape[0]).ravel()
        p = np.sort(q)[::-1]
    else:
        key = np.zeros(R.shape[0], np.int64)
        for j in range(R.shape[1]):
            key = key * E + R[:, j]
        p = np.sort(np.bincount(key, minlength=E ** R.shape[1]))[::-1] / R.shape[0]
    cum = np.cumsum(p)
    ks = sorted(set(int(round(x)) for x in np.logspace(0, np.log10(p.size), 60)))
    return [[k, float(cum[k - 1])] for k in ks]


def independent_routes(R):
    """the same per-block loads, blocks chosen independently: routes for 50 / 90 / 99% of tokens"""
    q = np.ones(1)
    for j in range(R.shape[1]):
        f = np.bincount(R[:, j], minlength=E) / R.shape[0]
        q = np.outer(q, f).ravel()
    p = np.sort(q[q > 0])[::-1]
    cum = np.cumsum(p)
    return dict(for50=int(np.searchsorted(cum, 0.5) + 1), for90=int(np.searchsorted(cum, 0.9) + 1),
                for99=int(np.searchsorted(cum, 0.99) + 1), entropy_bits=float(-(p * np.log2(p)).sum()))


@torch.no_grad()
def data_cmd():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    model = load_moe(RUN, None, dev, fname="model_bf16.pt")
    assert model.variant == "perlayer"
    N, M = model.N, model.M
    D = Data("data_mix", dev)
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "routes.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        pos, codes, dist = z["pos"], z["codes"], z["dist"]
    else:
        pos, codes, dist = collect(model, D)
        np.savez(cache, pos=pos, codes=codes, dist=dist)
    table = model.pl_table.cpu().numpy()                     # (M, 64) code -> expert
    ex = np.stack([table[j][codes[:, j]] for j in range(M)], 1).astype(np.int64)
    ids_all = np.fromfile(os.path.join("data_mix", "val.bin"), dtype=np.uint16).astype(np.int64)
    src_all = np.fromfile(os.path.join("data_mix", "val_src.bin"), dtype=np.uint8).astype(np.int64)
    ids, src = ids_all[pos], src_all[pos]
    prev = np.where(pos > 0, ids_all[np.maximum(pos - 1, 0)], 1)
    P = vocab_props(os.path.join("data_mix", "tokenizer.json"))
    typ = token_types(P, prev, ids).astype(np.int64)
    eot_pos = np.flatnonzero(ids_all == 1)
    tok = tokenizer()
    ab = json.load(open(os.path.join(RUN, "analysis.json")))["ablation"]
    rng = np.random.default_rng(21)
    n = ids.size
    print(f"{n} tokens", flush=True)
    blocks = []
    for j in range(M):
        cfreq = np.bincount(codes[:, j], minlength=table.shape[1])
        order = np.lexsort((dist[:, j].astype(np.float32), codes[:, j]))   # by code, then distance to its centre
        starts = np.searchsorted(codes[:, j][order], np.arange(table.shape[1]))
        exps = []
        for e in range(E):
            m = ex[:, j] == e
            ne = int(m.sum())
            srcs = (np.bincount(src[m], minlength=3) / max(1, ne)).tolist()
            types = (np.bincount(typ[m], minlength=7) / max(1, ne)).tolist()
            tc = np.bincount(ids[m], minlength=len(P["strs"]))
            top = np.argsort(-tc)[:20]
            top_tokens = [dict(t=disp(P["disp"][t]), n=int(tc[t]), share=float(tc[t] / max(1, ne))) for t in top if tc[t]]
            # typical contexts: the nearest tokens to the centres of the expert's most used codes, round robin
            ecodes = [c for c in np.argsort(-cfreq) if table[j][c] == e and cfreq[c] > 0]
            ctx, seen, r = [], {}, 0
            while len(ctx) < 10 and r < 20 and ecodes:
                for c in ecodes:
                    if len(ctx) >= 10:
                        break
                    k = starts[c] + r
                    if k >= (starts[c + 1] if c + 1 < starts.size else n) :
                        continue
                    i = order[k]
                    t = int(ids[i])
                    if seen.get(t, 0) >= 2:
                        continue
                    seen[t] = seen.get(t, 0) + 1
                    cx = context(tok, ids_all, int(pos[i]), eot_pos)
                    cx.update(code=int(c), src=int(src[i]), d=round(float(dist[i, j]), 2))
                    ctx.append(cx)
                r += 1
            a = ab["blocks"][j][e]
            exps.append(dict(e=e, load=ne / n, src=srcs, types=types, top=top_tokens, ctx=ctx,
                             codes=[int(c) for c in ecodes], code_share=[float(cfreq[c] / n) for c in ecodes],
                             ablate=a["d_ce_src"], ablate_all=a["d_ce"],
                             label=short_label(srcs, types, top_tokens[0]["t"] if top_tokens else "",
                                               top_tokens[0]["share"] if top_tokens else 0)))
        blocks.append(dict(block=N + 1 + j, experts=exps))
        print(f"block {N + 1 + j}: " + "; ".join(f"E{x['e']} {100 * x['load']:.1f}% {x['label']}" for x in exps), flush=True)
    # texts: 3 fragments per source, each run from its document start (up to 512 tokens of context)
    texts = []
    vb = D.meta["val_bounds"]
    for s in range(3):
        docs = [p + 1 for p in eot_pos if vb[s] <= p + 1 < vb[s + 1]]
        frs = []
        for q in (0.2, 0.5, 0.8):
            k = int(q * len(docs))
            while k < len(docs) - 1:
                d0, d1 = docs[k], docs[k + 1] - 1
                if d1 - d0 >= 170:
                    break
                k += 1
            a = d0
            if s == 1:                                       # code: start at the first def / class, not the licence
                pc_doc = pieces(tok, ids_all[d0:min(d1, d0 + 1500)])
                txt, cum = "", []
                for t in pc_doc:
                    cum.append(len(txt))
                    txt += t
                marks = ("\ndef ", "\nclass ", "\n    def ")
                m = min([x for x in (txt.find(k) for k in marks) if x >= 0], default=-1)
                if m >= 0:
                    a = d0 + int(np.searchsorted(cum, m + 1, side="right") - 1)
                a = min(a, d1 - 150)
            b = min(a + 150, d1)
            lo = max(d0, b - CTX)
            x = torch.tensor([[1] + ids_all[lo:b].tolist()], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, aux = model(x, {"states": True})
            cs = torch.stack(aux["pl_codes"], -1)[0].cpu().numpy()       # (T+1, M)
            off = 1 + (a - lo)
            pc = pieces(tok, ids_all[a:b])
            toks = [dict(s=pc[i], c=cs[off + i].tolist(), e=[int(table[j][cs[off + i, j]]) for j in range(M)])
                    for i in range(b - a)]
            frs.append(dict(doc_pos=int(d0), start=int(a - d0), tokens=toks))
        texts.append(dict(src=s, name=SRC_RU[s], fragments=frs))
    # routes
    key = np.zeros(n, np.int64)
    for j in range(M):
        key = key * E + ex[:, j]
    rc = np.bincount(key, minlength=E ** M)
    routes = []
    for k in np.argsort(-rc)[:24]:
        if rc[k] == 0:
            break
        r = [int(k // E ** (M - 1 - j) % E) for j in range(M)]
        idx = np.flatnonzero(key == k)
        sm = (np.bincount(src[idx], minlength=3) / idx.size).tolist()
        ex_ctx, seen = [], set()
        for i in rng.permutation(idx)[:200]:
            if int(ids[i]) in seen:
                continue
            seen.add(int(ids[i]))
            cx = context(tok, ids_all, int(pos[i]), eot_pos)
            cx["src"] = int(src[i])
            ex_ctx.append(cx)
            if len(ex_ctx) == 3:
                break
        routes.append(dict(r=r, n=int(rc[k]), share=float(rc[k] / n), src=sm, ex=ex_ctx))
    # diversity: perlayer against hash (the same token positions)
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    hx = np.stack([books["hash"][j].numpy()[ids] for j in range(M)], 1).astype(np.int64)
    div = {}
    for name, R in (("perlayer", ex), ("hash", hx)):
        div[name] = dict(adjacent_nmi=[nmi(R[:, j], R[:, j + 1]) for j in range(M - 1)],
                         first_last_nmi=nmi(R[:, 0], R[:, M - 1]), routes=route_stats(R),
                         independent=independent_routes(R),
                         loads=[(np.bincount(R[:, j], minlength=E) / n).tolist() for j in range(M)],
                         nmi_source=[nmi(R[:, j], src) for j in range(M)],
                         cum=coverage(R), cum_independent=coverage(R, True),
                         trans=[(np.bincount(R[:, j] * E + R[:, j + 1], minlength=E * E).reshape(E, E) / n).tolist()
                                for j in range(M - 1)])
        print(f"{name}: adjacent NMI " + " ".join(f"{x:.3f}" for x in div[name]["adjacent_nmi"]) +
              f"; routes {div[name]['routes']}; independent {div[name]['independent']}", flush=True)
    div["perlayer"]["adjacent_code_nmi"] = [nmi(codes[:, j], codes[:, j + 1]) for j in range(M - 1)]
    res = dict(N=N, M=M, tokens=int(n), windows=int(D.starts().numel()), blocks=blocks, texts=texts, routes=routes,
               diversity=div, show_diversity=True, ablate_windows=ab["windows"], src_names=SRC_RU, type_names=TYPES_RU,
               val_ce=json.load(open(os.path.join(RUN, "analysis.json")))["val_ce"])
    json.dump(res, open(DATA_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"-> {DATA_JSON} ({os.path.getsize(DATA_JSON) / 1e3:.0f} KB)", flush=True)


def page_cmd():
    data = open(DATA_JSON, encoding="utf-8").read()
    ws = os.path.join("report", "r21_ws.json")                 # the «пробелы в коде» card (dd.r21_ws report)
    if os.path.exists(ws):
        d = json.loads(data)
        d["ws"] = json.load(open(ws, encoding="utf-8"))
        data = json.dumps(d, ensure_ascii=False)
    t = open(TEMPLATE, encoding="utf-8").read()
    assert t.count("/*DATA*/null") == 1
    data = data.replace("</", "<\\/").replace("�", "\\ufffd")   # partial UTF-8 tokens: a JSON escape, not the raw char
    open(PAGE, "w", encoding="utf-8").write(t.replace("/*DATA*/null", data))
    print(f"-> {PAGE} ({os.path.getsize(PAGE) / 1e3:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "page"])
    a = ap.parse_args()
    data_cmd() if a.cmd == "data" else page_cmd()
