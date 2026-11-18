"""Round 6, part 2: evaluation of TinyStories runs.
  python -m dd.ts_eval calib runs_ts/A_s0           -> state scale of A per block, suggested grid (delta, levels)
  python -m dd.ts_eval eval  runs_ts/<run> [A_run]  -> <run>/ts_eval.json, <run>/tok.npz, <run>/generations.json
Per-token analysis uses the first N_WIN non-overlapping val windows (identical for every run); A's per-token loss
on the same tokens is the "difficulty" reference (A must be evaluated first)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from .model import Cfg, Model
from .ts_train import load_stream

CTX, N_WIN, BS = 512, 600, 16
FUNC = set("the a an and to was of in he she it they his her with on for that is said but at so you i her him "
           "them their there this then as be had have were be not do did all one".split())
PROMPTS = ["Once upon a time, there was a little", "Lily and Tom went to the park. They",
           "One day, a big dog named Max", "The sun was hot, so Sam wanted to", "Mom said, \"",
           "Tim found a small box under his bed. Inside the box"]


def data_dir(run):
    """the token data the run was trained on (dd.ts_train --data; data_ts for the TinyStories runs)"""
    try:
        return json.load(open(os.path.join(run, "log.json")))["result"]["args"].get("data", "data_ts")
    except (OSError, KeyError, ValueError):
        return "data_ts"


def load_run(run, dev="cuda"):
    ck = torch.load(os.path.join(run, "model.pt"), map_location=dev)
    cfg = Cfg(**ck["cfg"])
    m = Model(cfg).to(dev)
    m.load_state_dict(ck["state"])
    return m.eval(), cfg


def val_starts(val):
    return torch.arange(0, val.numel() - CTX - 1, CTX, device=val.device)


def batches(val, starts):
    ar = torch.arange(CTX + 1, device=val.device)
    for i in range(0, starts.numel(), BS):
        yield val[starts[i:i + BS, None] + ar].long()


def token_types(tok, freq, ids_prev, ids):
    """Mutually exclusive categories (priority: punct > sentence start > name > function > rare > other)."""
    v = vocab_masks(tok, freq)
    alpha, cap, ends, quote = v["alpha"], v["cap"], v["ends"], v["quote"]
    sent = alpha[ids] & ((ids_prev == 1) | ends[ids_prev] | (quote[ids_prev] & cap[ids]))
    typ = np.full(ids.shape, "other", dtype=object)
    for t, mask in (("rare", v["rare"][ids]), ("function", v["func"][ids] & alpha[ids]), ("name", v["name"][ids]),
                    ("sentence_start", sent), ("punct", v["punct"][ids])):
        typ[mask] = t
    return typ


def vocab_masks(tok, freq):
    """Per-vocabulary-id boolean properties used by the token categories."""
    V = tok.get_vocab_size()
    s = [tok.decode([i]) for i in range(V)]
    st = [x.strip() for x in s]
    punct = np.array([bool(x) and not any(c.isalnum() for c in x) for x in st])
    alpha = np.array([bool(x) and x.isalpha() for x in st])
    word_start = np.array([x.startswith(" ") for x in s])
    cap = np.array([bool(x) and x[0].isupper() for x in st])
    func = np.array([x.lower() in FUNC for x in st])
    ends = np.array([x.rstrip("\"'").endswith((".", "!", "?")) for x in st])
    quote = np.array([x in ("\"", "'") for x in st])                     # opening quote before a capital = new sentence
    # rare: alphabetic tokens outside the 2000 most frequent alphabetic tokens
    order = [i for i in np.argsort(-freq) if alpha[i]]
    common = np.zeros(V, bool)
    common[order[:2000]] = True
    rare = alpha & ~common
    # name: capitalised word-start token that is not a function word and whose lower-case twin (" lily" for " Lily")
    # is at most half as frequent -- so "Let", "We", "Now" are not names, "Lily", "Max", "Tim" are
    low_freq = np.zeros(V, np.int64)
    for i in np.flatnonzero(cap & alpha & word_start):
        t = tok.id_to_token(int(i))                                       # byte-level form, e.g. 'ĠLily'
        j = tok.token_to_id(t[0] + t[1:].lower())
        low_freq[i] = freq[j] if j is not None else 0
    name = cap & alpha & word_start & ~func & (low_freq <= 0.5 * freq)
    return dict(s=s, st=st, punct=punct, alpha=alpha, word_start=word_start, cap=cap, func=func, ends=ends,
                quote=quote, rare=rare, name=name)


def auroc(score, label):
    label = label.astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def calib(run):
    model, cfg = load_run(run)
    val = load_stream(os.path.join(data_dir(run), "val.bin"), "cuda")
    st = val_starts(val)[:128]
    acc = None
    for idx in batches(val, st):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(idx, collect=True)
        xs = [x.float().flatten(0, 1) for x in aux["xs"]]
        rec = [dict(rms=float(x.pow(2).mean().sqrt()), q99=float(x.abs().quantile(0.99)) if x.numel() < 2 ** 24 else
                    float(x.abs().flatten()[::7].quantile(0.99)), q999=float(x.abs().flatten()[::7].quantile(0.999)))
               for x in xs]
        acc = rec if acc is None else [{k: a[k] + r[k] for k in a} for a, r in zip(acc, rec)]
    nb = math.ceil(st.numel() / BS)
    acc = [{k: v / nb for k, v in a.items()} for a in acc]
    for i, a in enumerate(acc):
        print(f"state {'emb' if i == 0 else f'after block {i - 1}'}: rms {a['rms']:.3f} |x| q99 {a['q99']:.3f} q99.9 {a['q999']:.3f}")
    inner = acc[1:-1]                      # outputs of blocks 0..L-2 (last block excluded, as in round 4)
    rms = float(np.mean([a["rms"] for a in inner]))
    q99 = float(np.max([a["q99"] for a in inner]))
    json.dump(dict(per_state=acc, inner_rms_mean=rms, inner_q99_max=q99), open(os.path.join(run, "calib.json"), "w"), indent=1)
    print(f"inner mean rms {rms:.3f}, max q99 {q99:.3f}")


@torch.no_grad()
def generate(model, tok, prompt, n=120, temp=0.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = [1] + tok.encode(prompt).ids
    for _ in range(n):
        x = torch.tensor([ids[-CTX:]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)[0][0, -1].float()
        nxt = int(lg.argmax()) if temp == 0 else int(torch.multinomial(F.softmax(lg / temp, -1), 1, generator=g))
        if nxt == 1:
            break
        ids.append(nxt)
    return tok.decode(ids[1:])


@torch.no_grad()
def evaluate(run, ref=None, gen=True):
    from tokenizers import Tokenizer
    model, cfg = load_run(run)
    torch.manual_seed(0)                       # round 7: random-mask controls draw their masks at evaluation too
    data = data_dir(run)
    tok = Tokenizer.from_file(os.path.join(data, "tokenizer.json"))
    freq = np.load(os.path.join(data, "freq.npy"))
    val = load_stream(os.path.join(data, "val.bin"), "cuda")
    starts = val_starts(val)
    res = dict(run=run, variant_cfg=dict(quant=cfg.quant, qg=cfg.qg, split=cfg.qg_split, parts=cfg.qg_split_parts,
                                         q_delta=cfg.q_delta, q_levels=cfg.q_levels,
                                         q_delta_layers=list(cfg.q_delta_layers), emb_scale=cfg.emb_scale,
                                         q_nogrid=cfg.q_nogrid, qg_ordered=cfg.qg_ordered,
                                         qg_mu_per_block=cfg.qg_mu_per_block, qg_mask=cfg.qg_mask,
                                         qg_static_k=list(cfg.qg_static_k), qg_randk_probs=list(cfg.qg_randk_probs),
                                         qg_k_range=list(cfg.qg_k_range), vq=cfg.vq, vq_k=cfg.vq_k, vq_norm=cfg.vq_norm,
                                         attract=cfg.attract, write_groups=[list(g) for g in cfg.write_groups],
                                         mat_weight=cfg.mat_weight, mat_strict=cfg.mat_strict,
                                         exit_weight=cfg.exit_weight))
    L, G, d = cfg.n_layers, cfg.depth_groups, cfg.d
    gs, D, K = d // G, cfg.q_delta, cfg.q_levels
    # ---------------- full validation loss (all non-overlapping windows)
    tot = n = 0.0
    T_loss, T_tok, T_prev, T_open, T_real = [], [], [], [], []
    lvl_hist = None
    max_closed = 0.0
    for bi, idx in enumerate(batches(val, starts)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, aux = model(idx, collect=bi * BS < N_WIN)
        tl = F.cross_entropy(lg[:, :-1].float().flatten(0, 1), idx[:, 1:].flatten(), reduction="none").view(idx.shape[0], -1)
        tot += float(tl.sum())
        n += tl.numel()
        if bi * BS < N_WIN:
            # per-token record for the prediction made AT position t (target t+1): the state at position t
            T_loss.append(tl.cpu().numpy().astype(np.float32))
            T_tok.append(idx.cpu().numpy().astype(np.int16))                # cur = [:, :-1], target = [:, 1:]
            if cfg.qg:
                op = torch.stack(aux["dd"], 2)[:, :-1]                     # B, T, L, G (forward value 0/1)
                rl = torch.stack(aux["rc"], 2)[:, :-1]
                bits = (op.round().to(torch.uint8) * (2 ** torch.arange(G, device=op.device, dtype=torch.uint8))).sum(-1)
                T_open.append(bits.to(torch.uint8).cpu().numpy())
                rb = (rl.round().to(torch.uint8) * (2 ** torch.arange(G, device=op.device, dtype=torch.uint8))).sum(-1)
                T_real.append(rb.to(torch.uint8).cpu().numpy())
                for l in range(L):
                    x0 = aux["xs"][l].view(*idx.shape, G, gs)
                    x1 = aux["xs"][l + 1].view(*idx.shape, G, gs)
                    cl = aux["dd"][l] == 0
                    if cl.any():
                        max_closed = max(max_closed, float((x1[cl] - x0[cl]).abs().max()))
            if cfg.quant:
                lv = torch.stack([torch.round(x.float() / cfg.qd(i)).clamp(-K, K)
                                  for i, x in enumerate(aux["xs"])]).long() + K                       # L+1,B,T,d
                h = torch.stack([torch.bincount(lv[i].flatten(), minlength=2 * K + 1) for i in range(L + 1)])
                lvl_hist = h if lvl_hist is None else lvl_hist + h
    res["val_ce"] = tot / n
    res["val_ppl"] = math.exp(res["val_ce"])
    if cfg.qg_mask in ("random", "position", "randk"):
        # round 7: the random mask is part of the evaluation -- full val CE for mask seeds 0, 1, 2
        ces = []
        for s in (0, 1, 2):
            torch.manual_seed(s)
            ts_ = ns_ = 0.0
            for idx in batches(val, starts):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_s, _ = model(idx)
                ts_ += float(F.cross_entropy(lg_s[:, :-1].float().flatten(0, 1), idx[:, 1:].flatten(), reduction="sum"))
                ns_ += idx[:, 1:].numel()
            ces.append(ts_ / ns_)
        res["val_ce_mask_seeds"] = ces
        res["val_ce_mask_mean"] = float(np.mean(ces))
        res["val_ce_mask_std"] = float(np.std(ces, ddof=1))
        torch.manual_seed(0)
    res["val_tokens"] = int(n)
    loss = np.concatenate(T_loss).ravel()
    W = np.concatenate(T_tok).astype(np.int64)                             # windows, CTX+1
    cur, nxt = W[:, :-1].ravel(), W[:, 1:].ravel()
    prv = np.concatenate([np.ones((W.shape[0], 1), np.int64), W[:, :-2]], 1).ravel()
    res["n_tok_analysis"] = int(loss.size)
    # Position t: the gate decides on the state of token cur[t]; its loss is the loss of predicting nxt[t].
    # Types are given for the predicted token (main) and for the current token.
    typ = token_types(tok, freq, cur, nxt)
    typ_in = token_types(tok, freq, prv, cur)
    types = ["sentence_start", "name", "rare", "punct", "function", "other"]
    res["type_share"] = {t: float((typ == t).mean()) for t in types}
    res["type_loss"] = {t: float(loss[typ == t].mean()) for t in types}
    save = dict(loss=loss.astype(np.float16), ids=nxt.astype(np.int16), cur=cur.astype(np.int16),
                typ=np.array([types.index(t) for t in typ], np.int8),
                typ_in=np.array([types.index(t) for t in typ_in], np.int8))
    ref_loss = None
    if ref and os.path.exists(os.path.join(ref, "tok.npz")):
        rz = np.load(os.path.join(ref, "tok.npz"))
        assert np.array_equal(rz["ids"], save["ids"])
        ref_loss = rz["loss"].astype(np.float32)
    if cfg.quant:
        hist = lvl_hist.cpu().numpy().astype(np.float64)
        p = hist / hist.sum(1, keepdims=True)
        res["levels"] = dict(edge_share=[float(r[0] + r[-1]) for r in p], zero_share=[float(r[K]) for r in p],
                             levels_99=[int((np.cumsum(np.sort(r)[::-1]) < 0.99).sum() + 1) for r in p],
                             hist=p.tolist())
    if cfg.qg:
        ob = np.concatenate(T_open).reshape(-1, L)
        rb = np.concatenate(T_real).reshape(-1, L)
        opn = np.unpackbits(ob[..., None], axis=-1, bitorder="little")[..., :G].astype(np.float32)   # N, L, G
        rea = np.unpackbits(rb[..., None], axis=-1, bitorder="little")[..., :G].astype(np.float32)
        save["open_bits"], save["real_bits"] = ob, rb
        tot_open = opn.sum((1, 2))
        depth = opn.sum(1)                                       # N, G: per-group depth after the last block
        res["qg"] = dict(
            paid_layer=opn.mean((0, 2)).tolist(), real_layer=rea.mean((0, 2)).tolist(),
            wasted_of_paid_layer=((opn - rea).sum((0, 2)) / np.maximum(opn.sum((0, 2)), 1)).tolist(),
            token_none_open_layer=(opn.sum(2) == 0).mean(0).tolist(),
            group_rate=opn.mean(0).tolist(),
            never_open_layer=(opn.mean(0) < 0.01).mean(1).tolist(), always_open_layer=(opn.mean(0) > 0.99).mean(1).tolist(),
            max_abs_change_closed_group=max_closed,
            open_total_mean=float(tot_open.mean()), depth_max_mean=float(depth.max(1).mean()),
            by_type={t: dict(open_total=float(tot_open[typ == t].mean()),
                             open_layer=opn[typ == t].mean((0, 2)).tolist()) for t in types},
            by_type_current={t: float(tot_open[typ_in == t].mean()) for t in types},
            spearman_open_vs_own_loss=float(spearmanr(tot_open, loss).statistic),
            auroc_open_top10_own=auroc(tot_open, loss >= np.quantile(loss, 0.9)),
            auroc_depthmax_top10_own=auroc(depth.max(1), loss >= np.quantile(loss, 0.9)),
        )
        if ref_loss is not None:
            res["qg"]["spearman_open_vs_A_loss"] = float(spearmanr(tot_open, ref_loss).statistic)
            res["qg"]["spearman_open_vs_A_loss_within_type"] = {
                t: float(spearmanr(tot_open[typ == t], ref_loss[typ == t]).statistic) for t in types}
            res["qg"]["auroc_open_top10_A"] = auroc(tot_open, ref_loss >= np.quantile(ref_loss, 0.9))
            qs = np.quantile(ref_loss, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(ref_loss, qs)
            res["qg"]["open_by_A_loss_quintile"] = [float(tot_open[bins == i].mean()) for i in range(5)]
        # train vs eval forward identical (deterministic gate)
        idx = next(batches(val, starts[:4]))
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lt = model(idx)[0]
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            le = model(idx)[0]
        res["qg"]["train_eval_max_abs_logit_diff"] = float((lt.float() - le.float()).abs().max())
        res["mu_final"] = model.mu.tolist()
    if ref_loss is not None:
        res["spearman_own_vs_A_loss"] = float(spearmanr(loss, ref_loss).statistic)
    np.savez_compressed(os.path.join(run, "tok.npz"), **save)
    # ---------------- generations (identical prompts; greedy and temperature 0.8)
    if gen:
        gens = [dict(prompt=p, greedy=generate(model, tok, p), temp08=generate(model, tok, p, temp=0.8, seed=i))
                for i, p in enumerate(PROMPTS)]
        json.dump(gens, open(os.path.join(run, "generations.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(res, open(os.path.join(run, "ts_eval.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("levels",)}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x != "--nogen"]
    if args[0] == "calib":
        calib(args[1])
    else:
        evaluate(args[1], args[2] if len(args) > 2 else None, gen="--nogen" not in sys.argv)
