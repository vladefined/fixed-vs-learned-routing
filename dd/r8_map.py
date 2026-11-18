"""Round 8: where does the model spend its computation?  Finished checkpoints only (A_s0, QG25_s0, QG50_s0), no
training.  The analysis windows are those of dd.ts_eval (the first 608 non-overlapping 512-token val windows); row t
of tok.npz = the state of token cur[t], whose loss is the loss of predicting the next token.
  python -m dd.r8_map lens    -> runs_ts/A_s0/lens.npz: A per token and block ||dx|| / ||x|| and the logit-lens top-1
                                 (final RMSNorm + tied unembedding) after the embedding and every block
  python -m dd.r8_map causal  -> runs_ts/QG50_s0/causal.json: in every block close half of the open groups (chosen at
                                 random among the open ones) on 20% of the tokens of every window:
                                 top = largest depth, bottom = smallest, random; ctx_top = largest depth relative to
                                 the token's own mean (context part); lowgate = same tokens and count as random, but the
                                 open groups with the lowest gate logit
  python -m dd.r8_map top3    -> runs_ts/<run>/top3.npz: top-3 next-token predictions of A, QG50, QG25
  python -m dd.r8_map viz     -> report/r8_viz_data.json: 30 val stories (colour of a generated token = the cost of
                                 the position that generated it), contrasts of frequent tokens, extremes
  python -m dd.r8_map report  -> report/r8_map.md"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from .ts_eval import BS, CTX, N_WIN, auroc, load_run, val_starts
from .ts_train import load_stream

RUNS = "runs_ts"
VAST = "runs_vast"
N_ANALYSIS = ((N_WIN + BS - 1) // BS) * BS          # 608 windows, as in tok.npz
FRAC = 0.2
MIN_SEEN = 20


def windows(dev="cuda"):
    val = load_stream(os.path.join("data_ts", "val.bin"), dev)
    st = val_starts(val)[:N_ANALYSIS]
    ar = torch.arange(CTX + 1, device=dev)
    return val[st[:, None] + ar].long()                                         # 608, 513


def tokz(name):
    return np.load(os.path.join(RUNS, name, "tok.npz"))


def bits_to_counts(ob, G=8):
    return np.unpackbits(ob[..., None], axis=-1, bitorder="little")[..., :G].sum(-1).astype(np.float32)   # N, L


def tok_mean(x, cur, V=8192):
    return (np.bincount(cur, x, V) / np.maximum(np.bincount(cur, minlength=V), 1))[cur]


def spear(a, b):
    return float(spearmanr(a, b).statistic)


# ------------------------------------------------------------------------------------------------ lens (A)
@torch.no_grad()
def lens():
    model, cfg = load_run(os.path.join(RUNS, "A_s0"))
    W = windows()
    L = cfg.n_layers
    REL, TOP, LOSS = [], [], []
    for i in range(0, W.shape[0], BS):
        idx = W[i:i + BS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, aux = model(idx, collect=True)
        xs = [x[:, :-1].float() for x in aux["xs"]]                               # L+1 states, B, T, d
        rel = torch.stack([(xs[l + 1] - xs[l]).norm(dim=-1) / xs[l].norm(dim=-1).clamp_min(1e-6) for l in range(L)], -1)
        top = []
        for x in xs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                top.append((model.nf(x) @ model.emb.weight.t()).argmax(-1))
        REL.append(rel.cpu().numpy().astype(np.float32))
        TOP.append(torch.stack(top, -1).cpu().numpy().astype(np.int16))
        LOSS.append(F.cross_entropy(lg[:, :-1].float().flatten(0, 1), idx[:, 1:].flatten(), reduction="none")
                    .view(idx.shape[0], -1).cpu().numpy())
    rel = np.concatenate(REL).reshape(-1, L)
    top = np.concatenate(TOP).reshape(-1, L + 1)
    loss = np.concatenate(LOSS).ravel()
    chg = top[:, 1:] != top[:, :-1]                                              # N, L: block l changes the top-1
    last = np.where(chg.any(1), L - np.argmax(chg[:, ::-1], 1), 0)               # 1..L = last block that changes it
    z = tokz("A_s0")
    np.savez_compressed(os.path.join(RUNS, "A_s0", "lens.npz"), rel=rel.astype(np.float16), top=top, chg=chg,
                        last=last.astype(np.int8))
    print(f"lens: loss vs tok.npz max |diff| {np.abs(loss - z['loss'].astype(np.float32)).max():.3g}; "
          f"top-1 of the last state == argmax? {(top[:, -1] >= 0).all()}; mean last-change block {last.mean():.2f}; "
          f"rel by block {np.round(rel.mean(0), 3).tolist()}", flush=True)


# ------------------------------------------------------------------------------------------------ causal (QG50)
@torch.no_grad()
def causal(run="QG50_s0", seeds=(0, 1, 2)):
    model, cfg = load_run(os.path.join(RUNS, run))
    L, G = cfg.n_layers, cfg.depth_groups
    W = windows()
    Nw, T1 = W.shape
    T = T1 - 1
    z, za = tokz(run), tokz("A_s0")
    a_loss = za["loss"].astype(np.float32)
    st = {}

    def hook(blk, m, zg):
        if st.get("sel") is None:
            return m
        l = blk.l
        opn = m > 0.5                                                             # B, T1, G
        k = opn.sum(-1)
        n = k // 2 + ((k % 2 == 1) & st["coin"][:, :, l]).long()                  # half of the open ones
        if st["mode"] == "low":
            score = zg.float()
        else:
            score = torch.rand(m.shape, device=m.device, generator=st["gen"])
        score = torch.where(opn, score, torch.full_like(score, float("inf")))
        rank = score.argsort(-1).argsort(-1)
        close = opn & (rank < n[..., None]) & st["sel"][..., None]
        st["closed"][:, :, l] += close.sum(-1)
        return m * (~close).to(m.dtype)

    def run_pass(sel=None, mode="rand", seed=0):
        """sel: Nw x T1 bool (cpu) or None.  Returns per-position loss (Nw*T), open counts (Nw*T, L), closed (Nw*T, L)."""
        gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
        cg = torch.Generator(device="cuda").manual_seed(2000 + seed)
        LOSS, OPN, CL = [], [], []
        for i in range(0, Nw, BS):
            idx = W[i:i + BS]
            B = idx.shape[0]
            st.update(sel=None if sel is None else sel[i:i + BS].cuda(), mode=mode, gen=gen,
                      coin=torch.rand(B, T1, L, device="cuda", generator=cg) < 0.5,
                      closed=torch.zeros(B, T1, L, dtype=torch.long, device="cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, aux = model(idx, ctl={"hc_gate_fn": hook})
            LOSS.append(F.cross_entropy(lg[:, :-1].float().flatten(0, 1), idx[:, 1:].flatten(), reduction="none")
                        .view(B, -1).cpu().numpy())
            OPN.append(torch.stack(aux["dd"], 2)[:, :-1].round().sum(-1).cpu().numpy())
            CL.append(st["closed"][:, :-1].cpu().numpy())
        return (np.concatenate(LOSS).ravel(), np.concatenate(OPN).reshape(-1, L),
                np.concatenate(CL).reshape(-1, L).astype(np.float32))

    base_loss, base_open, _ = run_pass()
    stored = bits_to_counts(z["open_bits"], G)
    depth = base_open.sum(1)                                                      # N: open groups over all blocks
    cur = z["cur"].astype(np.int64)
    ctx_depth = depth - tok_mean(depth, cur)
    rng = np.random.default_rng(0)
    k = int(round(FRAC * T))

    def pick(score):
        """per window: the k positions with the largest score (ties at random); position T (no target) never"""
        s = score.reshape(Nw, T) + 1e-6 * rng.random((Nw, T))
        sel = np.zeros((Nw, T1), bool)
        top = np.argsort(-s, 1)[:, :k]
        np.put_along_axis(sel, top, True, 1)
        return torch.from_numpy(sel)

    qb = np.quantile(a_loss, [0.2, 0.4, 0.6, 0.8])
    abin = np.digitize(a_loss, qb)

    def summarise(name, sel, loss, opn, closed):
        s = sel[:, :T].numpy().ravel()
        d = loss - base_loss
        cl = closed.sum(1)
        out = dict(name=name, n_touched=int(s.sum()),
                   dce_all=float(d.mean()), dce_touched=float(d[s].mean()), dce_untouched=float(d[~s].mean()),
                   closed_groups=int(cl.sum()), closed_per_touched=float(cl[s].mean()),
                   closed_share_of_open=float(cl[s].sum() / max(base_open[s].sum(), 1)),
                   touched_none_closed=float((cl[s] == 0).mean()),
                   touched_depth=float(depth[s].mean()), touched_A_loss=float(a_loss[s].mean()),
                   touched_base_loss=float(base_loss[s].mean()),
                   dce_per_closed_group=float(d[s].sum() / max(cl.sum(), 1)),
                   dce_total_per_closed_group=float(d.sum() / max(cl.sum(), 1)),
                   by_A_bin=[dict(bin=b, n=int((s & (abin == b)).sum()),
                                  dce=float(d[s & (abin == b)].mean()) if (s & (abin == b)).any() else None,
                                  closed=float(cl[s & (abin == b)].mean()) if (s & (abin == b)).any() else None)
                             for b in range(5)])
        print(f"{name:>14}: dCE all {out['dce_all']:+.4f} touched {out['dce_touched']:+.4f} untouched "
              f"{out['dce_untouched']:+.4f}; closed {out['closed_per_touched']:.2f}/touched token "
              f"({100 * out['closed_share_of_open']:.0f}% of open); depth {out['touched_depth']:.1f}", flush=True)
        return out

    res = dict(run=run, frac=FRAC, n_windows=Nw, n_tok=int(base_loss.size),
               check=dict(base_open_equal_tok_npz=float((base_open == stored).mean()),
                          base_loss_max_abs_diff_tok_npz=float(np.abs(base_loss - z["loss"].astype(np.float32)).max()),
                          base_ce=float(base_loss.mean())),
               A_bins=qb.tolist(), conditions=[])
    print(f"baseline: CE {base_loss.mean():.4f}; open counts equal tok.npz {res['check']['base_open_equal_tok_npz']:.4f}",
          flush=True)
    plan = [("top", pick(depth), "rand", 0), ("bottom", pick(-depth), "rand", 0), ("ctx_top", pick(ctx_depth), "rand", 0)]
    for s in seeds:
        sel = pick(rng.random(depth.size))
        plan += [(f"random_s{s}", sel, "rand", s), (f"lowgate_s{s}", sel, "low", s)]
    for name, sel, mode, s in plan:
        loss, opn, closed = run_pass(sel, mode, s)
        res["conditions"].append(summarise(name, sel, loss, opn, closed))
    json.dump(res, open(os.path.join(RUNS, run, "causal.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ honesty stats
def honesty():
    zl = np.load(os.path.join(RUNS, "A_s0", "lens.npz"))
    rel, last, chg = zl["rel"].astype(np.float32), zl["last"].astype(np.float32), zl["chg"]
    za = tokz("A_s0")
    cur = za["cur"].astype(np.int64)
    a_loss = za["loss"].astype(np.float32)
    rel_tot = rel[:, 1:].sum(1)        # blocks 2..8: block 1 is relative to the embedding, whose norm is tiny (rel ~29)
    out = {}
    cnt = np.bincount(cur, minlength=8192)
    seen = cnt[cur] >= MIN_SEEN

    def r2_token(x):
        r = x - tok_mean(x, cur)
        return float(1 - r[seen].var() / x[seen].var())

    out["A"] = dict(rel_block_mean=rel.mean(0).tolist(), last_change_mean=float(last.mean()),
                    last_change_hist=np.bincount(last.astype(int), minlength=9).tolist(),
                    change_rate_block=chg.mean(0).tolist(),
                    r2_token=dict(rel_total=r2_token(rel_tot), last_change=r2_token(last), A_loss=r2_token(a_loss)))
    for run in ("QG50_s0", "QG25_s0"):
        c = bits_to_counts(tokz(run)["open_bits"])                               # N, L
        tot = c.sum(1)
        wt = lambda x: x - tok_mean(x, cur)                                       # noqa: E731
        o = dict(r2_token_open_total=r2_token(tot))
        o["all"] = dict(rel_total=spear(tot, rel_tot), last_change=spear(tot, last), A_loss=spear(tot, a_loss))
        o["within_token"] = dict(rel_total=spear(wt(tot), wt(rel_tot)), last_change=spear(wt(tot), wt(last)),
                                 A_loss=spear(wt(tot), wt(a_loss)))
        tm = lambda x: np.bincount(cur, x, 8192)[cnt >= MIN_SEEN] / cnt[cnt >= MIN_SEEN]     # noqa: E731
        o["token_means"] = dict(rel_total=spear(tm(tot), tm(rel_tot)), last_change=spear(tm(tot), tm(last)),
                                A_loss=spear(tm(tot), tm(a_loss)))
        o["block"] = [dict(rel=spear(c[:, l], rel[:, l]), rel_within=spear(wt(c[:, l]), wt(rel[:, l])),
                           auroc_A_top1_changes=auroc(c[:, l], chg[:, l]),
                           auroc_within=auroc(wt(c[:, l]), chg[:, l])) for l in range(c.shape[1])]
        # does QG open more in the block where A's top-1 settles?  open count of the token in that block vs its mean
        has = last > 0
        li = np.clip(last.astype(int) - 1, 0, c.shape[1] - 1)
        o["open_in_settle_block_minus_token_mean_block"] = float((c[np.arange(len(c)), li] - c.mean(1))[has].mean())
        # cross-block: QG block l vs A block j (rank corr matrix)
        o["block_matrix"] = [[spear(c[:, l], rel[:, j]) for j in range(rel.shape[1])] for l in range(c.shape[1])]
        out[run] = o
    json.dump(out, open(os.path.join(RUNS, "A_s0", "honesty.json"), "w"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------ top-3 predictions
@torch.no_grad()
def top3(runs=("A_s0", "QG50_s0", "QG25_s0")):
    """top-3 next-token predictions (ids, probabilities) at every analysis position -> runs_ts/<run>/top3.npz"""
    W = windows()
    for run in runs:
        model, _ = load_run(os.path.join(RUNS, run))
        I, P = [], []
        for i in range(0, W.shape[0], BS):
            idx = W[i:i + BS]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(idx)[0][:, :-1]
            p, ix = F.softmax(lg.float(), -1).topk(3, -1)
            I.append(ix.cpu().numpy().astype(np.int16))
            P.append(p.cpu().numpy().astype(np.float16))
        np.savez_compressed(os.path.join(RUNS, run, "top3.npz"), ids=np.concatenate(I).reshape(-1, 3),
                            p=np.concatenate(P).reshape(-1, 3))
        print(f"top3 {run}: done", flush=True)


# ------------------------------------------------------------------------------------------------ viz data
def viz(n_st=30, lo=60, hi=260, out="report", n_ctx=12, n_freq=20, n_ext=40, min_A_loss=0.1, seed=0):
    """Generated-token view: the colour of token t+1 = the computation at position t that generated it.
    Row r of tok.npz: cur[r] is the input token, nxt[r] the generated one; its cost = open groups at row r."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join("data_ts", "tokenizer.json"))
    ld = lambda z: {k: z[k] for k in z.files}                                     # noqa: E731  (NpzFile re-reads on every access)
    za, z25, z50 = ld(tokz("A_s0")), ld(tokz("QG25_s0")), ld(tokz("QG50_s0"))
    zl = ld(np.load(os.path.join(RUNS, "A_s0", "lens.npz")))
    tA, t50, t25 = (ld(np.load(os.path.join(RUNS, r, "top3.npz"))) for r in ("A_s0", "QG50_s0", "QG25_s0"))
    zsi, zst = ld(np.load(os.path.join(VAST, "QG25staticinit_s0", "tok.npz"))), ld(np.load(os.path.join(VAST, "QG25static_s0", "tok.npz")))
    abl = {}                                                     # round 9: causal maps (dd.r9_diag ablate), row -> index
    for name, base in (("A", RUNS), ("static", VAST)):
        pth = os.path.join(base, "A_s0" if name == "A" else "QG25static_s0", "ablate.npz")
        if os.path.exists(pth):
            zz = ld(np.load(pth))
            abl[name] = (dict(zip(zz["rows"].tolist(), range(zz["rows"].size))), zz)
    assert np.array_equal(zsi["ids"], za["ids"]) and np.array_equal(zst["ids"], za["ids"])
    cur, nxt = za["cur"].astype(np.int64), za["ids"].astype(np.int64)
    N = cur.size
    pos = np.arange(N) % CTX
    win = np.arange(N) // CTX
    # story start inside the window: index of the last EOT row at or before r (-1 if the story began before the window)
    last_eot = np.full(N, -1)
    le = -1
    for r in range(N):
        if pos[r] == 0:
            le = -1
        if cur[r] == 1:
            le = r
        last_eot[r] = le
    full_ctx = (last_eot >= 0) & (cur != 1)                  # story start visible, not the story's first token
    cost50 = bits_to_counts(z50["open_bits"]).sum(1).astype(int)
    cost25 = bits_to_counts(z25["open_bits"]).sum(1).astype(int)
    la = za["loss"].astype(np.float64)
    used = set()

    def r3(a):
        return np.round(np.asarray(a, np.float64), 3).tolist()

    def top(z, r):
        used.update(int(i) for i in z["ids"][r])
        return [z["ids"][r].astype(int).tolist(), r3(z["p"][r])]

    def example(r):
        s = max(last_eot[r] + 1, r - n_ctx + 1)              # context: cur[s..r] (same story), generated nxt[r]
        ctx = cur[s:r + 1].astype(int).tolist()
        used.update(ctx)
        used.add(int(nxt[r]))
        return dict(ctx=ctx, cut=bool(s > last_eot[r] + 1), tok=int(nxt[r]), c50=int(cost50[r]), c25=int(cost25[r]),
                    b50=z50["open_bits"][r].astype(int).tolist(), la=round(float(la[r]), 3),
                    l50=round(float(z50["loss"][r]), 3), a3=top(tA, r), q3=top(t50, r))

    # ---------------- stories: complete stories inside one window (rows p .. q-1, cur[p] = EOT, nxt[q-1] = EOT)
    eot = np.flatnonzero(cur == 1)
    cands = []
    for p in eot:
        q = p + 1
        while q < N and win[q] == win[p] and cur[q] != 1:
            q += 1
        if q < N and win[q] == win[p] and nxt[q - 1] == 1 and lo <= q - p <= hi:
            cands.append((p, q))
    pick = [cands[i] for i in np.linspace(0, len(cands) - 1, n_st).round().astype(int)]
    stories = []
    for p, q in pick:
        r = np.arange(p, q - 1)                                # generating rows of the displayed tokens nxt[p .. q-2]
        used.update(int(i) for i in nxt[r])
        stories.append(dict(
            ids=nxt[r].astype(int).tolist(), o25=z25["open_bits"][r].astype(int).tolist(),
            o50=z50["open_bits"][r].astype(int).tolist(), la=r3(za["loss"][r]), l25=r3(z25["loss"][r]),
            l50=r3(z50["loss"][r]), rel=r3(zl["rel"][r]), last=zl["last"][r].astype(int).tolist(),
            a3=[top(tA, i) for i in r], q3=[top(t50, i) for i in r], g3=[top(t25, i) for i in r],
            osi=zsi["open_bits"][r].astype(int).tolist(), lsi=r3(zsi["loss"][r]), lst=r3(zst["loss"][r]),
            **({} if "A" not in abl else dict(
                ca_own=[r3(abl["A"][1]["own"][abl["A"][0][int(i)]]) for i in r],        # 8 blocks x (attn, FFN, whole)
                ca_seq=[r3(abl["A"][1]["seq"][abl["A"][0][int(i)]]) for i in r])),
            **({} if "static" not in abl else dict(
                cs_seq=[r3(abl["static"][1]["seq"][abl["static"][0][int(i)]][:, 2]) for i in r]))))
    rng = np.random.default_rng(seed)
    tie = rng.random(N)

    def pick_rows(rows, key, k):
        """k rows with the smallest key (ties at random), from different windows, without repeated context + token"""
        rows = rows[np.lexsort((tie[rows], key[rows]))]
        out, seen, texts = [], set(), set()
        for r in rows:
            t = (tuple(cur[max(last_eot[r] + 1, r - n_ctx + 1):r + 1]), int(nxt[r]))   # no repeated context + token
            if win[r] not in seen and t not in texts:
                out.append(int(r))
                seen.add(win[r])
                texts.add(t)
            if len(out) == k:
                break
        return out

    # ---------------- contrasts: frequent generated tokens, cheapest / dearest appearances under QG50
    want = [" the", " a", " he", " she", " was", " and", ".", ",", " He", " She", " to", " it"]
    fixed = [tok.token_to_id(tok.id_to_token(tok.encode(w).ids[0])) for w in want if len(tok.encode(w).ids) == 1]
    cnt = np.bincount(nxt[full_ctx], minlength=8192)
    cnt[1] = 0
    order = [int(i) for i in np.argsort(-cnt) if cnt[i] > 0]
    chosen = list(dict.fromkeys(fixed + order))[:n_freq]
    chosen.sort(key=lambda v: -cnt[v])
    contrasts = []
    for v in chosen:
        rows = np.flatnonzero(full_ctx & (nxt == v))
        c = cost50[rows]
        contrasts.append(dict(tok=int(v), n=int(rows.size), mean=round(float(c.mean()), 2),
                              q=[int(x) for x in np.percentile(c, [5, 50, 95])],
                              cheap=[example(r) for r in pick_rows(rows, cost50, 3)],
                              dear=[example(r) for r in pick_rows(rows, -cost50, 3)]))
        used.add(int(v))
    # ---------------- extremes over all generated tokens with A loss >= min_A_loss
    rows = np.flatnonzero(full_ctx & (la >= min_A_loss) & (nxt != 1))
    extremes = dict(min_A_loss=min_A_loss, n=int(rows.size),
                    n_at_min=int((cost50[rows] == cost50[rows].min()).sum()),
                    n_at_max=int((cost50[rows] == cost50[rows].max()).sum()),
                    cmin=int(cost50[rows].min()), cmax=int(cost50[rows].max()),
                    dear=[example(r) for r in pick_rows(rows, -cost50, n_ext)],
                    cheap=[example(r) for r in pick_rows(rows, cost50, n_ext)])
    dec = {}
    for i in sorted(used):
        dec[str(i)] = tok.decode([i]) if i != 1 else "¶"
    meta = dict(n_candidates=len(cands), n_rows=int(N), mean_cost50=float(cost50[full_ctx].mean()),
                mean_cost25=float(cost25[full_ctx].mean()), n_ctx=n_ctx, static_bits=static_bits().astype(int).tolist())
    json.dump(dict(meta=meta, vocab=dec, stories=stories, contrasts=contrasts, extremes=extremes),
              open(os.path.join(out, "r8_viz_data.json"), "w", encoding="utf-8"), ensure_ascii=False,
              separators=(",", ":"))
    print(f"viz: {len(stories)} stories ({sum(len(s['ids']) for s in stories)} tokens), {len(contrasts)} contrast "
          f"tokens, extremes from {rows.size} rows (cost {extremes['cmin']}..{extremes['cmax']}, "
          f"{extremes['n_at_min']} at min, {extremes['n_at_max']} at max)", flush=True)


# ------------------------------------------------------------------------------------------------ deviation from static
def static_bits(L=8, G=8, s=2):
    return np.array([sum(1 << ((s * l + j) % G) for j in range(s)) for l in range(L)], np.uint8)


def deviation(run="QG25staticinit_s0", base=VAST, min_count=200):
    """how far a staticinit model's mask moved from the static start: per decision (token x block x group)"""
    from tokenizers import Tokenizer
    from .ts_eval import vocab_masks
    from .ts_adapt import story_pos, categories
    z = np.load(os.path.join(base, run, "tok.npz"))
    za = tokz("A_s0")
    ob = z["open_bits"]                                                           # N, L
    N, L = ob.shape
    G = 8
    sb = static_bits(L, G)
    opn = np.unpackbits(ob[..., None], axis=-1, bitorder="little")[..., :G].astype(bool)          # N, L, G
    sta = np.unpackbits(sb[None, :, None], axis=-1, bitorder="little")[..., :G].astype(bool)      # 1, L, G
    extra = opn & ~sta                                                            # opened a non-static group
    drop = ~opn & sta                                                             # closed a static group
    dev = (extra | drop).sum((1, 2))                                              # per position, 0..64
    cur, nxt = za["cur"].astype(np.int64), za["ids"].astype(np.int64)
    a_loss = za["loss"].astype(np.float32)
    tok = Tokenizer.from_file(os.path.join("data_ts", "tokenizer.json"))
    v = vocab_masks(tok, np.load(os.path.join("data_ts", "freq.npy")))
    W = np.concatenate([cur.reshape(-1, CTX), nxt.reshape(-1, CTX)[:, -1:]], 1)
    pos_w = story_pos(W)
    prv = np.concatenate([np.full((W.shape[0], 1), -1), W[:, :-2]], 1).ravel()
    pos_cur, pos_nxt = pos_w[:, :-1].ravel(), pos_w[:, 1:].ravel()
    cats = dict(input=categories(v, np.where(prv < 0, 1, prv), cur, pos_cur), generated=categories(v, cur, nxt, pos_nxt))
    res = dict(run=run, n=int(N),
               share_diff=float((extra | drop).mean()), share_extra=float(extra.mean()), share_drop=float(drop.mean()),
               open_share=float(opn.mean()), tokens_any_diff=float((dev > 0).mean()), dev_mean=float(dev.mean()),
               dev_hist=np.bincount(dev, minlength=17)[:17].tolist(),
               block=[dict(extra=float(extra[:, l].mean() * G / (G - sta[0, l].sum())), drop=float(drop[:, l].mean() * G / sta[0, l].sum()),
                           diff=float((extra | drop)[:, l].mean())) for l in range(L)],
               group_rate=opn.mean(0).round(4).tolist(), static=sta[0].astype(int).tolist(),
               rho_dev_A_loss=spear(dev, a_loss), rho_dev_A_loss_within=spear(dev - tok_mean(dev, cur), a_loss - tok_mean(a_loss, cur)))
    cnt = np.bincount(cur, minlength=8192)
    res["r2_token"] = float(1 - (dev - tok_mean(dev.astype(float), cur))[cnt[cur] >= 20].var() / dev[cnt[cur] >= 20].var())
    res["by_category"] = {view: {k: dict(share=float(m.mean()), dev=float(dev[m].mean()), extra=float(extra[m].sum((1, 2)).mean()),
                                         drop=float(drop[m].sum((1, 2)).mean())) for k, m in cm.items()} for view, cm in cats.items()}
    bins = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100), (101, 10 ** 6)]
    res["by_story_pos"] = [dict(lo=lo, hi=hi, share=float(((pos_cur >= lo) & (pos_cur <= hi)).mean()),
                                dev=float(dev[(pos_cur >= lo) & (pos_cur <= hi)].mean())) for lo, hi in bins]
    for key, ids in (("input", cur), ("generated", nxt)):
        c = np.bincount(ids, minlength=8192)
        m = np.bincount(ids, dev, 8192) / np.maximum(c, 1)
        ok = np.flatnonzero(c >= min_count)
        top = ok[np.argsort(-m[ok])][:15]
        low = ok[np.argsort(m[ok])][:15]
        res[f"tokens_{key}"] = dict(most=[(tok.decode([int(i)]), int(c[i]), round(float(m[i]), 2)) for i in top],
                                    least=[(tok.decode([int(i)]), int(c[i]), round(float(m[i]), 2)) for i in low])
    json.dump(res, open(os.path.join(base, run, "deviation.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{run}: decisions differing from static {100 * res['share_diff']:.1f}% (extra {100 * res['share_extra']:.1f}%, "
          f"dropped {100 * res['share_drop']:.1f}%), tokens with any difference {100 * res['tokens_any_diff']:.1f}%, "
          f"mean {res['dev_mean']:.2f}/64; rho(dev, A loss) {res['rho_dev_A_loss']:.2f}", flush=True)
    return res



# ------------------------------------------------------------------------------------------------ staticinit / hybrid
def staticinit_tables():
    from .report_r7 import pooled_sd, seeds, stats, verdict
    rows = [("QG25", "обученный гейт", seeds(RUNS, "QG25")),
            ("QG25-static", "2 из 8 групп на блок, фиксированные", seeds(RUNS, "QG25static")),
            ("QG25-noramp", "гейт QG25, цель 25% с первого шага", seeds(RUNS, "QG25noramp")),
            ("QG25-staticinit", "гейт QG25, старт с маски static", seeds(RUNS, "QG25staticinit")),
            ("QG25-hybrid", "группа 2l всегда открыта + гейт на остальных", seeds(RUNS, "QG25hybrid")),
            ("QG25-ctrl", "гейт QG25 + поток управления c ∈ ℝ⁶⁴ без сетки", seeds(RUNS, "QG25ctrl"))]
    st = {n: stats(rs) for n, _, rs in rows if rs}
    L = ["## 3. Старт со static и гибрид (20k шагов, полный val)\n",
         "Сравнение — по среднему по сидам обучения; 2 × SD — порог «обнаружен» (SD объединённое; при одном сиде — SD "
         "второй группы).\n",
         "| прогон | маска | сидов | val CE: среднее ± SD | по сидам | − static | 2 × SD | вывод | − QG25 | 2 × SD | вывод |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, desc, rs in rows:
        if not rs:
            L.append(f"| {n} | {desc} | 0 | — | — | — | — | — | — | — | — |")
            continue
        m, sd, k = st[n]
        cells = []
        for ref in ("QG25-static", "QG25"):
            if ref == n or ref not in st:
                cells += ["—", "—", "—"]
                continue
            d = m - st[ref][0]
            thr = 2 * pooled_sd(st[n], st[ref])
            cells += [f"{d:+.4f}", "—" if np.isnan(thr) else f"{thr:.4f}", verdict(d, thr, f"{n} хуже", f"{n} лучше")]
        L.append(f"| {n} | {desc} | {k} | {m:.4f}" + (f" ± {sd:.4f}" if k > 1 else "") + " | " +
                 " / ".join(f"{r['ce']:.4f}" for r in rs) + " | " + " | ".join(cells) + " |")
    L.append("")
    devs = [json.load(open(os.path.join(r["run"], "deviation.json"))) for r in seeds(RUNS, "QG25staticinit")
            if os.path.exists(os.path.join(r["run"], "deviation.json"))]
    if devs:
        L += ["### Насколько маска QG25-staticinit ушла от стартовой (static)\n",
              "Решение = (позиция × блок × группа). «Лишнее открытие» — открыта группа вне static; «закрыта static» — "
              "закрыта одна из двух групп static этого блока. Доли «по блокам» — от числа таких групп в блоке "
              "(6 не-static и 2 static).\n",
              "| сид | решений не как static | лишние открытия | закрытые static | позиций хоть с одним отличием | "
              "отличий на позицию (из 64) | ρ(отличия, потеря A) | то же внутри токена | R² токена |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in devs:
            L.append(f"| {d['run'].rsplit('_', 1)[1]} | {100 * d['share_diff']:.1f}% | {100 * d['share_extra']:.1f}% | "
                     f"{100 * d['share_drop']:.1f}% | {100 * d['tokens_any_diff']:.1f}% | {d['dev_mean']:.2f} | "
                     f"{d['rho_dev_A_loss']:.2f} | {d['rho_dev_A_loss_within']:.2f} | {d['r2_token']:.2f} |")
        d = devs[0]
        L += ["", f"По блокам ({d['run']}): доля открытых не-static групп / доля закрытых static групп\n",
              "| блок | " + " | ".join(str(l + 1) for l in range(len(d["block"]))) + " |",
              "|---|" + "---|" * len(d["block"]),
              "| лишние открытия | " + " | ".join(f"{100 * b['extra']:.1f}%" for b in d["block"]) + " |",
              "| закрытые static | " + " | ".join(f"{100 * b['drop']:.1f}%" for b in d["block"]) + " |", "",
              "По позиции в истории (отличий на позицию): " + "; ".join(
                  f"{b['lo']}–{b['hi'] if b['hi'] < 10 ** 6 else '…'}: {b['dev']:.2f}" for b in d["by_story_pos"]), "",
              "По типу токена (среднее число отличий; вход — токен на входе позиции, сгенерированный — её выход):\n",
              "| тип | вход: доля | вход: отличий | сгенерированный: отличий |", "|---|---|---|---|"]
        for k in d["by_category"]["input"]:
            L.append(f"| {k} | {100 * d['by_category']['input'][k]['share']:.1f}% | {d['by_category']['input'][k]['dev']:.2f} | "
                     f"{d['by_category']['generated'].get(k, {}).get('dev', float('nan')):.2f} |")
        for key, lab in (("input", "на входе"), ("generated", "сгенерированные")):
            t = d[f"tokens_{key}"]
            L += ["", f"Токены {lab} (≥ 200 раз), больше всего отличий: " + ", ".join(f"«{x[0]}» {x[2]}" for x in t["most"][:10]),
                  "меньше всего: " + ", ".join(f"«{x[0]}» {x[2]}" for x in t["least"][:10])]
        L.append("")
    return L


# ------------------------------------------------------------------------------------------------ Wikipedia vs TinyStories
def wiki(out="report"):
    """val CE and the gap to A of Q0 / static / QG25 on Wikipedia next to TinyStories; adaptivity of QG25 on both"""
    def ev(path):
        p = os.path.join(path, "ts_eval.json")
        return json.load(open(p)) if os.path.exists(p) else None
    ts = {"A": ["runs_ts/A_s0"], "Q0": ["runs_ts/Q0_s0"], "static": [f"runs_vast/QG25static_s{s}" for s in range(3)],
          "QG25": ["runs_ts/QG25_s0", "runs_ts/QG25_s1"]}
    wk = {"A": ["runs_vast/A_wiki_s0"], "Q0": ["runs_vast/Q0_wiki_s0"], "static": ["runs_vast/QG25static_wiki_s0"],
          "QG25": ["runs_vast/QG25_wiki_s0"]}
    def ce(paths):
        v = [e["val_ce"] for e in (ev(p) for p in paths) if e]
        return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
    T = {k: ce(v) for k, v in ts.items()}
    Wk = {k: ce(v) for k, v in wk.items()}
    L = ["# Википедия против TinyStories: Q0, static, QG25 (20k шагов, d = 512, 8 блоков)\n",
         "TinyStories: свой BPE 8192, выборка окон с возвращением; Википедия (20231101.en, шарды 0–3): GPT-2 BPE "
         "(id + 2), окна без повторов. Сетка на каждом наборе — от своей A по одному рецепту. Полный val.\n",
         "| | TinyStories: val CE | сидов | − A | % разрыва QG25 − A | Википедия: val CE | сидов | − A | % разрыва QG25 − A |",
         "|---|---|---|---|---|---|---|---|---|"]
    gT, gW = T["QG25"][0] - T["A"][0], Wk["QG25"][0] - Wk["A"][0]
    for k in ("A", "Q0", "static", "QG25"):
        (a, na), (b, nb) = T[k], Wk[k]
        L.append(f"| {k} | {a:.4f} | {na} | {a - T['A'][0]:+.4f} | {100 * (a - T['A'][0]) / gT:.0f}% | "
                 f"{b:.4f} | {nb} | {b - Wk['A'][0]:+.4f} | {100 * (b - Wk['A'][0]) / gW:.0f}% |")
    L += ["", f"static − QG25: TinyStories {T['static'][0] - T['QG25'][0]:+.4f}, Википедия {Wk['static'][0] - Wk['QG25'][0]:+.4f} "
          f"(отрицательно — static лучше).", ""]
    # adaptivity of QG25 (as in r6_adaptivity): difficulty = loss of A on the same token
    L += ["## Адаптивность QG25\n", "| | ρ(открыто, потеря A) все | внутри токена | средние по токенам | R² самого токена | "
          "открыто по квинтилям потери A (из 64) |", "|---|---|---|---|---|---|"]
    for name, q, a in (("TinyStories", "runs_ts/QG25_s0", "runs_ts/A_s0"), ("Википедия", "runs_vast/QG25_wiki_s0", "runs_vast/A_wiki_s0")):
        if not (os.path.exists(os.path.join(q, "tok.npz")) and os.path.exists(os.path.join(a, "tok.npz"))):
            L.append(f"| {name} | — | — | — | — | — |")
            continue
        zq, za = np.load(os.path.join(q, "tok.npz")), np.load(os.path.join(a, "tok.npz"))
        assert np.array_equal(zq["ids"], za["ids"])
        tot = bits_to_counts(zq["open_bits"]).sum(1)
        al = za["loss"].astype(np.float64)
        cur = za["cur"].view(np.uint16).astype(np.int64)      # tok.npz stores ids as int16: GPT-2 ids > 32767 wrap
        V = int(cur.max()) + 1
        tm = lambda x: (np.bincount(cur, x, V) / np.maximum(np.bincount(cur, minlength=V), 1))[cur]   # noqa: E731
        cnt = np.bincount(cur, minlength=V)
        seen = cnt[cur] >= MIN_SEEN
        r2 = float(1 - (tot - tm(tot))[seen].var() / tot[seen].var())
        ok = cnt >= MIN_SEEN
        tmeans = lambda x: np.bincount(cur, x, V)[ok] / cnt[ok]                                        # noqa: E731
        qb = np.digitize(al, np.quantile(al, [0.2, 0.4, 0.6, 0.8]))
        L.append(f"| {name} | {spear(tot, al):.2f} | {spear(tot - tm(tot), al - tm(al)):.2f} | "
                 f"{spear(tmeans(tot), tmeans(al)):.2f} | {r2:.2f} | " +
                 " / ".join(f"{tot[qb == i].mean():.1f}" for i in range(5)) + " |")
    L.append("")
    open(os.path.join(out, "wiki_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))



# ------------------------------------------------------------------------------------------------ report tables
def report(out="report"):
    h = json.load(open(os.path.join(RUNS, "A_s0", "honesty.json")))
    c = json.load(open(os.path.join(RUNS, "QG50_s0", "causal.json")))
    L = ["# Раунд 8: карта вычислений QG — таблицы\n",
         "Готовые чекпойнты A_s0, QG25_s0, QG50_s0; 608 окон val по 512 токенов (311 296 позиций), как в tok.npz. "
         "ρ — Спирмен. «Внутри токена» — по отклонениям от среднего этого токена (как в r6_adaptivity); «средние по "
         "токенам» — по средним значениям токенов, встреченных ≥ 20 раз. Изменение состояния A — ‖Δx‖/‖x‖ по блокам "
         "2–8: в блоке 1 знаменатель — эмбеддинг с маленькой нормой (среднее отношение 29 против 0.5–1 дальше).\n",
         "## 1. Насколько карта QG совпадает с тем, где считает A\n",
         "| число открытых групп ↔ | QG50: все токены | внутри токена | средние по токенам | QG25: все | внутри | средние |",
         "|---|---|---|---|---|---|---|"]
    lab = dict(rel_total="изменение состояния A (сумма по блокам 2–8)",
               last_change="последний блок, где меняется топ-1 A (logit lens)", A_loss="потеря A (для сравнения)")
    q5, q2 = h["QG50_s0"], h["QG25_s0"]
    for k in ("rel_total", "last_change", "A_loss"):
        L.append(f"| {lab[k]} | {q5['all'][k]:.2f} | {q5['within_token'][k]:.2f} | {q5['token_means'][k]:.2f} | "
                 f"{q2['all'][k]:.2f} | {q2['within_token'][k]:.2f} | {q2['token_means'][k]:.2f} |")
    L += ["", "Доля дисперсии, объяснённая самим токеном (R² средних по токену): "
          f"открыто групп QG50 {q5['r2_token_open_total']:.2f}, QG25 {q2['r2_token_open_total']:.2f}; изменение "
          f"состояния A {h['A']['r2_token']['rel_total']:.2f}; последний блок смены топ-1 A "
          f"{h['A']['r2_token']['last_change']:.2f}; потеря A {h['A']['r2_token']['A_loss']:.2f}.\n",
          "### По блокам (QG50, блок l ↔ тот же блок A)\n",
          "| блок | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |", "|---|---|---|---|---|---|---|---|---|",
          "| ρ(открыто, ‖Δx‖/‖x‖ A) | " + " | ".join(f"{b['rel']:.2f}" for b in q5["block"]) + " |",
          "| то же внутри токена | " + " | ".join(f"{b['rel_within']:.2f}" for b in q5["block"]) + " |",
          "| AUROC: открыто → топ-1 A меняется в этом блоке | " + " | ".join(f"{b['auroc_A_top1_changes']:.2f}" for b in q5["block"]) + " |",
          "| то же внутри токена | " + " | ".join(f"{b['auroc_within']:.2f}" for b in q5["block"]) + " |",
          "| доля токенов, у которых A меняет топ-1 в блоке | " + " | ".join(f"{x:.2f}" for x in h["A"]["change_rate_block"]) + " |",
          "",
          f"Матрица ρ «блок QG50 l ↔ блок A j»: на диагонали в среднем "
          f"{np.mean([q5['block_matrix'][i][i] for i in range(8)]):.2f}, вне диагонали "
          f"{np.mean([q5['block_matrix'][i][j] for i in range(8) for j in range(8) if i != j]):.2f}. В блоке, где у A в "
          f"последний раз меняется топ-1, QG50 открывает на {q5['open_in_settle_block_minus_token_mean_block']:+.2f} группы "
          "больше своего среднего по блокам у этого токена.\n",
          "## 2. Причинная проверка на QG50\n",
          f"В каждом блоке у 20% токенов каждого окна закрыта половина открытых гейтом групп (при нечётном числе — "
          f"с вероятностью ½ на одну больше); выбор групп — случайный среди открытых, у lowgate — с наименьшим логитом "
          f"гейта. Базовая CE {c['check']['base_ce']:.4f}; решения гейта в базовом проходе совпадают с tok.npz на "
          f"{100 * c['check']['base_open_equal_tok_npz']:.1f}%. ΔCE — рост потери; «тронутые» — позиции, где закрывали "
          "(их потеря — на предсказании следующего токена); «прочие» — остальные позиции тех же окон.\n",
          "| условие | глубина тронутых (из 64) | потеря A тронутых | закрыто групп на тронутый токен | доля от открытых | "
          "ΔCE все | ΔCE тронутые | ΔCE прочие | ΔCE тронутых на закрытую группу |", "|---|---|---|---|---|---|---|---|---|"]
    C = {x["name"]: x for x in c["conditions"]}

    def agg(prefix):
        xs = [x for n, x in C.items() if n.startswith(prefix)]
        return xs

    def fmt(xs, k, f="{:+.3f}"):
        v = [x[k] for x in xs]
        return f.format(np.mean(v)) + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")
    rows = [("top: наибольшая глубина", agg("top")), ("bottom: наименьшая глубина", agg("bottom")),
            ("random: случайные (3 сида)", agg("random_")), ("ctx_top: наибольшая глубина сверх средней для токена", agg("ctx_top")),
            ("lowgate: те же токены, что random; группы с наименьшим гейтом (3 сида)", agg("lowgate_"))]
    for name, xs in rows:
        L.append(f"| {name} | {fmt(xs, 'touched_depth', '{:.1f}')} | {fmt(xs, 'touched_A_loss', '{:.2f}')} | "
                 f"{fmt(xs, 'closed_per_touched', '{:.2f}')} | {fmt(xs, 'closed_share_of_open', '{:.2f}')} | "
                 f"{fmt(xs, 'dce_all')} | {fmt(xs, 'dce_touched')} | {fmt(xs, 'dce_untouched')} | "
                 f"{fmt(xs, 'dce_per_closed_group')} |")
    qb = c["A_bins"]
    edges = [f"< {qb[0]:.2f}"] + [f"{qb[i]:.2f}–{qb[i + 1]:.2f}" for i in range(3)] + [f"> {qb[3]:.2f}"]
    L += ["", "### ΔCE тронутых токенов внутри одинаковых по трудности групп (квинтили потери A; в скобках — закрыто "
          "групп на токен, n — число тронутых токенов)\n",
          "| потеря A | " + " | ".join(n for n, _ in rows) + " |", "|---|" + "---|" * len(rows)]
    for b in range(5):
        cells = []
        for _, xs in rows:
            d = [x["by_A_bin"][b]["dce"] for x in xs if x["by_A_bin"][b]["dce"] is not None]
            cl = [x["by_A_bin"][b]["closed"] for x in xs if x["by_A_bin"][b]["closed"] is not None]
            n = int(np.mean([x["by_A_bin"][b]["n"] for x in xs]))
            cells.append(f"{np.mean(d):+.2f} ({np.mean(cl):.1f}; n {n})" if d else "—")
        L.append(f"| {edges[b]} | " + " | ".join(cells) + " |")
    L.append("")
    L += staticinit_tables()
    open(os.path.join(out, "r8_map_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"lens": lens, "causal": causal, "honesty": lambda: print(json.dumps(honesty())[:3000]), "viz": viz,
     "report": report, "top3": top3, "deviation": lambda: deviation(*sys.argv[2:]), "wiki": wiki}[cmd]()
