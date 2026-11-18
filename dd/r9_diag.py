"""Round 9: (1) does the straight-through gradient tell the gate the truth? (2) causal map of computation without a gate.
Finished checkpoints only, home GPU, fp32 (TF32 matmuls) so that small effects keep their sign.
  python -m dd.r9_diag gate     -> runs_ts/QG25_s0/gate_diag.npz: for ~200 val positions x 8 blocks x 8 groups:
        promised change of the loss when the decision is switched = dL/dm * (switch sign: +1 open, -1 close)
        (dL/dm through the straight-through gate exactly as in training; CE only, the mu * cost term is reported apart)
        real change: switch only that decision and recompute the loss
        two versions: 'fixed' -- all other decisions frozen (pure linearisation test);
                      'free'  -- the other gates react (as in training: the gradient also flows through their surrogates)
        losses: the token's own loss (prediction made at that position) and the summed loss of the window
  python -m dd.r9_diag ablate   -> runs_ts/{A_s0, ...}/ablate.npz and runs_vast/QG25static_s0/ablate.npz:
        for every position of the 30 viz stories + 1024 random positions and every block: drop the block's write at
        that position only (attention / FFN / whole block), loss increase on the own token and on the whole window
  python -m dd.r9_diag report   -> report/r9_tables.md"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from .model import doc_causal_mask, quantize
from .r8_map import CTX, MIN_SEEN, RUNS, VAST, bits_to_counts, tok_mean, tokz, windows
from .ts_eval import load_run

DEV = "cuda"


def ce_vec(lg, idx):
    return F.cross_entropy(lg[:, :-1].float().flatten(0, 1), idx[:, 1:].flatten(), reduction="none").view(idx.shape[0], -1)


# ------------------------------------------------------------------------------------------------ 1. gate diagnostics
def gate(run="QG25_s0", n_win=25, per_win=8, seed=0):
    torch.backends.cuda.matmul.allow_tf32 = True
    model, cfg = load_run(os.path.join(RUNS, run))
    model.float()
    L, G = cfg.n_layers, cfg.depth_groups
    W = windows()
    rng = np.random.default_rng(seed)
    wins = rng.choice(W.shape[0], n_win, replace=False)
    st = {}

    def hook(blk, m, zg):
        mode, l = st["mode"], blk.l
        if mode == "base":
            st["m"][l] = m.detach().clone()
            return m
        if mode == "grad_fixed":                       # all decisions = frozen leaves
            leaf = st["base"][l].clone().requires_grad_(True)
            st["leaf"][l] = leaf
            return leaf
        if mode == "grad_free":                        # zero perturbation on top of the live straight-through mask
            e = torch.zeros_like(m, requires_grad=True)
            st["leaf"][l] = e
            return m + e
        if mode in ("flip_fixed", "flip_free"):
            mm = (st["base"][l] if mode == "flip_fixed" else m).clone()
            k = st["blocks"] == l                       # batch copies that switch a decision in this block
            if k.any():
                b = torch.nonzero(k).squeeze(1)
                g = st["groups"][k]
                mm[b, st["t"], g] = 1.0 - mm[b, st["t"], g].detach()
            return mm
        return m

    rows = []
    for w in wins:
        idx = W[w:w + 1]
        ts = np.sort(rng.choice(np.arange(32, CTX - 1), per_win, replace=False))
        # baseline decisions and losses
        st.update(mode="base", m={})
        with torch.no_grad():
            lg = model(idx, ctl={"hc_gate_fn": hook})[0]
        base_m = [st["m"][l] for l in range(L)]
        base_ce = ce_vec(lg, idx)[0]
        for t in ts:
            t = int(t)
            out = {}
            for gmode in ("grad_fixed", "grad_free"):
                for which in ("tok", "seq"):
                    st.update(mode=gmode, base=base_m, leaf={})
                    lg = model(idx, ctl={"hc_gate_fn": hook})[0]
                    ce = ce_vec(lg, idx)[0]
                    loss = ce[t] if which == "tok" else ce.sum()
                    gr = torch.autograd.grad(loss, [st["leaf"][l] for l in range(L)])
                    out[f"g_{gmode[5:]}_{which}"] = torch.stack([gr[l][0, t] for l in range(L)]).cpu().numpy()   # L, G
            # real switches: 64 copies, copy k switches (block k // G, group k % G) at position t
            blocks = torch.arange(L * G, device=DEV) // G
            groups = torch.arange(L * G, device=DEV) % G
            for fmode in ("flip_fixed", "flip_free"):
                st.update(mode=fmode, base=[b.expand(L * G, -1, -1) for b in base_m], blocks=blocks, groups=groups, t=t)
                with torch.no_grad():
                    lg = model(idx.expand(L * G, -1), ctl={"hc_gate_fn": hook})[0]
                ce = ce_vec(lg, idx.expand(L * G, -1))
                out[f"r_{fmode[5:]}_tok"] = (ce[:, t] - base_ce[t]).view(L, G).cpu().numpy()
                out[f"r_{fmode[5:]}_seq"] = (ce.sum(1) - base_ce.sum()).view(L, G).cpu().numpy()
            m0 = torch.stack([base_m[l][0, t] for l in range(L)]).cpu().numpy()          # L, G (0/1)
            rows.append(dict(win=int(w), t=t, m=m0, **out))
        print(f"window {w}: {len(rows)} positions done", flush=True)
    keys = [k for k in rows[0] if k not in ("win", "t")]
    np.savez_compressed(os.path.join(RUNS, run, "gate_diag.npz"), win=np.array([r["win"] for r in rows]),
                        t=np.array([r["t"] for r in rows]), mu=float(model.mu[0]),
                        **{k: np.stack([r[k] for r in rows]) for k in keys})
    print("saved", len(rows), "positions", flush=True)


# ------------------------------------------------------------------------------------------------ 2. ablation maps
def forward_abl(model, idx, abl_t, abl_l, abl_kind):
    """manual forward of A (plain residual) or a QG model with a fixed mask (static), with one ablation per batch row:
    at position abl_t[b] in block abl_l[b] drop the attention write (kind 0), the FFN write (1) or the whole block (2);
    abl_l = -1 -> no ablation.  Returns logits."""
    cfg = model.cfg
    B, T = idx.shape
    rope = model.rope(T, idx.device)
    mask = doc_causal_mask(idx)
    ar = torch.arange(B, device=idx.device)
    x = model.emb(idx)
    if cfg.emb_scale != 1.0:
        x = x * cfg.emb_scale
    if cfg.quant:
        x = quantize(x.float(), cfg.qd(0), cfg.q_levels)[0]
        G = cfg.depth_groups
        depth = torch.zeros(B, T, G, device=idx.device)
    for l, blk in enumerate(model.blocks):
        sel = abl_l == l
        keep_a = torch.ones(B, T, 1, device=idx.device)
        keep_u = torch.ones(B, T, 1, device=idx.device)
        keep_a[ar[sel & (abl_kind != 1)], abl_t[sel & (abl_kind != 1)]] = 0.0
        keep_u[ar[sel & (abl_kind != 0)], abl_t[sel & (abl_kind != 0)]] = 0.0
        if not cfg.quant:                                                         # A
            a = blk.attn(blk.n1(x), rope, mask=mask) * keep_a
            y = x + a
            z = blk.n2(y)
            u = blk.w2(F.silu(blk.w1(z)) * blk.w3(z)) * keep_u
            x = y + u
        else:                                                                     # QG with a fixed (static) mask
            a = blk.attn(blk.n1(x) + blk.depth_lookup(blk.dt_a, depth).to(x.dtype), rope, mask=mask) * keep_a
            zin = blk.n2(x + a) + blk.depth_lookup(blk.dt_f, depth).to(x.dtype)
            u = blk.w2(F.silu(blk.w1(zin)) * blk.w3(zin)) * keep_u
            x_c, _ = quantize(x + (a + u).float(), cfg.qd(l + 1), cfg.q_levels)
            m = blk.static_mask.expand(B, T, G).clone()
            whole = sel & (abl_kind == 2)
            m[ar[whole], abl_t[whole]] = 0.0                                      # the block writes nothing there
            mf = blk.expand(m)
            x = mf * x_c + (1 - mf) * x
            depth = depth + m
    return model.nf(x) @ model.emb.weight.t()


def story_rows():
    d = json.load(open(os.path.join("report", "r8_viz_data.json"), encoding="utf-8"))
    za = tokz("A_s0")
    cur, nxt = za["cur"].astype(np.int64), za["ids"].astype(np.int64)
    # recover the generating rows of the viz stories: the viz stores nxt ids; find them as in r8_map.viz
    N = cur.size
    win = np.arange(N) // CTX
    eot = np.flatnonzero(cur == 1)
    cands = []
    for p in eot:
        q = p + 1
        while q < N and win[q] == win[p] and cur[q] != 1:
            q += 1
        if q < N and win[q] == win[p] and nxt[q - 1] == 1 and 60 <= q - p <= 260:
            cands.append((p, q))
    pick = [cands[i] for i in np.linspace(0, len(cands) - 1, len(d["stories"])).round().astype(int)]
    out = []
    for (p, q), s in zip(pick, d["stories"]):
        r = np.arange(p, q - 1)
        assert nxt[r].tolist() == s["ids"]
        out.append(r)
    return out


@torch.no_grad()
def ablate(runs=(("A_s0", RUNS), ("QG25static_s0", VAST)), n_rand=1024, seed=0, bs=96):
    torch.backends.cuda.matmul.allow_tf32 = True
    W = windows()
    srows = story_rows()
    rng = np.random.default_rng(seed)
    all_rows = np.concatenate(srows)
    pool = np.setdiff1d(np.flatnonzero((np.arange(W.shape[0] * CTX) % CTX) >= 32), all_rows)
    rand_rows = np.sort(rng.choice(pool, n_rand, replace=False))
    rows = np.concatenate([all_rows, rand_rows])                               # global row = window * CTX + t
    for run, base in runs:
        model, cfg = load_run(os.path.join(base, run))
        model.float()
        L = cfg.n_layers
        lg0 = forward_abl(model, W[:4], torch.zeros(4, dtype=torch.long, device=DEV), torch.full((4,), -1, device=DEV),
                          torch.zeros(4, dtype=torch.long, device=DEV))
        lgm = model(W[:4])[0]
        print(f"{run}: manual forward vs model, max |logit diff| {float((lg0 - lgm).abs().max()):.2e}", flush=True)
        own = np.zeros((rows.size, L, 3), np.float32)
        seq = np.zeros((rows.size, L, 3), np.float32)
        by_win = {}
        for i, r in enumerate(rows):
            by_win.setdefault(int(r // CTX), []).append(i)
        for w, ii in by_win.items():
            idx1 = W[w:w + 1]
            ce0 = ce_vec(forward_abl(model, idx1, torch.zeros(1, dtype=torch.long, device=DEV),
                                     torch.full((1,), -1, device=DEV), torch.zeros(1, dtype=torch.long, device=DEV)), idx1)[0]
            jobs = [(i, int(rows[i] % CTX), l, k) for i in ii for l in range(L) for k in range(3)]
            for s in range(0, len(jobs), bs):
                jb = jobs[s:s + bs]
                tt = torch.tensor([j[1] for j in jb], device=DEV)
                ll = torch.tensor([j[2] for j in jb], device=DEV)
                kk = torch.tensor([j[3] for j in jb], device=DEV)
                idx = idx1.expand(len(jb), -1)
                ce = ce_vec(forward_abl(model, idx, tt, ll, kk), idx)
                d_own = (ce[torch.arange(len(jb), device=DEV), tt] - ce0[tt]).cpu().numpy()
                d_seq = (ce.sum(1) - ce0.sum()).cpu().numpy()
                for n, (i, t, l, k) in enumerate(jb):
                    own[i, l, k], seq[i, l, k] = d_own[n], d_seq[n]
        np.savez_compressed(os.path.join(base, run, "ablate.npz"), rows=rows, n_story=all_rows.size, own=own, seq=seq)
        print(f"{run}: {rows.size} positions x {L} blocks x 3 ablations; mean own dCE per block "
              f"{np.round(own[..., 2].mean(0), 3).tolist()}", flush=True)


# ------------------------------------------------------------------------------------------------ report
def _corr(a, b):
    return (float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)) if a.size > 2 else (float("nan"),) * 2


def gate_tables(run="QG25_s0", eps=(1e-4, 1e-3, 1e-2)):
    z = np.load(os.path.join(RUNS, run, "gate_diag.npz"))
    m = z["m"]                                                       # P, L, G (0/1)
    s = 1.0 - 2.0 * m                                                # +1 = open a closed group, -1 = close an open one
    mu = float(z["mu"])
    P, L, G = m.shape
    out = [f"## 1. Говорит ли градиент гейту правду ({run}, {P} позиций × {L} блоков × {G} групп = {m.size} решений)\n",
           "Обещанное изменение потери при переключении решения = dL/dm × (+1 открыть / −1 закрыть), dL/dm — через "
           "straight-through, как при обучении (только CE; член μ·цена отдельно). Реальное — переключено только это "
           "решение, потеря пересчитана. «Остальные заморожены»: все прочие решения гейта фиксированы (чистая "
           "проверка линейного прогноза); «гейты реагируют»: прочие гейты пересчитываются, как в модели, и градиент "
           "тоже идёт через их суррогаты, как при обучении. fp32. Совпадение знака — среди решений с |реальное| > ε.\n",
           f"Цена одного решения в тех же единицах (на токен): μ / 64 = {mu / 64:.4f} нат (μ = {mu:.3f}).\n",
           "| вариант | потеря | Пирсон | Спирмен | знак совпал: ε=1e-4 | ε=1e-3 | ε=1e-2 | доля |реальное| > 1e-3 | "
           "медиана |обещ.| | медиана |реальн.| |", "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for v, vl in (("fixed", "остальные заморожены"), ("free", "гейты реагируют")):
        for w, wl in (("tok", "этот токен"), ("seq", "вся последовательность")):
            pr = (z[f"g_{v}_{w}"] * s).ravel()
            re = z[f"r_{v}_{w}"].ravel()
            res[(v, w)] = (pr, re)
            p, sp = _corr(pr, re)
            sg = [float((np.sign(pr[np.abs(re) > e]) == np.sign(re[np.abs(re) > e])).mean()) for e in eps]
            out.append(f"| {vl} | {wl} | {p:.2f} | {sp:.2f} | " + " | ".join(f"{100 * x:.0f}%" for x in sg) +
                       f" | {100 * (np.abs(re) > 1e-3).mean():.0f}% | {np.median(np.abs(pr)):.4f} | {np.median(np.abs(re)):.4f} |")
    out.append("")
    mm = m.ravel()
    blk = np.broadcast_to(np.arange(L)[None, :, None], m.shape).ravel()
    cost = (mu / 64) * (1 - 2 * mm)                                  # opening costs +mu/64, closing saves it
    for v, w, title in (("free", "seq", "гейты реагируют, вся последовательность"),
                        ("fixed", "tok", "остальные заморожены, этот токен")):
        pr, re = res[(v, w)]
        out += [f"### По типу решения и блокам ({title}; знак — при |реальное| > 1e-3)\n",
                "| срез | решений | Спирмен | знак совпал | обещано: «переключение снизит потерю» | реально: снизит | "
                "реально снизит с учётом μ·цены |", "|---|---|---|---|---|---|---|"]
        for name, sel in [("открытые (переключение = закрыть)", mm == 1), ("закрытые (переключение = открыть)", mm == 0)] + \
                         [(f"блок {l + 1}", blk == l) for l in range(L)]:
            a, b = pr[sel], re[sel]
            big = np.abs(b) > 1e-3
            out.append(f"| {name} | {int(sel.sum())} | {_corr(a, b)[1]:.2f} | "
                       f"{100 * (np.sign(a[big]) == np.sign(b[big])).mean():.0f}% | {100 * (a < 0).mean():.0f}% | "
                       f"{100 * (b < 0).mean():.0f}% | {100 * ((b + cost[sel]) < 0).mean():.0f}% |")
        out.append("")
    return out


def _within(x, cu):
    cnt = np.bincount(cu, minlength=int(cu.max()) + 1)
    return x - (np.bincount(cu, x, cnt.size) / np.maximum(cnt, 1))[cu], cnt[cu] >= 3


def ablate_tables():
    za, zq = tokz("A_s0"), tokz("QG25_s0")
    cur = za["cur"].astype(np.int64)
    a = np.load(os.path.join(RUNS, "A_s0", "ablate.npz"))
    st = np.load(os.path.join(VAST, "QG25static_s0", "ablate.npz"))
    assert np.array_equal(a["rows"], st["rows"])
    rows = a["rows"]
    c = bits_to_counts(zq["open_bits"])[rows]                       # QG25 open groups per block at these positions
    L = c.shape[1]
    n_story = int(a["n_story"])
    rnd = np.arange(rows.size) >= n_story                           # the random positions (unbiased sample)
    out = ["## 2. Причинная карта без гейта: что стоит выкинуть запись блока на одной позиции\n",
           f"A и static (s0); {rows.size} позиций (все позиции 30 историй визуализации + {int(rnd.sum())} случайных). "
           "На позиции t в блоке l выкинута запись внимания, FFN или всего блока (у static — запись в его 2 группы), "
           "прочие позиции и блоки не тронуты. ΔCE — рост потери на собственном токене (предсказание, сделанное в t) и "
           "суммарно по окну. Статистика — по случайным позициям.\n",
           "| модель | часть | ΔCE своего токена, блоки 1–8 | ΔCE по окну, блоки 1–8 |", "|---|---|---|---|"]
    for name, z in (("A", a), ("static", st)):
        for k, kl in ((0, "внимание"), (1, "FFN"), (2, "весь блок")):
            out.append(f"| {name} | {kl} | " + " ".join(f"{x:+.3f}" for x in z["own"][rnd, :, k].mean(0)) + " | " +
                       " ".join(f"{x:+.3f}" for x in z["seq"][rnd, :, k].mean(0)) + " |")
    out += ["", "### Совпадает ли с картой QG25 (число открытых групп в том же блоке на той же позиции), Спирмен\n",
            "| сравнение | все пары (позиция × блок) | по блокам 1–8 | суммарно по позиции | внутри токена (сумма) |",
            "|---|---|---|---|---|"]
    rr, cc = rows[rnd], c[rnd]
    cu = cur[rr]
    for name, z in (("A: весь блок, ΔCE по окну", a["seq"][rnd, :, 2]), ("A: весь блок, ΔCE своего токена", a["own"][rnd, :, 2]),
                    ("A: внимание, по окну", a["seq"][rnd, :, 0]), ("A: FFN, по окну", a["seq"][rnd, :, 1]),
                    ("static: весь блок, по окну", st["seq"][rnd, :, 2])):
        tc, tz = cc.sum(1), z.sum(1)
        dc, ok = _within(tc, cu)
        dz, _ = _within(tz, cu)
        out.append(f"| {name} | {spearmanr(cc.ravel(), z.ravel()).statistic:.2f} | " +
                   " ".join(f"{spearmanr(cc[:, l], z[:, l]).statistic:.2f}" for l in range(L)) +
                   f" | {spearmanr(tc, tz).statistic:.2f} | {spearmanr(dc[ok], dz[ok]).statistic:.2f} |")
    sa, ss = a["seq"][rnd, :, 2].sum(1), st["seq"][rnd, :, 2].sum(1)
    out += ["", f"A и static между собой (суммарный ΔCE по окну на позиции): Спирмен {spearmanr(sa, ss).statistic:.2f}; "
            "по блокам: " + " ".join(f"{spearmanr(a['seq'][rnd, l, 2], st['seq'][rnd, l, 2]).statistic:.2f}" for l in range(L)) + ".",
            ""]
    x = a["seq"][:, :, 2].sum(1)
    r, ok = _within(x, cur[rows])
    out += [f"Доля разброса суммарного ΔCE A (по окну), объяснённая самим токеном (токены, встреченные ≥ 3 раз среди "
            f"{rows.size} позиций): {1 - r[ok].var() / x[ok].var():.2f}.", ""]
    return out


def report(out="report"):
    L = ["# Раунд 9: обучение гейта и причинная карта вычислений\n"] + gate_tables() + ablate_tables()
    open(os.path.join(out, "r9_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

# ------------------------------------------------------------------------------------------------ 3. blocks needed (logit lens)
LENS_RUNS = (("A_s0", RUNS), ("A_s1", VAST), ("QG25static_s0", VAST), ("QG25_s0", RUNS))
TAUS = (0.05, 0.1, 0.25, 0.5)


@torch.no_grad()
def lens_kl(runs=LENS_RUNS):
    """KL(final || lens_l) for every analysis position and state l = 0 (embedding) .. L (final, = 0):
    lens_l = softmax(final RMSNorm(x_l) @ embedding^T), the model's own output head on the intermediate state."""
    W = windows()
    for run, base in runs:
        if not os.path.exists(os.path.join(base, run, "model.pt")):
            print(f"{run}: no model, skipped", flush=True)
            continue
        model, cfg = load_run(os.path.join(base, run))
        KL = []
        for i in range(0, W.shape[0], 16):
            idx = W[i:i + 16]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, aux = model(idx, collect=True)
            xs = [x[:, :-1].float() for x in aux["xs"]]
            E = model.emb.weight.float()
            lp = [F.log_softmax(model.nf(x) @ E.t(), -1) for x in xs[-1:]]
            lpf = lp[0]
            pf = lpf.exp()
            kl = [((pf * (lpf - F.log_softmax(model.nf(x) @ E.t(), -1))).sum(-1)) for x in xs]
            KL.append(torch.stack(kl, -1).reshape(-1, len(xs)).cpu().numpy().astype(np.float32))
        KL = np.concatenate(KL)
        np.savez_compressed(os.path.join(base, run, "lens_kl.npz"), kl=KL.astype(np.float16))
        print(f"{run}: KL to final by state {np.round(np.median(KL, 0), 3).tolist()} (median)", flush=True)


def needed(kl, tau):
    """earliest state l such that KL stays <= tau from l on (0 = embedding .. L = final)"""
    tail = np.maximum.accumulate(kl[:, ::-1], axis=1)[:, ::-1]
    return (tail > tau).sum(1)


def agree_tables(out="report"):
    cur = tokz("A_s0")["cur"].astype(np.int64)
    K = {}
    for run, base in LENS_RUNS:
        p = os.path.join(base, run, "lens_kl.npz")
        if os.path.exists(p):
            K[run] = np.load(p)["kl"].astype(np.float32)
    names = {"A_s0": "A s0", "A_s1": "A s1", "QG25static_s0": "static", "QG25_s0": "QG25"}
    L = ["## 3. Сколько блоков нужно позиции: logit lens, согласие между моделями\n",
         "Мера: самый ранний блок, начиная с которого KL(финальное ‖ lens) ≤ τ и дальше не поднимается (0 = уже после "
         "эмбеддинга, 8 = только на выходе); lens — финальная RMSNorm и выходной слой модели на промежуточном "
         "состоянии. Плюс гладкий вариант: сумма KL по состояниям 0–7. Все 311 296 позиций анализа. Потолок согласия — "
         "A s0 против A s1 (та же архитектура и данные, другой сид).\n",
         "| модель | среднее «нужно блоков» при τ = 0.05 / 0.1 / 0.25 / 0.5 | медиана KL по состояниям 0–8 |", "|---|---|---|"]
    for r, kl in K.items():
        L.append(f"| {names[r]} | " + " / ".join(f"{needed(kl, t).mean():.2f}" for t in TAUS) + " | " +
                 " ".join(f"{x:.2f}" for x in np.median(kl, 0)) + " |")
    L.append("")
    pairs = [("A_s0", "A_s1"), ("A_s0", "QG25static_s0"), ("A_s1", "QG25static_s0"), ("A_s0", "QG25_s0"), ("A_s1", "QG25_s0"),
             ("QG25static_s0", "QG25_s0")]
    L += ["| пара | ρ «нужно блоков», τ = 0.05 / 0.1 / 0.25 / 0.5 | точное совпадение при τ = 0.1 | ρ суммы KL | "
          "ρ внутри токена (сумма KL) | ρ средних по токенам (сумма KL) |", "|---|---|---|---|---|---|"]
    cnt = np.bincount(cur, minlength=8192)
    seen = cnt[cur] >= MIN_SEEN
    okv = cnt >= MIN_SEEN
    for a, b in pairs:
        if a not in K or b not in K:
            continue
        ka, kb = K[a], K[b]
        sa, sb = ka[:, :-1].sum(1), kb[:, :-1].sum(1)
        wa, wb = sa - tok_mean(sa, cur), sb - tok_mean(sb, cur)
        ma, mb = np.bincount(cur, sa, 8192)[okv] / cnt[okv], np.bincount(cur, sb, 8192)[okv] / cnt[okv]
        L.append(f"| {names[a]} — {names[b]} | " + " / ".join(f"{spearmanr(needed(ka, t), needed(kb, t)).statistic:.2f}" for t in TAUS) +
                 f" | {100 * (needed(ka, 0.1) == needed(kb, 0.1)).mean():.0f}% | {spearmanr(sa, sb).statistic:.2f} | "
                 f"{spearmanr(wa[seen], wb[seen]).statistic:.2f} | {spearmanr(ma, mb).statistic:.2f} |")
    # QG25's gate map against the lens measure of each A / static
    if "QG25_s0" in K:
        c = bits_to_counts(tokz("QG25_s0")["open_bits"]).sum(1)
        L += ["", "Карта гейта QG25 (число открытых групп на позиции) против меры «нужно блоков»:\n",
              "| модель | ρ с «нужно блоков» при τ = 0.1 | ρ с суммой KL | внутри токена |", "|---|---|---|---|"]
        for r in ("A_s0", "A_s1", "QG25static_s0", "QG25_s0"):
            if r in K:
                s = K[r][:, :-1].sum(1)
                L.append(f"| {names[r]} | {spearmanr(c, needed(K[r], 0.1)).statistic:.2f} | {spearmanr(c, s).statistic:.2f} | "
                         f"{spearmanr((c - tok_mean(c, cur))[seen], (s - tok_mean(s, cur))[seen]).statistic:.2f} |")
    L.append("")
    # causal maps (ablate.npz on the same rows): A s0 vs A s1 = ceiling
    ab = {}
    for run, base in (("A_s0", RUNS), ("A_s1", VAST), ("QG25static_s0", VAST)):
        p = os.path.join(base, run, "ablate.npz")
        if os.path.exists(p):
            ab[run] = np.load(p)
    if len(ab) >= 2:
        rows = ab["A_s0"]["rows"]
        rnd = np.arange(rows.size) >= int(ab["A_s0"]["n_story"])
        cq = bits_to_counts(tokz("QG25_s0")["open_bits"])[rows][rnd]
        L += ["## 4. Причинная карта (выкидывание записи блока на позиции): согласие между моделями\n",
              f"Случайные {int(rnd.sum())} позиций; ΔCE по окну без записи всего блока. «Все пары» — позиция × блок; "
              "блок 1 у A исключён из «все пары» (его FFN незаменим у обоих сидов и задаёт масштаб).\n",
              "| пара | все пары (блоки 2–8) | по блокам 1–8 | сумма по позиции | внутри токена (сумма) |", "|---|---|---|---|---|"]
        cu = cur[rows][rnd]
        def wtk(x):
            cn = np.bincount(cu, minlength=int(cu.max()) + 1)
            return x - (np.bincount(cu, x, cn.size) / np.maximum(cn, 1))[cu], cn[cu] >= 3
        maps = {k: v["seq"][rnd, :, 2] for k, v in ab.items()}
        maps["QG25 (открыто групп)"] = cq
        for a, b in (("A_s0", "A_s1"), ("A_s0", "QG25static_s0"), ("A_s1", "QG25static_s0"), ("A_s0", "QG25 (открыто групп)"),
                     ("A_s1", "QG25 (открыто групп)"), ("QG25static_s0", "QG25 (открыто групп)")):
            if a not in maps or b not in maps:
                continue
            x, y = maps[a], maps[b]
            dx, ok = wtk(x.sum(1))
            dy, _ = wtk(y.sum(1))
            L.append(f"| {names.get(a, a)} — {names.get(b, b)} | {spearmanr(x[:, 1:].ravel(), y[:, 1:].ravel()).statistic:.2f} | " +
                     " ".join(f"{spearmanr(x[:, l], y[:, l]).statistic:.2f}" for l in range(x.shape[1])) +
                     f" | {spearmanr(x.sum(1), y.sum(1)).statistic:.2f} | {spearmanr(dx[ok], dy[ok]).statistic:.2f} |")
        L.append("")
    return L


def agree(out="report"):
    L = ["# Раунд 9: потолок согласия карт (A s0 против A s1) и согласие A со static и QG25\n"] + agree_tables(out)
    open(os.path.join(out, "r9_agree.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

# ------------------------------------------------------------------------------------------------ 5. gate training variants (10.14)
def gate_any(run, base, n_win=25, per_win=8, seed=0):
    """the same 200 positions as gate(); works for straight-through gates (QG25, mulimit) and for the H-Net-style
    smooth gates.  Saves promise_* directly: STE -> dL/dm * (1 - 2m); smooth -> for open groups -s * dL/ds (removing the
    write fraction), for closed groups 0 (no gradient path: marked by has_path = 0)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    model, cfg = load_run(os.path.join(base, run))
    model.float()
    smooth = bool(getattr(cfg, "qg_smooth", ""))
    L, G = cfg.n_layers, cfg.depth_groups
    W = windows()
    rng = np.random.default_rng(seed)
    wins = rng.choice(W.shape[0], n_win, replace=False)
    st = {}

    def gate_fn(blk, m, zg):
        mode, l = st["mode"], blk.l
        if mode == "base":
            st["m"][l] = m.detach().clone()
            return m
        if mode.startswith("flip"):
            mm = (st["base"][l] if mode == "flip_fixed" else m).clone()
            k = st["blocks"] == l
            if k.any():
                b = torch.nonzero(k).squeeze(1)
                g = st["groups"][k]
                mm[b, st["t"], g] = 1.0 - mm[b, st["t"], g].detach()
            return mm
        if smooth:
            return st["base"][l] if mode == "grad_fixed" else m
        if mode == "grad_fixed":
            leaf = st["base"][l].clone().requires_grad_(True)
            st["leaf"][l] = leaf
            return leaf
        e = torch.zeros_like(m, requires_grad=True)                  # grad_free
        st["leaf"][l] = e
        return m + e

    def s_fn(blk, sg):
        mode, l = st["mode"], blk.l
        if mode == "base":
            st["s"][l] = sg.detach().clone()
            return sg
        if mode == "grad_fixed":
            leaf = st["sbase"][l].clone().requires_grad_(True)
            st["leaf"][l] = leaf
            return leaf
        if mode == "grad_free":
            e = torch.zeros_like(sg, requires_grad=True)
            st["leaf"][l] = e
            return sg + e
        return sg

    ctl = {"hc_gate_fn": gate_fn}
    if smooth:
        ctl["smooth_s_fn"] = s_fn
    rows = []
    for w in wins:
        idx = W[w:w + 1]
        ts = np.sort(rng.choice(np.arange(32, CTX - 1), per_win, replace=False))
        st.update(mode="base", m={}, s={})
        with torch.no_grad():
            lg = model(idx, ctl=dict(ctl))[0]
        base_m = [st["m"][l] for l in range(L)]
        base_s = [st["s"][l] for l in range(L)] if smooth else None
        base_ce = ce_vec(lg, idx)[0]
        for t in ts:
            t = int(t)
            out = {}
            m0 = torch.stack([base_m[l][0, t] for l in range(L)]).cpu().numpy()
            for gmode in ("grad_fixed", "grad_free"):
                for which in ("tok", "seq"):
                    st.update(mode=gmode, base=base_m, sbase=base_s, leaf={})
                    lg = model(idx, ctl=dict(ctl))[0]
                    ce = ce_vec(lg, idx)[0]
                    loss = ce[t] if which == "tok" else ce.sum()
                    gr = torch.autograd.grad(loss, [st["leaf"][l] for l in range(L)], allow_unused=True)
                    g = torch.stack([(gr[l][0, t] if gr[l] is not None else torch.zeros(G, device=DEV)) for l in range(L)])
                    g = g.detach().cpu().numpy()
                    if smooth:
                        s0 = torch.stack([base_s[l][0, t] for l in range(L)]).cpu().numpy()
                        out[f"p_{gmode[5:]}_{which}"] = np.where(m0 > 0.5, -s0 * g, 0.0)
                    else:
                        out[f"p_{gmode[5:]}_{which}"] = g * (1.0 - 2.0 * m0)
            blocks = torch.arange(L * G, device=DEV) // G
            groups = torch.arange(L * G, device=DEV) % G
            for fmode in ("flip_fixed", "flip_free"):
                st.update(mode=fmode, base=[b.expand(L * G, -1, -1) for b in base_m], blocks=blocks, groups=groups, t=t)
                with torch.no_grad():
                    lg = model(idx.expand(L * G, -1), ctl=dict(ctl))[0]
                ce = ce_vec(lg, idx.expand(L * G, -1))
                out[f"r_{fmode[5:]}_tok"] = (ce[:, t] - base_ce[t]).view(L, G).cpu().numpy()
                out[f"r_{fmode[5:]}_seq"] = (ce.sum(1) - base_ce.sum()).view(L, G).cpu().numpy()
            rows.append(dict(win=int(w), t=t, m=m0, has_path=(m0 > 0.5) if smooth else np.ones_like(m0, bool), **out))
    keys = [k for k in rows[0] if k not in ("win", "t")]
    np.savez_compressed(os.path.join(base, run, "gate_diag2.npz"), win=np.array([r["win"] for r in rows]),
                        t=np.array([r["t"] for r in rows]), mu=float(model.mu[0]), smooth=smooth,
                        **{k: np.stack([r[k] for r in rows]) for k in keys})
    print(f"{run}: gate diagnostics on {len(rows)} positions saved", flush=True)


@torch.no_grad()
def pdist(run, base, n_win=64):
    """distribution of p = sigmoid(z) of the open groups on the first n_win analysis windows"""
    model, cfg = load_run(os.path.join(base, run))
    Z = []

    def fn(blk, m, zg):
        Z.append((blk.l, zg.detach()[:, :-1].float().cpu()))
        return m
    W = windows()[:n_win]
    for i in range(0, W.shape[0], 16):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(W[i:i + 16], ctl={"hc_gate_fn": fn})
    L = cfg.n_layers
    p = [torch.sigmoid(torch.cat([z for l, z in Z if l == b]).flatten()).numpy() for b in range(L)]
    allp = np.concatenate(p)
    op = allp[allp > 0.5]
    res = dict(open_share=float((allp > 0.5).mean()), p_gt_09=float((op > 0.9).mean()), p_05_06=float((op < 0.6).mean()),
               p_median=float(np.median(op)), closed_p_lt_01=float((allp[allp <= 0.5] < 0.1).mean()),
               block_open=[float((x > 0.5).mean()) for x in p], block_p_gt_09=[float((x[x > 0.5] > 0.9).mean()) for x in p])
    json.dump(res, open(os.path.join(base, run, "pdist.json"), "w"), indent=1)
    print(run, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not k.startswith("block")}), flush=True)
    return res


GATE_RUNS = (("QG25_s0", RUNS), ("QG25mulimit_s0", VAST), ("QG25hnetfix_s0", VAST), ("QG25dimmerfix_s0", VAST))


def gate_variants():
    for run, base in GATE_RUNS:
        if os.path.exists(os.path.join(base, run, "model.pt")):
            if not os.path.exists(os.path.join(base, run, "gate_diag2.npz")):
                gate_any(run, base)
            pdist(run, base)


def g14_tables():
    from .report_r7 import pooled_sd, seeds, stats, verdict
    names = dict(QG25_s0="QG25", QG25mulimit_s0="QG25-mulimit", QG25hnetfix_s0="QG25-smooth-hnet",
                 QG25dimmerfix_s0="QG25-smooth-dimmer")
    out = ["## 10.14: обучение гейта по мотивам H-Net и ограничение μ — таблицы\n"]
    st = {"static": stats(seeds(RUNS, "QG25static")), "QG25": stats(seeds(RUNS, "QG25")),
          "staticinit": stats(seeds(RUNS, "QG25staticinit"))}
    out += ["| прогон | сидов | val CE | − static | − QG25 | − staticinit | открыто по блокам 1–8 | μ: максимум (шаг) / конец |",
            "|---|---|---|---|---|---|---|---|"]
    for key, lab in (("static", "static"), ("staticinit", "staticinit"), ("QG25", "QG25")):
        m, sd, k = st[key]
        out.append(f"| {lab} | {k} | {m:.4f}" + (f" ± {sd:.4f}" if k > 1 else "") + " | | | | | |")
    for run, base in GATE_RUNS[1:]:
        p = os.path.join(base, run)
        if not os.path.exists(os.path.join(p, "ts_eval.json")):
            out.append(f"| {names[run]} | 0 | — | | | | | |")
            continue
        E, Lg = json.load(open(os.path.join(p, "ts_eval.json"))), json.load(open(os.path.join(p, "log.json")))
        ce = E["val_ce"]
        mu = [(h["step"], h["mu"][0]) for h in Lg["train_log"] if "mu" in h]
        mx = max(mu, key=lambda x: x[1])
        cells = []
        for ref in ("static", "QG25", "staticinit"):
            d = ce - st[ref][0]
            thr = 2 * pooled_sd((ce, float("nan"), 1), st[ref])
            cells.append(f"{d:+.4f} ({verdict(d, thr, 'хуже', 'лучше')})")
        out.append(f"| {names[run]} | 1 | {ce:.4f} | " + " | ".join(cells) + " | " +
                   " ".join(f"{x:.2f}" for x in E["qg"]["paid_layer"]) + f" | {mx[1]:.2f} ({mx[0]}) / {mu[-1][1]:.2f} |")
    out.append("")
    out += ["### Уверенность решений: p = sigmoid(z) у открытых групп (64 окна val)\n",
            "| прогон | открыто | медиана p открытых | p > 0.9 | p в 0.5–0.6 | закрытые с p < 0.1 | p > 0.9 по блокам 1–8 |",
            "|---|---|---|---|---|---|---|"]
    for run, base in GATE_RUNS:
        f = os.path.join(base, run, "pdist.json")
        if os.path.exists(f):
            d = json.load(open(f))
            out.append(f"| {names[run]} | {d['open_share']:.3f} | {d['p_median']:.3f} | {100 * d['p_gt_09']:.0f}% | "
                       f"{100 * d['p_05_06']:.0f}% | {100 * d['closed_p_lt_01']:.0f}% | " +
                       " ".join(f"{100 * x:.0f}%" for x in d["block_p_gt_09"]) + " |")
    out.append("")
    out += ["### Обещание градиента против реального переключения (те же 200 позиций; гейты реагируют, эффект на окно)\n",
            "Для smooth-вариантов обещание закрыть открытую группу = −s·dL/ds; у закрытых групп пути градиента нет (обещание "
            "0), они в «знак» не входят и показаны отдельно. Знак — среди решений с |реальное| > 10⁻³.\n",
            "| прогон | знак: всё (решения с путём) | Спирмен | знак: закрытия | знак: открытия | знак по блокам 1–8 | "
            "доля решений без пути градиента |", "|---|---|---|---|---|---|---|"]
    for run, base in GATE_RUNS:
        f2 = os.path.join(base, run, "gate_diag2.npz")
        f1 = os.path.join(base, run, "gate_diag.npz")
        if os.path.exists(f2):
            z = np.load(f2)
            pr, re, m = z["p_free_seq"], z["r_free_seq"], z["m"]
            hp = z["has_path"].astype(bool)
        elif os.path.exists(f1):
            z = np.load(f1)
            m = z["m"]
            pr, re = z["g_free_seq"] * (1 - 2 * m), z["r_free_seq"]
            hp = np.ones_like(m, bool)
        else:
            continue
        big = (np.abs(re) > 1e-3) & hp
        sg = lambda sel: f"{100 * (np.sign(pr[sel]) == np.sign(re[sel])).mean():.0f}%" if sel.any() else "—"   # noqa: E731
        blk = np.broadcast_to(np.arange(m.shape[1])[None, :, None], m.shape)
        out.append(f"| {names[run]} | {sg(big)} | {spearmanr(pr[hp], re[hp]).statistic:.2f} | {sg(big & (m > 0.5))} | "
                   f"{sg(big & (m < 0.5))} | " + " ".join(sg(big & (blk == l)) for l in range(m.shape[1])) +
                   f" | {100 * (~hp).mean():.0f}% |")
    out.append("")
    return out


def g14(out="report"):
    L = g14_tables()
    open(os.path.join(out, "r9_gate_variants.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    {"gate": gate, "ablate": ablate, "report": report, "lens": lens_kl, "agree": agree,
     "ablate_s1": lambda: ablate(runs=(("A_s1", VAST),)), "gate_variants": gate_variants, "g14": g14}[sys.argv[1]]()
