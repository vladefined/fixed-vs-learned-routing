"""Evaluate a trained run:  python -m dd.evaluate runs/B_m_s0 [data_dir]
Writes <run>/eval.json and <run>/eval_arrays.npz."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from .model import Cfg, Model

DEV = torch.device("cuda")
THR = 0.01


def load_run(run):
    ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
    cfg = Cfg(**ck["cfg"])
    m = Model(cfg)
    m.load_state_dict(ck["state"])
    return m.to(DEV).eval(), cfg


def load_split(data, name):
    z = np.load(os.path.join(data, f"{name}.npz"))
    return {k: z[k] for k in z.files}


@torch.no_grad()
def ce_of(model, rows, bs=256, **kw):
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], bs):
        idx = rows[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx, **kw)
        tgt = idx[:, 1:]
        m = tgt != 0
        tot += float(F.cross_entropy(logits[:, :-1][m].float(), tgt[m], reduction="sum"))
        n += int(m.sum())
    return tot / n


@torch.no_grad()
def quick_struct(model, te, n_rows=600):
    """Agreement / answer / closure accuracy on the first n_rows test rows (learning curves)."""
    was = model.training
    model.eval()
    idx = torch.tensor(te["tok"][:n_rows].astype(np.int64), device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = torch.cat([model(idx[i:i + 200])[0].float() for i in range(0, n_rows, 200)])
    lp, tgt = logits[:, :-1], idx[:, 1:]
    out = {}
    for key in ("agree", "copy", "close"):
        a = torch.tensor(te[key][:n_rows, 1:], device=DEV).bool()
        lg, t = lp[a], tgt[a]
        if key == "agree":
            fl = torch.tensor(te["flip"][:n_rows, 1:], device=DEV).long()[a]
            ok = (lg.gather(1, t[:, None]) > lg.gather(1, fl[:, None])).squeeze(1)
            out["agree"] = float(ok.float().mean())
            att = torch.tensor(te["attractor"][:n_rows, 1:], device=DEV)[a].bool()
            out["agree_att"] = float(ok[att].float().mean())
        else:
            ok = lg.argmax(-1) == t
            out[key] = float(ok.float().mean())
            if key == "copy" and "nsw" in te:
                mk = "qm" if "qm" in te else "nsw"          # v5: swaps before each question
                nsw = torch.tensor(te[mk][:n_rows, 1:], device=DEV)[a]
                for lo, hi in ((0, 0), (1, 2), (3, 4), (5, 6), (4, 6)):
                    s = (nsw >= lo) & (nsw <= hi)
                    if s.any():
                        out[f"ans_m{lo}{hi}"] = float(ok[s].float().mean())
    model.train(was)
    return out


# ------------------------------------------------------------------ statistics helpers
def ols_cluster(y, X, clusters, names):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    _, inv = np.unique(clusters, return_inverse=True)
    S = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(S, inv, X * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return {n: dict(beta=float(b), se=float(s), t=float(b / s) if s > 0 else 0.0)
            for n, b, s in zip(names, beta, se) if not n.startswith("cls_")}


def residualize(v, Z):
    b, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ b


def demean_by(v, inv):
    return v - (np.bincount(inv, weights=v) / np.bincount(inv))[inv]


def partial_spearman(a, b, Z, clusters, n_boot=200, seed=0, within=False):
    """Rank partial correlation of a, b given nuisance Z; sentence-bootstrap 95% CI.
    within=True also removes sentence means (pure within-sentence relation)."""
    ra = residualize(stats.rankdata(a), Z)
    rb = residualize(stats.rankdata(b), Z)
    uniq, inv = np.unique(clusters, return_inverse=True)
    if within:
        ra, rb = demean_by(ra, inv), demean_by(rb, inv)
    r = float(np.corrcoef(ra, rb)[0, 1])
    G = len(uniq)
    sums = [np.bincount(inv, weights=v, minlength=G) for v in (ra, rb, ra * rb, ra * ra, rb * rb)]
    cnt = np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        Nw = (w * cnt).sum()
        sa, sb, sab, saa, sbb = [(w * s).sum() / Nw for s in sums]
        boots.append((sab - sa * sb) / math.sqrt((saa - sa ** 2) * (sbb - sb ** 2)))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return dict(r=r, ci=[float(lo), float(hi)])


def probe_token(Xtr, ytr, Xte, yte, n_cls, steps=500):
    Xtr = torch.tensor(Xtr, device=DEV, dtype=torch.float32)
    Xte = torch.tensor(Xte, device=DEV, dtype=torch.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = torch.tensor(ytr, device=DEV), torch.tensor(yte, device=DEV)
    W = torch.zeros(Xtr.shape[1], n_cls, device=DEV, requires_grad=True)
    b = torch.zeros(n_cls, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((Xte @ W + b).argmax(-1) == yte).float().mean())


def kurt_stats(x):
    x = x.double()

    def tok_kurt(v):
        c = v - v.mean(1, keepdim=True)
        return float(((c ** 4).mean(1) / (c ** 2).mean(1) ** 2).mean())

    def coord_kurt(v):
        c = v - v.mean(0, keepdim=True)
        return float(((c ** 4).mean(0) / (c ** 2).mean(0) ** 2).mean())

    def hoyer(v):
        n = v.shape[1]
        r = v.abs().sum(1) / (v.norm(dim=1) + 1e-12)
        return float(((math.sqrt(n) - r) / (math.sqrt(n) - 1)).mean())

    g = torch.Generator().manual_seed(0)
    R, _ = torch.linalg.qr(torch.randn(x.shape[1], x.shape[1], generator=g, dtype=torch.float64))
    xr = x @ R
    return dict(tok_kurt=tok_kurt(x), tok_kurt_rot=tok_kurt(xr), coord_kurt=coord_kurt(x),
                coord_kurt_rot=coord_kurt(xr), hoyer=hoyer(x), hoyer_rot=hoyer(xr))


def decompose(cnt_wc, n_w):
    """Frozen-mask variance decomposition.  cnt_wc[w, c] = #tokens of type w where unit c froze.
    Returns shares of total variance: static (which unit), lexical (which word), contextual (rest)."""
    N = n_w.sum()
    p_c = cnt_wc.sum(0) / N
    pbar = p_c.mean()
    tot = pbar * (1 - pbar)
    if tot <= 0:
        return dict(p=float(pbar), static=float("nan"), lexical=float("nan"), context=float("nan"))
    ok = n_w > 0
    p_wc = cnt_wc[ok] / n_w[ok, None]
    wt = (n_w[ok] / N)[:, None]
    S = ((p_c - pbar) ** 2).mean()
    Lx = (wt * (p_wc - p_c[None]) ** 2).sum(0).mean()
    C = (wt * p_wc * (1 - p_wc)).sum(0).mean()
    t = S + Lx + C
    return dict(p=float(pbar), static=float(S / t), lexical=float(Lx / t), context=float(C / t),
                always_frozen=float((p_c > 0.99).mean()), never_frozen=float((p_c < 0.01).mean()))


def by_bins(vals, key, keys):
    return {int(k): float(vals[key == k].mean()) for k in keys if (key == k).any()}


# ------------------------------------------------------------------ main
def main(run, data="data4"):
    model, cfg = load_run(run)
    meta = json.load(open(os.path.join(data, "meta.json")))
    V = len(meta["vocab"])
    cls_names = sorted(set(meta["cls"]))
    cls_of = np.array([cls_names.index(c) for c in meta["cls"]])
    te, va = load_split(data, "test"), load_split(data, "val")
    rows = torch.tensor(te["tok"].astype(np.int64), device=DEV)
    vrows = torch.tensor(va["tok"].astype(np.int64), device=DEV)
    L, d = cfg.n_layers, cfg.d
    G = cfg.depth_groups if cfg.budget else 0
    NG = cfg.norm_groups if cfg.norm_state else 1
    res = dict(run=os.path.basename(run.rstrip("/\\")), budget=cfg.budget, G=G, d=d, L=L,
               norm_state=cfg.norm_state, update=cfg.update)

    # ---------------------------------------------------------------- loss
    res["val_ce"] = ce_of(model, vrows)
    res["test_ce"] = ce_of(model, rows)
    n_tgt = int((te["tok"][:, 1:] != 0).sum())
    res["test_entropy_floor"] = float(-te["sent_lp"].sum() / n_tgt)
    res["val_entropy_floor"] = float(-va["sent_lp"].sum() / int((va["tok"][:, 1:] != 0).sum()))
    res["test_excess_ce"] = res["test_ce"] - res["test_entropy_floor"]
    res["val_excess_ce"] = res["val_ce"] - res["val_entropy_floor"]

    mask_np = te["tok"] != 0
    N = int(mask_np.sum())
    flat = {k: te[k][mask_np] for k in te if te[k].ndim == 2}
    closer_ids = torch.tensor([i for i, c in enumerate(meta["cls"]) if c == "close"], device=DEV)

    # per-token arrays
    pred_ok = np.zeros(N, bool)            # argmax == next token (at position of the *target*)
    tgt_nll = np.zeros(N, np.float32)      # -log p(token | prefix), at the target's position
    flip_ok = np.zeros(N, bool)            # agreement: correct form beats other number
    close_ok = np.zeros(N, bool)
    fin_depth = np.zeros(N, np.float32)
    eff = np.zeros(N, np.float32)          # sum_l mean_c min(dd,1)
    grp_depth = np.zeros((N, G), np.float32) if cfg.budget and G <= 16 else None
    dd_mean = np.zeros((N, L), np.float32)
    froz_dd = np.zeros((N, L), np.float32)
    froz_rel = np.zeros((N, L), np.float32)
    relchg = np.zeros((N, L), np.float32)  # RMS(dx)/RMS(x_prev) per token per layer
    agree4 = np.zeros((L, 4))              # coord-level: both, dd-only, rel-only, neither
    gate_hist = np.zeros((L, 50))
    dd_hist = np.zeros((L, 80))            # dd in [0, 8]
    ddmin_hist = np.zeros((L, 50))
    unit_zero = np.zeros((L, max(G, 1)))
    unit_logit = np.zeros((L, max(G, 1)))
    unit_final = np.zeros(max(G, 1))
    cnt_rel = torch.zeros(L, V, d, device=DEV)
    cnt_dd = torch.zeros(L, V, max(G, 1), device=DEV)
    n_w = torch.zeros(V, device=DEV)
    feats = []
    logits_hook = {}
    hooks = []
    if cfg.budget:
        for l, b in enumerate(model.blocks):
            hooks.append(b.wd.register_forward_hook(
                lambda mod, i, o, l=l: logits_hook.__setitem__(l, o.float())))
    off = 0
    with torch.no_grad():
        for i in range(0, rows.shape[0], 128):
            idx = rows[i:i + 128]
            B = idx.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(idx, collect=True)
            logits = logits.float()
            m = idx != 0
            n = int(m.sum())
            sl = slice(off, off + n)
            # predictions for token t are made at position t-1 -> shift right
            lsm = F.log_softmax(logits[:, :-1], -1)
            tgt = idx[:, 1:]
            nll = torch.zeros_like(idx, dtype=torch.float32)
            okp = torch.zeros_like(idx, dtype=torch.bool)
            nll[:, 1:] = -lsm.gather(-1, tgt[..., None]).squeeze(-1)
            okp[:, 1:] = lsm.argmax(-1) == tgt
            fl = torch.tensor(te["flip"][i:i + B].astype(np.int64), device=DEV).clamp_min(0)
            fok = torch.zeros_like(okp)
            fok[:, 1:] = lsm.gather(-1, tgt[..., None]).squeeze(-1) > lsm.gather(-1, fl[:, 1:, None]).squeeze(-1)
            lc = lsm[..., closer_ids]
            other = lc.masked_fill(closer_ids == tgt[..., None], float("-inf")).max(-1).values
            cok = torch.zeros_like(okp)
            cok[:, 1:] = lsm.gather(-1, tgt[..., None]).squeeze(-1) > other
            pred_ok[sl] = okp[m].cpu().numpy()
            tgt_nll[sl] = nll[m].cpu().numpy()
            flip_ok[sl] = fok[m].cpu().numpy()
            close_ok[sl] = cok[m].cpu().numpy()
            tok_m = idx[m]
            n_w += torch.bincount(tok_m, minlength=V).float()
            xs = aux["xs"]
            for l in range(L):
                x0, x1 = xs[l].float(), xs[l + 1].float()
                dx = (x1 - x0)
                if NG > 1:
                    rms = x0.view(B, -1, NG, d // NG).pow(2).mean(-1).sqrt().repeat_interleave(d // NG, -1)
                else:
                    rms = x0.pow(2).mean(-1, keepdim=True).sqrt()
                rel = dx.abs() / (rms + 1e-12)
                fr = (rel < THR)[m]                                            # n, d
                froz_rel[sl, l] = fr.float().mean(-1).cpu().numpy()
                relchg[sl, l] = (dx.pow(2).mean(-1).sqrt() / x0.pow(2).mean(-1).sqrt().clamp_min(1e-12))[m].cpu().numpy()
                cnt_rel[l].index_add_(0, tok_m, fr.float())
                if cfg.budget:
                    dd = aux["dd"][l][m]                                       # n, G
                    zd = dd < THR
                    froz_dd[sl, l] = zd.float().mean(-1).cpu().numpy()
                    dd_mean[sl, l] = dd.mean(-1).cpu().numpy()
                    zc = model.blocks[l].expand(zd.float()[None])[0].bool() if G != d else zd
                    agree4[l] += [float((zc & fr).sum()), float((zc & ~fr).sum()),
                                  float((~zc & fr).sum()), float((~zc & ~fr).sum())]
                    cnt_dd[l].index_add_(0, tok_m, zd.float())
                    unit_zero[l] += zd.sum(0).double().cpu().numpy()
                    zl = logits_hook[l][m]
                    unit_logit[l] += zl.sum(0).double().cpu().numpy()
                    gate_hist[l] += np.histogram(torch.sigmoid(zl).cpu().numpy(), bins=np.linspace(0, 1, 51))[0]
                    dd_hist[l] += np.histogram(dd.clamp_max(7.999).cpu().numpy(), bins=np.linspace(0, 8, 81))[0]
                    ddmin_hist[l] += np.histogram(dd.clamp_max(0.9999).cpu().numpy(), bins=np.linspace(0, 1, 51))[0]
                    eff[sl] += dd.clamp_max(1.0).mean(-1).cpu().numpy()
            if cfg.budget:
                fd = aux["depth"][m]
                fin_depth[sl] = fd.mean(-1).cpu().numpy()
                unit_final += fd.sum(0).double().cpu().numpy()
                if grp_depth is not None:
                    grp_depth[sl] = fd.cpu().numpy()
            if off < 80_000:
                feats.append(xs[-1].float()[m].cpu())
            off += n
    for h in hooks:
        h.remove()

    # ---------------------------------------------------------------- checks
    ag = flat["agree"] == 1
    att = flat["attractor"] == 1
    kind = flat["kind"]
    res["agree_acc"] = float(flip_ok[ag].mean())
    res["agree_acc_attractor"] = float(flip_ok[ag & att].mean())
    kind_names = {0: "bare_subj", 1: "subj_PP", 2: "subj_RC", 3: "in_subjRC", 4: "in_objRC"}
    res["agree_by_kind"] = {kn: float(flip_ok[ag & (kind == k)].mean()) for k, kn in kind_names.items()
                            if (ag & (kind == k)).any()}
    res["agree_by_kind_attractor"] = {kn: float(flip_ok[ag & att & (kind == k)].mean())
                                      for k, kn in kind_names.items() if (ag & att & (kind == k)).any()}
    cl = flat["close"] == 1
    res["close_type_acc"] = float(close_ok[cl].mean())
    res["close_top1_acc"] = float(pred_ok[cl].mean())
    an = flat["copy"] > 0                       # answer token
    nsw, nk = flat.get("nsw"), flat.get("nk")
    res["answer_acc"] = float(pred_ok[an].mean())
    res["answer_nll"] = float(tgt_nll[an].mean())
    if nsw is not None:
        res["answer_acc_by_m"] = by_bins(pred_ok[an], nsw[an], range(0, 7))
        res["answer_nll_by_m"] = by_bins(tgt_nll[an], nsw[an], range(0, 7))
        res["answer_acc_by_k"] = by_bins(pred_ok[an], nk[an], range(1, 6))
        res["answer_acc_by_k_m"] = {int(k): by_bins(pred_ok[an & (nk == k)], nsw[an & (nk == k)], range(0, 7))
                                    for k in np.unique(nk[an])}
    res["n_answer"] = int(an.sum())
    # excess CE split by part of the sentence (floor is unknown per part, so report mean NLL)
    part = flat.get("part")
    if part is not None:
        res["nll_by_part"] = {p: float(tgt_nll[(part == k) & (flat["pos"] > 0)].mean())
                              for k, p in ((0, "intro"), (1, "clause"), (2, "story"))}

    # ---------------------------------------------------------------- spent depth vs number of swaps
    aq = flat["ansq"] == 1 if "ansq" in flat else np.zeros(N, bool)
    ev = flat["ev"] if "ev" in flat else np.zeros(N, int)
    tot_rel = relchg.sum(1)
    if nsw is None:                       # data without a swap story (language v2/v3)
        nsw, nk = np.full(N, -1), np.full(N, -1)
    sp = {}
    for nm, sel in (("answer_pos", aq), ("event_tokens", ev > 0)):
        sp[nm] = {"relchg_sum": by_bins(tot_rel[sel], nsw[sel], range(0, 7)) if nsw is not None else {}}
        if cfg.budget:
            sp[nm]["final_depth"] = by_bins(fin_depth[sel], nsw[sel], range(0, 7))
            sp[nm]["eff_update"] = by_bins(eff[sel], nsw[sel], range(0, 7))
            sp[nm]["dd_layer_by_m"] = {int(k): dd_mean[sel & (nsw == k)].mean(0).tolist()
                                       for k in range(0, 7) if (sel & (nsw == k)).any()}
        sp[nm]["relchg_layer_by_m"] = {int(k): relchg[sel & (nsw == k)].mean(0).tolist()
                                       for k in range(0, 7) if (sel & (nsw == k)).any()}
    # event tokens by event index (position in the story) and by token kind
    evm = ev > 0
    sp["event_by_index"] = {"relchg_sum": by_bins(tot_rel[evm], ev[evm], range(1, 7))}
    swap_id = meta["vocab"].index("swap") if "swap" in meta["vocab"] else -1
    is_swap = flat["tok"] == swap_id
    sp["swap_token"] = {"relchg_sum": by_bins(tot_rel[is_swap], nsw[is_swap], range(1, 7)) if nsw is not None else {}}
    if cfg.budget:
        sp["event_by_index"]["final_depth"] = by_bins(fin_depth[evm], ev[evm], range(1, 7))
        sp["swap_token"]["final_depth"] = by_bins(fin_depth[is_swap], nsw[is_swap], range(1, 7))
        # correct vs wrong answers at the same m
        ans_pos_ok = np.zeros(N, bool)
        idx_an = np.where(an)[0]
        ans_pos_ok[idx_an - 1] = pred_ok[idx_an]   # answer position is the token before the answer
        sp["answer_pos_correct_vs_wrong"] = {
            int(k): [float(fin_depth[aq & (nsw == k) & ans_pos_ok].mean()) if (aq & (nsw == k) & ans_pos_ok).any() else None,
                     float(fin_depth[aq & (nsw == k) & ~ans_pos_ok].mean()) if (aq & (nsw == k) & ~ans_pos_ok).any() else None]
            for k in range(0, 7)}
        # regression with sentence-level controls: depth at answer position ~ m + k + length
        if aq.any():
            y = fin_depth[aq].astype(np.float64)
            Xr = np.column_stack([np.ones(aq.sum()), nsw[aq], nk[aq], flat["pos"][aq]])
            sp["answer_pos_ols"] = ols_cluster((y - y.mean()) / (y.std() + 1e-12), Xr,
                                               flat["sid"][aq], ["const", "m", "k", "pos"])
            sp["answer_pos_spearman_m"] = float(stats.spearmanr(y, nsw[aq])[0])
    res["spent"] = sp

    # ---------------------------------------------------------------- frozen units, two definitions
    res["frozen_rel_layer"] = froz_rel.mean(0).tolist()
    res["relchg_layer"] = relchg.mean(0).tolist()
    nw = n_w.cpu().numpy()
    res["decomp_rel"] = [decompose(cnt_rel[l].cpu().numpy().astype(np.float64), nw) for l in range(L)]
    if cfg.budget:
        res["frozen_dd_layer"] = froz_dd.mean(0).tolist()
        a4 = agree4 / agree4.sum(1, keepdims=True)
        res["frozen_agreement"] = [dict(both=float(r[0]), dd_only=float(r[1]), rel_only=float(r[2]),
                                        neither=float(r[3])) for r in a4]
        res["decomp_dd"] = [decompose(cnt_dd[l].cpu().numpy().astype(np.float64), nw) for l in range(L)]
        res["dd_layer_mean"] = dd_mean.mean(0).tolist()
        res["final_depth_mean"] = float(fin_depth.mean())
        res["eff_update_mean"] = float(eff.mean())
        res["unit_zero_layer"] = (unit_zero / N).tolist()
        res["unit_logit_layer"] = (unit_logit / N).tolist()
        res["unit_final_depth"] = (unit_final / N).tolist()
        res["static_zero_frac_layer"] = ((unit_zero / N) > 0.99).mean(1).tolist()
        res["dead_gate_frac_layer"] = (((unit_zero / N) > 0.99) & ((unit_logit / N) < -10)).mean(1).tolist()
        sub = vrows[:1500]
        base = ce_of(model, sub)
        res["dce_dd_zeroed"] = {str(t): ce_of(model, sub, ctl={"dd_zero_below": t}) - base
                                for t in (0.01, 0.05, 0.1)}
    # ---------------------------------------------------------------- block removal & axis alignment
    sub = vrows[:1500]
    base = ce_of(model, sub)
    res["block_removal_dce"] = [ce_of(model, sub, skip={l}) - base for l in range(L)]
    xf = torch.cat(feats)[:60000]
    res["kurt_final"] = kurt_stats(xf)
    res["norm_final"] = float(xf.norm(dim=-1).mean())

    # ---------------------------------------------------------------- probes: which token, per group
    gs = d // 8
    ytok = flat["tok"][:xf.shape[0]].astype(np.int64)
    sid = flat["sid"][:xf.shape[0]]
    tr = sid < np.quantile(sid, 0.7)
    if cfg.budget and G == d:
        order = np.argsort(unit_final)
        groups = [(f"depthbin{h}", order[h * gs:(h + 1) * gs].tolist()) for h in range(8)]
    else:
        groups = [(f"g{h}", list(range(h * gs, (h + 1) * gs))) for h in range(8)]
    X = xf.numpy()
    res["probe_full"] = probe_token(X[tr], ytok[tr], X[~tr], ytok[~tr], V)
    res["probe_groups"] = {gn: probe_token(X[tr][:, c], ytok[tr], X[~tr][:, c], ytok[~tr], V) for gn, c in groups}
    if cfg.budget and G > 1:
        uf = unit_final / N
        res["probe_group_depth"] = {gn: float(uf[c].mean()) if G == d else float(uf[int(gn[1:])])
                                    for gn, c in groups}

    # ---------------------------------------------------------------- structure (clause tokens only)
    arrays = dict(gate_hist=gate_hist, dd_hist=dd_hist, ddmin_hist=ddmin_hist, froz_rel=froz_rel.mean(0))
    if cfg.budget:
        cpart = (flat["part"] == 1) if "part" in flat else np.ones(N, bool)
        cls = cls_of[flat["tok"]][cpart]
        pos = flat["pos"][cpart].astype(np.float64)
        Z = np.column_stack([np.eye(len(cls_names))[cls], pos, pos ** 2, np.log1p(pos)])
        clusters = flat["sid"][cpart]
        td = flat["tree_depth"][cpart].astype(np.float64)
        ed = flat["emb_depth"][cpart].astype(np.float64)
        y = fin_depth[cpart].astype(np.float64)
        st = dict(n_tokens=int(cpart.sum()))
        st["spearman_tree_raw"] = float(stats.spearmanr(y, td)[0])
        st["partial_tree"] = partial_spearman(y, td, Z, clusters)
        st["partial_emb"] = partial_spearman(y, ed, Z, clusters)
        st["partial_tree_within"] = partial_spearman(y, td, Z, clusters, within=True)
        st["partial_emb_within"] = partial_spearman(y, ed, Z, clusters, within=True)
        st["partial_tree_dd_by_layer"] = [partial_spearman(dd_mean[cpart, l].astype(np.float64), td, Z,
                                                           clusters, n_boot=0)["r"] for l in range(L)]
        Xr = np.column_stack([Z, td, flat["opens"][cpart], flat["closes"][cpart], flat["is_head"][cpart]])
        names = [f"cls_{c}" for c in cls_names] + ["pos", "pos2", "logpos", "tree_depth", "opens",
                                                    "closes", "is_head"]
        st["ols_final_depth_std"] = ols_cluster((y - y.mean()) / (y.std() + 1e-12), Xr, clusters, names)
        st["mean_by_class"] = {c: float(y[cls == k].mean()) for k, c in enumerate(cls_names) if (cls == k).any()}
        st["by_tree_depth"] = {int(v): float(y[td == v].mean()) for v in np.unique(td)}
        st["by_emb_depth"] = {int(v): float(y[ed == v].mean()) for v in np.unique(ed)}
        hd = flat["is_head"][cpart]
        st["head"] = [float(y[hd == 0].mean()), float(y[hd == 1].mean())]
        ki, ka = cls_names.index("noun"), cls_names.index("adj")
        st["noun_vs_adj"] = [float(y[cls == ki].mean()), float(y[cls == ka].mean())]
        nn_ = cls == ki
        closes = flat["closes"][cpart]
        st["noun_close_le1_vs_gt1"] = [float(y[nn_ & (closes <= 1)].mean()), float(y[nn_ & (closes > 1)].mean())]
        if grp_depth is not None and G > 1:
            gd = grp_depth[cpart].astype(np.float64)
            st["group_partial_tree"] = [partial_spearman(gd[:, h], td, Z, clusters, n_boot=0)["r"] for h in range(G)]
            st["group_final_depth"] = grp_depth.mean(0).tolist()
            st["group_final_depth_by_part"] = {p: grp_depth[flat["part"] == k].mean(0).tolist()
                                               for k, p in ((0, "intro"), (1, "clause"), (2, "story"))}
        if cfg.attn_depth_bias:
            st["slopes"] = [b.slope.detach().cpu().tolist() for b in model.blocks]
        res["structure"] = st
        keep = flat["sid"] < flat["sid"].min() + 1500
        arrays.update(tok_sid=flat["sid"][keep], tok_pos=flat["pos"][keep], tok=flat["tok"][keep],
                      tok_fin=fin_depth[keep], tok_dd=dd_mean[keep], tok_tree=flat["tree_depth"][keep],
                      unit_zero=unit_zero / N, unit_final=unit_final / N)
    json.dump(res, open(os.path.join(run, "eval.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(run, "eval_arrays.npz"), **arrays)
    show = {k: res.get(k) for k in ("val_excess_ce", "test_excess_ce", "agree_acc", "agree_acc_attractor",
                                    "close_top1_acc", "answer_acc", "final_depth_mean", "dce_dd_zeroed")}
    print(json.dumps(show))
    if "answer_acc_by_m" in res:
        print("answer acc by m", {k: round(v, 3) for k, v in res["answer_acc_by_m"].items()},
              "by k", {k: round(v, 3) for k, v in res["answer_acc_by_k"].items()})
    print("frozen(rel) by layer", np.round(res["frozen_rel_layer"], 3).tolist())
    if cfg.budget:
        print("frozen(dd) by layer", np.round(res["frozen_dd_layer"], 3).tolist())
        print("depth@answer by m", {k: round(v, 3) for k, v in res["spent"]["answer_pos"]["final_depth"].items()})
        print("partial tree", res["structure"]["partial_tree"], "within", res["structure"]["partial_tree_within"])


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
