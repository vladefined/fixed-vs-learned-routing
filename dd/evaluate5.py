"""Round-3 evaluation (language v5, group variants).  python -m dd.evaluate5 runs5/E_s0 [data5]
Writes <run>/eval5.json (+ eval5_arrays.npz)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import (DEV, by_bins, ce_of, kurt_stats, load_run, load_split, partial_spearman,
                       probe_token)

THR = 0.01


def auroc(score, label):
    """P(score of a positive > score of a negative); label True = positive."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def forward_rows(model, rows, bs=128, **kw):
    for i in range(0, rows.shape[0], bs):
        idx = rows[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = model(idx, **kw)
        yield i, idx, logits.float(), aux


@torch.no_grad()
def answer_eval(model, rows, te, n_rows=None, **kw):
    """Answer accuracy (overall, by qm) + CE on the given rows, with optional ctl overrides."""
    n_rows = rows.shape[0] if n_rows is None else n_rows
    ok_all, qm_all, nll, cnt = [], [], 0.0, 0
    for i, idx, logits, _ in forward_rows(model, rows[:n_rows], **kw):
        lsm = F.log_softmax(logits[:, :-1], -1)
        tgt = idx[:, 1:]
        m = tgt != 0
        nll += float(-lsm.gather(-1, tgt[..., None]).squeeze(-1)[m].sum())
        cnt += int(m.sum())
        a = torch.tensor(te["copy"][i:i + idx.shape[0], 1:] > 0, device=DEV)
        ok_all.append((lsm.argmax(-1) == tgt)[a].cpu().numpy())
        qm_all.append(te["qm"][i:i + idx.shape[0], 1:][te["copy"][i:i + idx.shape[0], 1:] > 0])
    ok, qm = np.concatenate(ok_all), np.concatenate(qm_all)
    return dict(ce=nll / cnt, acc=float(ok.mean()), acc_by_m=by_bins(ok, qm, range(7)))


def flop_model(cfg, ctx):
    """Per-token, per-block FLOPs (multiply-adds x2) and the part that a standing group could skip
    in a group-blocked block (head h <-> group h, FFN split into G slices)."""
    d, hid, H = cfg.d, cfg.ffn_hidden, cfg.n_heads
    hd = d // H
    ffn = 2 * 3 * d * hid
    qkv, out = 2 * 3 * d * d, 2 * d * d
    scores = 2 * 2 * ctx * d                     # QK^T and AV over ctx keys
    attn = qkv + out + scores
    # per group (= head) skippable when it stands: its FFN slice, its query projection, its share of
    # scores/AV, its output projection slice.  K and V must still be produced for other queries.
    skip_attn_per_group = (2 * d * hd) + (2 * hd * d) + 2 * 2 * ctx * hd
    return dict(ffn=ffn, attn=attn, ffn_per_group=ffn / H, attn_skip_per_group=skip_attn_per_group)


def main(run, data="data5"):
    model, cfg = load_run(run)
    meta = json.load(open(os.path.join(data, "meta.json")))
    V = len(meta["vocab"])
    cls_names = sorted(set(meta["cls"]))
    cls_of = np.array([cls_names.index(c) for c in meta["cls"]])
    te, va = load_split(data, "test"), load_split(data, "val")
    rows = torch.tensor(te["tok"].astype(np.int64), device=DEV)
    vrows = torch.tensor(va["tok"].astype(np.int64), device=DEV)
    L, d = cfg.n_layers, cfg.d
    Gd = cfg.depth_groups if cfg.budget else 0
    NGr = cfg.n_heads                                   # groups for relative-change analysis
    gs = d // NGr
    res = dict(run=os.path.basename(run.rstrip("/\\")), budget=cfg.budget, G=Gd, L=L)

    # ------------------------------------------------------------ loss
    res["val_ce"] = ce_of(model, vrows)
    nv = int((va["tok"][:, 1:] != 0).sum())
    res["val_excess_ce"] = res["val_ce"] + float(va["sent_lp"].sum()) / nv
    res["test_ce"] = ce_of(model, rows)
    nt = int((te["tok"][:, 1:] != 0).sum())
    res["test_excess_ce"] = res["test_ce"] + float(te["sent_lp"].sum()) / nt

    # ------------------------------------------------------------ main pass
    mask_np = te["tok"] != 0
    N = int(mask_np.sum())
    flat = {k: te[k][mask_np] for k in te if te[k].ndim == 2}
    pred_ok = np.zeros(N, bool)
    tgt_nll = np.zeros(N, np.float32)
    flip_ok = np.zeros(N, bool)
    relchg_tok = np.zeros((N, L), np.float32)                 # RMS(dx)/RMS(x) per token
    rel_g = np.zeros((N, L, NGr), np.float16)                 # per group relative change
    dd_g = np.zeros((N, L, max(Gd, 1)), np.float16)
    dep_g = np.zeros((N, L, max(Gd, 1)), np.float16)          # depth after each block
    norm_layer = np.zeros(L + 1)
    maxmed = np.zeros(L + 1)
    coord_ms = np.zeros((L + 1, d))
    coord_mean = np.zeros((L + 1, d))
    feats = []
    off = 0
    for i, idx, logits, aux in forward_rows(model, rows, collect=True):
        B = idx.shape[0]
        m = idx != 0
        n = int(m.sum())
        sl = slice(off, off + n)
        lsm = F.log_softmax(logits[:, :-1], -1)
        tgt = idx[:, 1:]
        okp = torch.zeros_like(idx, dtype=torch.bool)
        nl = torch.zeros_like(idx, dtype=torch.float32)
        okp[:, 1:] = lsm.argmax(-1) == tgt
        nl[:, 1:] = -lsm.gather(-1, tgt[..., None]).squeeze(-1)
        fl = torch.tensor(te["flip"][i:i + B].astype(np.int64), device=DEV).clamp_min(0)
        fok = torch.zeros_like(okp)
        fok[:, 1:] = lsm.gather(-1, tgt[..., None]).squeeze(-1) > lsm.gather(-1, fl[:, 1:, None]).squeeze(-1)
        pred_ok[sl], tgt_nll[sl], flip_ok[sl] = okp[m].cpu().numpy(), nl[m].cpu().numpy(), fok[m].cpu().numpy()
        xs = [x.float() for x in aux["xs"]]
        for l, x in enumerate(xs):
            xm = x[m]
            norm_layer[l] += float(xm.norm(dim=-1).sum())
            ax = xm.abs()
            maxmed[l] += float((ax.max(-1).values / ax.median(-1).values.clamp_min(1e-12)).sum())
            coord_ms[l] += xm.pow(2).sum(0).double().cpu().numpy()
            coord_mean[l] += xm.sum(0).double().cpu().numpy()
        for l in range(L):
            dx = xs[l + 1] - xs[l]
            relchg_tok[sl, l] = (dx.pow(2).mean(-1).sqrt() / xs[l].pow(2).mean(-1).sqrt().clamp_min(1e-12))[m].cpu().numpy()
            dxg = dx.view(B, -1, NGr, gs).pow(2).mean(-1).sqrt()
            xg = xs[l].view(B, -1, NGr, gs).pow(2).mean(-1).sqrt().clamp_min(1e-12)
            rel_g[sl, l] = (dxg / xg)[m].cpu().numpy().astype(np.float16)
            if cfg.budget:
                dd_g[sl, l] = aux["dd"][l][m].cpu().numpy().astype(np.float16)
                dep_g[sl, l] = aux["depths"][l][m].cpu().numpy().astype(np.float16)
        if off < 80_000:
            feats.append(xs[-1][m].cpu())
        off += n
    res["norm_layer"] = (norm_layer / N).tolist()
    res["maxabs_over_median_layer"] = (maxmed / N).tolist()
    rms_c = np.sqrt(coord_ms / N)
    res["coord_rms_max_over_median_layer"] = (rms_c.max(1) / np.median(rms_c, 1)).tolist()
    res["coord_dc_fraction_layer"] = (np.abs(coord_mean / N) / np.maximum(rms_c, 1e-12)).max(1).tolist()

    # ------------------------------------------------------------ checks: answers, agreement
    an = flat["copy"] > 0
    qm = flat["qm"]
    res["answer_acc"] = float(pred_ok[an].mean())
    res["answer_acc_by_m"] = by_bins(pred_ok[an], qm[an], range(7))
    res["answer_nll_by_m"] = by_bins(tgt_nll[an], qm[an], range(7))
    res["answer_acc_by_k"] = by_bins(pred_ok[an], flat["nk"][an], (3, 4))
    res["agree_acc"] = float(flip_ok[flat["agree"] == 1].mean())
    res["close_top1"] = float(pred_ok[flat["close"] == 1].mean())

    # ------------------------------------------------------------ spent depth at the answer position
    aq = np.where(flat["ansq"] == 1)[0]
    wrong = ~pred_ok[aq + 1]                       # answer token follows the question token
    qm_aq = qm[aq]
    spent = {"relchg_sum": relchg_tok[aq].sum(1)}
    if cfg.budget:
        spent["final_depth"] = dep_g[aq, -1].astype(np.float32).mean(-1)
        spent["eff_update"] = np.minimum(dd_g[aq].astype(np.float32), 1.0).sum(1).mean(-1)
    res["spent_by_m"] = {k: by_bins(v, qm_aq, range(7)) for k, v in spent.items()}
    res["auroc_wrong"] = {k: dict(all=auroc(v, wrong),
                                  by_m={int(mm): auroc(v[qm_aq == mm], wrong[qm_aq == mm]) for mm in range(7)})
                          for k, v in spent.items()}
    res["n_wrong_by_m"] = {int(mm): int(wrong[qm_aq == mm].sum()) for mm in range(7)}
    res["spent_correct_vs_wrong"] = {k: {int(mm): [float(v[(qm_aq == mm) & ~wrong].mean()) if ((qm_aq == mm) & ~wrong).any() else None,
                                                   float(v[(qm_aq == mm) & wrong].mean()) if ((qm_aq == mm) & wrong).any() else None]
                                         for mm in range(7)} for k, v in spent.items()}

    # ------------------------------------------------------------ standing groups (two definitions)
    st_rel = rel_g < THR                                          # N, L, 8
    res["standing_rel_layer"] = st_rel.mean((0, 2)).tolist()
    if cfg.budget:
        st_dd = dd_g < THR
        res["standing_dd_layer"] = st_dd.mean((0, 2)).tolist()
        res["standing_agreement_layer"] = [dict(both=float((st_dd[:, l] & st_rel[:, l]).mean()),
                                                dd_only=float((st_dd[:, l] & ~st_rel[:, l]).mean()),
                                                rel_only=float((~st_dd[:, l] & st_rel[:, l]).mean()))
                                           for l in range(L)]
        sub = vrows[:1500]
        base = ce_of(model, sub)
        res["dce_zero_dd"] = ce_of(model, sub, ctl={"dd_zero_below": THR}) - base
        res["dce_zero_rel"] = ce_of(model, sub, ctl={"zero_rel_below": THR}) - base
        ar = answer_eval(model, rows, te, n_rows=2500)
        ar_rel = answer_eval(model, rows, te, n_rows=2500, ctl={"zero_rel_below": THR})
        ar_dd = answer_eval(model, rows, te, n_rows=2500, ctl={"dd_zero_below": THR})
        res["zeroing_answer"] = dict(base=ar["acc"], zero_rel=ar_rel["acc"], zero_dd=ar_dd["acc"],
                                     base_by_m=ar["acc_by_m"], zero_rel_by_m=ar_rel["acc_by_m"])
        # ------------------------------------------------------------ adaptive computation
        R = cfg.n_layers if cfg.budget_global else None
        dep_last = dep_g[:, -1].astype(np.float32)
        res["final_depth_mean"] = float(dep_last.mean())
        res["final_depth_group"] = dep_last.mean(0).tolist()
        # last block in which a (token, group) still moves (by real relative change); -1 = never moves
        mv = ~st_rel[:, :, :Gd] if Gd == NGr else ~st_rel
        last_move = np.where(mv.any(1), L - 1 - np.argmax(mv[:, ::-1], axis=1), -1)      # N, 8
        res["last_move_hist"] = [float((last_move == b).mean()) for b in range(-1, L)]
        if cfg.budget_global:
            exh = dep_g.astype(np.float32) >= (cfg.n_layers - 0.01)                       # N, L, G
            first_exh = np.where(exh.any(1), np.argmax(exh, axis=1), L)                  # L = never
            res["exhausted_by_block"] = [float((first_exh <= b).mean()) for b in range(L)]
            res["exhausted_by_m_answer"] = {int(mm): float((first_exh[aq][qm_aq == mm] < L).mean())
                                            for mm in range(7) if (qm_aq == mm).any()}
        res["last_move_by_m_answer"] = by_bins(last_move[aq].mean(-1), qm_aq, range(7))
        cls = cls_of[flat["tok"]]
        res["last_move_by_class"] = {c: float(last_move[cls == k].mean()) for k, c in enumerate(cls_names)
                                     if (cls == k).any()}
        res["standing_by_class"] = {c: float(st_rel[cls == k].mean()) for k, c in enumerate(cls_names)
                                    if (cls == k).any()}
        # compute that could be skipped
        ctx = float((flat["pos"] + 1).mean())
        fm = flop_model(cfg, ctx)
        frac_rel, frac_dd = float(st_rel.mean()), float(st_dd.mean())
        tot = fm["ffn"] + fm["attn"]
        res["skip_estimate"] = dict(
            ctx=ctx, frac_standing_rel=frac_rel, frac_standing_dd=frac_dd,
            ffn_saving=frac_rel, attn_saving=frac_rel * NGr * fm["attn_skip_per_group"] / fm["attn"],
            block_saving=frac_rel * (fm["ffn"] + NGr * fm["attn_skip_per_group"]) / tot,
            ffn_share_of_block=fm["ffn"] / tot)

    # ------------------------------------------------------------ representation: probes and overlap
    xf = torch.cat(feats)[:60000].numpy()
    ytok = flat["tok"][:xf.shape[0]].astype(np.int64)
    sid = flat["sid"][:xf.shape[0]]
    tr = sid < np.quantile(sid, 0.7)
    res["probe_full"] = probe_token(xf[tr], ytok[tr], xf[~tr], ytok[~tr], V)
    res["probe_groups"] = [probe_token(xf[tr][:, h * gs:(h + 1) * gs], ytok[tr], xf[~tr][:, h * gs:(h + 1) * gs],
                                       ytok[~tr], V) for h in range(NGr)]
    if cfg.budget and Gd == NGr:
        res["group_depth"] = dep_g[:, -1].astype(np.float32).mean(0).tolist()
        order = np.argsort(res["group_depth"])
        res["probe_shallow4_deep4"] = [float(np.mean([res["probe_groups"][h] for h in order[:4]])),
                                       float(np.mean([res["probe_groups"][h] for h in order[4:]]))]
    rng = np.random.default_rng(0)
    toks, sids = flat["tok"], flat["sid"]
    types, counts = np.unique(toks, return_counts=True)
    frequent = types[counts >= 20]
    jac = []
    for l in range(L):
        M = st_rel[:, l]
        same, diff = [], []
        for _ in range(4000):
            w = rng.choice(frequent)
            ii = np.where(toks == w)[0]
            a, b = rng.choice(ii, 2, replace=False)
            if sids[a] != sids[b]:
                u = (M[a] | M[b]).sum()
                same.append((M[a] & M[b]).sum() / u if u else np.nan)
            c, e = rng.integers(0, N, 2)
            if toks[c] != toks[e]:
                u = (M[c] | M[e]).sum()
                diff.append((M[c] & M[e]).sum() / u if u else np.nan)
        p = M.mean()
        jac.append(dict(p=float(p), same_word=float(np.nanmean(same)), diff_word=float(np.nanmean(diff)),
                        chance=float(p / (2 - p)) if p > 0 else 0.0))
    res["jaccard_standing"] = jac
    res["kurt_final"] = kurt_stats(torch.tensor(xf[:40000]))

    # ------------------------------------------------------------ gate predictability before the FFN
    if cfg.budget and Gd == NGr and cfg.gate_type not in ("hc", "ste"):
        res["gate_prediction"] = gate_prediction(model, cfg, data, rows, te, V)

    json.dump(res, open(os.path.join(run, "eval5.json"), "w"), indent=1)
    show = {k: res.get(k) for k in ("val_excess_ce", "answer_acc", "final_depth_mean", "dce_zero_rel", "dce_zero_dd")}
    print(json.dumps(show))
    print("acc by m", {k: round(v, 3) for k, v in res["answer_acc_by_m"].items()})
    print("standing rel", np.round(res["standing_rel_layer"], 3).tolist(),
          "dd", np.round(res.get("standing_dd_layer", []), 3).tolist())
    if "gate_prediction" in res:
        gp = res["gate_prediction"]
        print("gate pred acc table/head", gp["table"]["acc"], gp["head"]["acc"],
              "substitution acc", gp["table"]["skip"]["acc"], gp["head"]["skip"]["acc"], "base", gp["base"]["acc"])


# ---------------------------------------------------------------------- gate prediction
@torch.no_grad()
def _collect_gates(model, rows, n_rows):
    """Run the model, record per block: token ids, FFN input z, true gate sigma, allowed."""
    rec = {l: dict(z=[], g=[], allowed=[], tok=[]) for l in range(model.cfg.n_layers)}
    cur = {}

    def fn(blk, g, z, allowed):
        m = cur["m"]
        r = rec[blk.l]
        r["z"].append(z.float()[m].cpu())
        r["g"].append(g.float()[m].cpu())
        r["allowed"].append(allowed.float()[m].cpu())
        r["tok"].append(cur["idx"][m].cpu())
        return g

    for i in range(0, n_rows, 128):
        idx = rows[i:i + 128]
        cur["idx"], cur["m"] = idx, idx != 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(idx, ctl={"gate_fn": fn})
    return {l: {k: torch.cat(v) for k, v in r.items()} for l, r in rec.items()}


@torch.no_grad()
def _eval_with(model, rows, te, vrows, ctl_fn, cur, n_rows=2500):
    ok_all, qm_all = [], []
    for i in range(0, n_rows, 128):
        idx = rows[i:i + 128]
        cur["idx"] = idx
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx, ctl={"gate_fn": ctl_fn})
        lsm = F.log_softmax(logits.float()[:, :-1], -1)
        tgt = idx[:, 1:]
        sel = te["copy"][i:i + idx.shape[0], 1:] > 0
        ok_all.append((lsm.argmax(-1) == tgt)[torch.tensor(sel, device=DEV)].cpu().numpy())
        qm_all.append(te["qm"][i:i + idx.shape[0], 1:][sel])
    ok, qmv = np.concatenate(ok_all), np.concatenate(qm_all)
    tot, n = 0.0, 0
    for i in range(0, vrows.shape[0], 128):
        idx = vrows[i:i + 128]
        cur["idx"] = idx
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx, ctl={"gate_fn": ctl_fn})
        tgt = idx[:, 1:]
        mm = tgt != 0
        tot += float(F.cross_entropy(logits.float()[:, :-1][mm], tgt[mm], reduction="sum"))
        n += int(mm.sum())
    return dict(acc=float(ok.mean()), acc_by_m=by_bins(ok, qmv, range(7)), val_ce=tot / n)


def gate_prediction(model, cfg, data, rows, te, V, n_train_rows=1500, n_test_rows=1500):
    """Predict each group's gate BEFORE the FFN: (a) token -> mean gate table (train),
    (b) linear head on the FFN input z = rmsnorm(x + a_out) + depth_emb (trained on true gates)."""
    train = torch.from_numpy(np.load(os.path.join(data, "train_tok.npy"))[:n_train_rows].astype(np.int64)).to(DEV)
    tr = _collect_gates(model, train, n_train_rows)
    tst = _collect_gates(model, rows, n_test_rows)
    L, G = cfg.n_layers, cfg.depth_groups
    tables, heads = [], []
    for l in range(L):
        tok, g = tr[l]["tok"], tr[l]["g"]
        s = torch.zeros(V, G).index_add_(0, tok, g)
        c = torch.zeros(V).index_add_(0, tok, torch.ones(tok.shape[0]))
        tab = torch.where(c[:, None] > 0, s / c.clamp_min(1)[:, None], g.mean(0, keepdim=True).expand(V, G))
        tables.append(tab.to(DEV))
        X = tr[l]["z"].to(DEV)
        Y = tr[l]["g"].to(DEV).clamp(1e-6, 1 - 1e-6)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        W = torch.zeros(X.shape[1], G, device=DEV, requires_grad=True)
        b = torch.logit(Y.mean(0)).clone().requires_grad_(True)
        with torch.enable_grad():
            opt = torch.optim.Adam([W, b], lr=0.02)
            for _ in range(400):
                loss = F.binary_cross_entropy_with_logits((X - mu) / sd @ W + b, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        heads.append((mu, sd, W.detach(), b.detach()))

    def pred(kind, l, tok=None, z=None):
        if kind == "table":
            return tables[l][tok]
        mu, sd, W, b = heads[l]
        return torch.sigmoid((z.float() - mu) / sd @ W + b)

    out = {}
    for kind in ("table", "head"):
        accs, prec, recl, rate = [], [], [], []
        for l in range(L):
            al = tst[l]["allowed"].to(DEV)
            ph = pred(kind, l, tok=tst[l]["tok"].to(DEV), z=tst[l]["z"].to(DEV))
            true_st = al * tst[l]["g"].to(DEV) < THR
            pred_st = al * ph < THR
            accs.append(float((true_st == pred_st).float().mean()))
            prec.append(float((true_st & pred_st).sum() / pred_st.sum().clamp_min(1)))
            recl.append(float((true_st & pred_st).sum() / true_st.sum().clamp_min(1)))
            rate.append(float(true_st.float().mean()))
        out[kind] = dict(acc_layer=accs, acc=float(np.mean(accs)), precision_standing=prec,
                         recall_standing=recl, standing_rate=rate)

    va = load_split(data, "val")
    vrows = torch.tensor(va["tok"][:1500].astype(np.int64), device=DEV)
    cur = {}
    out["base"] = _eval_with(model, rows, te, vrows, lambda blk, g, z, allowed: g, cur)
    for kind in ("table", "head"):
        for mode in ("skip", "full"):
            def fn(blk, g, z, allowed, kind=kind, mode=mode):
                ph = pred(kind, blk.l, tok=cur["idx"], z=z).to(g.dtype)
                if mode == "full":
                    return ph
                # skip mode: groups predicted to stand get gate 0 (not computed); others keep the true gate
                return torch.where(allowed * ph < THR, torch.zeros_like(g), g)
            out[kind][mode] = _eval_with(model, rows, te, vrows, fn, cur)
            out[kind][mode]["frac_skipped"] = None
    return out


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
