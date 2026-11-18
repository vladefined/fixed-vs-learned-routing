"""Round-4 specific evaluation (complements dd.evaluate5, which must run first for the common metrics).
python -m dd.evaluate_r4 runs6/QL1w_s0 [data5]   -> <run>/eval_r4.json

Q  (quantised state): level occupancy / edges / dead levels, levels crossed per block, exactness of standing
   coordinates, "depth" = sum of levels crossed (per coordinate, per group) at the answer vs m, AUROC,
   token-table prediction of standing coordinates/groups before the FFN and the cost of substituting it,
   skippable compute.
K-STE: shift shares, train == eval check, shifted groups at the answer vs m, AUROC, whole-FFN skip, and the
   checkpoint trajectory of the shift placement (agreement with the final placement / with the previous one).
+free: what the free (unnormalised) coordinates carry: correlations with m, number of names, position,
   swaps so far; linear probe of m from free vs normalised coordinates; ablation of free-coordinate updates.
I': sparse coordinates (share of coordinates whose median |x_i| < 0.1 mean |x_i|) -- also for references.
QG (round 5): paid vs real group changes, wasted openings, exactness of standing groups, level occupancy, dead /
   always-active groups, shifts at the answer vs m, AUROC, token-table prediction of the decision and the cost of
   substituting it, skippable compute (round-4-comparable estimate and the one realisable with the gate placed
   after attention), checkpoint trajectory of the decision placement."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import DEV, by_bins, ce_of, load_run, load_split
from .evaluate5 import answer_eval, auroc, flop_model
from .model import Cfg, Model


def family(cfg):
    if getattr(cfg, "qg", ""):
        return "QG"
    if cfg.quant:
        return "Q"
    if cfg.gate_type == "ste":
        return "KSTE"
    if cfg.free_per_group:
        return "free"
    if cfg.rel_cost and cfg.rel_scale == "median":
        return "Ip"
    return "other"


def swaps_so_far(te):
    """Running max of the swap-event index within each sentence (0 before the first swap)."""
    ev, sid = te["ev"].astype(np.int64), te["sid"]
    out = np.zeros_like(ev)
    for r in range(ev.shape[0]):
        cur, cs = 0, None
        for t in range(ev.shape[1]):
            if sid[r, t] != cs:
                cur, cs = 0, sid[r, t]
            cur = max(cur, ev[r, t])
            out[r, t] = cur
    return out


@torch.no_grad()
def run_rows(model, rows, bs=128, **kw):
    for i in range(0, rows.shape[0], bs):
        idx = rows[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = model(idx, **kw)
        yield i, idx, logits.float(), aux


def answer_ok(logits, idx, te, i):
    lsm = F.log_softmax(logits[:, :-1], -1)
    tgt = idx[:, 1:]
    ok = torch.zeros_like(idx, dtype=torch.bool)
    ok[:, 1:] = lsm.argmax(-1) == tgt
    return ok


# ====================================================================================== Q
@torch.no_grad()
def eval_q(model, cfg, te, rows, vrows, data, V, n_rows=1500):
    L, d, K, D = cfg.n_layers, cfg.d, cfg.q_levels, cfg.q_delta
    G = cfg.n_heads
    gs = d // G
    occ = np.zeros((L + 1, 2 * K + 1))
    n_hist = np.zeros((L, 4))                      # n = 0, 1, 2, >=3
    l1 = np.zeros(L)
    l0 = np.zeros(L)
    gstand = np.zeros(L)                           # share of (token, group) with all coords standing
    max_stand_change = 0.0
    max_n_vs_dx = 0.0
    ntok = 0
    dep_aq, m_aq, ok_aq, depg_aq = [], [], [], []
    coord_dep_aq = []
    for i, idx, logits, aux in run_rows(model, rows[:n_rows], collect=True):
        B = idx.shape[0]
        msk = idx != 0
        ntok += int(msk.sum())
        for l, x in enumerate(aux["xs"]):
            lv = torch.round(x.float() / D)[msk].long() + K
            occ[l] += torch.bincount(lv.flatten(), minlength=2 * K + 1).cpu().numpy()
        for l in range(L):
            n = aux["dd"][l]
            nm = n[msk]
            l1[l] += float(nm.sum())
            l0[l] += float((nm > 0).float().sum())
            n_hist[l] += np.array([float((nm == 0).sum()), float((nm == 1).sum()), float((nm == 2).sum()),
                                   float((nm >= 3).sum())])
            gstand[l] += float((nm.view(-1, G, gs).amax(-1) == 0).float().sum())
            dx = (aux["xs"][l + 1] - aux["xs"][l])[msk]
            st = nm == 0
            if st.any():
                max_stand_change = max(max_stand_change, float(dx[st].abs().max()))
            max_n_vs_dx = max(max_n_vs_dx, float((dx.abs() / D - nm).abs().max()))
        aq = torch.tensor(te["ansq"][i:i + B] == 1, device=DEV)
        if aq.any():
            ok = answer_ok(logits, idx, te, i)
            okn = torch.zeros_like(ok)
            okn[:, :-1] = ok[:, 1:]                          # answer token follows the question token
            ntot = torch.stack(aux["dd"]).sum(0)             # B,T,d: levels crossed per coordinate
            dep_aq.append(ntot.mean(-1)[aq].cpu().numpy())
            depg_aq.append(ntot.view(B, -1, G, gs).mean(-1)[aq].cpu().numpy())
            coord_dep_aq.append((ntot > 0).float().mean(-1)[aq].cpu().numpy())
            m_aq.append(te["qm"][i:i + B][te["ansq"][i:i + B] == 1])
            ok_aq.append(okn[aq].cpu().numpy())
    res = {}
    occ_n = occ / occ.sum(1, keepdims=True)
    res["level_occupancy"] = occ_n.tolist()                   # L+1 x 17, level -K..K
    res["edge_share_layer"] = (occ_n[:, 0] + occ_n[:, -1]).tolist()
    res["zero_share_layer"] = occ_n[:, K].tolist()
    res["dead_levels_layer"] = [int((r < 1e-4).sum()) for r in occ_n]
    res["levels_used_99_layer"] = [int((np.cumsum(np.sort(r)[::-1]) < 0.99).sum() + 1) for r in occ_n]
    tot = ntok * d
    res["l1_layer"] = (l1 / tot).tolist()
    res["moved_share_layer"] = (l0 / tot).tolist()
    res["n_hist_layer"] = (n_hist / tot).tolist()
    res["group_standing_layer"] = (gstand / (ntok * G)).tolist()
    res["max_abs_change_standing_coord"] = max_stand_change
    res["max_abs_n_minus_dx_over_delta"] = max_n_vs_dx
    dep, mm, okv = np.concatenate(dep_aq), np.concatenate(m_aq), np.concatenate(ok_aq)
    depg = np.concatenate(depg_aq)
    cd = np.concatenate(coord_dep_aq)
    wrong = ~okv
    res["answer_depth_by_m"] = by_bins(dep, mm, range(7))            # mean levels crossed per coordinate
    res["answer_moved_coord_share_by_m"] = by_bins(cd, mm, range(7))
    res["answer_group_depth_by_m"] = {int(k): depg[mm == k].mean(0).tolist() for k in range(7) if (mm == k).any()}
    res["auroc_wrong"] = dict(all=auroc(dep, wrong),
                              by_m={int(k): auroc(dep[mm == k], wrong[mm == k]) for k in range(7)})
    res["n_wrong_by_m"] = {int(k): int(wrong[mm == k].sum()) for k in range(7)}

    # --------------------------------------------- standing prediction before the FFN (token table)
    train = torch.from_numpy(np.load(os.path.join(data, "train_tok.npy"))[:1500].astype(np.int64)).to(DEV)
    cnt = torch.zeros(V, device=DEV)
    st_c = [torch.zeros(V, d, device=DEV) for _ in range(L)]
    st_g = [torch.zeros(V, G, device=DEV) for _ in range(L)]
    for _, idx, _, aux in run_rows(model, train, collect=True):
        msk = idx != 0
        tok = idx[msk]
        cnt.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float32))
        for l in range(L):
            s = (aux["dd"][l][msk] == 0).float()
            st_c[l].index_add_(0, tok, s)
            st_g[l].index_add_(0, tok, (s.view(-1, G, gs).amin(-1)))
    tab_c = [s / cnt.clamp_min(1)[:, None] for s in st_c]
    tab_g = [s / cnt.clamp_min(1)[:, None] for s in st_g]
    pred = {}
    for kind, tabs, unit in (("coord", tab_c, 1), ("group", tab_g, gs)):
        acc, prec, rec, rate, prate = [], [], [], [], []
        for l in range(L):
            tp = fp = fn = tn = 0.0
            for _, idx, _, aux in run_rows(model, rows[:n_rows], collect=True):
                msk = idx != 0
                tok = idx[msk]
                n = aux["dd"][l][msk]
                true = (n == 0) if unit == 1 else (n.view(-1, G, gs).amax(-1) == 0)
                p = tabs[l][tok] > 0.5
                tp += float((true & p).sum()); fp += float((~true & p).sum())
                fn += float((true & ~p).sum()); tn += float((~true & ~p).sum())
            tot_ = tp + fp + fn + tn
            acc.append((tp + tn) / tot_); prec.append(tp / max(tp + fp, 1)); rec.append(tp / max(tp + fn, 1))
            rate.append((tp + fn) / tot_); prate.append((tp + fp) / tot_)
        pred[kind] = dict(acc_layer=acc, precision_standing=prec, recall_standing=rec, standing_rate=rate,
                          predicted_standing_rate=prate)

    cur = {}

    def make_fn(kind):
        tabs = tab_c if kind == "coord" else tab_g

        def fn(blk, upd, x):
            p = tabs[blk.l][cur["idx"]] > 0.5                       # B,T,(d or G)
            if kind == "group":
                p = p.repeat_interleave(gs, dim=-1)
            return torch.where(p, torch.zeros_like(upd), upd)
        return fn

    def eval_ctl(ctl):
        ok_all, qm_all, tot_, n_ = [], [], 0.0, 0
        for i in range(0, 2500, 128):
            idx = rows[i:i + 128]
            cur["idx"] = idx
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(idx, ctl=ctl)
            lsm = F.log_softmax(logits.float()[:, :-1], -1)
            sel = te["copy"][i:i + idx.shape[0], 1:] > 0
            ok_all.append((lsm.argmax(-1) == idx[:, 1:])[torch.tensor(sel, device=DEV)].cpu().numpy())
            qm_all.append(te["qm"][i:i + idx.shape[0], 1:][sel])
        for i in range(0, vrows.shape[0], 128):
            idx = vrows[i:i + 128]
            cur["idx"] = idx
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(idx, ctl=ctl)
            tgt = idx[:, 1:]
            mm_ = tgt != 0
            tot_ += float(F.cross_entropy(logits.float()[:, :-1][mm_], tgt[mm_], reduction="sum"))
            n_ += int(mm_.sum())
        ok, qmv = np.concatenate(ok_all), np.concatenate(qm_all)
        return dict(acc=float(ok.mean()), acc_by_m=by_bins(ok, qmv, range(7)), val_ce=tot_ / n_)

    pred["base"] = eval_ctl(None)
    for kind in ("coord", "group"):
        pred[kind]["skip"] = eval_ctl({"q_upd_fn": make_fn(kind)})
    res["standing_prediction"] = pred

    # --------------------------------------------- compute that could be skipped (group-blocked block)
    ctx = float((te["pos"][te["tok"] != 0] + 1).mean())
    fm = flop_model(cfg, ctx)
    tot_f = fm["ffn"] + fm["attn"]

    def saving(frac):
        return dict(ffn=frac, attn=frac * G * fm["attn_skip_per_group"] / fm["attn"],
                    block=frac * (fm["ffn"] + G * fm["attn_skip_per_group"]) / tot_f)
    gs_post = float(np.mean(res["group_standing_layer"]))
    gp = pred["group"]
    res["skip_estimate"] = dict(post_hoc=saving(gs_post),
                                table_predicted=saving(float(np.mean(gp["predicted_standing_rate"]))),
                                frac_group_standing=gs_post)
    return res


# ====================================================================================== QG
@torch.no_grad()
def qg_stats(model, cfg, te, rows, n_rows=1500, with_answer=True):
    """aux['dd'][l] = m (gate decision, or moved for nogate), aux['rc'][l] = real (some coordinate changed)."""
    L, G, d, K, D = cfg.n_layers, cfg.depth_groups, cfg.d, cfg.q_levels, cfg.q_delta
    gs = d // G
    paid, real, wasted, none_open = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
    rate = np.zeros((L, G))
    occ = np.zeros((L + 1, 2 * K + 1))
    ntok = 0
    max_stand = 0.0
    a_s, a_r, a_m, a_ok = [], [], [], []
    decisions = []
    for i, idx, logits, aux in run_rows(model, rows[:n_rows], collect=True):
        B = idx.shape[0]
        msk = idx != 0
        ntok += int(msk.sum())
        for l, x in enumerate(aux["xs"]):
            lv = torch.round(x.float() / D)[msk].long() + K
            occ[l] += torch.bincount(lv.flatten(), minlength=2 * K + 1).cpu().numpy()
        decisions.append(torch.stack(aux["dd"], 1).transpose(1, 2)[msk].bool().cpu())     # n, L, G
        for l in range(L):
            mm, rr = aux["dd"][l][msk], aux["rc"][l][msk]
            paid[l] += float(mm.sum())
            real[l] += float(rr.sum())
            wasted[l] += float(((mm > 0) & (rr == 0)).sum())
            none_open[l] += float((mm.sum(-1) == 0).sum())
            rate[l] += mm.sum(0).cpu().numpy()
            x0 = aux["xs"][l].view(B, -1, G, gs)[msk]
            x1 = aux["xs"][l + 1].view(B, -1, G, gs)[msk]
            st = mm == 0
            if st.any():
                max_stand = max(max_stand, float((x1 - x0)[st].abs().max()))
        if with_answer:
            aq = torch.tensor(te["ansq"][i:i + B] == 1, device=DEV)
            if aq.any():
                ok = answer_ok(logits, idx, te, i)
                okn = torch.zeros_like(ok)
                okn[:, :-1] = ok[:, 1:]
                a_s.append(torch.stack(aux["dd"]).sum(0).sum(-1)[aq].cpu().numpy())
                a_r.append(torch.stack(aux["rc"]).sum(0).sum(-1)[aq].cpu().numpy())
                a_m.append(te["qm"][i:i + B][te["ansq"][i:i + B] == 1])
                a_ok.append(okn[aq].cpu().numpy())
    r = rate / ntok
    occ_n = occ / occ.sum(1, keepdims=True)
    out = dict(paid_layer=(paid / (ntok * G)).tolist(), real_layer=(real / (ntok * G)).tolist(),
               paid=float(paid.sum() / (ntok * G * L)), real=float(real.sum() / (ntok * G * L)),
               wasted_of_paid_layer=(wasted / np.maximum(paid, 1)).tolist(),
               token_none_open_layer=(none_open / ntok).tolist(),
               always_open_frac_layer=(r > 0.99).mean(1).tolist(), never_open_frac_layer=(r < 0.01).mean(1).tolist(),
               group_rate=r.tolist(), max_abs_change_of_standing_group=max_stand,
               level_occupancy=occ_n.tolist(), edge_share_layer=(occ_n[:, 0] + occ_n[:, -1]).tolist(),
               zero_share_layer=occ_n[:, K].tolist(), dead_levels_layer=[int((q < 1e-4).sum()) for q in occ_n])
    if with_answer:
        s_, r_, m, ok = (np.concatenate(v) for v in (a_s, a_r, a_m, a_ok))
        out["answer_shifts_by_m"] = by_bins(s_, m, range(7))
        out["answer_real_by_m"] = by_bins(r_, m, range(7))
        out["auroc_wrong"] = dict(all=auroc(s_, ~ok), by_m={int(k): auroc(s_[m == k], ~ok[m == k]) for k in range(7)})
        out["n_wrong_by_m"] = {int(k): int((~ok)[m == k].sum()) for k in range(7)}
    return out, torch.cat(decisions)


def qg_flops(cfg, ctx, frac_stand, frac_tok_none):
    """Skippable share of block compute.
    comparable: the round-4 group-blocked estimate (a standing group skips its FFN slice and its head's query,
                scores and output slice) -- the gate is assumed to be known before attention.
    realisable: the gate reads rmsnorm(x + a), so attention (all heads) must run first.
                QG:       standing group skips its rows of w2; a token with no open group skips the whole FFN.
                QG-split: standing group skips its FFN block (w1, w3, w2 slices).
                The gate head itself is added."""
    fm = flop_model(cfg, ctx)
    d, hid, G = cfg.d, cfg.ffn_hidden, cfg.depth_groups
    gs = d // G
    tot = fm["ffn"] + fm["attn"]
    gate = 2 * d * G
    comp = frac_stand * (fm["ffn"] + G * fm["attn_skip_per_group"]) / tot
    if cfg.split_ffn:
        ffn_skip = frac_stand * fm["ffn"]
    else:
        w13 = 2 * 2 * d * hid
        w2g = 2 * hid * gs
        # tokens with no open group: whole FFN; otherwise w2 rows of the standing groups
        ffn_skip = frac_tok_none * fm["ffn"] + (frac_stand * G - frac_tok_none * G) * w2g
        ffn_skip = max(ffn_skip, 0.0)
    real = (ffn_skip - (gate if cfg.qg == "gate" else 0)) / tot
    return dict(comparable_block=comp, realisable_block=real if cfg.qg == "gate" else 0.0,
                realisable_ffn=ffn_skip / fm["ffn"] if cfg.qg == "gate" else 0.0,
                ffn_share_of_block=fm["ffn"] / tot, ctx=ctx)


@torch.no_grad()
def eval_qg(model, cfg, te, rows, vrows, data, V, run, n_rows=1500):
    L, G, d = cfg.n_layers, cfg.depth_groups, cfg.d
    gs = d // G
    res = {}
    st, final_dec = qg_stats(model, cfg, te, rows, n_rows)
    res["stats"] = st
    idx = rows[:64]
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lt, _ = model(idx)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        le, _ = model(idx)
    res["train_eval_max_abs_logit_diff"] = float((lt.float() - le.float()).abs().max())
    res["mu_final"] = model.mu.tolist()

    # ------------------------------------------- token table "token -> decision" per (block, group)
    train = torch.from_numpy(np.load(os.path.join(data, "train_tok.npy"))[:1500].astype(np.int64)).to(DEV)
    cnt = torch.zeros(V, device=DEV)
    op = [torch.zeros(V, G, device=DEV) for _ in range(L)]
    for _, idx, _, aux in run_rows(model, train, collect=True):
        msk = idx != 0
        tok = idx[msk]
        cnt.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float32))
        for l in range(L):
            op[l].index_add_(0, tok, aux["dd"][l][msk].float())
    tab = [(o / cnt.clamp_min(1)[:, None]) > 0.5 for o in op]          # predicted open
    acc, prec, rec, prate, rate_ = [], [], [], [], []
    for l in range(L):
        tp = fp = fn = tn = 0.0                                           # positive = standing
        for _, idx, _, aux in run_rows(model, rows[:n_rows], collect=True):
            msk = idx != 0
            true = aux["dd"][l][msk] == 0
            p = ~tab[l][idx[msk]]
            tp += float((true & p).sum()); fp += float((~true & p).sum())
            fn += float((true & ~p).sum()); tn += float((~true & ~p).sum())
        tot_ = tp + fp + fn + tn
        acc.append((tp + tn) / tot_); prec.append(tp / max(tp + fp, 1)); rec.append(tp / max(tp + fn, 1))
        rate_.append((tp + fn) / tot_); prate.append((tp + fp) / tot_)
    pred = dict(acc_layer=acc, precision_standing=prec, recall_standing=rec, standing_rate=rate_,
                predicted_standing_rate=prate)
    cur = {}

    def gate_fn(blk, m, zg):                     # QG: decisions from the table
        return tab[blk.l][cur["idx"]].float()

    def upd_fn(blk, upd, x):                     # nogate: predicted-standing groups get no update
        stand = (~tab[blk.l][cur["idx"]]).repeat_interleave(gs, dim=-1)
        return torch.where(stand, torch.zeros_like(upd), upd)

    def eval_ctl(ctl):
        ok_all, qm_all, tot_, n_ = [], [], 0.0, 0
        for i in range(0, 2500, 128):
            idx = rows[i:i + 128]
            cur["idx"] = idx
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(idx, ctl=ctl)
            lsm = F.log_softmax(logits.float()[:, :-1], -1)
            sel = te["copy"][i:i + idx.shape[0], 1:] > 0
            ok_all.append((lsm.argmax(-1) == idx[:, 1:])[torch.tensor(sel, device=DEV)].cpu().numpy())
            qm_all.append(te["qm"][i:i + idx.shape[0], 1:][sel])
        for i in range(0, vrows.shape[0], 128):
            idx = vrows[i:i + 128]
            cur["idx"] = idx
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(idx, ctl=ctl)
            tgt = idx[:, 1:]
            mm_ = tgt != 0
            tot_ += float(F.cross_entropy(logits.float()[:, :-1][mm_], tgt[mm_], reduction="sum"))
            n_ += int(mm_.sum())
        ok, qmv = np.concatenate(ok_all), np.concatenate(qm_all)
        return dict(acc=float(ok.mean()), acc_by_m=by_bins(ok, qmv, range(7)), val_ce=tot_ / n_)

    pred["base"] = eval_ctl(None)
    pred["skip"] = eval_ctl({"hc_gate_fn": gate_fn} if cfg.qg == "gate" else {"q_upd_fn": upd_fn})
    res["table_prediction"] = pred

    # ------------------------------------------- skippable compute
    ctx = float((te["pos"][te["tok"] != 0] + 1).mean())
    fs = 1.0 - st["paid"]
    fnone = float(np.mean(st["token_none_open_layer"]))
    res["skip"] = dict(post=qg_flops(cfg, ctx, fs, fnone),
                       table=qg_flops(cfg, ctx, float(np.mean(prate)), 0.0),
                       frac_group_standing=fs, frac_token_none_open=fnone)

    # ------------------------------------------- checkpoint trajectory of the decision placement
    traj, prev = [], None
    for ck in sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))):
        c = torch.load(ck, map_location="cpu")
        mdl = Model(Cfg(**c["cfg"]))
        mdl.load_state_dict(c["state"])
        mdl = mdl.to(DEV).eval()
        s, dec = qg_stats(mdl, cfg, te, rows, n_rows=600, with_answer=True)
        fd = final_dec[:dec.shape[0]]
        traj.append(dict(step=int(c["step"]), paid_layer=s["paid_layer"], real_layer=s["real_layer"],
                         agree_final=float((dec == fd).float().mean()),
                         agree_prev=None if prev is None else float((dec == prev).float().mean()),
                         answer_shifts_by_m=s["answer_shifts_by_m"]))
        prev = dec
        print(f"  ckpt {traj[-1]['step']}: paid {np.round(s['paid_layer'], 2).tolist()} agree_final "
              f"{traj[-1]['agree_final']:.3f}", flush=True)
    res["ckpt_traj"] = traj
    return res


# ====================================================================================== K-STE
@torch.no_grad()
def kste_stats(model, cfg, te, rows, n_rows=1500, with_answer=True):
    L, G = cfg.n_layers, cfg.depth_groups
    gs = cfg.d // G
    shift = np.zeros(L)
    none_shift = np.zeros(L)
    rate = np.zeros((L, G))
    ntok = 0
    max_stand = 0.0
    a_s, a_m, a_ok = [], [], []
    decisions = []
    for i, idx, logits, aux in run_rows(model, rows[:n_rows], collect=True):
        B = idx.shape[0]
        msk = idx != 0
        ntok += int(msk.sum())
        dec = torch.stack(aux["dd"], 1)                       # B, L, T, G
        decisions.append(dec.transpose(1, 2)[msk].bool().cpu())  # n, L, G
        for l in range(L):
            mm = aux["dd"][l][msk]
            shift[l] += float(mm.sum())
            none_shift[l] += float((mm.sum(-1) == 0).float().sum())
            rate[l] += mm.sum(0).cpu().numpy()
            x0 = aux["xs"][l].view(B, -1, G, gs)[msk]
            x1 = aux["xs"][l + 1].view(B, -1, G, gs)[msk]
            st = mm == 0
            if st.any():
                max_stand = max(max_stand, float((x1 - x0)[st].abs().max()))
        if with_answer:
            aq = torch.tensor(te["ansq"][i:i + B] == 1, device=DEV)
            if aq.any():
                ok = answer_ok(logits, idx, te, i)
                okn = torch.zeros_like(ok)
                okn[:, :-1] = ok[:, 1:]
                tot = torch.stack(aux["dd"]).sum(0).sum(-1)
                a_s.append(tot[aq].cpu().numpy())
                a_m.append(te["qm"][i:i + B][te["ansq"][i:i + B] == 1])
                a_ok.append(okn[aq].cpu().numpy())
    r = rate / ntok
    out = dict(shift_frac_layer=(shift / (ntok * G)).tolist(), shift_frac=float(shift.sum() / (ntok * G * L)),
               token_no_shift_layer=(none_shift / ntok).tolist(),
               always_shift_frac_layer=(r > 0.99).mean(1).tolist(), never_shift_frac_layer=(r < 0.01).mean(1).tolist(),
               group_rate=r.tolist(), max_abs_change_of_standing_group=max_stand)
    if with_answer:
        s, m, ok = np.concatenate(a_s), np.concatenate(a_m), np.concatenate(a_ok)
        out["answer_shifts_by_m"] = by_bins(s, m, range(7))
        out["auroc_wrong"] = dict(all=auroc(s, ~ok), by_m={int(k): auroc(s[m == k], ~ok[m == k]) for k in range(7)})
        out["n_wrong_by_m"] = {int(k): int((~ok)[m == k].sum()) for k in range(7)}
    return out, torch.cat(decisions)


@torch.no_grad()
def eval_kste(model, cfg, te, rows, vrows, run):
    res = {}
    st, final_dec = kste_stats(model, cfg, te, rows)
    res["stats"] = st
    # train == eval (deterministic gates)
    idx = rows[:64]
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lt, _ = model(idx)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        le, _ = model(idx)
    res["train_eval_max_abs_logit_diff"] = float((lt.float() - le.float()).abs().max())
    res["mu_final"] = model.mu.tolist()
    # checkpoint trajectory of the shift placement
    traj = []
    prev = None
    for ck in sorted(glob.glob(os.path.join(run, "ckpt_*.pt"))):
        c = torch.load(ck, map_location="cpu")
        mdl = Model(Cfg(**c["cfg"]))
        mdl.load_state_dict(c["state"])
        mdl = mdl.to(DEV).eval()
        s, dec = kste_stats(mdl, cfg, te, rows, n_rows=600, with_answer=True)
        fd = final_dec[:dec.shape[0]]
        rec = dict(step=int(c["step"]), shift_frac_layer=s["shift_frac_layer"],
                   agree_final=float((dec == fd).float().mean()),
                   agree_final_layer=(dec == fd).float().mean((0, 2)).tolist(),
                   agree_prev=None if prev is None else float((dec == prev).float().mean()),
                   answer_shifts_by_m=s["answer_shifts_by_m"],
                   acc=answer_eval(mdl, rows, te, n_rows=1200)["acc_by_m"])
        prev = dec
        traj.append(rec)
        print(f"  ckpt {rec['step']}: agree_final {rec['agree_final']:.3f} agree_prev {rec['agree_prev']} "
              f"shifts@ans {[round(v, 1) for v in rec['answer_shifts_by_m'].values()]}", flush=True)
    res["ckpt_traj"] = traj
    return res


# ====================================================================================== free coordinates
@torch.no_grad()
def eval_free(model, cfg, te, rows, vrows, n_rows=2500):
    L, d, G = cfg.n_layers, cfg.d, cfg.norm_groups
    gs, f = d // G, cfg.free_per_group
    free_idx = np.array([g * gs + gs - f + j for g in range(G) for j in range(f)])
    norm_idx = np.array([i for i in range(d) if i not in set(free_idx)])
    sofar = swaps_so_far(te)
    Xq = {l: [] for l in range(1, L + 1)}
    keys_q = dict(qm=[], nk=[], pos=[], qi=[])
    Xs = {l: [] for l in range(1, L + 1)}
    keys_s = dict(sofar=[], pos=[], nk=[])
    rms = np.zeros((L + 1, 2))
    mx = np.zeros(L + 1)
    ntok = 0
    for i, idx, logits, aux in run_rows(model, rows[:n_rows], collect=True):
        B = idx.shape[0]
        msk = (idx != 0).cpu().numpy()
        aq = te["ansq"][i:i + B] == 1
        story = (te["part"][i:i + B] == 2) & msk
        ntok += int(msk.sum())
        for l, x in enumerate(aux["xs"]):
            xn = x.float().cpu().numpy()
            xm = xn[msk]
            rms[l, 0] += float((xm[:, free_idx] ** 2).sum())
            rms[l, 1] += float((xm[:, norm_idx] ** 2).sum())
            mx[l] = max(mx[l], float(np.abs(xm[:, free_idx]).max()))
            if l >= 1:
                Xq[l].append(xn[aq])
                Xs[l].append(xn[story][::4])
        for k in keys_q:
            keys_q[k].append(te[k][i:i + B][aq])
        keys_s["sofar"].append(sofar[i:i + B][story][::4])
        keys_s["pos"].append(te["pos"][i:i + B][story][::4])
        keys_s["nk"].append(te["nk"][i:i + B][story][::4])
    res = dict(free_idx=free_idx.tolist())
    res["rms_free_layer"] = np.sqrt(rms[:, 0] / (ntok * len(free_idx))).tolist()
    res["rms_norm_layer"] = np.sqrt(rms[:, 1] / (ntok * len(norm_idx))).tolist()
    res["max_abs_free_layer"] = mx.tolist()
    kq = {k: np.concatenate(v).astype(np.float64) for k, v in keys_q.items()}
    ks = {k: np.concatenate(v).astype(np.float64) for k, v in keys_s.items()}

    def corr(X, y):
        X = X - X.mean(0)
        y = y - y.mean()
        den = np.sqrt((X ** 2).sum(0) * (y ** 2).sum()) + 1e-12
        return (X * y[:, None]).sum(0) / den

    def probe_r2(X, y, seed=0):
        """Ridge regression R^2 on a 70/30 split."""
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(y))
        tr, tst = p[:int(0.7 * len(y))], p[int(0.7 * len(y)):]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd
        A1 = np.c_[A, np.ones(len(tr))]
        w = np.linalg.solve(A1.T @ A1 + 1e-2 * np.eye(A1.shape[1]), A1.T @ y[tr])
        pr = np.c_[(X[tst] - mu) / sd, np.ones(len(tst))] @ w
        return float(1 - ((y[tst] - pr) ** 2).sum() / ((y[tst] - y[tst].mean()) ** 2).sum())

    rng = np.random.default_rng(1)
    corr_q, corr_s, probes = {}, {}, {}
    for l in range(1, L + 1):
        X = np.concatenate(Xq[l]).astype(np.float64)
        Xst = np.concatenate(Xs[l]).astype(np.float64)
        corr_q[l] = {k: corr(X[:, free_idx], v).tolist() for k, v in kq.items()}
        corr_q[l]["norm_max_abs"] = {k: float(np.abs(corr(X[:, norm_idx], v)).max()) for k, v in kq.items()}
        corr_s[l] = {k: corr(Xst[:, free_idx], v).tolist() for k, v in ks.items()}
        corr_s[l]["norm_max_abs"] = {k: float(np.abs(corr(Xst[:, norm_idx], v)).max()) for k, v in ks.items()}
        rand16 = [rng.choice(norm_idx, len(free_idx), replace=False) for _ in range(3)]
        probes[l] = dict(
            m_free=probe_r2(X[:, free_idx], kq["qm"]),
            m_norm16=float(np.mean([probe_r2(X[:, r], kq["qm"]) for r in rand16])),
            m_all=probe_r2(X, kq["qm"]),
            sofar_free=probe_r2(Xst[:, free_idx], ks["sofar"]),
            sofar_norm16=float(np.mean([probe_r2(Xst[:, r], ks["sofar"]) for r in rand16])),
            pos_free=probe_r2(Xst[:, free_idx], ks["pos"]),
            pos_norm16=float(np.mean([probe_r2(Xst[:, r], ks["pos"]) for r in rand16])),
            nk_free=probe_r2(X[:, free_idx], kq["nk"]),
            nk_norm16=float(np.mean([probe_r2(X[:, r], kq["nk"]) for r in rand16])))
    res["corr_question"] = corr_q
    res["corr_story"] = corr_s
    res["probe_r2"] = probes
    # ablation: free coordinates receive no update (stay at their embedding value)
    fmask = torch.zeros(d, dtype=torch.bool, device=DEV)
    fmask[torch.tensor(free_idx, device=DEV)] = True

    def free_fn(blk, t):
        return torch.where(fmask, torch.zeros_like(t), t)
    base = answer_eval(model, rows, te, n_rows=2500)
    base["val_ce"] = ce_of(model, vrows)
    abl = answer_eval(model, rows, te, n_rows=2500, ctl={"free_fn": free_fn})
    abl["val_ce"] = ce_of(model, vrows, ctl={"free_fn": free_fn})
    res["ablate_free_updates"] = dict(base=base, ablated=abl)
    return res


# ====================================================================================== sparse coordinates
@torch.no_grad()
def sparse_coords(model, cfg, rows, n_rows=800):
    """Share of coordinates whose median |x_i| < 0.1 * mean |x_i| (most tokens near zero), per block input;
    and (I') the share of coordinates where the cost floor is active."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, aux = model(rows[:n_rows], collect=True)
    m = rows[:n_rows] != 0
    out = dict(sparse_layer=[], near0_majority_layer=[])
    for l in range(cfg.n_layers):
        ax = aux["xs"][l].float()[m].abs()
        med, mean = ax.median(0).values, ax.mean(0)
        rms = ax.pow(2).mean(0).sqrt()
        out["sparse_layer"].append(float((med < 0.1 * mean).float().mean()))
        out["near0_majority_layer"].append(float(((ax < 0.05 * rms).float().mean(0) > 0.5).float().mean()))
    blk0 = model.blocks[0]
    if hasattr(blk0, "s_med"):
        out["floor_active_layer"] = [float((b.s_med < cfg.med_floor * b.s_mabs).float().mean()) for b in model.blocks]
    return out


def main(run, data="data5"):
    model, cfg = load_run(run)
    meta = json.load(open(os.path.join(data, "meta.json")))
    V = len(meta["vocab"])
    te, va = load_split(data, "test"), load_split(data, "val")
    rows = torch.tensor(te["tok"].astype(np.int64), device=DEV)
    vrows = torch.tensor(va["tok"][:1500].astype(np.int64), device=DEV)
    fam = family(cfg)
    res = dict(family=fam, sparse=sparse_coords(model, cfg, rows))
    if fam == "QG":
        res["qg"] = eval_qg(model, cfg, te, rows, vrows, data, V, run)
        s = res["qg"]["stats"]
        print("paid", np.round(s["paid_layer"], 3).tolist(), "real", np.round(s["real_layer"], 3).tolist(),
              "wasted/paid", np.round(s["wasted_of_paid_layer"], 3).tolist(), "| max change standing group",
              s["max_abs_change_of_standing_group"])
        print("answer shifts by m", {k: round(v, 2) for k, v in s["answer_shifts_by_m"].items()},
              "AUROC", round(s["auroc_wrong"]["all"], 3), "| edge share", np.round(s["edge_share_layer"], 3).tolist())
        tp = res["qg"]["table_prediction"]
        print("table pred acc", np.round(tp["acc_layer"], 3).tolist(), "substitution acc", tp["base"]["acc"], "->",
              tp["skip"]["acc"], "| skip", res["qg"]["skip"]["post"])
    elif fam == "Q":
        res["q"] = eval_q(model, cfg, te, rows, vrows, data, V)
        q = res["q"]
        print("levels crossed / block", np.round(q["l1_layer"], 3).tolist(), "moved share",
              np.round(q["moved_share_layer"], 3).tolist())
        print("edge share", np.round(q["edge_share_layer"], 3).tolist(), "dead levels", q["dead_levels_layer"],
              "| max change of standing coord", q["max_abs_change_standing_coord"])
        print("answer depth by m", {k: round(v, 3) for k, v in q["answer_depth_by_m"].items()},
              "AUROC", round(q["auroc_wrong"]["all"], 3))
        sp = q["standing_prediction"]
        print("table-skip acc coord/group", sp["coord"]["skip"]["acc"], sp["group"]["skip"]["acc"],
              "base", sp["base"]["acc"])
    elif fam == "KSTE":
        res["kste"] = eval_kste(model, cfg, te, rows, vrows, run)
        s = res["kste"]["stats"]
        print("shift frac", np.round(s["shift_frac_layer"], 3).tolist(), "answer shifts by m",
              {k: round(v, 2) for k, v in s["answer_shifts_by_m"].items()},
              "| train-eval logit diff", res["kste"]["train_eval_max_abs_logit_diff"])
    elif fam == "free":
        res["free"] = eval_free(model, cfg, te, rows, vrows)
        fr = res["free"]
        print("free rms by layer", np.round(fr["rms_free_layer"], 3).tolist(), "max", np.round(fr["max_abs_free_layer"], 2).tolist())
        print("probe R2 (last layer)", {k: round(v, 3) for k, v in fr["probe_r2"][cfg.n_layers].items()})
        ab = fr["ablate_free_updates"]
        print("ablate free updates: acc", ab["base"]["acc"], "->", ab["ablated"]["acc"])
    print("sparse coords", res["sparse"])
    json.dump(res, open(os.path.join(run, "eval_r4.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
